#!/usr/bin/env python3
"""Temporal LiDAR Navigation (TLN) training script.

Each sample is a triplet of LiDAR scans ordered oldest→newest:
    [scan(t-offset2), scan(t-offset1), scan(t)]  →  [steering(t), speed(t)]

Key design choices
------------------
* Commands are interpolated onto LiDAR timestamps (np.interp) so that every
  scan is paired with the command that was actually issued at that time,
  regardless of topic publish rates.  Without this, pairing the i-th scan with
  the i-th command (different topics, different Hz) produces severe temporal
  misalignment that causes the trained model to react late.
* Dataset augmentation: horizontal mirror (reverse scan, negate steering) then
  turn/straight undersampling to prevent the model from learning "go straight".
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from tln_variants.utils import read_ros2_bag

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Input bags
    bag_paths: List[str] = field(default_factory=lambda: [
        'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
        'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
        'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3',
    ])

    # Output paths
    model_dir:  str = './Models'
    fig_dir:    str = './Figures'
    model_name: str = 'TLN_temporal_compact'

    # Training hyper-parameters
    lr:          float = 5e-5
    batch_size:  int   = 64
    epochs:      int   = 20
    train_ratio: float = 0.85
    patience:    int   = 5

    # LiDAR pre-processing
    lidar_downsample: int = 2   # keep every Nth range value

    # Temporal context (frames back at ~40 Hz)
    offset1: int = 30           # ~0.75 s ago
    offset2: int = 60           # ~1.50 s ago

    # Dataset balancing
    turn_threshold: float = 0.05  # |steering| above this → "turning" sample


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bags(
    bag_paths:  List[str],
    downsample: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate multiple bags."""
    lidar_list, steer_list, speed_list = [], [], []
    for path in bag_paths:
        lidar, steer, speed, _ = read_ros2_bag(path, downsample)
        print(f'  {len(lidar):>6,} scans  ←  {path}')
        lidar_list.append(lidar)
        steer_list.append(steer)
        speed_list.append(speed)
    return (np.concatenate(lidar_list, axis=0),
            np.concatenate(steer_list, axis=0),
            np.concatenate(speed_list, axis=0))


# ── Temporal triplets ─────────────────────────────────────────────────────────

def build_temporal_triplets(
    lidar:   np.ndarray,  # (N, L)
    labels:  np.ndarray,  # (N, 2)
    offset1: int,
    offset2: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack 3-frame sequences ordered oldest → newest.

    For timestep t:
        X[t] = [scan(t-offset2), scan(t-offset1), scan(t)]
        Y[t] = labels(t)

    Shape: X (M, 3, L, 1),  Y (M, 2)
    """
    t_idx = np.arange(offset2, len(lidar))
    X = np.stack([
        lidar[t_idx - offset2],   # oldest context
        lidar[t_idx - offset1],   # mid context
        lidar[t_idx],             # current scan
    ], axis=1)                    # (M, 3, L)
    X = np.expand_dims(X, axis=-1)  # (M, 3, L, 1)
    Y = labels[t_idx]               # (M, 2)
    return X.astype(np.float32), Y.astype(np.float32)


# ── Dataset augmentation ──────────────────────────────────────────────────────

def mirror_dataset(
    X: np.ndarray,  # (N, 3, L, 1)
    Y: np.ndarray,  # (N, 2) – [steering, speed]
) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal mirror augmentation: reverse scan order, negate steering.

    Reversing the LiDAR array is equivalent to reflecting the scene left↔right.
    The correct response in the mirrored scene is the negated steering angle;
    speed is unchanged.  Returns original + mirrored samples concatenated.
    """
    X_mirror      = X[:, :, ::-1, :]   # flip scan left↔right across all 3 frames
    Y_mirror      = Y.copy()
    Y_mirror[:, 0] *= -1               # negate steering

    return (np.concatenate([X, X_mirror], axis=0),
            np.concatenate([Y, Y_mirror], axis=0))


def balance_turn_straight(
    X:               np.ndarray,
    Y:               np.ndarray,
    turn_threshold:  float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample the majority class so turns and straights are equal in count.

    Straight: |steering| <= turn_threshold
    Turning:  |steering| >  turn_threshold
    """
    rng = np.random.default_rng(SEED)

    steering      = Y[:, 0]
    turn_mask     = np.abs(steering) > turn_threshold
    n_turn        = int(turn_mask.sum())
    n_straight    = int((~turn_mask).sum())
    target        = min(n_turn, n_straight)

    if target == 0:
        print('  Balance: one class is empty — skipping.')
        return X, Y

    turn_idx     = np.where( turn_mask)[0]
    straight_idx = np.where(~turn_mask)[0]
    sel = np.concatenate([
        rng.choice(turn_idx,     target, replace=False),
        rng.choice(straight_idx, target, replace=False),
    ])
    discarded = len(X) - 2 * target
    print(f'  Balance: {n_turn} turn + {n_straight} straight '
          f'→ {target} each  ({discarded} discarded)')
    return X[sel], Y[sel]


# ── Speed scaling ─────────────────────────────────────────────────────────────

def scale_speed(
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Normalise the speed column (index 1) to [0, 1] using training stats."""
    s_min  = float(y_train[:, 1].min())
    s_max  = float(y_train[:, 1].max())
    denom  = (s_max - s_min) or 1.0

    def _scale(y: np.ndarray) -> np.ndarray:
        out        = y.copy()
        out[:, 1]  = (y[:, 1] - s_min) / denom
        return out

    return _scale(y_train), _scale(y_test), s_min, s_max


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """2-D Conv model operating over a (time_steps=3, lidar_length, 1) input."""
    return tf.keras.Sequential([
        # Per-scan feature extraction (kernel height=1 → independent per timestep)
        tf.keras.layers.Conv2D(24, (1, 10), strides=(1, 4), activation='relu',
                               input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(36, (1, 5), strides=(1, 2), activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(48, (1, 5), strides=(1, 2), activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # Cross-timestep fusion (kernel height=3 spans all 3 frames)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (1, 3), activation='relu'),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh'),
    ], name='TLN_temporal')


# ── Evaluation helpers ────────────────────────────────────────────────────────

def huber_loss_np(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    err = np.abs(y_true - y_pred)
    return float(np.mean(np.where(err <= delta, 0.5 * err**2,
                                  delta * (err - 0.5 * delta))))


def save_loss_curve(history: tf.keras.callbacks.History, path: str) -> None:
    plt.figure()
    plt.plot(history.history['loss'],     label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Loss curve  →  {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir,   exist_ok=True)

    print('GPU available:', bool(tf.config.list_physical_devices('GPU')))

    # ── Load ─────────────────────────────────────────────────────────────────
    print('\nLoading bags …')
    lidar, steer, speed = load_bags(cfg.bag_paths, downsample=cfg.lidar_downsample)
    print(f'Total: {len(lidar):,} scans  |  scan length L={lidar.shape[1]}')

    labels = np.stack([steer, speed], axis=1)   # (N, 2)

    # ── Train / test split (preserve temporal order for triplet construction) ─
    N_train          = int(cfg.train_ratio * len(lidar))
    x_tr, y_tr_raw   = lidar[:N_train],  labels[:N_train]
    x_te, y_te_raw   = lidar[N_train:],  labels[N_train:]

    y_tr, y_te, s_min, s_max = scale_speed(y_tr_raw, y_te_raw)
    print(f'Speed range (train): [{s_min:.4f}, {s_max:.4f}]')

    # ── Build temporal triplets ───────────────────────────────────────────────
    print('\nBuilding temporal triplets …')
    X_tr, Y_tr = build_temporal_triplets(x_tr, y_tr, cfg.offset1, cfg.offset2)
    X_te, Y_te = build_temporal_triplets(x_te, y_te, cfg.offset1, cfg.offset2)
    print(f'  Train: {X_tr.shape}   Test: {X_te.shape}')

    # ── Augmentation (training only) ─────────────────────────────────────────
    print('\nApplying mirror augmentation …')
    X_tr, Y_tr = mirror_dataset(X_tr, Y_tr)
    print(f'  After mirror: {X_tr.shape}')

    print('\nBalancing turn / straight …')
    X_tr, Y_tr = balance_turn_straight(X_tr, Y_tr, cfg.turn_threshold)

    X_tr, Y_tr = shuffle(X_tr, Y_tr, random_state=SEED)
    X_te, Y_te = shuffle(X_te, Y_te, random_state=SEED)
    print(f'\nFinal  train: {X_tr.shape}   test: {X_te.shape}')

    # ── Build & compile ───────────────────────────────────────────────────────
    model = build_model(input_shape=X_tr.shape[1:])   # (3, L, 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=tf.keras.losses.Huber(),
    )
    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        X_tr, Y_tr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_te, Y_te),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg.patience,
            restore_best_weights=True,
        )],
        verbose=1,
    )
    print(f'\nTraining time: {int(time.time() - t0)} s')

    save_loss_curve(history, os.path.join(cfg.fig_dir, 'loss_curve.png'))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('\n── Evaluation ───────────────────────────────────────────────────')
    print(f'Test loss (Keras Huber): {model.evaluate(X_te, Y_te, verbose=0):.6f}')

    Y_pred = model.predict(X_te, verbose=0)
    print(f'Steering Huber: {huber_loss_np(Y_te[:, 0], Y_pred[:, 0]):.6f}')
    print(f'Speed    Huber: {huber_loss_np(Y_te[:, 1], Y_pred[:, 1]):.6f}')

    # ── Export TFLite ─────────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops  = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_path = os.path.join(cfg.model_dir, f'{cfg.model_name}_noquantized.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(converter.convert())
    print(f'\nSaved TFLite model  →  {tflite_path}')


if __name__ == '__main__':
    main()
