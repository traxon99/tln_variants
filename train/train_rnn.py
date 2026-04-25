#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

sys.path.insert(0, os.path.dirname(__file__))
from train_utils import read_ros2_bag


# Keras model imports
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Bidirectional, LSTM, Dense, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#========================================================
# Sequence builder (unchanged)
#========================================================
def create_lidar_sequences(lidar_data, servo_data, speed_data, timestamps, sequence_length=5):
    """
    Build sliding-window sequences of LiDAR frames with time-delta features
    and corresponding steering/speed targets.
    """
    X, y = [], []
    num_ranges = lidar_data.shape[1]

    for i in range(len(lidar_data) - sequence_length):
        # stack the raw scans [seq_len x num_ranges]
        frames = np.stack(lidar_data[i : i + sequence_length], axis=0)  # (seq_len, num_ranges)

        # compute deltas dt between frames, shape (seq_len, 1)
        dt = np.diff(timestamps[i : i + sequence_length + 1]).reshape(sequence_length, 1)

        # replicate dt across all range bins: becomes (seq_len, num_ranges)
        dt_tiled = np.repeat(dt, num_ranges, axis=1)

        # now stack channels: (seq_len, num_ranges, 2)
        seq = np.concatenate([
            frames[..., None],      # (seq_len, num_ranges, 1)
            dt_tiled[..., None]     # (seq_len, num_ranges, 1)
        ], axis=2)

        X.append(seq)
        y.append([servo_data[i + sequence_length], speed_data[i + sequence_length]])

    return np.array(X), np.array(y)


#========================================================
# Model definition (RNN + Attention)
#========================================================
def build_spatiotemporal_model(seq_len, num_ranges):
    inp = Input(shape=(seq_len, num_ranges, 2), name='lidar_sequence')
    x = TimeDistributed(Conv1D(24, 10, strides=4, activation='relu'))(inp)
    x = TimeDistributed(Conv1D(36, 8, strides=4, activation='relu'))(x)
    x = TimeDistributed(Conv1D(48, 4, strides=2, activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    q = Dense(64)(lstm_out)
    k = Dense(64)(lstm_out)
    v = Dense(64)(lstm_out)
    attn = Attention()([q, v, k])
    context = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(attn)
    # context = tf.keras.layers.GlobalAveragePooling1D()(attn)
    # context = tf.keras.layers.GlobalMaxPooling1D()(attn)
    out = Dense(2, activation='tanh', name='controls')(context)
    return Model(inp, out, name='RNN_Attention_Controller')

#========================================================
# Main
#========================================================
if __name__ == '__main__':
    # Check for GPU
    print('GPU AVAILABLE:', bool(tf.config.list_physical_devices('GPU')))

    # --- Parameters ---
    bag_paths = [
    # 'Dataset/car_Dataset/controller_slow_5min/controller_slow_5min_0.db3',
    # 'Dataset/car_Dataset/controller_slow_10min/controller_slow_10.db3',
    # 'Dataset/car_Dataset/controller_fast_5min/controller_fast_5min_0.db3'
    # 'Dataset/car_Dataset/fast_5min/fast_5min_0.db3',
    # 'Dataset/car_Dataset/fast_10min/fast_10min_0.db3',
    # 'Dataset/car_Dataset/fast_sequential/fast_sequential_0.db3'
    'Dataset/f2/f2.db3',
    'Dataset/f4/f4.db3',
    'Dataset/out/out.db3'
    # 'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3',
    # 'Dataset/5_min_Spiel_sim2/5_min_Spiel_sim2_0.db3',
    # 'Dataset/5_min_Spiel_sim_rvrs/5_min_Spiel_sim_rvrs_0.db3'
    # 'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
    # 'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
    # 'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3'
    ]

    seq_len    = 5
    batch_size = 128
    lr         = 1e-5
    epochs     = 30

    # --- Load & concatenate all bags ---
    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for pth in bag_paths:
        l, s, sp, ts = read_ros2_bag(pth)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)
        all_ts.extend(ts)

    all_lidar = np.array(all_lidar)
    all_servo = np.array(all_servo)
    all_speed = np.array(all_speed)
    all_ts    = np.array(all_ts)

    # Normalize speed 0→1
    min_s, max_s = all_speed.min(), all_speed.max()
    all_speed = (all_speed - min_s) / (max_s - min_s)

    # Build sequences
    X, y = create_lidar_sequences(all_lidar, all_servo, all_speed, all_ts, seq_len)
    n_samples, _, num_ranges, _ = X.shape
    print(f'Total sequences: {n_samples}, ranges per scan: {num_ranges}')

    # Shuffle and split
    X, y = shuffle(X, y, random_state=42)
    split = int(0.85 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build & compile model
    model = build_spatiotemporal_model(seq_len, num_ranges)
    model.compile(optimizer=Adam(lr), loss='huber')
    print(model.summary())

    # Train
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )
    print(f'Training done in {int(time.time() - t0)}s')

    # Plot loss curve
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('Figures/loss_curve.png')
    plt.close()

    # Convert & save TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    os.makedirs('Models', exist_ok=True)
    # with open('Models/RNN_Attn_Controller.tflite', 'wb') as f:
    with open('Models/test.tflite', 'wb') as f:
        f.write(tflite_model)
    print('TFLite model saved.')

    # Final evaluation
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Final test loss: {test_loss:.4f}')