#!/usr/bin/env python3
import os
import time
import csv
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, TimeDistributed, Conv1D,
                                     MaxPooling1D, Flatten, LSTM, Dense, Dropout)
from tensorflow.keras.optimizers import Adam

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
# from nav_msgs.msg import Odometry  # optional

# ========================================================
# Config / Reproducibility
# ========================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Uncomment to force CPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
print('GPU AVAILABLE:', gpu_available)

# Folders
MODEL_DIR = "./Models"
FIG_DIR = "./Figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Training params
MODEL_NAME = "TLN_temporal_compact"
LOSS_FIG_PATH = os.path.join(FIG_DIR, "loss_curve.png")
LR = 5e-5
BATCH_SIZE = 64
EPOCHS = 20
HZ = 40               # used only by (optional) TFLite eval
DOWNSAMPLE = 2        # lidar downsample factor
# temporal indices: now, ~1.55s, ~3.1s ago at 40 Hz
# (t, t-62, t-125). We’ll build forward indices to keep code simple: (i, i+62, i+125)
OFFSET1 = 62
OFFSET2 = 125

# ========================================================
# Utils
# ========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    """Linear mapping with safe handling when x_min==x_max."""
    denom = (x_max - x_min)
    if denom == 0:
        return np.full_like(x, (y_min + y_max) / 2.0)
    return (x - x_min) / denom * (y_max - y_min) + y_min

def huber_loss_np(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss)

def read_ros2_bag(bag_path, downsample=2):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        t = t_ns * 1e-9

        if topic in ('scan', 'Lidar'):
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(np.array(msg.ranges, dtype=np.float32),
                                    posinf=0.0, neginf=0.0)
            if downsample and downsample > 1:
                cleaned = cleaned[::downsample]
            lidar_data.append(cleaned)
            timestamps.append(t)
        elif topic in ('drive', 'Ackermann'):
            msg = deserialize_message(serialized_msg, AckermannDriveStamped)
            servo_data.append(float(msg.drive.steering_angle))
            speed_data.append(float(msg.drive.speed))
        # elif topic == 'odom':
        #     msg = deserialize_message(serialized_msg, Odometry)
        #     servo_data.append(float(msg.twist.twist.angular.z))
        #     speed_data.append(float(msg.twist.twist.linear.x))

    return np.array(lidar_data, dtype=np.float32), \
           np.array(servo_data, dtype=np.float32), \
           np.array(speed_data, dtype=np.float32), \
           np.array(timestamps, dtype=np.float64)

# If scans can vary in length, pad/trim them to a fixed size:
# def pad_or_trim(arrs, target_len):
#     out = []
#     for a in arrs:
#         if len(a) >= target_len:
#             out.append(a[:target_len])
#         else:
#             pad = np.zeros(target_len, dtype=a.dtype)
#             pad[:len(a)] = a
#             out.append(pad)
#     return np.stack(out, axis=0)

def build_temporal_triplets(x_seq, y_seq, offsets=(OFFSET1, OFFSET2)):
    """
    Build sequences of shape (N', 3, L) aligned to y at t (current frame).
    We construct [x[i], x[i+o1], x[i+o2]] and target y[i+o2] so inputs and targets align.
    """
    o1, o2 = offsets
    N = len(x_seq)
    max_i = N - o2
    X_triplets = []
    y_aligned = []
    for i in range(max_i):
        X_triplets.append([x_seq[i], x_seq[i+o1], x_seq[i+o2]])
        y_aligned.append(y_seq[i+o2])
    X_triplets = np.array(X_triplets, dtype=np.float32)     # (N', 3, L)
    y_aligned = np.array(y_aligned, dtype=np.float32)       # (N', 2)
    # add channel dim for Conv1D: (N', 3, L, 1)
    X_triplets = np.expand_dims(X_triplets, axis=-1)
    return X_triplets, y_aligned

# ========================================================
# Load Dataset (bags)
# ========================================================
bag_paths = [
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_1lap/test_map_1lap.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_non_obstr/test_map_non_obstr.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_obstr/test_map_obstr.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr1db3/jfr1.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr2db3/jfr2.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv5_opp/jfrv5_opp.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv6_opp/jfrv6_opp.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv6_nonobstr_edited/JFRv6_nonobstr_edited.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv5_nonobstr_edited/JFRv5_nonobstr_edited.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_non_obstr/test_map_non_obstr.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv6_1lap/JFRv6_1lap.db3',
]

lidar_all, servo_all, speed_all = [], [], []

for p in bag_paths:
    l, s, sp, _ = read_ros2_bag(p, downsample=DOWNSAMPLE)
    print(f'Loaded {len(l)} scans from {p}')
    # Basic sanity: ensure we have same length among l/s/sp if they are sampled differently
    N = min(len(l), len(s), len(sp))
    if N == 0:
        continue
    lidar_all.append(l[:N])
    servo_all.append(s[:N])
    speed_all.append(sp[:N])

if not lidar_all:
    raise RuntimeError("No data loaded. Check bag paths and topics.")

lidar_all = np.concatenate(lidar_all, axis=0)
servo_all = np.concatenate(servo_all, axis=0)
speed_all = np.concatenate(speed_all, axis=0)

# ========================================================
# Shuffle & Split
# ========================================================
# Pack labels (servo, speed) together for shuffling in sync
labels_all = np.stack([servo_all, speed_all], axis=1)  # (N, 2)

L = lidar_all.shape[1]
print(f'num_lidar_range_values: {L}')

lidar_shuf, labels_shuf = shuffle(lidar_all, labels_all, random_state=SEED)

train_ratio = 0.85
N_total = len(lidar_shuf)
N_train = int(train_ratio * N_total)

x_train_1d = lidar_shuf[:N_train]     # (N_train, L)
y_train_raw = labels_shuf[:N_train]   # (N_train, 2)
x_test_1d  = lidar_shuf[N_train:]
y_test_raw = labels_shuf[N_train:]

# ========================================================
# Scale speed to [0,1] using training stats only
# ========================================================
speed_min = np.min(y_train_raw[:,1])
speed_max = np.max(y_train_raw[:,1])
y_train = np.stack([y_train_raw[:,0],
                    linear_map(y_train_raw[:,1], speed_min, speed_max, 0, 1)], axis=1)
y_test  = np.stack([y_test_raw[:,0],
                    linear_map(y_test_raw[:,1], speed_min, speed_max, 0, 1)], axis=1)

print(f'Min_speed (train): {speed_min:.4f}')
print(f'Max_speed (train): {speed_max:.4f}')
print(f'Train shapes: x={x_train_1d.shape}, y={y_train.shape}')
print(f'Test  shapes: x={x_test_1d.shape},  y={y_test.shape}')

# ========================================================
# Build temporal triplets separately for train and test
# ========================================================
X_train, Y_train = build_temporal_triplets(x_train_1d, y_train, offsets=(OFFSET1, OFFSET2))
X_test,  Y_test  = build_temporal_triplets(x_test_1d,  y_test,  offsets=(OFFSET1, OFFSET2))

print(f'X_train: {X_train.shape} (N, 3, {L}, 1)')
print(f'Y_train: {Y_train.shape} (N, 2)')
print(f'X_test : {X_test.shape}')
print(f'Y_test : {Y_test.shape}')

# Final input shape for Keras
input_shape = (3, L, 1)

# ========================================================
# Model (compact Conv1D + stacked LSTMs with dropout)
# ========================================================
model = Sequential(name="tln_temporal_compact")
model.add(BatchNormalization(input_shape=input_shape))

# Spatial feature extractor per frame
model.add(TimeDistributed(Conv1D(16, kernel_size=5, activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Conv1D(32, kernel_size=3, activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))

# Temporal modeling
model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))

# Regression head (servo, speed)
model.add(Dense(2, activation='linear'))

optimizer = Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss='huber')  # same as your setup
model.summary(print_fn=lambda s: print(s))

# ========================================================
# Train
# ========================================================
t0 = time.time()
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, Y_test),
    verbose=1
)
print(f'=============> {int(time.time() - t0)} seconds <=============')

# Plot training/validation loss
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(LOSS_FIG_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved loss curve to {LOSS_FIG_PATH}")

# ========================================================
# Evaluate
# ========================================================
print("\n==========================================")
print("Model Evaluation")
print("==========================================")
test_loss = model.evaluate(X_test, Y_test, verbose=0)
print(f'Overall Test Loss (Keras Huber) = {test_loss:.6f}')

Y_pred = model.predict(X_test, verbose=0)
overall_huber = huber_loss_np(Y_test, Y_pred)
print(f'Overall Huber Loss (numpy): {overall_huber:.6f}')

# Per-output diagnostics
servo_huber = huber_loss_np(Y_test[:, 0], Y_pred[:, 0])
speed_huber = huber_loss_np(Y_test[:, 1], Y_pred[:, 1])
print(f"Servo   Huber: {servo_huber:.6f}")
print(f"Speed   Huber: {speed_huber:.6f}")

# ========================================================
# Save TFLite (default + Select TF Ops to keep LSTM)
# ========================================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
tflite_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_noquantized.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"Saved: {tflite_path}")

# # Optional: INT8 full integer quantization (needs representative dataset)
# rep_ds = tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32)).batch(1)
# def representative_data_gen():
#     for x in rep_ds.take(500):
#         yield [x]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
# quant_model = converter.convert()
# quant_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_int8.tflite")
# with open(quant_path, 'wb') as f:
#     f.write(quant_model)
# print(f"Saved: {quant_path}")

print("End")
