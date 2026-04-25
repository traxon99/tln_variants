"""
Shared utilities for F1TENTH model training scripts.

Import from here instead of copy-pasting across training scripts.
All functions are pure Python/NumPy/TF — no ROS2 required at import time;
rosbag2_py is imported lazily inside read_ros2_bag() so the rest of this
module remains importable in a standard Python environment.
"""
import os
import time
import csv

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as sk_shuffle


# ---- Data Loading: CSV ----

def load_csv_dataset(paths, downscale_factor=2, column_format='speed_servo_extra_lidar'):
    """
    Load F1TENTH demonstration data from CSV files.

    Args:
        paths: List of CSV file paths.
        downscale_factor: Stride for LiDAR downsampling (default 2).
        column_format: 'speed_servo_extra_lidar' (columns: speed, servo, extra, lidar)
                       or 'speed_servo_lidar'    (columns: speed, servo, lidar)

    Returns:
        lidar (np.ndarray), servo (np.ndarray), speed (np.ndarray), max_speed (float)
    """
    lidar_data, servo_data, speed_data = [], [], []
    max_speed = 0.0
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Dataset not found: {path}')
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                spd = float(row[0])
                srv = float(row[1])
                if column_format == 'speed_servo_lidar':
                    raw_lidar = row[2]
                else:
                    raw_lidar = row[3]
                lidar = list(map(float, raw_lidar.translate(str.maketrans('', '', '[]')).split(',')))
                if spd > max_speed:
                    max_speed = spd
                speed_data.append(spd)
                servo_data.append(srv)
                lidar_data.append(lidar[::downscale_factor])
    # Drop last sample (partial scan guard)
    return (
        np.array(lidar_data[:-1]),
        np.array(servo_data[:-1]),
        np.array(speed_data[:-1]),
        max_speed,
    )


# ---- Data Loading: ROS2 Bags ----

def read_ros2_bag(bag_path, lidar_topic=None, drive_topic=None, downscale_factor=2):
    """
    Read a ROS2 SQLite3 bag file and extract synchronized LiDAR/drive data.

    Args:
        bag_path: Path to .db3 bag file.
        lidar_topic: Topic name for LaserScan (None = auto-detect 'Lidar' or 'scan').
        drive_topic: Topic name for AckermannDriveStamped (None = auto-detect).
        downscale_factor: Stride for LiDAR downsampling.

    Returns:
        lidar (np.ndarray), servo (np.ndarray), speed (np.ndarray), timestamps (np.ndarray)
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan
    from ackermann_msgs.msg import AckermannDriveStamped

    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_topics = {lidar_topic} if lidar_topic else {'Lidar', 'scan'}
    drive_topics = {drive_topic} if drive_topic else {'Ackermann', 'drive'}

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if topic in lidar_topics:
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(np.array(msg.ranges), posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::downscale_factor])
            timestamps.append(t)
        elif topic in drive_topics:
            msg = deserialize_message(serialized_msg, AckermannDriveStamped)
            servo_data.append(msg.drive.steering_angle)
            speed_data.append(msg.drive.speed)

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(timestamps),
    )


# ---- Normalization ----

def normalize_speed(speed_data, max_speed=None, min_speed=0.0):
    """
    Normalize speed to [0, 1].

    Args:
        speed_data: np.ndarray of raw speed values.
        max_speed: Upper bound (inferred from data if None).
        min_speed: Lower bound (default 0).

    Returns:
        (normalized np.ndarray, max_speed float)
    """
    if max_speed is None:
        max_speed = float(speed_data.max())
    if max_speed == min_speed:
        raise ValueError('max_speed == min_speed; cannot normalize speed.')
    return (speed_data - min_speed) / (max_speed - min_speed), max_speed


# ---- Train/Test Split ----

def split_dataset(lidar, servo, speed, train_ratio=0.85, random_state=42):
    """
    Shuffle and split into train/test.

    Returns:
        X_train, X_test, y_train, y_test  (np.ndarray each)
        y columns: [servo, speed]
    """
    labels = np.stack([servo, speed], axis=1)
    lidar_s, labels_s = sk_shuffle(lidar, labels, random_state=random_state)
    n = int(train_ratio * len(lidar_s))
    return lidar_s[:n], lidar_s[n:], labels_s[:n], labels_s[n:]


# ---- Model Architecture ----

def build_tln_model(num_ranges):
    """Standard Conv1D TLN model. Input shape: (num_ranges, 1)."""
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu', input_shape=(num_ranges, 1)),
        tf.keras.layers.Conv1D(36, 8, strides=4, activation='relu'),
        tf.keras.layers.Conv1D(48, 4, strides=2, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh'),
    ])


# ---- Training ----

def train_model(model, X_train, y_train, X_test, y_test,
                epochs=20, batch_size=64, lr=5e-5):
    """Compile (huber loss, Adam), fit, and return history."""
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(lr), loss='huber')
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    print(f'Training done in {int(time.time() - t0)}s')
    return history


# ---- Visualization ----

def plot_loss_curve(history, output_path='./Figures/loss_curve.png'):
    """Save a training/validation loss curve PNG."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f'Loss curve saved to {output_path}')


# ---- TFLite Export ----

def export_tflite(model, output_path, quantize_int8=False, representative_data=None):
    """
    Convert and save a Keras model to TFLite.

    Args:
        model: Trained tf.keras.Model.
        output_path: Destination .tflite file path.
        quantize_int8: Enable post-training int8 quantization.
        representative_data: np.ndarray of representative float32 inputs (required for int8).
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_int8:
        if representative_data is None:
            raise ValueError('representative_data is required for int8 quantization')

        def _rep_gen():
            for x in tf.data.Dataset.from_tensor_slices(
                    representative_data.astype(np.float32)).batch(1):
                yield [x]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model saved to {output_path}')


# ---- Evaluation ----

def huber_loss_np(y_true, y_pred, delta=1.0):
    """NumPy Huber loss for post-training evaluation (not used during training)."""
    error = np.abs(y_true - y_pred)
    return float(np.mean(np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))))


def evaluate_tflite(model_path, test_lidar, test_labels, hz=40):
    """
    Run TFLite inference over the test set and report timing stats.

    Args:
        model_path: Path to .tflite file.
        test_lidar: np.ndarray of shape (N, num_ranges).
        test_labels: np.ndarray of shape (N, 2) — [servo, speed].
        hz: Target control rate for deadline checking.

    Returns:
        y_pred (np.ndarray shape (N, 2)), inference_times_us (list of floats)
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()
    period = 1.0 / hz

    preds, times_us = [], []
    for row in test_lidar:
        inp = np.expand_dims(np.expand_dims(row, -1), 0).astype(np.float32)
        t0 = time.time()
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        dur = time.time() - t0
        times_us.append(dur * 1e6)
        if dur > period:
            print(f'Deadline miss: {dur * 1000:.2f} ms')
        preds.append(out[0])

    arr = np.array(times_us)
    p99 = np.percentile(arr, 99)
    trimmed = arr[arr < p99]
    print(f'Model: {model_path}')
    print(f'Avg inference: {np.mean(trimmed):.2f} us, Max: {np.max(trimmed):.2f} us')
    return np.array(preds), times_us
