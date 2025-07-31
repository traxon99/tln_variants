#!/usr/bin/env python3
import os
import time
import warnings
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# ROS 2 bag imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

# Keras model imports
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Bidirectional, LSTM, Dense, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#========================================================
# Utility functions (from your first block)
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def read_ros2_bag(bag_path):
    """
    Reads a ROS 2 bag via rosbag2_py and returns
    lidar scans, steering (angular.z), speeds (linear.x), and timestamps.
    """
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []


    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        # Convert nanoseconds to seconds
        t = t_ns * 1e-9

        # if topic == 'scan':
        if topic == 'Lidar': #TLN orig
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::2])
            timestamps.append(t)
        elif topic == 'Ackermann': #TLN orig
        # elif topic == 'drive':
            msg = deserialize_message(serialized_msg, AckermannDriveStamped)
 
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            
            servo_data.append(data)

            speed_data.append(s_data)
        # elif topic == 'odom':
        #     msg = deserialize_message(serialized_msg, Odometry)
        #     servo_data.append(msg.twist.twist.angular.z)
        #     speed_data.append(msg.twist.twist.linear.x)
        #     # align timestamp for control measurements too
        #     # (you can choose to append t here or ignore if you only need LIDAR dt)
        #     # timestamps.append(t)

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(timestamps)
    )

    ## Data Imbalance Stuff

    # #make all data lists into np arrays

    # lidar_data_arr = np.array(lidar_data)
    # servo_data_arr = np.array(servo_data)
    # speed_data_arr = np.array(speed_data)
    # timestamps_arr = np.array(timestamps)
    # print(np.flip(lidar_data_arr).shape)
    # # adding flipped data to balance out right and left turn differences
    
    # final_lidar_data = np.concatenate([lidar_data_arr, np.flip(lidar_data_arr, axis=1)], axis=0)    
    # final_servo_data = np.concatenate([servo_data_arr, np.negative(servo_data_arr)], axis=0)
    # final_speed_data = np.concatenate([speed_data_arr, speed_data_arr], axis=0)
    # final_timestamps = np.concatenate([timestamps_arr, timestamps_arr], axis=0)
    # print(final_lidar_data.shape)
    # print(final_servo_data.shape)
    # print(final_speed_data.shape)
    # print(final_timestamps.shape)

    # return (
    #     final_lidar_data,
    #     final_servo_data,
    #     final_speed_data,
    #     final_timestamps
    # )

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
    all_speed = linear_map(all_speed, min_s, max_s, 0, 1)

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