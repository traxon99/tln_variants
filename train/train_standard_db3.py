#!/usr/bin/env python3
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# ROS 2 bag imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

#========================================================
# Utility functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def read_ros2_bag(bag_path):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        t = t_ns * 1e-9

        if topic == 'scan' or topic == 'Lidar':
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::2])
            timestamps.append(t)
        elif topic == 'drive' or topic == 'Ackermann':
            msg = deserialize_message(serialized_msg, AckermannDriveStamped)
            servo_data.append(msg.drive.steering_angle)
            speed_data.append(msg.drive.speed)
        # elif topic == 'odom':
        #     msg = deserialize_message(serialized_msg, Odometry)
        #     servo_data.append(msg.twist.twist.angular.z)
        #     speed_data.append(msg.twist.twist.linear.x)

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(timestamps)
    )


#========================================================
# Main
#========================================================
if __name__ == '__main__':
    print('GPU AVAILABLE:', bool(tf.config.list_physical_devices('GPU')))

    # Parameters
    bag_paths = [
        # 'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
        # # 'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
        # # 'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3'
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/forza_pf_map/forza_pf_map_0.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP/Forza_GLC_smile_PP_0.db3', #evil
        '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3', #could be bad... or good
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_glc_ot_ez_3laps/Forza_glc_ot_ez_3laps_0.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_small_3laps/Forza_GLC_smile_small_3laps_0.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_hangar_1905_v0_1lap/Forza_hangar_1905_v0_1lap_0.db3',
        '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr1db3/jfr1.db3',
        '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr2db3/jfr2.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map/test_map.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/out/out.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f2/f2.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f4/f4.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map_opp/test_map_opp.db3',
        '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv5_opp/jfrv5_opp.db3',
        '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv6_opp/jfrv6_opp.db3',
        # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map_obstacles_good/test_map_obstacles_good.db3'
    ]
    batch_size = 64
    lr = 5e-5
    num_epochs = 20
    model_name = 'lidar_imitation_model'
    loss_figure_path = f'./Models/{model_name}_loss.png'

    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for pth in bag_paths:
        l, s, sp, ts = read_ros2_bag(pth)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)
        all_ts.extend(ts)

    lidar = np.array(all_lidar)
    servo = np.array(all_servo)
    speed = np.array(all_speed)

    speed = linear_map(speed, speed.min(), speed.max(), 0, 1)

    # Shuffle data
    lidar, servo, speed = shuffle(lidar, servo, speed, random_state=42)

    # Train/test split
    lidar_train, lidar_test, servo_train, servo_test, speed_train, speed_test = train_test_split(
        lidar, servo, speed, test_size=0.2, random_state=42
    )

    train_data = np.stack((servo_train, speed_train), axis=-1)
    test_data = np.stack((servo_test, speed_test), axis=-1)

    # Reshape input
    lidar_train = lidar_train[..., np.newaxis]
    lidar_test = lidar_test[..., np.newaxis]

    # Model
    num_lidar_range_values = lidar.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
        tf.keras.layers.Conv1D(36, 8, strides=4, activation='relu'),
        tf.keras.layers.Conv1D(48, 4, strides=2, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])

    # Compile
    loss_function = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer, loss=loss_function)
    model.summary()

    # Train
    start_time = time.time()
    history = model.fit(
        lidar_train, train_data,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(lidar_test, test_data)
    )
    print(f'Training Time: {int(time.time() - start_time)} seconds')

    # Plot Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    os.makedirs(os.path.dirname(loss_figure_path), exist_ok=True)
    plt.savefig(loss_figure_path)
    plt.close()

    # Evaluate
    print("\nModel Evaluation")
    test_loss = model.evaluate(lidar_test, test_data)
    print(f'Overall Test Loss: {test_loss:.4f}')

    predictions = model.predict(lidar_test)
    huber = tf.keras.losses.Huber()
    print(f"Servo Test Huber Loss: {huber(test_data[:, 0], predictions[:, 0]).numpy():.4f}")
    print(f"Speed Test Huber Loss: {huber(test_data[:, 1], predictions[:, 1]).numpy():.4f}")

    # Save Model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    os.makedirs('./Models', exist_ok=True)
    with open(f'./Models/{model_name}_noquantized.tflite', 'wb') as f:
        f.write(tflite_model)
        print(f"{model_name}_noquantized.tflite saved.")