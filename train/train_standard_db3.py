#!/usr/bin/env python3
"""
TLN training from ROS2 db3 bag files.
Usage: python3 train_standard_db3.py
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from train_utils import read_ros2_bag, build_tln_model, export_tflite, huber_loss_np

if __name__ == '__main__':
    print('GPU AVAILABLE:', bool(tf.config.list_physical_devices('GPU')))

    bag_paths = [
        'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
        'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
        'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3',
    ]
    BATCH_SIZE = 64
    LR = 5e-5
    EPOCHS = 10
    MODEL_NAME = 'TLN_db3'
    LOSS_FIG = f'./Models/{MODEL_NAME}_loss.png'

    all_lidar, all_servo, all_speed = [], [], []
    for pth in bag_paths:
        l, s, sp, _ = read_ros2_bag(pth)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)

    lidar = np.array(all_lidar)
    servo = np.array(all_servo)
    speed = np.array(all_speed)

    print(f'Max speed: {speed.max():.2f}')
    speed = (speed - speed.min()) / (speed.max() - speed.min())

    lidar, servo, speed = shuffle(lidar, servo, speed, random_state=42)

    lidar_train, lidar_test, servo_train, servo_test, speed_train, speed_test = train_test_split(
        lidar, servo, speed, test_size=0.2, random_state=42
    )

    train_data = np.stack((servo_train, speed_train), axis=-1)
    test_data = np.stack((servo_test, speed_test), axis=-1)
    lidar_train = lidar_train[..., np.newaxis]
    lidar_test = lidar_test[..., np.newaxis]

    model = build_tln_model(lidar.shape[1])

    def weighted_mse(y_true, y_pred):
        steer_weight = 1.0 + 5.0 * tf.abs(y_true[:, 0])
        speed_loss = tf.square(y_true[:, 1] - y_pred[:, 1])
        steer_loss = tf.square(y_true[:, 0] - y_pred[:, 0]) * steer_weight
        return tf.reduce_mean(steer_loss + speed_loss)

    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=tf.keras.losses.Huber())
    model.summary()

    t0 = time.time()
    history = model.fit(
        lidar_train, train_data,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(lidar_test, test_data),
    )
    print(f'Training time: {int(time.time() - t0)}s')

    os.makedirs(os.path.dirname(LOSS_FIG), exist_ok=True)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(LOSS_FIG)
    plt.close()

    print('\nModel Evaluation')
    test_loss = model.evaluate(lidar_test, test_data)
    print(f'Overall Test Loss: {test_loss:.4f}')
    preds = model.predict(lidar_test)
    print(f'Huber Loss: {huber_loss_np(test_data, preds):.4f}')

    os.makedirs('./Models', exist_ok=True)
    export_tflite(model, f'./Models/{MODEL_NAME}_noquantized.tflite')

    print('End')
