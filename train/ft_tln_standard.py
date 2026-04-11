import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tln_variants.utils import find_db3_files, read_ros2_bag, linear_map

bag_paths = [
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_ccw_forza/3_26_ccw_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_forza/3_26_cw_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_edge_forza/3_26_cw_edge_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_edge2_forza/3_26_cw_edge2_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_edge3_forza/3_26_cw_edge3_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_obstacles_forza/3_26_cw_obstacles_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/2_27_hard_forza/2_27_hard_forza.db3'
]


DOWNSCALE_FACTOR = 2



# ---- 1. Create a pretrained model (stand-in for your saved .h5) ----
def main():
    
    model_name = "TLN_2conv"
    
    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for pth in bag_paths:
        l, s, sp, ts = read_ros2_bag(pth, downsample=DOWNSCALE_FACTOR)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)
        all_ts.extend(ts)
    
    lidar = np.array(all_lidar)

    
    servo = np.array(all_servo)
    speed = np.array(all_speed)
    print(speed.max())
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
    len_lidar = lidar.shape[1]

    base_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu', input_shape=(len_lidar, 1)),
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

    lr: float =  5e-5 / 10
    huber = tf.keras.losses.Huber()

    # ---- 2. Load and fine tune ----
    model = keras.models.load_model("/home/jackson/sim_ws/src/tln_variants/train/Models/test_preft.keras")


    # Freeze only the first conv layer(s), let later ones adapt
    for layer in model.layers:
        if "conv" in layer.name:
            layer.trainable = False

    # Then selectively unfreeze the last conv layer
    model.layers[4].trainable = True  #last convolutional layer

    # Recompile with a smaller learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=huber
    )
    model.summary()

    model.fit(lidar_train, train_data, epochs=10, validation_split=0.2)

    # ---- 3. Save fine-tuned model ----
    model.save("finetuned.keras")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    os.makedirs('./Models', exist_ok=True)
    with open(f'./Models/{model_name}_fine_tuned.tflite', 'wb') as f:
        f.write(tflite_model)
        print(f"{model_name}_fine_tuned.tflite saved.")

    

if __name__ == '__main__':
    main()