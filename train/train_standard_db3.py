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

from tln_variants.utils import find_db3_files, read_ros2_bag, linear_map

DOWNSCALE_FACTOR = 2



#========================================================
# Main
#========================================================
if __name__ == '__main__':
    print('GPU AVAILABLE:', bool(tf.config.list_physical_devices('GPU')))
    #"Good" model

    # TLN Standard
    bag_paths = find_db3_files('/home/jackson/sim_ws/src/tln_variants/train/Dataset/TLN_Original_Dataset')


    # Bag path for decent model TLN_Forza WITH CUSTOM LOSS - Current prelim results
    # bag_paths = [
    #     # 'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
    #     # # 'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
    #     # # 'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3'
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/forza_pf_map/forza_pf_map_0.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP/Forza_GLC_smile_PP_0.db3', #evil
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3', #could be bad... or good
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_glc_ot_ez_3laps/Forza_glc_ot_ez_3laps_0.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_small_3laps/Forza_GLC_smile_small_3laps_0.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_hangar_1905_v0_1lap/Forza_hangar_1905_v0_1lap_0.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr1db3/jfr1.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr2db3/jfr2.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map/test_map.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/out/out.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f2/f2.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f4/f4.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map_opp/test_map_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv5_opp/jfrv5_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv6_opp/jfrv6_opp.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map_obstacles_good/test_map_obstacles_good.db3'
    # ]
    
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_oval_12_4_25/lab_oval_12_4_25.db3'
    # ]
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/2_20_easy_forza/2_20_easy_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_oval_12_4_25/lab_oval_12_4_25.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/2_20_easy_forza2/2_20_easy_forza2.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/2_27_hard_forza/2_27_hard_forza.db3'
    # ]
    
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_ccw_forza/3_26_ccw_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_forza/3_26_cw_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_edge_forza/3_26_cw_edge_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_edge2_forza/3_26_cw_edge2_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_edge3_forza/3_26_cw_edge3_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_obstacles_forza/3_26_cw_obstacles_forza.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_oval_12_4_25/lab_oval_12_4_25.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/2_27_hard_forza/2_27_hard_forza.db3'

    # ]
    
    # bag_paths = [
    #     'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
    #     'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
    #     'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3'
    # ]
    #TLN Standard
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/out/out.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f2/f2.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f4/f4.db3',
    # ]
    
    #Bad
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3', #could be bad... or good
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr1db3/jfr1.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr2db3/jfr2.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv5_opp/jfrv5_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv6_opp/jfrv6_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_test_real_dataset/lab_test_real_no_obst/lab_test_real_no_obst.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_test_real_dataset/lab_test_real_obst/lab_test_real_obst.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_test_real_dataset/lab_test_real_obst2/lab_test_real_obst2.db3'
        
    # ]

    #Everything

    # bag_paths = [
    #     'Dataset/5_min_austin_sim/5_min_austin_sim_0.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_1lap/JFRv5_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_edgecases/JFRv5_edgecases.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_obstacle_overtake/JFRv5_obstacle_overtake.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv6_1lap/JFRv6_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Test_map_1lap/Test_map_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3',
    #             # # 'Dataset/5_min_moscow_sim/5_min_moscow_sim_0.db3',
    #     # 'Dataset/5_min_Spiel_sim/5_min_Spiel_sim_0.db3'
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/forza_pf_map/forza_pf_map_0.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP/Forza_GLC_smile_PP_0.db3', #evil
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3', #could be bad... or good
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_glc_ot_ez_3laps/Forza_glc_ot_ez_3laps_0.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_small_3laps/Forza_GLC_smile_small_3laps_0.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_hangar_1905_v0_1lap/Forza_hangar_1905_v0_1lap_0.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr1db3/jfr1.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr2db3/jfr2.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map/test_map.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/out/out.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f2/f2.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/f4/f4.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map_opp/test_map_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv5_opp/jfrv5_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv6_opp/jfrv6_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/test_map_obstacles_good/test_map_obstacles_good.db3'
    # ]

    # #GOLDENIsh
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_1lap/test_map_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_non_obstr/test_map_non_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_obstr/test_map_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_GLC_smile_PP_edgecases/Forza_GLC_smile_PP_edgecases_0.db3', #could be bad... or good
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr1db3/jfr1.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfr2db3/jfr2.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv5_opp/jfrv5_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/Forza_dataset/jfrv6_opp/jfrv6_opp.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv6_nonobstr_edited/JFRv6_nonobstr_edited.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv5_nonobstr_edited/JFRv5_nonobstr_edited.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_non_obstr/test_map_non_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv6_1lap/JFRv6_1lap.db3',
        

    # ]
    
    # #CLose, but slow
    # bag_paths = [
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_1lap/test_map_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_non_obstr/test_map_non_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_obstr/test_map_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_1lap/JFRv5_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv6_1lap/JFRv6_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_edgecases/JFRv5_edgecases.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_obstacle_overtake/JFRv5_obstacle_overtake.db3'
    # ]
    
    
    # # Slow
    # bag_paths = [
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_1lap/test_map_1lap.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_non_obstr/test_map_non_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/test_map_nonobstr_obstr_norm/test_map_obstr/test_map_obstr.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv5_nonobstr_edited/JFRv5_nonobstr_edited.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv6_edited/JFRv6_edited.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv6_nonobstr_edited/JFRv6_nonobstr_edited.db3',
    #     '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5-6_nonobstr/JFRv6_obstr_edited/JFRv6_obstr_edited.db3',
    #     # '/home/jackson/sim_ws/src/tln_variants/train/Dataset/JFRv5_1lap/JFRv5_1lap.db3'
    # ] 


    batch_size = 64
    lr = 5e-5
    num_epochs = 10# 20 #10
    model_name = 'test'
    loss_figure_path = f'./Models/{model_name}_loss.png'

    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for pth in bag_paths:
        l, s, sp, ts = read_ros2_bag(pth, downsample=DOWNSCALE_FACTOR)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)
        all_ts.extend(ts)
    
    lidar = np.array(all_lidar)#[:-1]

    #add noise
    # noise = np.random.normal(0,0.15,lidar.shape)   
    # print(noise[1])
    
    # print(noise.shape)

    # lidar = lidar + noise
    
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

    def weighted_mse(y_true, y_pred):
        steer_weight = 1.0 + 5.0 * tf.abs(y_true[:, 0])  # higher weight for turns
        speed_loss = tf.square(y_true[:, 1] - y_pred[:, 1])
        steer_loss = tf.square(y_true[:, 0] - y_pred[:, 0]) * steer_weight
        return tf.reduce_mean(steer_loss + speed_loss)

    optimizer = tf.keras.optimizers.Adam(lr)
    huber = tf.keras.losses.Huber()
    model.compile(optimizer=optimizer, loss=huber)
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
    
    model.save(f"./Models/{model_name}_preft.keras")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    os.makedirs('./Models', exist_ok=True)
    with open(f'./Models/{model_name}_noquantized.tflite', 'wb') as f:
        f.write(tflite_model)
        print(f"{model_name}_noquantized.tflite saved.")