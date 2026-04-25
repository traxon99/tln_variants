#!/usr/bin/env python3
"""
TLN (TinyLidarNet) training from CSV datasets.
Usage: python3 train_standard_csv.py
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from train_utils import (
    load_csv_dataset, normalize_speed, split_dataset,
    build_tln_model, train_model, plot_loss_curve,
    export_tflite, evaluate_tflite, huber_loss_np,
)

# ---- Configuration ----
MODEL_NAME = 'TLN_standard'
DATASET_PATHS = [
    './dataset/5_min_0ms_min.csv',
    './dataset/5_min_no_crash.csv',
    './dataset/5laps-decent.csv',
    './dataset/aus_good_2.csv',
    './dataset/aus_good_3.csv',
    './dataset/extra_driving.csv',
    './dataset/ftg.csv',
    './dataset/good_clk.csv',
    './dataset/good_countclk.csv',
    './dataset/more_good_driving.csv',
    './dataset/mostly_const_speed.csv',
    './dataset/rounded_turn_agg.csv',
    './dataset/rounded_turn_faster.csv',
    './dataset/rounded_turn_good_2.csv',
    './dataset/rounded_turn_good.csv',
    './dataset/rounded_turn_med.csv',
    './dataset/sharp_corner_one.csv',
    './dataset/sharp_corner_two.csv',
    './dataset/turn_good.csv',
    './dataset/turn_low_speed.csv',
    './dataset/good_countclk.csv',
    './dataset/good_clk.csv',
]
DOWNSCALE = 2
LR = 5e-5
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.85
HZ = 40

if __name__ == '__main__':
    print('GPU:', bool(tf.config.list_physical_devices('GPU')))
    os.makedirs('./Models', exist_ok=True)
    os.makedirs('./Figures', exist_ok=True)

    lidar, servo, speed, max_speed = load_csv_dataset(DATASET_PATHS, downscale_factor=DOWNSCALE)
    speed, _ = normalize_speed(speed, max_speed=max_speed)

    X_train, X_test, y_train, y_test = split_dataset(lidar, servo, speed, TRAIN_RATIO)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    model = build_tln_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, LR)
    plot_loss_curve(history, f'./Figures/{MODEL_NAME}_loss.png')

    noquant_path = f'./Models/{MODEL_NAME}_noquantized.tflite'
    int8_path = f'./Models/{MODEL_NAME}_int8.tflite'
    export_tflite(model, noquant_path)
    export_tflite(model, int8_path, quantize_int8=True, representative_data=X_train)

    for path in [noquant_path, int8_path]:
        y_pred, _ = evaluate_tflite(path, X_test, y_test, HZ)
        print(f'Huber Loss [{path}]: {huber_loss_np(y_test, y_pred):.4f}')

    print('End')
