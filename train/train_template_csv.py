#!/usr/bin/env python3
"""
TLN training template from CSV datasets.
Usage: python3 train_template_csv.py
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from train_utils import (
    load_csv_dataset, normalize_speed, split_dataset,
    build_tln_model, train_model, plot_loss_curve,
    export_tflite, evaluate_tflite, huber_loss_np,
)

# ---- Configuration ----
MODEL_NAME = 'TLN'
DATASET_PATHS = [
    './dataset/dataset_ftg_spielberg_bothdir.csv',
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

    lidar, servo, speed, max_speed = load_csv_dataset(DATASET_PATHS, downscale_factor=DOWNSCALE,
                                                       column_format='speed_servo_lidar')
    speed, _ = normalize_speed(speed, max_speed=max_speed)

    X_train, X_test, y_train, y_test = split_dataset(lidar, servo, speed, TRAIN_RATIO)
    import numpy as np
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
