#!/usr/bin/env python3
"""
Dataset evaluation script for pre-training analysis of db3 rosbag files.
Generates speed vs time plots and summary metrics for each bag and the combined dataset.

Usage:
    python3 eval_dataset.py

Edit the bag_paths list below to point to your .db3 files.
Plots are saved to data_eval/plots/ (overwritten each run).
Metrics are saved to data_eval/metrics.txt.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from ackermann_msgs.msg import AckermannDriveStamped

# ============================================================
# Configure bag paths here
# ============================================================
bag_paths = [
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_ccw_forza/3_26_ccw_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/3_26/3_26_cw_forza/3_26_cw_forza.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/lab_oval_12_4_25/lab_oval_12_4_25.db3',
    '/home/jackson/sim_ws/src/tln_variants/train/Dataset/2_27_hard_forza/2_27_hard_forza.db3',
]

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
METRICS_FILE = os.path.join(os.path.dirname(__file__), 'metrics.txt')

SPEED_Y_MIN = 0.0
SPEED_Y_MAX = 10.0

# F1TENTH max steering angle (radians)
STEER_Y_MIN = -0.4189
STEER_Y_MAX = 0.4189


# ============================================================
# Bag reading
# ============================================================
def read_drive_from_bag(bag_path):
    """Read steering angle and speed from drive/Ackermann topic in a db3 bag."""
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    speeds, steers, timestamps = [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        if topic in ('drive', '/drive', 'Ackermann', '/Ackermann'):
            msg = deserialize_message(serialized_msg, AckermannDriveStamped)
            speeds.append(msg.drive.speed)
            steers.append(msg.drive.steering_angle)
            timestamps.append(t_ns * 1e-9)

    return np.array(speeds), np.array(steers), np.array(timestamps)


# ============================================================
# Plotting
# ============================================================
def plot_speed_vs_time(speeds, timestamps, bag_name, out_path):
    """Save a speed vs time plot with a static 0-10 m/s y-axis."""
    t0 = timestamps[0] if len(timestamps) > 0 else 0.0
    t_rel = timestamps - t0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, speeds, linewidth=0.8, color='steelblue')
    ax.set_ylim(SPEED_Y_MIN, SPEED_Y_MAX)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title(f'Speed vs Time — {bag_name}')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  Plot saved: {out_path}')


def plot_steer_vs_time(steers, timestamps, bag_name, out_path):
    """Save a steering angle vs time plot with a static y-axis."""
    t0 = timestamps[0] if len(timestamps) > 0 else 0.0
    t_rel = timestamps - t0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, steers, linewidth=0.8, color='darkorange')
    ax.set_ylim(STEER_Y_MIN, STEER_Y_MAX)
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering Angle (rad)')
    ax.set_title(f'Steering Angle vs Time — {bag_name}')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  Plot saved: {out_path}')


# ============================================================
# Metrics helpers
# ============================================================
def compute_metrics(speeds, steers):
    if len(speeds) == 0:
        return {}
    return {
        'samples': len(speeds),
        'avg_speed': float(np.mean(speeds)),
        'max_speed': float(np.max(speeds)),
        'min_speed': float(np.min(speeds)),
        'std_speed': float(np.std(speeds)),
        'avg_steer': float(np.mean(steers)),
        'max_steer': float(np.max(steers)),
        'min_steer': float(np.min(steers)),
        'std_steer': float(np.std(steers)),
    }


def format_metrics_block(name, m):
    lines = [
        f'  Bag          : {name}',
        f'  Samples      : {m["samples"]}',
        f'  Speed avg    : {m["avg_speed"]:.4f} m/s',
        f'  Speed max    : {m["max_speed"]:.4f} m/s',
        f'  Speed min    : {m["min_speed"]:.4f} m/s',
        f'  Speed std    : {m["std_speed"]:.4f} m/s',
        f'  Steer avg    : {m["avg_steer"]:.4f} rad',
        f'  Steer max    : {m["max_steer"]:.4f} rad',
        f'  Steer min    : {m["min_steer"]:.4f} rad',
        f'  Steer std    : {m["std_steer"]:.4f} rad',
    ]
    return '\n'.join(lines)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_speeds, all_steers = [], []
    per_bag_metrics = []

    with open(METRICS_FILE, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('DATASET EVALUATION METRICS\n')
        f.write('=' * 60 + '\n\n')

        for bag_path in bag_paths:
            bag_name = os.path.splitext(os.path.basename(bag_path))[0]
            print(f'Reading: {bag_path}')

            if not os.path.exists(bag_path):
                msg = f'  [WARNING] File not found, skipping: {bag_path}\n'
                print(msg.strip())
                f.write(msg + '\n')
                continue

            speeds, steers, timestamps = read_drive_from_bag(bag_path)

            if len(speeds) == 0:
                msg = f'  [WARNING] No drive messages found in: {bag_path}\n'
                print(msg.strip())
                f.write(msg + '\n')
                continue

            # Speed vs time plot
            plot_path = os.path.join(PLOTS_DIR, f'{bag_name}_speed.png')
            plot_speed_vs_time(speeds, timestamps, bag_name, plot_path)

            # Steering vs time plot
            steer_plot_path = os.path.join(PLOTS_DIR, f'{bag_name}_steer.png')
            plot_steer_vs_time(steers, timestamps, bag_name, steer_plot_path)

            # Per-bag metrics
            m = compute_metrics(speeds, steers)
            per_bag_metrics.append((bag_name, m))

            f.write(f'[{bag_name}]\n')
            f.write(format_metrics_block(bag_name, m) + '\n\n')

            all_speeds.extend(speeds.tolist())
            all_steers.extend(steers.tolist())

        # Combined metrics
        if all_speeds:
            all_speeds = np.array(all_speeds)
            all_steers = np.array(all_steers)
            combined = compute_metrics(all_speeds, all_steers)

            f.write('=' * 60 + '\n')
            f.write('[COMBINED — ALL BAGS]\n')
            f.write(format_metrics_block('all bags', combined) + '\n')
            f.write('=' * 60 + '\n')

            # Combined speed vs time (concatenated, no meaningful time axis)
            combined_ts = np.arange(len(all_speeds), dtype=float)

            combined_plot_path = os.path.join(PLOTS_DIR, '_combined_speed.png')
            plot_speed_vs_time(all_speeds, combined_ts, 'Combined (all bags)', combined_plot_path)

            combined_steer_path = os.path.join(PLOTS_DIR, '_combined_steer.png')
            plot_steer_vs_time(all_steers, combined_ts, 'Combined (all bags)', combined_steer_path)

            print(f'\nMetrics saved: {METRICS_FILE}')
            print(f'\n--- COMBINED METRICS ---')
            print(format_metrics_block('all bags', combined))
        else:
            print('[ERROR] No data loaded from any bag.')


if __name__ == '__main__':
    main()
