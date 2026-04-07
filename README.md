# tln_variants

A ROS2 package for F1TENTH autonomous racing using deep neural network controllers. The package provides several variants of LiDAR-based neural network controllers trained via imitation learning, along with data collection, training, and dataset evaluation tooling.

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Package Setup and Building](#package-setup-and-building)
- [Simulation vs Physical Deployment](#simulation-vs-physical-deployment)
- [Nodes and Executables](#nodes-and-executables)
- [Operational Procedures](#operational-procedures)
  - [Running a Controller](#running-a-controller)
  - [Recording a Dataset](#recording-a-dataset)
  - [Evaluating a Dataset](#evaluating-a-dataset)
  - [Training a Model](#training-a-model)
  - [Testing the Joystick](#testing-the-joystick)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

`tln_variants` implements two families of DNN controllers for F1TENTH racing:

| Family | Description |
|---|---|
| **TLN** (Temporal LiDAR Network) | 1D-CNN operating on a single downsampled LiDAR scan. Fast, low-latency, suitable for 40 Hz control. |
| **RLN** (Recurrent LiDAR Network) | Conv1D + Bidirectional LSTM + Attention operating on a sliding window of LiDAR frames. Captures temporal dynamics for smoother, more stable control. |

Both families output a **(steering angle, speed)** pair and run inference using TFLite on-device.

A classical **Follow-the-Gap (FTG)** node is also included as a baseline and as a data-collection driver.

---

## Dependencies

### ROS2 Packages
- `rclpy`
- `ackermann_msgs`
- `sensor_msgs`
- `nav_msgs`
- `std_msgs`
- `geometry_msgs`
- `message_filters`
- `sick_scan_xd` *(physical platform only — SICK LiDAR driver)*

### Python Libraries
```
tensorflow >= 2.x
numpy
matplotlib
scikit-learn
rosbag2_py       # for training from db3 bags
rclpy            # for training from db3 bags
```

Install Python dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Package Setup and Building

Note: The simulation environment for this project is [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros). Please follow installation steps there to use the simulation features in this project. 

### Adding tln_variants to Your Workspace

1. Clone into your ROS2 workspace `src/` directory:
   ```bash
   cd ~/sim_ws/src
   git clone https://github.com/traxon99/tln_variants.git
   ```

2. Build the package:
   ```bash
   cd ~/sim_ws
   colcon build --packages-select tln_variants
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

---

## Simulation vs Physical Deployment

The package uses separate config files and launch files to distinguish between simulation and physical deployments.

| Setting | Simulation | Physical |
|---|---|---|
| Config file | `config/tln_standard_sim.yaml` | `config/tln_standard_physical.yaml` |
| Launch file | `launch/tln_standard_sim.launch.py` | `launch/tln_standard_physical.launch.py` |
| LiDAR noise injection | **Enabled** (Gaussian, σ=0.5) | **Disabled** |
| Default max speed | 8.0 m/s | 6.0 m/s |
| Odometry topic | `/ego_racecar/odom` | `/pf/viz/inferred_pose` |
| LiDAR driver | F1TENTH gym simulator | `sick_scan_xd` |

### Why noise injection in simulation?
The simulator produces clean, noiseless LiDAR data. Adding Gaussian noise during sim inference bridges the sim-to-real gap so that models trained on real data also perform well in simulation, and vice versa.

### Config parameters (`tln_standard_*.yaml`)
```yaml
tln_standard:
  ros__parameters:
    sim: true/false        # Enables/disables noise injection
    min_speed: 1.0         # Minimum output speed (m/s)
    max_speed: 8.0         # Maximum output speed (m/s)
    downscale_factor: 2    # Use every Nth LiDAR beam
    model_path: '...'      # Absolute path to .tflite model file
```

---

## Nodes and Executables

| Executable | Class | Description |
|---|---|---|
| `tln_standard` | `TLNStandard` | TLN baseline — single scan, no velocity input. |
| `tln_vel` | `TLNStandard` | TLN with odometry velocity as a second input to the model. |
| `tln_override` | `TLNStandardOverride` | TLN with partial joystick override for speed and/or steering. **Note: not fully functional.** |
| `rln_sim` | `RLNSim` | RLN using a 5-frame temporal scan buffer with time-delta channel. Intended for simulation. |
| `rln_no_ts` | `AutonomousNode` | RLN without timestamp channel. Runs at 40 Hz on a timer. Includes joystick kill-switch (A button). |
| `ftg` | `FtgNode` | Follow-the-Gap classical controller. Used as a baseline and for collecting training data. Based on F1Tenth Benchmarks implementation |
| `collect` | `DataCollection` | Records synchronized (LiDAR, drive command, odometry) rows to a timestamped CSV file. |
| `collect_rosbags` | — | Records data to ROS2 db3 bag format. |
| `joy_test` | `JoyListener` | Prints raw joystick axes and buttons — useful for verifying controller connectivity. |

### ROS2 Topics

| Topic | Type | Direction |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Subscribed by all controllers |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Published by all controllers |
| `/ego_racecar/odom` | `nav_msgs/Odometry` | Subscribed (sim) for velocity feedback |
| `/pf/viz/inferred_pose` | `geometry_msgs/PoseStamped` | Subscribed (physical) for distance tracking |
| `/joy` | `sensor_msgs/Joy` | Subscribed for joystick override / kill-switch |

---

## Operational Procedures

### Running a Controller

**Simulation:**
```bash
ros2 launch tln_variants tln_standard_sim.launch.py
```

**Physical car:**
```bash
ros2 launch tln_variants tln_standard_physical.launch.py
```

Before launching, update `model_path` in the appropriate config file to point to the `.tflite` model you want to use.

**Running other executables directly:**
```bash
ros2 run tln_variants ftg
ros2 run tln_variants rln_sim
ros2 run tln_variants rln_no_ts
```

**Joystick kill-switch (`rln_no_ts`):** Press the **A button** on the controller to toggle between autonomous mode and manual override (manual mode stops drive commands from being published).

---

### Recording a Dataset

Datasets can be recorded as CSV files or ROS2 bags.

**CSV recording** (synchronized LiDAR + drive + odometry):
```bash
ros2 run tln_variants collect
```
- Run a controller or drive manually in a separate terminal before starting collection.
- The CSV is saved to `src/tln_variants/train/dataset/recording-HH:MM:SS.csv`.
- **Rename the file immediately after recording** to something meaningful (e.g., `forza_ccw_5min.csv`).
- Each row contains: `speed, steering_angle, velocity_magnitude, [lidar_ranges...]`

**ROS2 bag recording:**
```bash
ros2 run tln_variants collect_rosbags
```
Bags are saved to `train/Dataset/` and can be used directly with the RLN training scripts.

---

### Evaluating a Dataset

Before training, inspect your rosbag datasets for quality issues (speed distribution, steering imbalance, coverage):

```bash
cd src/tln_variants/data_eval
python3 eval_dataset.py
```

Edit the `bag_paths` list inside `eval_dataset.py` to point to your `.db3` files. The script produces:

- `data_eval/plots/<bag_name>_speed.png` — speed vs time
- `data_eval/plots/<bag_name>_steer.png` — steering angle vs time
- `data_eval/plots/<bag_name>_steer_hist.png` — steering histogram (shows left/right imbalance)
- `data_eval/metrics.txt` — per-bag and combined summary statistics

**Tip:** Aim for a roughly balanced steering histogram (left vs right) before training to avoid biased models. If the dataset is highly imbalanced, consider mirroring the data (flip LiDAR + negate steering) in the training script.

---

### Training a Model

All training scripts are in `train/`. Run them from the `train/` directory.

#### TLN — Standard (CSV input)

```bash
cd src/tln_variants/train
python3 train_standard_csv.py
```

- Reads CSV datasets listed in `dataset_path`.
- Trains a 1D-CNN (`Conv1D → Dense → Dense(2, tanh)`) to predict `[steering, speed]`.
- Saves both a float32 and an int8-quantized TFLite model to `train/Models/`.
- Plots the loss curve to `train/Figures/loss_curve.png`.

Key hyperparameters (edit at top of script):
```python
lr = 5e-5
batch_size = 64
num_epochs = 20
down_sample_param = 2   # LiDAR downsampling factor (must match node config)
```

#### RLN — Recurrent (db3 bag input)

```bash
cd src/tln_variants/train
python3 train_rnn.py
```

- Reads one or more `.db3` rosbag files listed in `bag_paths`.
- Builds sliding-window sequences of 5 LiDAR frames with inter-frame time-delta channels.
- Trains a `TimeDistributed Conv1D → BiLSTM → Attention → Dense(2, tanh)` model.
- Saves a TFLite model to `train/Models/`.

Key hyperparameters:
```python
seq_len    = 5       # Number of frames per sequence
batch_size = 128
lr         = 1e-5
epochs     = 30
```

#### Model output format

All models output a 2-element vector: `[steering_angle, speed]` in the range `[-1, 1]` (tanh activation). The nodes apply a linear remap at inference time:

- **Speed**: remapped from `[0, 1]` to `[min_speed, max_speed]` (see config).
- **Steering**: passed through directly (TLN) or remapped from `[-1, 1]` to `[-0.52, 0.52]` rad (RLN).

---

### Testing the Joystick

To verify your joystick is connected and publishing correctly before a run:

```bash
ros2 run tln_variants joy_test
```

This prints the raw axes and button values from `/joy`. Confirm the axes indices match what the override nodes expect (left stick up/down = axis 1, right stick left/right = axis 3).

---

## Results

> The models in `models/` were trained on F1TENTH simulation data across tracks including Spielberg, Austin, and Moscow.

| Model | Architecture | Training data | Notes |
|---|---|---|---|
| `TLN_M_noquantized.tflite` | TLN (standard) | Medium size: Uses every 2 scans | General-purpose baseline |
| `f1_tenth_model_temporal_M_noquantized.tflite` | TLN (temporal) | Medium size: Uses every 2 scans | Temporal variant |
| `RLN_TLN_M.tflite` | RLN (BiLSTM+Attn) | Medium size: Uses every 2 scans | Combined real/sim training |
| `RLN_with_TLN_data.tflite` | RLN | TLN physical data | RLN trained on sim-collected data |

**Inference timing (TLN standard, CPU):** ~100–500 µs per scan at 40 Hz — well within the 25 ms deadline.

---

## Future Work

- [ ] Fix `tln_standard_override.py` joystick override logic (currently non-functional as noted in source).
- [ ] Add a unified, config-driven node that replaces the proliferation of separate node variants.
- [ ] Complete `data_collection.py` joystick-gated recording (`CONTROLLER` flag and `joy_callback` are stubs).
- [ ] Add sim-to-real evaluation metrics (lap time, crash rate) for each model variant.
- [ ] Replace hardcoded absolute `model_path` values in nodes with ROS2 parameters loaded from config.
- [ ] Add a `collect_rosbags` launch file with configurable topic list.
- [ ] Investigate data augmentation (LiDAR mirroring) to reduce steering histogram imbalance.
- [ ] Explore quantization-aware training to close the accuracy gap between float32 and int8 models.
