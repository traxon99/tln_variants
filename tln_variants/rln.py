#!/usr/bin/env python3
import os
import time
from collections import deque
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from tln_variants.node_utils import linear_map as _linear_map, preprocess_scan, STEER_LIMIT, RLN_SPEED_OUT_MIN, RLN_SPEED_OUT_MAX
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import tensorflow as tf
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
import matplotlib.pyplot as plt



class RLNNode(Node):
    def __init__(self, use_timestamps=True):
        super().__init__('autonomous_node')
        self.use_timestamps = use_timestamps

        # Add a mutex for thread safety
        self.buffer_lock = Lock()

        # ——— Load the TFLite model ———
        try:
            model_path = '/home/jackson/sim_ws/src/tln_variants/train/Models/test.tflite'
            # Check if model file exists
            if not os.path.exists(model_path):
                self.get_logger().error(f'Model file not found: {model_path}')
                raise FileNotFoundError(f'Model file not found: {model_path}')
                
            self.get_logger().info(f'Loading model from: {model_path}')
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Inspect input shape: (batch, seq_len, num_ranges, num_ch)
            inp = self.interpreter.get_input_details()[0]
            _, seq_len, num_ranges, num_ch = inp['shape']
            self.seq_len = int(seq_len)
            self.num_ranges = int(num_ranges)
            self.num_ch = int(num_ch)  # should be 2 after your training fix
            
            self.get_logger().info(f'Model expects input shape: {inp["shape"]}, dtype: {inp["dtype"]}')
            
            self.input_index = inp['index']
            self.output_details = self.interpreter.get_output_details()
            self.get_logger().info(f'Model output details: {self.output_details}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

        # ——— Ring buffers for the last seq_len scans & timestamps ———
        self.buff_scans = deque(maxlen=self.seq_len)       # holds np.array(scan)
        self.buff_ts    = deque(maxlen=self.seq_len)       # one timestamp per scan

        # Manual override parameter
        self.declare_parameter('is_joy', False)
        self.prev_button = 0

        # Loop timing
        self.hz = 40.0
        self.period = 1.0 / self.hz
        self.start_ts = time.time()
        self.init_ts = time.time()
        self.total_distance = 0.0
        self.start_position = None

        # QoS for LiDAR (best-effort)
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # ——— ROS2 Subscriptions & Publishers ———
        self.create_subscription(
            Joy, '/joy', self.button_callback, 10
        )
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, lidar_qos
        )
        self.create_subscription(
            PoseStamped, '/pf/viz/inferred_pose',
            self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # Timer for control loop
        self.create_timer(self.period, self.control_loop)

        self.get_logger().info(
            f'Node ready: seq_len={self.seq_len}, '
            f'ranges={self.num_ranges}, channels={self.num_ch}'
        )

        self.recorded_ts = []
        self.recorded_speed = []
        self.recorded_steer = []


    def lidar_callback(self, msg: LaserScan):
        try:
            scan = preprocess_scan(msg.ranges, self.num_ranges, add_noise=True)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.buffer_lock:
                self.buff_scans.append(scan)
                self.buff_ts.append(t)
        except Exception as e:
            self.get_logger().error(f'Error in lidar_callback: {e}')

    def button_callback(self, msg: Joy):
        try:
            curr = msg.buttons[0]  # A button
            if curr == 1 and curr != self.prev_button:
                new_val = not self.get_parameter('is_joy').value
                self.set_parameters([Parameter('is_joy',
                                            Parameter.Type.BOOL,
                                            new_val)])
            self.prev_button = curr
        except Exception as e:
            self.get_logger().error(f'Error in button_callback: {e}')

    def odom_callback(self, msg: PoseStamped):
        try:
            pos = np.array([msg.pose.position.x, msg.pose.position.y])
            if self.start_position is None:
                self.start_position = pos
                return
            self.total_distance += np.linalg.norm(pos - self.start_position)
            self.start_position = pos
        except Exception as e:
            self.get_logger().error(f'Error in odom_callback: {e}')

    def dnn_output(self):
        try:
            # Need exactly seq_len scans to run inference
            with self.buffer_lock:
                if len(self.buff_scans) < self.seq_len:
                    self.get_logger().debug(f'Not enough scans: {len(self.buff_scans)}/{self.seq_len}')
                    return 0.0, 0.0
                
                # Make copies to work with outside the lock
                scans_list = list(self.buff_scans)
                ts_list = list(self.buff_ts)
            
            # 1) Stack scans: shape (seq_len, num_ranges)
            scans = np.stack(scans_list, axis=0)
            
            # Check for NaN/Inf values
            if np.isnan(scans).any() or np.isinf(scans).any():
                self.get_logger().warn('NaN or Inf values found in scans, replacing with zeros')
                scans = np.nan_to_num(scans, nan=0.0, posinf=0.0, neginf=0.0)

            if self.use_timestamps:
                # 2) Build Δt array from real ROS timestamps
                ts = np.array(ts_list, dtype=np.float32)
                if not np.all(np.diff(ts) >= 0):
                    self.get_logger().warn('Non-monotonic timestamps detected')
                    diffs = np.ones(self.seq_len-1, dtype=np.float32) * 0.025  # 40Hz fallback
                else:
                    diffs = np.diff(ts)
                dt_full = np.zeros(self.seq_len, dtype=np.float32)
                dt_full[1:] = diffs
                dt_tiled = np.repeat(dt_full[:, None], self.num_ranges, axis=1)
                seq = np.stack([scans, dt_tiled], axis=2)
                inp = seq[None, ...].astype(np.float32)
            else:
                # Single-channel input — no timestamp encoding
                inp = scans[None, ..., None].astype(np.float32)
            
            # Debug logging
            self.get_logger().debug(f'Input shape: {inp.shape}, dtype: {inp.dtype}')
            self.get_logger().debug(f'Input range: {np.min(inp)} to {np.max(inp)}')

            # 6) Run TFLite
            self.interpreter.set_tensor(self.input_index, inp)
            start = time.time()
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]
            inf_ms = (time.time() - start) * 1000.0
            
            self.get_logger().debug(f'Inference: {inf_ms:.1f} ms, Output: {out}, shape: {out.shape}')

            return float(out[0]), float(out[1])
            
        except Exception as e:
            self.get_logger().error(f'Error in dnn_output: {e}')
            return 0.0, 0.0

    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        return _linear_map(x, x_min, x_max, y_min, y_max)

    def control_loop(self):
        try:
            joy = self.get_parameter('is_joy').value
            self.get_logger().info(
                f'Manual: {"ON" if joy else "OFF"} | '
                f'Dist: {self.total_distance:.2f} m'
            )

            if not joy:
                steer, speed = self.dnn_output()

                # Reverse training normalization:
                speed = self.linear_map(speed, 0.0, 1.0, -0.5, 9)
                steer = self.linear_map(steer, -1.0, 1.0, -0.52, 0.52) #-0.34, 0.34)

                msg = AckermannDriveStamped()
                msg.drive.speed = speed #* 1.2 #??
                msg.drive.steering_angle = steer #* 1.5
                self.drive_pub.publish(msg)

                curr_plot_t = time.time() - self.init_ts
                self.recorded_ts.append(curr_plot_t)
                self.recorded_speed.append(msg.drive.speed)
                self.recorded_steer.append(msg.drive.steering_angle)


            # Deadline check
            dur = time.time() - self.start_ts
            if dur > self.period:
                self.get_logger().warn(
                    f'Deadline miss: {dur*1000:.1f} ms'
                )
            self.start_ts = time.time()

        except Exception as e:
            self.get_logger().error(f'Error in control_loop: {e}')

def plot_results(t_unused, speed, steer, filename="speed_steering_plot.png"):
    plt.figure(figsize=(10, 5))
    
    timesteps = range(len(speed))  # or len(steer) — they should match

    plt.subplot(2, 1, 1)
    plt.plot(timesteps, speed, label='Speed [m/s]')
    plt.ylabel("Speed")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timesteps, steer, label='Steering Angle [rad]', color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Steering")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RLNNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Fatal error: {e}")
    except KeyboardInterrupt:
        print("Node stopped by keyboard interrupt")
    finally:
        if 'node' in locals():
            plot_results(
                node.recorded_ts,
                node.recorded_speed,
                node.recorded_steer,
                filename="Figures/RNN_speed_steering_plot.png"
            )
            node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()