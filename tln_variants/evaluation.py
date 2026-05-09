import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Evaluation(Node):
    def __init__(self):
        super().__init__('evaluation')

        self.declare_parameter('ego_scan_topic',  '/autodrive/roboracer_1/lidar')
        self.declare_parameter('ego_odom_topic',  '/autodrive/roboracer_1/odom')
        self.declare_parameter('steering_topic',  '/autodrive/roboracer_1/steering_command')
        self.declare_parameter('throttle_topic',  '/autodrive/roboracer_1/throttle_command')
        self.declare_parameter('lap_count_topic',       '/autodrive/roboracer_1/lap_count')
        self.declare_parameter('collision_count_topic', '/autodrive/roboracer_1/collision_count')
        self.declare_parameter('model_name',      'model')
        self.declare_parameter('max_laps',        3)
        self.declare_parameter('warmup_laps',     1)
        self.declare_parameter('centerline_path', '')
        self.declare_parameter('map_path',        '')
        self.declare_parameter('map_img_ext',     '.png')
        self.declare_parameter('output_dir',      'eval_results/')

        scan_topic            = self.get_parameter('ego_scan_topic').value
        odom_topic            = self.get_parameter('ego_odom_topic').value
        steer_topic           = self.get_parameter('steering_topic').value
        throttle_topic        = self.get_parameter('throttle_topic').value
        lap_count_topic       = self.get_parameter('lap_count_topic').value
        collision_count_topic = self.get_parameter('collision_count_topic').value
        self.name            = self.get_parameter('model_name').value
        self.max_laps        = self.get_parameter('max_laps').value
        self.warmup_laps     = self.get_parameter('warmup_laps').value
        self.centerline_path = self.get_parameter('centerline_path').value
        self.map_name        = self.get_parameter('map_path').value
        self.map_img_ext     = self.get_parameter('map_img_ext').value
        self.output_dir      = self.get_parameter('output_dir').value

        if not self.centerline_path:
            self.get_logger().fatal('centerline_path parameter is required.')
            raise ValueError('centerline_path not set')
        self.path_data = np.loadtxt(self.centerline_path, delimiter=',', usecols=(0, 1), skiprows=1)

        # State
        self.start             = True       # waiting for car to move
        self.lap_count         = 0
        self.lap_times         = []
        self.starting_progress = 0.0
        self.relative_progress = 0.0
        self.crash              = False
        self.collision_count    = 0
        self._prev_collision_count = 0
        self._t_collisions: list[float] = []
        self.done              = False
        self._t0_ns            = None       # wall-clock origin for all elapsed times
        self._lap_start_ns     = None       # reset each time a lap (incl. warmup) completes
        self._prev_lap_count   = 0

        # Per-sample time series (all laps combined)
        self._t_odom:     list[float] = []   # elapsed s
        self._speeds:     list[float] = []
        self._progresses: list[float] = []
        self._xs:         list[float] = []
        self._ys:         list[float] = []
        self._lap_ids:    list[int]   = []   # which lap this sample belongs to

        self._t_steer:    list[float] = []
        self._steerings:  list[float] = []
        self._lap_steer:  list[int]   = []

        self._t_throttle: list[float] = []
        self._throttles:  list[float] = []
        self._lap_thr:    list[int]   = []

        self._t_lidar:    list[float] = []
        self._min_lidar:  list[float] = []

        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Odometry,  odom_topic,            self._odom_cb,            10)
        self.create_subscription(LaserScan, scan_topic,            self._scan_cb,            sensor_qos)
        self.create_subscription(Float32,   steer_topic,           self._steer_cb,           10)
        self.create_subscription(Float32,   throttle_topic,        self._throttle_cb,        10)
        self.create_subscription(Int32,     lap_count_topic,       self._lap_count_cb,       10)
        self.create_subscription(Int32,     collision_count_topic, self._collision_count_cb, 10)

        self.get_logger().info(
            f'Evaluation ready  model={self.name}  max_laps={self.max_laps}'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        now = self.get_clock().now().nanoseconds
        if self._t0_ns is None:
            return 0.0
        return (now - self._t0_ns) * 1e-9

    def _update_progress(self, x, y):
        dists = np.linalg.norm(self.path_data - np.array([x, y]), axis=1)
        closest_idx = int(np.argmin(dists))
        absolute = closest_idx / (len(self.path_data) - 1)
        if self.start:
            self.starting_progress = absolute
        rel = absolute - self.starting_progress
        if rel < 0:
            rel += 1.0
        self.relative_progress = rel
        print(f'\rLap: {self.lap_count}  Progress: {rel:.0%}   ', end='', flush=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        vx  = msg.twist.twist.linear.x
        vy  = msg.twist.twist.linear.y
        spd = float((vx**2 + vy**2) ** 0.5)

        self._update_progress(x, y)

        if self.start and spd > 0.0:
            now = self.get_clock().now().nanoseconds
            self._t0_ns        = now
            self._lap_start_ns = now
            self.start         = False
            self.get_logger().info(
                f'Car moving — warmup lap 1 of {self.warmup_laps} in progress'
                if self.warmup_laps > 0 else 'Recording started'
            )

        if not self.start:
            t = self._elapsed()
            self._t_odom.append(t)
            self._speeds.append(spd)
            self._progresses.append(self.relative_progress)
            self._xs.append(x)
            self._ys.append(y)
            self._lap_ids.append(self.lap_count)

    def _lap_count_cb(self, msg: Int32):
        count = int(msg.data)
        if count <= self._prev_lap_count or self._lap_start_ns is None:
            return

        now_ns  = self.get_clock().now().nanoseconds
        lap_s   = (now_ns - self._lap_start_ns) * 1e-9
        self._prev_lap_count = count

        if count <= self.warmup_laps:
            remaining = self.warmup_laps - count
            self.get_logger().info(
                f'Warmup lap {count} complete ({lap_s:.3f}s) — '
                + (f'{remaining} warmup lap(s) remaining' if remaining else 'starting recording')
            )
            self._reset_recording(now_ns)
            return

        # Real lap
        self._lap_start_ns  = now_ns
        self.lap_count      = count - self.warmup_laps
        self.lap_times.append(lap_s)
        self.get_logger().info(f'Lap {self.lap_count} complete  time={lap_s:.3f}s')
        if self.lap_count >= self.max_laps:
            self.wrap_up()

    def _reset_recording(self, now_ns: int):
        """Discard all data collected so far and restart the clock. Called after each warmup lap."""
        self._t0_ns        = now_ns
        self._lap_start_ns = now_ns
        self._prev_collision_count = self.collision_count  # ignore pre-warmup collisions
        self._t_collisions.clear()
        self._t_odom.clear();     self._speeds.clear()
        self._progresses.clear(); self._xs.clear(); self._ys.clear()
        self._lap_ids.clear()
        self._t_steer.clear();    self._steerings.clear();  self._lap_steer.clear()
        self._t_throttle.clear(); self._throttles.clear();  self._lap_thr.clear()
        self._t_lidar.clear();    self._min_lidar.clear()

    def _scan_cb(self, msg: LaserScan):
        if self.start:
            return
        valid = [r for r in msg.ranges if r > 0.01]
        if valid:
            self._t_lidar.append(self._elapsed())
            self._min_lidar.append(float(min(valid)))

    def _collision_count_cb(self, msg: Int32):
        count = int(msg.data)
        if count > self._prev_collision_count and not self.start:
            new_hits = count - self._prev_collision_count
            self.collision_count = count
            self._prev_collision_count = count
            t = self._elapsed()
            for _ in range(new_hits):
                self._t_collisions.append(t)
            self.crash = count > 0
            self.get_logger().warn(
                f'Collision #{count}  progress={self.relative_progress:.0%}  t={t:.1f}s'
            )

    def _steer_cb(self, msg: Float32):
        if not self.start:
            self._t_steer.append(self._elapsed())
            self._steerings.append(float(msg.data))
            self._lap_steer.append(self.lap_count)

    def _throttle_cb(self, msg: Float32):
        if not self.start:
            self._t_throttle.append(self._elapsed())
            self._throttles.append(float(msg.data))
            self._lap_thr.append(self.lap_count)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def wrap_up(self):
        if self.done:
            return
        self.done = True

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir   = os.path.join(self.output_dir, f'{self.name}_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)

        t_odom   = np.array(self._t_odom)
        speeds   = np.array(self._speeds)
        progs    = np.array(self._progresses)
        xs       = np.array(self._xs)
        ys       = np.array(self._ys)
        lap_ids  = np.array(self._lap_ids)

        t_steer  = np.array(self._t_steer)
        steers   = np.array(self._steerings)
        lap_steer= np.array(self._lap_steer)

        t_thr    = np.array(self._t_throttle)
        throttles= np.array(self._throttles)
        lap_thr  = np.array(self._lap_thr)

        t_lidar     = np.array(self._t_lidar)
        min_lidar   = np.array(self._min_lidar)
        t_collisions= np.array(self._t_collisions)

        # ── metrics.txt ──────────────────────────────────────────────
        avg_spd  = float(np.mean(speeds)) if speeds.size else 0.0
        max_spd  = float(np.max(speeds))  if speeds.size else 0.0
        avg_thr  = float(np.mean(throttles)) if throttles.size else 0.0
        avg_steer= float(np.mean(np.abs(steers))) if steers.size else 0.0
        avg_time = float(np.mean(self.lap_times)) if self.lap_times else 0.0

        with open(os.path.join(run_dir, 'metrics.txt'), 'w') as f:
            f.write(f'Model:             {self.name}\n')
            f.write(f'Timestamp:         {timestamp}\n')
            f.write(f'Collisions:        {self.collision_count}\n')
            f.write(f'Collision times:   {[f"{t:.1f}s" for t in self._t_collisions]}\n')
            f.write(f'Laps completed:    {self.lap_count}\n')
            f.write(f'Lap times (s):     {[f"{t:.3f}" for t in self.lap_times]}\n')
            f.write(f'Avg lap time (s):  {avg_time:.3f}\n')
            f.write(f'Best lap (s):      {min(self.lap_times):.3f}\n' if self.lap_times else '')
            f.write(f'Avg speed (m/s):   {avg_spd:.4f}\n')
            f.write(f'Max speed (m/s):   {max_spd:.4f}\n')
            f.write(f'Avg throttle:      {avg_thr:.4f}\n')
            f.write(f'Avg |steering|:    {avg_steer:.4f}\n')

            if self.lap_times:
                f.write('\nPer-lap summary:\n')
                for lap_i, lap_t in enumerate(self.lap_times):
                    mask = lap_ids == lap_i
                    ls = speeds[mask]
                    sm = steers[lap_steer == lap_i] if steers.size else np.array([])
                    th = throttles[lap_thr == lap_i] if throttles.size else np.array([])
                    f.write(
                        f'  Lap {lap_i+1}: '
                        f'time={lap_t:.3f}s  '
                        f'avg_spd={np.mean(ls):.3f}  '
                        f'max_spd={np.max(ls):.3f}  '
                        f'avg_thr={np.mean(th):.3f}  '
                        f'avg_|steer|={np.mean(np.abs(sm)):.3f}\n'
                        if ls.size and th.size and sm.size else
                        f'  Lap {lap_i+1}: time={lap_t:.3f}s\n'
                    )

        self.get_logger().info(f'Metrics saved → {run_dir}/metrics.txt')

        # ── timeseries.csv ───────────────────────────────────────────
        with open(os.path.join(run_dir, 'timeseries.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_s', 'lap', 'speed_mps', 'progress', 'x', 'y'])
            for row in zip(t_odom, lap_ids, speeds, progs, xs, ys):
                w.writerow([f'{v:.4f}' for v in row])

        # ── colour palette ───────────────────────────────────────────
        n_laps  = max(self.lap_count, 1)
        palette = plt.cm.tab10(np.linspace(0, 0.9, n_laps))

        def lap_color(i): return palette[i % len(palette)]

        # ── Plot 1: speed vs time ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        for lap_i in range(n_laps):
            m = lap_ids == lap_i
            if np.any(m):
                ax.plot(t_odom[m], speeds[m], color=lap_color(lap_i),
                        lw=1.2, label=f'Lap {lap_i+1}')
        for lt in np.cumsum(self.lap_times):
            ax.axvline(lt, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Speed [m/s]')
        ax.set_title(f'{self.name} — Speed vs Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, 'speed_vs_time.png'), dpi=150)
        plt.close(fig)

        # ── Plot 2: speed vs track progress ──────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        for lap_i in range(n_laps):
            m = lap_ids == lap_i
            if np.any(m):
                ax.plot(progs[m], speeds[m], color=lap_color(lap_i),
                        lw=1.2, alpha=0.8, label=f'Lap {lap_i+1}')
        ax.set_xlabel('Track Progress [0–1]')
        ax.set_ylabel('Speed [m/s]')
        ax.set_xlim(0, 1)
        ax.set_title(f'{self.name} — Speed vs Progress')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, 'speed_vs_progress.png'), dpi=150)
        plt.close(fig)

        # ── Plot 3: throttle vs time ──────────────────────────────────
        if throttles.size:
            fig, ax = plt.subplots(figsize=(10, 4))
            for lap_i in range(n_laps):
                m = lap_thr == lap_i
                if np.any(m):
                    ax.plot(t_thr[m], throttles[m], color=lap_color(lap_i),
                            lw=1.0, label=f'Lap {lap_i+1}')
            for lt in np.cumsum(self.lap_times):
                ax.axvline(lt, color='gray', ls='--', lw=0.8)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Throttle [0–1]')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{self.name} — Throttle vs Time')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, 'throttle_vs_time.png'), dpi=150)
            plt.close(fig)

        # ── Plot 4: steering vs time ──────────────────────────────────
        if steers.size:
            fig, ax = plt.subplots(figsize=(10, 4))
            for lap_i in range(n_laps):
                m = lap_steer == lap_i
                if np.any(m):
                    ax.plot(t_steer[m], steers[m], color=lap_color(lap_i),
                            lw=1.0, label=f'Lap {lap_i+1}')
            for lt in np.cumsum(self.lap_times):
                ax.axvline(lt, color='gray', ls='--', lw=0.8)
            ax.axhline(0, color='black', lw=0.6)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Steering [−1 … +1]')
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f'{self.name} — Steering vs Time')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, 'steering_vs_time.png'), dpi=150)
            plt.close(fig)

        # ── Plot 5: steering distribution ────────────────────────────
        if steers.size:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(steers, bins=60, range=(-1, 1), color='steelblue', edgecolor='white', lw=0.4)
            ax.set_xlabel('Steering command')
            ax.set_ylabel('Count')
            ax.set_title(f'{self.name} — Steering Distribution')
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, 'steering_distribution.png'), dpi=150)
            plt.close(fig)

        # ── Plot 6: min LiDAR distance vs time ───────────────────────
        if min_lidar.size:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_lidar, min_lidar, color='tomato', lw=0.8)
            for lt in np.cumsum(self.lap_times):
                ax.axvline(lt, color='gray', ls='--', lw=0.8)
            for tc in t_collisions:
                ax.axvline(tc, color='red', lw=1.2, alpha=0.8)
            if t_collisions.size:
                ax.axvline(t_collisions[0], color='red', lw=1.2, alpha=0.8,
                           label=f'Collision ×{len(t_collisions)}')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Min LiDAR range [m]')
            ax.set_title(f'{self.name} — Wall Proximity vs Time')
            if t_collisions.size:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, 'proximity_vs_time.png'), dpi=150)
            plt.close(fig)

        # ── Plot 7: lap summary bar chart ─────────────────────────────
        if self.lap_times:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            lap_labels = [f'Lap {i+1}' for i in range(len(self.lap_times))]
            axes[0].bar(lap_labels, self.lap_times,
                        color=[lap_color(i) for i in range(len(self.lap_times))])
            axes[0].set_ylabel('Lap time [s]')
            axes[0].set_title('Lap Times')
            axes[0].grid(True, alpha=0.3, axis='y')

            per_lap_avg = []
            for lap_i in range(len(self.lap_times)):
                m = lap_ids == lap_i
                per_lap_avg.append(float(np.mean(speeds[m])) if np.any(m) else 0.0)
            axes[1].bar(lap_labels, per_lap_avg,
                        color=[lap_color(i) for i in range(len(self.lap_times))])
            axes[1].set_ylabel('Avg speed [m/s]')
            axes[1].set_title('Average Speed per Lap')
            axes[1].grid(True, alpha=0.3, axis='y')

            fig.suptitle(self.name)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, 'lap_summary.png'), dpi=150)
            plt.close(fig)

        # ── Plot 8: trajectory coloured by speed ──────────────────────
        if xs.size > 1:
            self._plot_trajectory(run_dir, xs, ys, speeds)

        self.get_logger().info(f'All results saved → {run_dir}/')
        raise SystemExit(0)

    def _plot_trajectory(self, run_dir, xs, ys, speeds):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Try to overlay map image if path is configured
        if self.map_name:
            yaml_path = f'{self.map_name}.yaml'
            try:
                with open(yaml_path) as f:
                    map_info = yaml.safe_load(f)
                resolution = map_info['resolution']
                origin     = map_info['origin']
                img_path   = os.path.join(os.path.dirname(self.map_name), map_info['image'])
                img        = plt.imread(img_path)
                h, w       = img.shape[:2]
                ax.imshow(img, cmap='gray', origin='lower',
                          extent=[origin[0], origin[0] + w * resolution,
                                  origin[1], origin[1] + h * resolution])
            except Exception as e:
                self.get_logger().warn(f'Map overlay skipped: {e}')

        points   = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm     = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
        lc       = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
        lc.set_array(speeds)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='Speed [m/s]')
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'{self.name} — Trajectory')
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, 'trajectory_map.png'), dpi=150)
        plt.close(fig)


def main(args=None):
    rclpy.init(args=args)
    node = Evaluation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
