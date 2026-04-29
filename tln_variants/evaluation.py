import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import yaml
#for graphing
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.collections import LineCollection
#for progress tracking
import csv
import os
import sys
from datetime import datetime

class Evaluation(Node):
    def __init__(self):
        super().__init__('evaluation')

        # Declare parameters
        self.declare_parameter("ego_scan_topic", "/scan")
        self.declare_parameter("ego_odom_topic", "/odom")

        # Evaluation parameters
        self.declare_parameter("model_name", "TLN")
        self.declare_parameter("max_laps", 1)
        self.declare_parameter("evaluation_lap", 0)
        self.declare_parameter("centerline_path", "")
        self.declare_parameter("map_path", "")
        self.declare_parameter("map_img_ext", ".png")

        scan_topic = self.get_parameter("ego_scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("ego_odom_topic").get_parameter_value().string_value
        
        self.name = self.get_parameter("model_name").value
        self.max_laps = self.get_parameter("max_laps").value
        self.eval_lap = self.get_parameter("evaluation_lap").value
        self.centerline_path = self.get_parameter("centerline_path").value
        self.map_name = self.get_parameter("map_path").value
        self.map_img_ext = self.get_parameter("map_img_ext").value

        
        
        self.output_dir = 'temp/'

        if not self.centerline_path:
            self.get_logger().fatal("centerline_path parameter is required. Run with a config file.")
            raise ValueError("centerline_path not set")
        self.path_data = np.loadtxt(self.centerline_path, delimiter=',', usecols=(0, 1))
        
        #starting point parameters
        self.starting_x = 0     # GYM -52
        self.starting_y = 0      # 0
        self.finish_line_radius = 2
        # self.max_laps = 5
        self.CRASH_THRESHOLD = 0.15

        #Do Not Touch :D
        self.start = True
        self.on_line = False
        self.previous_step_on_line = False
        self.lap_count = 0
        self.lap_times = []
        self.starting_progress = 0.0
        self.current_progress = 0.0
        self.relative_progress = 0.0
        self.absolute_progress = 0.0

        self.crash = False
        self.done = False

        self.speeds = []
        self.progresses = []
        self.xs = []
        self.ys = []


        self.odom_subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10)

        
        self.get_logger().info(f"Evaluation Node Started.\nTesting: {self.name}\n")
        self.get_logger().info(f"Evaluation Parameters:\n"
                               f"Map Name:\t{self.map_name}\n"
                               f"Evaluated Lap:\t{self.eval_lap}\n"
                               f"Max Laps:\t{self.max_laps}\n")

    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_mag = (vel_x**2 + vel_y**2)**0.5 # velocity magnitude
        
        self.update_progress(x, y)
        if (self.start == False) and (self.lap_count < 1) and self.relative_progress < 1.0:
            self.progresses.append(self.relative_progress)
            self.speeds.append(vel_mag)
            self.xs.append(x)
            self.ys.append(y)
        

        # Check if the racecar has crossed the finish line
        distance_from_start = ((x - self.starting_x)**2 + (y - self.starting_y)**2)**0.5
        
        #check to see if car is at starting circle (not line technically)
        self.on_line = distance_from_start <= self.finish_line_radius

        #see if car left finish circle
        if (self.start == True) and (vel_mag > 0):             
            self.get_logger().info("Timer started")
            self.lap_time = self.get_clock().now().nanoseconds
            self.start = False

        #check if car crossed finish line, not during start state
        # todo: clean this logic up
        elif not(self.previous_step_on_line) and self.on_line and not(self.start):
            # add to lap count
            self.lap_count += 1
            # lap time from ros clock
            self.lap_time = self.get_clock().now().nanoseconds - self.lap_time
            # add time in seconds 
            self.lap_times.append(self.ns_2_s(self.lap_time))
            
            #print lap count and time
            self.get_logger().info(f"Laps completed: {self.lap_count}, Lap Time: {self.ns_2_s(self.lap_time)}")
            
            #restart clock
            self.lap_time = self.get_clock().now().nanoseconds



        if self.lap_count == self.max_laps:
            self.wrap_up()

        # t-1 status of car on line or not
        self.previous_step_on_line = self.on_line
    
    def ns_2_s(self, ns):
        #nanoseconds to seconds
        return ns / 1_000_000_000
    
    def scan_callback(self, msg):

        min_distance = min(msg.ranges)
        if (min_distance < self.CRASH_THRESHOLD) and not(self.crash):
            self.crash = True
            self.get_logger().info(f"Crash Detected, Progress: {self.relative_progress:.0%}")
            self.wrap_up()
    
    
    def wrap_up(self):
        if self.done:
            return
        self.done = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"{self.name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        avg_speed = float(np.mean(self.speeds)) if self.speeds else 0.0
        avg_time = float(np.mean(self.lap_times)) if self.lap_times else 0.0

        print(f"Lap times:\t{self.lap_times}\n"
              f"Average time:\t{avg_time}\n"
              f"Average speed:\t{avg_speed}")

        metrics_path = os.path.join(run_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Model:         {self.name}\n")
            f.write(f"Map:           {self.map_name}\n")
            f.write(f"Timestamp:     {timestamp}\n")
            f.write(f"Crashed:       {self.crash}\n")
            f.write(f"Laps:          {self.lap_count}\n")
            f.write(f"Lap times:     {self.lap_times}\n")
            f.write(f"Average time:  {avg_time:.4f} s\n")
            f.write(f"Average speed: {avg_speed:.4f} m/s\n")
            f.write(f"Max speed:     {float(np.max(self.speeds)) if self.speeds else 0.0:.4f} m/s\n")
            f.write(f"Min speed:     {float(np.min(self.speeds)) if self.speeds else 0.0:.4f} m/s\n")
        self.get_logger().info(f"Metrics saved to {metrics_path}")

        # Plot 1: speed vs. track progress
        fig1, ax1 = plt.subplots()
        ax1.set_ylim(0.0, 10.0)
        ax1.set_xlim(0.0, 1.0)
        ax1.plot(self.progresses, self.speeds)
        ax1.set_xlabel("Track Progress")
        ax1.set_ylabel("Speed")
        ax1.set_title(f"{self.name}: Speed vs. Track Progress on {self.map_name}")
        ax1.grid(True)
        plot1_path = os.path.join(run_dir, "speed_vs_progress.png")
        fig1.savefig(plot1_path, dpi=200, bbox_inches='tight')
        plt.close(fig1)
        self.get_logger().info(f"Plot saved to {plot1_path}")

        # --- Load map yaml ---
        yaml_path = f"{self.map_name}.yaml"
        with open(yaml_path, 'r') as f:
            map_info = yaml.safe_load(f)

        resolution = map_info['resolution']
        origin = map_info['origin']

        # --- Load image ---
        img_path = os.path.join(os.path.dirname(self.map_name), map_info['image'])
        img = plt.imread(img_path)

        # Compute image extents in world coordinates
        height, width = img.shape[:2]
        x_min = origin[0]
        x_max = origin[0] + width * resolution
        y_min = origin[1]
        y_max = origin[1] + height * resolution

        fig2, ax2 = plt.subplots()
        ax2.imshow(img,
                cmap='gray',
                origin='lower',
                extent=[x_min, x_max, y_max, y_min])
        points = np.array([self.xs, self.ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(vmin=1, vmax=8)
        cmap = plt.get_cmap("viridis")

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(self.speeds)
        lc.set_linewidth(2)

        ax2.add_collection(lc)
        plt.colorbar(lc, ax=ax2, label="Speed [m/s]")
        plot2_path = os.path.join(run_dir, "trajectory_map.png")
        fig2.savefig(plot2_path, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        self.get_logger().info(f"Plot saved to {plot2_path}")
        raise SystemExit(0)


    def update_progress(self, x, y):
        target = np.array([x, y])

        # Compute Euclidean distances from path points to the target
        dists = np.linalg.norm(self.path_data - target, axis=1)

        # Index of the closest point on the path
        closest_idx = np.argmin(dists)

        # Progress as a fraction of total path length
        self.absolute_progress = closest_idx / (self.path_data.shape[0] - 1)

        if self.start == True:
            self.starting_progress = self.absolute_progress

        self.relative_progress = (1 - self.starting_progress) + self.absolute_progress
        if self.relative_progress >= 1:
            self.relative_progress -= 1
        print(f"\rLap: {self.lap_count}  Progress: {self.relative_progress:.0%}   ", end='', flush=True)



def main(args=None):
    rclpy.init(args=args)
    node = Evaluation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
