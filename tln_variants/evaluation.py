import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

#for graphing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
#for progress tracking
import csv
#
import os
import sys

class Evaluation(Node):
    def __init__(self, name = "Planner", map=None, max_laps=3, eval_lap=1):
        super().__init__('evaluation_node')
        
        #extract evaluation parameters from Node
        self.name = name
        self.map_name = map
        self.max_laps = max_laps
        self.eval_lap = eval_lap

        #load path data from centerline csv file
        # self.path_data = np.loadtxt(f'{os.getcwd()}{map_name}_centerline.csv', delimiter=',', usecols=(0, 1))
        self.path_data = np.loadtxt('/home/jackson/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_centerline.csv', delimiter=',', usecols=(0, 1))
        
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

        self.speeds = []
        self.progresses = []


        # Pass the callback group when creating the subscription
        self.odom_subscription = self.create_subscription(
            Odometry, 
            '/ego_racecar/odom', 
            self.odom_callback, 
            10
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
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
        #Evaluation needs to be ceased. All times and data needs to be saved and the node shut down
        print(f"Lap times:\t{self.lap_times}\n"
              f"Average time:\t{np.mean(self.lap_times)}\n"
              f"Average speed:\t{np.mean(self.speeds)}")
        fig, ax = plt.subplots()
        ax.axes.set_ylim(0.0,10.0)
        ax.axes.set_xlim(0.0,1.0)
        plt.plot(self.progresses, self.speeds,)
        plt.xlabel("Track Progress")
        plt.ylabel("Speed")
        plt.title(f"{self.name}: Speed vs. Track Progress on {self.map_name}")
        plt.grid(True)
        plt.show()


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
            print(f"Starting progrss: {self.starting_progress:.0%}")

        self.relative_progress = (1 - self.starting_progress) + self.absolute_progress
        if self.relative_progress >= 1:
            self.relative_progress -= 1
        print(f"Lap: {self.lap_count}\tProgress: {self.relative_progress:.0%}")
        # if round(self.relative_progress, 3) % 0.25 == 0:
            # print(round(self.relative_progress, 2))



def main(args=None):
    map = None
    max_laps = None
    eval_lap = None
    try:
        name = str(sys.argv[1])
        map = str(sys.argv[2])
        max_laps = int(sys.argv[3])
        eval_lap = int(sys.argv[4])
        assert eval_lap <= max_laps
    except:
        raise ValueError()

    rclpy.init(args=args)
    node = Evaluation(name=name, map=map, max_laps=max_laps, eval_lap=eval_lap)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

# def analyze(avg_speeds, lap_times):
#     return fastest_time, slowest_time, avg_lap_time

if __name__ == '__main__':
    main()
