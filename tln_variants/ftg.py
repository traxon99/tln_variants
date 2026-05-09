import rclpy
from math import pi
from rclpy.node import Node
from statistics import mean
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Float32
import numpy as np

class FtgNode(Node):
    def __init__(self):
        super().__init__('ftg_node')
        self.steering_publisher = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_publisher = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.scan_callback, 10)
        self.get_logger().info('Jackson FTGNode has been started.')

        #adapted parameters from BDEvan5 f1tenth benchmarks FTG implementation
        self.bubble_radius = 270
        self.preprocess_conv_size = 3
        self.best_point_conv_size = 80
        self.max_lidar_dist = 10.0
        self.fast_speed = 0.17
        self.straights_speed = 0.09
        self.corners_speed = 0.09
        self.straights_steering_angle = 0.174 
        self.fast_steering_angle = 0.0785 #0.0785
        self.safe_threshold = 6
        self.max_steer = 0.52 #0.52


    def scan_callback(self, msg):
        self.get_logger().info(f'Scan received {len(msg.ranges)}')
        proc_ranges = self.preprocess_lidar(msg.ranges)
        
        closest = np.argmin(proc_ranges)

        #eliminate all points inside of bubble
        min_index = closest - self.bubble_radius
        max_index = closest + self.bubble_radius
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        
        if abs(steering_angle) > self.straights_steering_angle:
            speed = self.corners_speed
        elif abs(steering_angle) > self.fast_steering_angle:
            speed = self.straights_speed
        else:
            speed = self.fast_speed


        self.publish_drive(speed, steering_angle)

    def preprocess_lidar(self, ranges):
            """ Preprocess the LiDAR scan array. Expert implementation includes:
                1.Setting each value to the mean over some window
                2.Rejecting high values (eg. > 3m)
            """
            self.radians_per_elem = (2 * np.pi) / len(ranges)
            # we won't use the LiDAR data from directly behind us
            proc_ranges = np.array(ranges[180:-180])
            # sets each value to the mean over a given window
            proc_ranges = np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), 'same') / self.preprocess_conv_size
            proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)
            return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        # print(slices)
        # max_len = slices[-1].stop - slices[-1].start
        # chosen_slice = slices[-1]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[::-1]:
            # print(sl)
            sl_len = sl.stop - sl.start
            if sl_len > self.safe_threshold:
                chosen_slice = sl
                # print("Slice choosen")
                return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.best_point_conv_size),'same') / self.best_point_conv_size
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the lidar data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        return steering_angle
    
    def publish_drive(self, speed, steering_angle):
        
        speed_msg = Float32()
        steering_msg = Float32()
        
        speed_msg.data = float(speed)
        steering_msg.data = float(steering_angle)
        # Pretty much a boilerplate publishing function
        self.steering_publisher.publish(steering_msg)
        self.throttle_publisher.publish(speed_msg)
        
        
        
        # Debug, if there was a debug mode lmao
        self.get_logger().info(f'Published command: speed={speed_msg.data}, steering_angle={steering_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = FtgNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
        node.publish_ackermann_drive(0,0)
    finally:
        node.publish_ackermann_drive(0,0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
