import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup  # Import callback group
from nav_msgs.msg import Odometry

class Evaluation(Node):
    def __init__(self):
        super().__init__('evaluation_node')
        
        # Define a callback group
        self.callback_group = ReentrantCallbackGroup()
        
        #starting point parameters
        self.starting_x = 5.0
        self.starting_y = 5.0
        self.finish_line_radius = 1
        
        # Pass the callback group when creating the subscription
        self.odom_subscription = self.create_subscription(
            Odometry, 
            '/ego_racecar/odom', 
            self.odom_callback, 
            10,
            callback_group=self.callback_group  # Assign callback group
        )
        
        self.get_logger().info("Evaluation Node Started.")
        self.start = True
        self.on_line = False
        self.previous_step_on_line = False
        self.lap_count = 0
        self.lap_times = []
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_mag = (vel_x**2 + vel_y**2)**0.5 # velocity magnitude

        #self.get_logger().info(f"x: {x}, y: {y}")

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
            self.lap_time = self.get_clock().now().nanoseconds - self.lap_time
            self.lap_times.append(self.lap_time)
            self.get_logger().info(f"Laps completed: {self.lap_count}, Lap Time: {self.ns_2_s(self.lap_time)}")
            self.lap_time = self.get_clock().now().nanoseconds

        # t-1 status of car on line or not
        self.previous_step_on_line = self.on_line
    def ns_2_s(self, ns):
        #nanoseconds to seconds
        return ns / 1_000_000_000


def main(args=None):
    rclpy.init(args=args)
    node = Evaluation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.get_logger().info(f"{node.lap_times}")
        node.destroy_node()
        rclpy.shutdown()
# def analyze(avg_speeds, lap_times):
#     return fastest_time, slowest_time, avg_lap_time

if __name__ == '__main__':
    main()
