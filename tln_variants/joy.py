import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header


#Const params
MAX_JOYSTICK = 32767
MAX_STEER = 0.52


class JoyNode(Node):
    def __init__(self):
        super().__init__('joy_node')
        self.min_speed = 2
        self.max_speed = 9

        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10  # Queue size
        )
        self.ackermann_publisher = self.create_publisher(
            AckermannDriveStamped, 
            '/drive', 
            10
        )
        

    def joy_callback(self, msg: Joy):

        #get values [0,1] from joysticks (raw)
        left_js_up_down = msg.axes[1] #/ MAX_JOYSTICK
        right_js_left_right = msg.axes[3] #/ MAX_JOYSTICK

        #convert to speed/steering angle
        speed = left_js_up_down * self.max_speed #+ 3
        steering_angle = right_js_left_right * MAX_STEER * 0.7
                
        self.publish_ackermann_drive(speed, steering_angle)


    def publish_ackermann_drive(self, speed, steering_angle):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDriveStamped message: speed={speed}, steering_angle={steering_angle}')
    
    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    

def main(args=None):
    rclpy.init(args=args)
    node = JoyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()