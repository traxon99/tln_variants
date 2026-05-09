import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Header, Float32


#Const params
MAX_JOYSTICK = 32767
MAX_STEER = 1


class JoyNode(Node):
    def __init__(self):
        super().__init__('joy_node')
        self.min_speed = 0
        self.max_speed = 0.2

        self.steering_publisher = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_publisher = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

    def joy_callback(self, msg: Joy):

        #get values [0,1] from joysticks (raw)
        left_y = msg.axes[1] #/ MAX_JOYSTICK
        right_x = msg.axes[2] #/ MAX_JOYSTICK

        #convert to speed/steering angle
        speed = left_y * self.max_speed #+ 3
        steering_angle = right_x * MAX_STEER
                
        self.publish_drive(speed, steering_angle)


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