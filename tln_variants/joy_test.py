### Node to test whether joystick is working or not.


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

class JoyListener(Node):
    def __init__(self):
        super().__init__('joy_listener')
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10  # Queue size
        )
        self.subscription  # Prevent unused variable warning

    def joy_callback(self, msg: Joy):
        self.get_logger().info(f'Axes: {msg.axes}, Buttons: {msg.buttons}')


def main(args=None):
    rclpy.init(args=args)
    node = JoyListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()