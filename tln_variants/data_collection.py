import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import csv
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy

hour = str(time.localtime().tm_hour)
min = str(time.localtime().tm_min)
sec = str(time.localtime().tm_sec)
EXPORT_PATH = 'src/tln_variants/train/dataset/recording-'+ hour + ':' + min + ':' + sec + '.csv'
#EXPORT_PATH = 'data'+ hour + '-' + min + '-' + sec + '.csv'

CONTROLLER = True

class DataCollection(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.csvfile = open(EXPORT_PATH,'w')
        self.writer = csv.writer(self.csvfile)

        #joy subscription
        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10  # Queue size
        )
        
        #Subscription list for message sync
        subscription_list = []

        self.scan_subscription = message_filters.Subscriber(
            self,
            LaserScan,
            '/scan', 
            qos_profile=10
        )
        self.ackermann_subscription = message_filters.Subscriber(
            self,
            AckermannDriveStamped, 
            '/drive', 
            qos_profile=10
        )
        self.odom_subscription = message_filters.Subscriber(
            self,
            Odometry,
            '/ego_racecar/odom',
            qos_profile=10
        )

        subscription_list.append(self.scan_subscription)
        subscription_list.append(self.ackermann_subscription)
        subscription_list.append(self.odom_subscription)     
        #self.get_logger().info(f"Subscribed to {subscription_list}")
        
        # DATA RECORDING GRANULARITY HERE. SLOP VALUE IS 5 MS
        # ts = message_filters.ApproximateTimeSynchronizer(subscription_list,10, 0.0005) # 5 ms
        ts = message_filters.ApproximateTimeSynchronizer(subscription_list, 10, 0.0005) # 90 ms

        self.get_logger().warn(f'Data recording Node Initialized.')

        ts.registerCallback(self.write)
        self.recording = False

    def joy_callback(self, msg):
        pass
    def write(self, scan_msg, ack_msg, odom_msg):
        vel_x = odom_msg.twist.twist.linear.x
        vel_y = odom_msg.twist.twist.linear.y
        vel_mag = (vel_x**2 + vel_y**2)**0.5
        if CONTROLLER == True: 
            pass
        self.get_logger().info("Data written")
        #self.get_logger().info(f"{list(scan_msg.ranges)}")
        # self.writer.writerow([ack_msg.drive.speed, ack_msg.drive.steering_angle, vel_mag, scan_msg.ranges])#, vel_mag])
        self.writer.writerow([ack_msg.drive.speed, ack_msg.drive.steering_angle, vel_mag, list(scan_msg.ranges)])#, vel_mag])

    def close_file(self):
        self.csvfile.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.get_logger().info('Data collection shutting down. Closing CSV...')
        node.close_file()
        node.get_logger().warn(f'Data saved at {EXPORT_PATH}! \n\nPlease rename dataset file to a meaningful name to avoid confusion!')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
