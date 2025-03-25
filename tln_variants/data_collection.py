import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import csv
import time
import sys


hour = str(time.localtime().tm_mon)
min = str(time.localtime().tm_min)
sec = str(time.localtime().tm_sec)
EXPORT_PATH = 'src/tln_variants/train/dataset/recording-'+ hour + ':' + min + ':' + sec + '.csv'
#EXPORT_PATH = 'data'+ hour + '-' + min + '-' + sec + '.csv'


class DataCollection(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.csvfile = open(EXPORT_PATH,'w')
        self.writer = csv.writer(self.csvfile)
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

        # DATA RECORDING GRANULARITY HERE. SLOP VALUE IS 5 MS
        ts = message_filters.ApproximateTimeSynchronizer([self.ackermann_subscription, self.scan_subscription],10, 0.0005) # 5 ms
        self.get_logger().warn(f'Data recording started.')
        ts.registerCallback(self.write)

    def write(self, ack_msg, scan_msg):
        # self.get_logger().info("Data written")
        #self.get_logger().info(f"{list(scan_msg.ranges)}")
        self.writer.writerow([ack_msg.drive.speed, ack_msg.drive.steering_angle, list(scan_msg.ranges)])
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
