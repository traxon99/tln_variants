import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import message_filters
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan, Joy
from nav_msgs.msg import Odometry
import sys
from rosbags.rosbag2 import Writer
from rosbags.serde import serialize_cdr
from copy import deepcopy

class DataCollectionNode(Node):

    STOP_AFTER_SEC = 5 * 60  # seconds

    def __init__(self, name):
        super().__init__('sim_data_collector')
        self.file_name = name
        self.start_time = self.get_clock().now()
        self.recording_active = True
        self.msg_counter = 0

        # Initialize rosbag writer
        self.writer = Writer(self.file_name + '.db3')
        self.writer.open()

        # QoS for sensors
        sensor_qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.lidar_sub = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=sensor_qos)
        self.ack_sub = message_filters.Subscriber(self, AckermannDriveStamped, '/drive', qos_profile=10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.ack_sub],
            queue_size=20,
            slop=0.05,
            allow_headerless=True
        )
        self.ts.registerCallback(self.sensor_callback)

        # Timer to stop recording
        self.stop_timer = self.create_timer(1.0, self._check_time)
        self.get_logger().info('Data collector node initialized!')

    def sensor_callback(self, scan_msg, drive_msg):
        if not self.recording_active:
            return

        try:
            # Clone scan_msg and convert array.array fields to lists
            scan_copy = deepcopy(scan_msg)
            scan_copy.ranges = list(scan_copy.ranges)
            scan_copy.intensities = list(scan_copy.intensities)

            # Use message header timestamp for nanoseconds
            scan_ts = scan_copy.header.stamp.sec * 1e9 + scan_copy.header.stamp.nanosec
            drive_ts = drive_msg.header.stamp.sec * 1e9 + drive_msg.header.stamp.nanosec

            self.writer.write('/scan', serialize_cdr(scan_copy, 'sensor_msgs/msg/LaserScan'), int(scan_ts))
            self.writer.write('/drive', serialize_cdr(drive_msg, 'ackermann_msgs/msg/AckermannDriveStamped'), int(drive_ts))

            self.msg_counter += 1
            if self.msg_counter % 10 == 0:
                self.get_logger().info(f'Writing data... ({self.msg_counter} messages recorded)')

        except Exception as e:
            self.get_logger().error(f'Recording error: {e}')
            
    def joy_callback(self, joy_msg: Joy):
        # PS4 square button is buttons[2]
        if joy_msg.buttons[2] == 1:
            # Toggle recording
            self.recording_active = not self.recording_active
            status = "RESUMED" if self.recording_active else "PAUSED"
            self.get_logger().info(f'Recording {status} via PS4 square button!')

    def _check_time(self):
        elapsed_sec = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if self.recording_active and elapsed_sec >= self.STOP_AFTER_SEC:
            self._stop_recording()

    def _stop_recording(self):
        self.recording_active = False
        self.writer.close()
        self.get_logger().info('Recording stopped after the time selected')


def main(args=None):
    rclpy.init(args=args)
    name = str(sys.argv[1])
    collector = DataCollectionNode(name)

    try:
        collector.get_logger().info('Starting data collection...')
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info('Keyboard interrupt received')
    finally:
        if collector.writer:
            collector.writer.close()
        collector.get_logger().info(f'Recording COMPLETE! Total messages: {collector.msg_counter}')
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
