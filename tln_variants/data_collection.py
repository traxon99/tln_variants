import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import message_filters
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Joy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class DataCollectionNode(Node):

    def __init__(self):
        super().__init__('data_collection')

        self.declare_parameter('name',                    'porto_expert_v1')
        self.declare_parameter('save_dir',                '/home/autodrive_devkit/src/tln_variants/train/dataset')
        self.declare_parameter('max_samples',             50000)
        self.declare_parameter('scan_topic',              '/autodrive/roboracer_1/lidar')
        self.declare_parameter('steer_topic',             '/autodrive/roboracer_1/steering_command')
        self.declare_parameter('throttle_topic',          '/autodrive/roboracer_1/throttle_command')
        self.declare_parameter('odom_topic',              '/autodrive/roboracer_1/odom')
        self.declare_parameter('sync_tolerance_ms',       25.0)
        self.declare_parameter('record_only_when_moving', True)
        self.declare_parameter('min_speed_threshold',     0.01)

        name           = self.get_parameter('name').value
        save_dir       = self.get_parameter('save_dir').value
        self.max_samples      = self.get_parameter('max_samples').value
        scan_topic     = self.get_parameter('scan_topic').value
        steer_topic    = self.get_parameter('steer_topic').value
        throttle_topic = self.get_parameter('throttle_topic').value
        odom_topic     = self.get_parameter('odom_topic').value
        slop           = self.get_parameter('sync_tolerance_ms').value / 1000.0
        self.only_moving      = self.get_parameter('record_only_when_moving').value
        self.min_speed        = self.get_parameter('min_speed_threshold').value

        os.makedirs(save_dir, exist_ok=True)
        bag_dir = os.path.join(save_dir, name)

        self.writer = SequentialWriter()
        self.writer.open(
            StorageOptions(uri=bag_dir, storage_id='sqlite3'),
            ConverterOptions('', ''),
        )
        self.writer.create_topic(
            TopicMetadata(name='scan',  type='sensor_msgs/msg/LaserScan',               serialization_format='cdr')
        )
        self.writer.create_topic(
            TopicMetadata(name='drive', type='ackermann_msgs/msg/AckermannDriveStamped', serialization_format='cdr')
        )

        self.msg_counter      = 0
        self.recording_active = True
        self._current_speed   = 0.0
        self._writer_closed   = False

        sensor_qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        lidar_sub    = message_filters.Subscriber(self, LaserScan, scan_topic,     qos_profile=sensor_qos)
        steer_sub    = message_filters.Subscriber(self, Float32,   steer_topic,    qos_profile=10)
        throttle_sub = message_filters.Subscriber(self, Float32,   throttle_topic, qos_profile=10)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, steer_sub, throttle_sub],
            queue_size=20,
            slop=slop,
            allow_headerless=True,
        )
        self._sync.registerCallback(self._sync_cb)

        self._odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self._joy_sub  = self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        self.get_logger().info(
            f'DataCollection ready → {bag_dir}/\n'
            f'Max samples: {self.max_samples}  |  '
            f'PS4 □ to pause/resume  |  CTRL+C to stop'
        )

    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._current_speed = (vx**2 + vy**2) ** 0.5

    def _joy_cb(self, msg: Joy):
        if msg.buttons[2] == 1:   # PS4 square
            self.recording_active = not self.recording_active
            state = 'RESUMED' if self.recording_active else 'PAUSED'
            self.get_logger().info(f'Recording {state}')

    def _sync_cb(self, scan_msg: LaserScan, steer_msg: Float32, throttle_msg: Float32):
        if not self.recording_active:
            return
        if self.only_moving and self._current_speed < self.min_speed:
            return
        if self.msg_counter >= self.max_samples:
            return

        timestamp = self.get_clock().now().nanoseconds

        # Write raw LaserScan — training script downsamples by factor 2 itself
        self.writer.write('scan', serialize_message(scan_msg), timestamp)

        # Construct AckermannDriveStamped from Float32 simulator commands:
        #   steering_angle: normalized steering  [-1, 1]
        #   speed:          normalized throttle  [ 0, 1]
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = scan_msg.header.stamp
        drive_msg.drive.steering_angle = float(steer_msg.data)
        drive_msg.drive.speed          = float(throttle_msg.data)
        self.writer.write('drive', serialize_message(drive_msg), timestamp)

        self.msg_counter += 1

        if self.msg_counter % 100 == 0:
            self.get_logger().info(f'Recorded {self.msg_counter} / {self.max_samples} samples')

        if self.msg_counter >= self.max_samples:
            self.get_logger().info('Max samples reached — stopping.')
            self._close()

    def _close(self):
        if self._writer_closed:
            return
        self._writer_closed = True
        self.writer.close()
        self.get_logger().info(f'Saved {self.msg_counter} samples → bag')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
