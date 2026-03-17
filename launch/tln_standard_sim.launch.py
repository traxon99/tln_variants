import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('tln_variants'),
        'config',
        'tln_standard_sim.yaml'
    )

    return LaunchDescription([
        Node(
            package='tln_variants',
            executable='tln_standard',
            name='tln_standard',
            parameters=[config],
            output='screen',
        )
    ])
