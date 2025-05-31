from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='pycarmaker',
        #     executable='CarmakerInterface',
        #     name='CarmakerInterface',
        #     output='screen'
        # ),
        # Node(
        #     package='camera_sensor',
        #     executable='CameraFramePublisher',
        #     name='CameraFramePublisher',
        #     output='screen'
        # ),
        Node(
            package='aeb_pkg',
            executable='sensor_fusion',
            name='sensor_fusion',
            output='screen'
        ),
        Node(
            package='aeb_pkg',
            executable='aeb_node',
            name='aeb_node',
            output='screen'
        ),
    ])