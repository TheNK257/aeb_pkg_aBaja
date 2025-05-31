#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Accel
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from inertial_msgs.msg import Pose

class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb_node')
        self.obstacle_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/obstacles', self.obstacle_callback, 10)
        self.vehicle_state_sub = self.create_subscription(
            Pose, '/vehicle/state', self.vehicle_state_callback, 10)
        self.control_pub = self.create_publisher(Accel, '/vehicle/control', 10)
        self.aeb_status_pub = self.create_publisher(Bool, '/aeb/status', 10)
        self.vehicle_speed = 10.0  # Default: 10 m/s (36 km/h)
        self.safe_distance_min = 3.0  # meters
        self.reaction_time = 0.5  # seconds
        self.ttc_threshold = 2.0  # seconds
        self.get_logger().info('AEBNode initialized')

    def vehicle_state_callback(self, msg):
        self.vehicle_speed = msg.velocity.x  # Forward speed from Pose.velocity.x

    def obstacle_callback(self, msg):
        distance = msg.pose.pose.position.x
        relative_velocity = msg.pose.covariance[0]  # Velocity stored in covariance[0]
        safe_distance = self.vehicle_speed * self.reaction_time + self.safe_distance_min
        
        # Calculate TTC
        ttc = distance / abs(relative_velocity) if relative_velocity < 0 else float('inf')
        
        # AEB decision
        control_msg = Accel()
        aeb_status = Bool()
        if distance < safe_distance or ttc < self.ttc_threshold:
            control_msg.linear.x = -5.0  # Deceleration in m/s^2
            aeb_status.data = True
            self.get_logger().warn(f'AEB Triggered: distance: {distance:.2f}m, TTC: {ttc:.2f}s')
        else:
            control_msg.linear.x = 0.0
            aeb_status.data = False
        
        self.control_pub.publish(control_msg)
        self.aeb_status_pub.publish(aeb_status)

def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()