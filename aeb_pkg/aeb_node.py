#!/usr/bin/env python3  
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from inertial_msgs.msg import Pose
from vehiclecontrol.msg import Control
import time

class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb_node')
        self.get_logger().info('AEBNode initialized with longitudinal control')

        self.obstacle_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/obstacles', self.obstacle_callback, 10)
        self.vehicle_state_sub = self.create_subscription(
            Pose, '/InertialData', self.vehicle_state_callback, 10)

        self.control_pub = self.create_publisher(Control, '/vehicle_control', 10)
        self.aeb_status_pub = self.create_publisher(Bool, '/aeb/status', 10)
        
        self.vehicle_speed = 0.0
        self.vehicle_position_x = 0.0
        self.last_position_x = 0.0
        
        self.target_speed = 30.0 * (1000.0 / 3600.0)
        self.acceleration = 2.0
        self.distance_threshold = 100.0
        
        self.safe_distance_min = 3.0
        self.reaction_time = 0.5
        self.ttc_threshold = 2.0
        self.brake_deceleration = -5.0
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.obstacle_velocity = 0.0
        
        self.simulation_ready = False
        self.position_update_count = 0  # Counter for consistent updates
        self.initial_delay = 15.0  # Increased to 15 seconds
        self.start_time = time.time()

        self.get_logger().info(f'Waiting {self.initial_delay} seconds for simulation to initialize...')
        while time.time() - self.start_time < self.initial_delay:
            time.sleep(0.1)

        self.get_logger().info('Initial delay complete, waiting for consistent /InertialData updates...')

    def vehicle_state_callback(self, msg):
        self.vehicle_speed = msg.velocity.x
        self.vehicle_position_x = msg.position.x
        
        if not self.simulation_ready:
            if self.vehicle_position_x > self.last_position_x and self.vehicle_position_x > 0.0:
                self.position_update_count += 1
                if self.position_update_count >= 10:  # Wait for 10 consistent updates
                    self.simulation_ready = True
                    self.get_logger().info('Simulation is ready (consistent position updates), starting control commands.')
            self.last_position_x = self.vehicle_position_x
            return
        
        self.publish_control()

    def obstacle_callback(self, msg):
        self.obstacle_detected = True
        self.obstacle_distance = msg.pose.pose.position.x
        self.obstacle_velocity = msg.pose.covariance[0]
        
        if self.simulation_ready:
            self.publish_control()

    def publish_control(self):
        control_msg = Control()
        aeb_status = Bool()
        
        acceleration = 0.0
        if self.obstacle_detected:
            safe_distance = self.vehicle_speed * self.reaction_time + self.safe_distance_min
            ttc = self.obstacle_distance / abs(self.obstacle_velocity) if self.obstacle_velocity < 0 else float('inf')
            
            if self.obstacle_distance < safe_distance or ttc < self.ttc_threshold:
                acceleration = self.brake_deceleration
                aeb_status.data = True
                self.get_logger().warn(f'AEB Triggered: distance: {self.obstacle_distance:.2f}m, TTC: {ttc:.2f}s')
            else:
                aeb_status.data = False
                acceleration = self.calculate_longitudinal_control()
        else:
            aeb_status.data = False
            acceleration = self.calculate_longitudinal_control()
        
        if acceleration >= 0:
            control_msg.throttle = min(acceleration / 2.0, 1.0)
            control_msg.brake = 0.0
        else:
            control_msg.throttle = 0.0
            control_msg.brake = min(-acceleration / 5.0, 1.0)
        
        control_msg.steering = 0.0
        control_msg.latswitch = 0
        control_msg.longswitch = 1
        
        self.control_pub.publish(control_msg)
        self.aeb_status_pub.publish(aeb_status)

    def calculate_longitudinal_control(self):
        if self.vehicle_position_x < self.distance_threshold and self.vehicle_speed < self.target_speed:
            acceleration = self.acceleration
            self.get_logger().info(f'Accelerating: speed={self.vehicle_speed:.2f} m/s, position={self.vehicle_position_x:.2f} m')
        else:
            if self.vehicle_speed > self.target_speed + 0.1:
                acceleration = -0.5
            elif self.vehicle_speed < self.target_speed - 0.1:
                acceleration = 0.5
            else:
                acceleration = 0.0
            self.get_logger().info(f'Maintaining speed: speed={self.vehicle_speed:.2f} m/s')
        return acceleration

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