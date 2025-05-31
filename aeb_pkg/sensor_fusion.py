#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from ultralytics import YOLO
from radar_msgs.msg import RadarTrackList
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import time

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        self.radar_sub = self.create_subscription(
            RadarTrackList, '/RadarObjects', self.radar_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/RGBImage', self.camera_callback, 10)
        self.detections_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.obstacles_pub = self.create_publisher(PoseWithCovarianceStamped, '/obstacles', 10)
        
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        
        self.mtx = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        self.T_camera_to_radar = np.array([0, 0, 0], dtype=np.float32)
        self.R_camera_to_radar = np.eye(3, dtype=np.float32)
        
        self.radar_data = None
        self.simulation_ready = False
        self.initial_delay = 15.0  # Increased to 15 seconds
        self.start_time = time.time()

        self.get_logger().info(f'Waiting {self.initial_delay} seconds for simulation to initialize...')
        while time.time() - self.start_time < self.initial_delay:
            time.sleep(0.1)
        self.simulation_ready = True
        self.get_logger().info('SensorFusionNode initialized and ready')

    def radar_callback(self, msg):
        if not self.simulation_ready:
            return
        self.radar_data = msg
        self.fuse_data()

    def camera_callback(self, msg):
        if not self.simulation_ready:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            results = self.model(cv_image)
            
            detections = Detection2DArray()
            detections.header = msg.header
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    detection = Detection2D()
                    detection.bbox.center.position.x = (box[0] + box[2]) / 2
                    detection.bbox.center.position.y = (box[1] + box[3]) / 2
                    detection.bbox.size_x = box[2] - box[0]
                    detection.bbox.size_y = box[3] - box[1]
                    detection.results.append(conf)
                    detections.detections.append(detection)
            
            self.detections_pub.publish(detections)
            self.fuse_data()
        except Exception as e:
            self.get_logger().error(f'Camera callback failed: {e}')

    def fuse_data(self):
        if self.radar_data is None:
            return
        
        obstacle = PoseWithCovarianceStamped()
        obstacle.header.stamp = self.get_clock().now().to_msg()
        obstacle.header.frame_id = "base_link"
        
        closest_dist = float('inf')
        closest_track = None
        for track in self.radar_data.objects:
            dist = track.x_distance
            if dist < closest_dist and dist > 0:
                closest_dist = dist
                closest_track = track
        
        if closest_track:
            obstacle.pose.pose.position.x = closest_track.x_distance
            obstacle.pose.covariance[0] = closest_track.vx
            self.obstacles_pub.publish(obstacle)
            self.get_logger().info(f'Published obstacle at distance: {closest_dist:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()