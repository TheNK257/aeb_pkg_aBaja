#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from radar_msgs.msg import RadarTrackList
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

class SensorFusion(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.radar_sub = self.create_subscription(
            RadarTrackList, '/RadarObjects', self.radar_callback, 10)
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10)
        self.obstacle_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/obstacles', 10)
        self.bridge = CvBridge()
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLOv8: {e}')
            rclpy.shutdown()
        self.radar_data = []
        self.frame_width = 640
        self.frame_height = 480
        self.mtx = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        self.T_camera_to_radar = np.array([0, 0, 0], dtype=np.float32)
        self.R_camera_to_radar = np.eye(3, dtype=np.float32)
        self.get_logger().info('SensorFusion initialized')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (self.frame_width, self.frame_height))
        
        # YOLO object detection
        results = self.model(cv_image)
        detections = Detection2DArray()
        detections.header = msg.header
        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].tolist()
            conf = box.conf.item()
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            if class_name in ['car', 'person', 'bicycle']:
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.position.x = x + w / 2
                detection.bbox.center.position.y = y + h / 2
                detection.bbox.size_x = w
                detection.bbox.size_y = h
                detection.results.append(Detection2D.Hypothesis(class_id=class_name, score=conf))
                detections.detections.append(detection)
        self.detection_pub.publish(detections)
        
        # Fuse with radar data
        self.fuse_sensors(detections)

    def radar_callback(self, msg):
        self.radar_data = msg.objects  # Update with latest radar tracks
        self.radar_data = [data for data in self.radar_data if 
                          (self.get_clock().now().to_msg().sec - msg.header.stamp.sec) < 1.0]

    def project_radar_to_image(self, radar_point):
        x = radar_point.x_distance
        y = radar_point.y_distance
        z = 0  # Assume ground plane
        point_radar = np.array([x, y, z], dtype=np.float32)
        point_camera = self.R_camera_to_radar @ point_radar + self.T_camera_to_radar
        x_cam, y_cam, z_cam = point_camera
        u = self.mtx[0, 0] * x_cam / z_cam + self.mtx[0, 2] if z_cam != 0 else 0
        v = self.mtx[1, 1] * y_cam / z_cam + self.mtx[1, 2] if z_cam != 0 else 0
        return u, v, x, radar_point.vx

    def fuse_sensors(self, detections):
        for detection in detections.detections:
            class_name = detection.results[0].class_id
            confidence = detection.results[0].score
            if confidence < 0.5 or class_name not in ['car', 'person', 'bicycle']:
                continue
            x, y = detection.bbox.center.position.x, detection.bbox.center.position.y
            w, h = detection.bbox.size_x, detection.bbox.size_y
            bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
            min_distance = float('inf')
            associated_velocity = 0.0
            associated = False
            for radar_point in self.radar_data:
                u, v, distance, velocity = self.project_radar_to_image(radar_point)
                if bbox[0] <= u <= bbox[2] and bbox[1] <= v <= bbox[3]:
                    associated = True
                    if distance < min_distance:
                        min_distance = distance
                        associated_velocity = velocity
            if associated:
                obstacle = PoseWithCovarianceStamped()
                obstacle.header = detection.header
                obstacle.pose.pose.position.x = min_distance
                obstacle.pose.pose.position.y = 0.0
                obstacle.pose.pose.position.z = 0.0
                obstacle.pose.covariance[0] = associated_velocity  # Store velocity in covariance[0]
                self.obstacle_pub.publish(obstacle)
                self.get_logger().info(
                    f'Obstacle: {class_name}, distance: {min_distance:.2f}m, velocity: {associated_velocity:.2f}m/s')

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()