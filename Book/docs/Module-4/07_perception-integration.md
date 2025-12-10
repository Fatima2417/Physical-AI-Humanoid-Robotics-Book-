---
sidebar_position: 7
---

# Perception Integration

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate multiple perception sensors into a unified system
- Implement sensor fusion techniques for enhanced robot perception
- Design perception pipelines for different robotics applications
- Apply machine learning for perception tasks in robotics
- Optimize perception processing for real-time applications
- Validate perception system performance and accuracy

## Introduction to Perception Integration

Perception is the cornerstone of robotic autonomy, enabling robots to understand and interact with their environment. Perception integration involves combining data from multiple sensors to create a comprehensive understanding of the world. This chapter explores how to effectively integrate various perception modalities for robotic applications.

### Perception Integration Architecture

```
Perception Integration Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Sensors   │───→│  Preprocessing  │───→│  Feature        │
│   • Cameras     │    │  • Filtering    │    │  Extraction     │
│   • LiDAR       │    │  • Calibration  │    │  • Object Det.  │
│   • IMU         │    │  • Alignment    │    │  • Segmentation │
│   • GPS         │    │  • Synchronization│   │  • Tracking     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │  Fusion &      │
                         │  Interpretation │
                         │  • Sensor      │
                         │  • Context     │
                         │  • Uncertainty │
                         └─────────────────┘
```

### Key Perception Modalities

1. **Vision (Cameras)**: Rich visual information, object recognition
2. **LiDAR**: Precise 3D geometry, distance measurements
3. **Radar**: Weather-resistant sensing, velocity measurements
4. **Sonar**: Short-range obstacle detection
5. **IMU**: Inertial measurements, orientation
6. **GPS**: Global positioning information

## Camera Perception Integration

### RGB Camera Integration

```python
# camera_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from typing import List, Dict, Any, Optional

class CameraPerceptionNode(Node):
    def __init__(self):
        super().__init__('camera_perception')

        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, 'camera/detections', 10)
        self.processed_image_pub = self.create_publisher(Image, 'camera/image_processed', 10)
        self.perception_status_pub = self.create_publisher(String, 'camera/perception_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.image_width = 640
        self.image_height = 480

        # Object detection model
        try:
            self.model = YOLO('yolov8n.pt')  # You can use yolov8x.pt for better accuracy
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Could not load YOLO model: {e}')
            self.model = None

        # Processing parameters
        self.confidence_threshold = 0.5
        self.enable_object_detection = True
        self.enable_segmentation = False
        self.enable_tracking = False

        # Threading for processing
        self.processing_lock = threading.Lock()
        self.latest_image = None

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('Camera Perception Node initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.image_width = msg.width
        self.image_height = msg.height

        self.get_logger().info(f'Camera calibration updated: {self.image_width}x{self.image_height}')

    def image_callback(self, msg):
        """Process incoming camera images."""
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Process image in separate thread to avoid blocking
        processing_args = (cv_image, msg.header)
        processing_thread = threading.Thread(
            target=self.process_image_threaded,
            args=processing_args,
            daemon=True
        )
        processing_thread.start()

        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Log performance periodically
        self.frame_count += 1
        if self.frame_count % 50 == 0:  # Every 50 frames
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'Camera processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}'
            )

    def process_image_threaded(self, cv_image, header):
        """Process image in separate thread."""
        with self.processing_lock:
            # Apply undistortion if calibration is available
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                cv_image = cv2.undistort(
                    cv_image,
                    self.camera_matrix,
                    self.distortion_coeffs,
                    None,
                    self.camera_matrix
                )

            # Process with YOLO if enabled
            if self.enable_object_detection and self.model is not None:
                results = self.model(
                    cv_image,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                # Convert YOLO results to ROS messages
                detections_msg = self.convert_yolo_to_ros_detections(results, header)

                # Publish detections
                self.detections_pub.publish(detections_msg)

                # Draw detections on image for visualization
                annotated_image = results[0].plot()
            else:
                annotated_image = cv_image

            # Publish processed image
            try:
                processed_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                processed_msg.header = header
                self.processed_image_pub.publish(processed_msg)
            except Exception as e:
                self.get_logger().error(f'Could not convert processed image: {e}')

    def convert_yolo_to_ros_detections(self, yolo_results, header):
        """Convert YOLO results to ROS Detection2DArray message."""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        for result in yolo_results:
            for box in result.boxes:
                detection = Detection2D()

                # Convert bounding box format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)
                width = float(x2 - x1)
                height = float(y2 - y1)

                detection.bbox.center.x = center_x
                detection.bbox.center.y = center_y
                detection.bbox.size_x = width
                detection.bbox.size_y = height

                # Create hypothesis
                hypothesis = ObjectHypothesisWithPose()
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]  # YOLO class names
                confidence = float(box.conf[0].item())

                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence

                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)

        return detections_msg

    def enable_object_detection(self, enable=True):
        """Enable or disable object detection."""
        self.enable_object_detection = enable
        status_msg = String()
        status_msg.data = f'Object detection: {\"enabled\" if enable else \"disabled\"}'
        self.perception_status_pub.publish(status_msg)
        self.get_logger().info(f'Object detection {status_msg.data}')

    def set_confidence_threshold(self, threshold):
        """Set detection confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.get_logger().info(f'Confidence threshold set to: {self.confidence_threshold}')

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraPerceptionNode()

    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Shutting down Camera Perception Node')
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Stereo Camera Integration

```python
# stereo_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from collections import deque
import threading

class StereoPerceptionNode(Node):
    def __init__(self):
        super().__init__('stereo_perception')

        # Publishers
        self.disparity_pub = self.create_publisher(DisparityImage, 'stereo/disparity', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'stereo/pointcloud', 10)
        self.depth_image_pub = self.create_publisher(Image, 'stereo/depth_image', 10)

        # Subscribers
        self.left_image_sub = self.create_subscription(
            Image, 'stereo/left/image_rect', self.left_image_callback, 10
        )
        self.right_image_sub = self.create_subscription(
            Image, 'stereo/right/image_rect', self.right_image_callback, 10
        )
        self.left_info_sub = self.create_subscription(
            CameraInfo, 'stereo/left/camera_info', self.left_info_callback, 10
        )
        self.right_info_sub = self.create_subscription(
            CameraInfo, 'stereo/right/camera_info', self.right_info_callback, 10
        )

        # CV Bridge
        self.bridge = CvBridge()

        # Stereo parameters
        self.left_image = None
        self.right_image = None
        self.left_camera_info = None
        self.right_camera_info = None

        # Stereo processing parameters
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # Must be divisible by 16
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Q matrix for point cloud reconstruction (computed from camera calibration)
        self.Q = None

        # Threading for processing
        self.processing_thread = threading.Thread(target=self.process_stereo_pair, daemon=True)
        self.processing_lock = threading.Lock()
        self.new_data_available = threading.Event()

        # Buffer for synchronization
        self.left_buffer = deque(maxlen=5)
        self.right_buffer = deque(maxlen=5)

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('Stereo Perception Node initialized')

    def left_image_callback(self, msg):
        """Process left camera image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Left image conversion error: {e}')
            return

        with self.processing_lock:
            self.left_buffer.append((cv_image, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))

        self.new_data_available.set()

    def right_image_callback(self, msg):
        """Process right camera image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Right image conversion error: {e}')
            return

        with self.processing_lock:
            self.right_buffer.append((cv_image, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))

        self.new_data_available.set()

    def left_info_callback(self, msg):
        """Process left camera calibration info."""
        self.left_camera_info = msg
        self.compute_stereo_parameters()

    def right_info_callback(self, msg):
        """Process right camera calibration info."""
        self.right_camera_info = msg
        self.compute_stereo_parameters()

    def compute_stereo_parameters(self):
        """Compute stereo rectification parameters."""
        if self.left_camera_info is None or self.right_camera_info is None:
            return

        # Extract camera matrices and distortion coefficients
        left_K = np.array(self.left_camera_info.k).reshape(3, 3)
        right_K = np.array(self.right_camera_info.k).reshape(3, 3)
        left_D = np.array(self.left_camera_info.d)
        right_D = np.array(self.right_camera_info.d)

        # Extract baseline (translation between cameras)
        # This assumes the extrinsic parameters are in the P matrix
        if len(self.left_camera_info.p) >= 12:
            baseline = -self.left_camera_info.p[3] / self.left_camera_info.p[0]  # P[0,3] / P[0,0]
        else:
            baseline = 0.1  # Default baseline if not available

        # Compute Q matrix for 3D reconstruction
        # This is a simplified version - in practice, you'd use stereo rectification
        focal_length = left_K[0, 0]  # fx
        center_x = left_K[0, 2]     # cx
        center_y = left_K[1, 2]     # cy

        self.Q = np.array([
            [1, 0, 0, -center_x],
            [0, 1, 0, -center_y],
            [0, 0, 0, focal_length],
            [0, 0, -1/baseline, 0]
        ])

        self.get_logger().info('Stereo parameters computed successfully')

    def process_stereo_pair(self):
        """Process stereo image pairs in background thread."""
        while rclpy.ok():
            try:
                self.new_data_available.wait(timeout=0.1)
                self.new_data_available.clear()

                with self.processing_lock:
                    # Find temporally matched pairs
                    matched_pairs = self.match_images_by_timestamp()

                if matched_pairs:
                    left_img, right_img, header = matched_pairs[0]

                    start_time = time.time()

                    # Compute disparity
                    disparity = self.stereo_matcher.compute(left_img, right_img).astype(np.float32) / 16.0

                    # Create disparity message
                    disparity_msg = DisparityImage()
                    disparity_msg.header = header

                    # Convert disparity to 32-bit float image
                    disparity_msg.image = self.bridge.cv2_to_imgmsg(disparity, encoding='32FC1')
                    disparity_msg.image.header = header

                    # Set disparity parameters
                    disparity_msg.f = self.left_camera_info.k[0] if self.left_camera_info else 320.0  # Focal length in pixels
                    disparity_msg.t = 0.1  # Baseline (meters) - should come from calibration

                    self.disparity_pub.publish(disparity_msg)

                    # Generate point cloud from disparity
                    if self.Q is not None:
                        pointcloud_msg = self.generate_pointcloud_from_disparity(disparity, header)
                        if pointcloud_msg:
                            self.pointcloud_pub.publish(pointcloud_msg)

                    # Generate depth image
                    depth_image = self.disparity_to_depth(disparity)
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
                    depth_msg.header = header
                    self.depth_image_pub.publish(depth_msg)

                    # Track performance
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)

                    if len(self.processing_times) > 100:
                        self.processing_times.pop(0)

                    # Log performance periodically
                    self.frame_count += 1
                    if self.frame_count % 20 == 0:
                        avg_time = sum(self.processing_times) / len(self.processing_times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0
                        self.get_logger().info(
                            f'Stereo processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}'
                        )

            except Exception as e:
                self.get_logger().error(f'Stereo processing error: {e}')

    def match_images_by_timestamp(self, max_time_diff=0.05):
        """Match left and right images by timestamp."""
        matched_pairs = []

        for left_img, left_ts in list(self.left_buffer):
            for right_img, right_ts in list(self.right_buffer):
                time_diff = abs(left_ts - right_ts)
                if time_diff < max_time_diff:
                    # Create header with average timestamp
                    avg_ts = (left_ts + right_ts) / 2.0
                    header = Header()
                    header.stamp.sec = int(avg_ts)
                    header.stamp.nanosec = int((avg_ts - int(avg_ts)) * 1e9)
                    header.frame_id = 'stereo_camera'  # Should come from camera info

                    matched_pairs.append((left_img, right_img, header))

                    # Remove processed images
                    self.left_buffer.remove((left_img, left_ts))
                    self.right_buffer.remove((right_img, right_ts))

                    break  # Process one pair at a time

        return matched_pairs

    def generate_pointcloud_from_disparity(self, disparity, header):
        """Generate point cloud from disparity image."""
        if self.Q is None:
            return None

        # Reproject disparity to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        # Get valid points (non-zero disparities)
        valid_mask = disparity > 0
        valid_points = points_3d[valid_mask]

        if len(valid_points) == 0:
            return None

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        return pc2.create_cloud(header, fields, valid_points)

    def disparity_to_depth(self, disparity):
        """Convert disparity to depth image."""
        # Depth = baseline * focal_length / disparity
        if self.Q is not None:
            baseline = abs(self.Q[3, 2])  # Baseline from Q matrix
            focal_length = self.Q[2, 3]   # Focal length from Q matrix
        else:
            baseline = 0.1  # Default baseline
            focal_length = 320.0  # Default focal length

        # Avoid division by zero
        depth = np.zeros_like(disparity)
        valid_disparities = disparity > 0
        depth[valid_disparities] = (baseline * focal_length) / disparity[valid_disparities]

        return depth

def main(args=None):
    rclpy.init(args=args)
    stereo_node = StereoPerceptionNode()

    # Start processing thread
    stereo_node.processing_thread.start()

    try:
        rclpy.spin(stereo_node)
    except KeyboardInterrupt:
        stereo_node.get_logger().info('Shutting down Stereo Perception Node')
    finally:
        stereo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## LiDAR Perception Integration

### Point Cloud Processing

```python
# lidar_perception.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import struct
from collections import deque
import threading
import time

class LiDARPerceptionNode(Node):
    def __init__(self):
        super().__init__('lidar_perception')

        # Publishers
        self.obstacles_pub = self.create_publisher(PointCloud2, 'lidar/obstacles', 10)
        self.ground_pub = self.create_publisher(PointCloud2, 'lidar/ground', 10)
        self.clustered_pub = self.create_publisher(PointCloud2, 'lidar/clustered', 10)
        self.obstacles_viz_pub = self.create_publisher(MarkerArray, 'lidar/obstacle_markers', 10)

        # Subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 'velodyne_points', self.pointcloud_callback, 10
        )

        # Processing parameters
        self.enable_ground_removal = True
        self.enable_clustering = True
        self.enable_obstacle_detection = True
        self.enable_visualization = True

        # Ground removal parameters
        self.ground_height_threshold = -0.1  # meters
        self.ransac_distance_threshold = 0.1
        self.ransac_max_iterations = 100

        # Clustering parameters
        self.cluster_epsilon = 0.5  # meters
        self.min_cluster_points = 10

        # Obstacle detection parameters
        self.obstacle_height_min = 0.1  # meters above ground
        self.obstacle_height_max = 2.0  # meters above ground
        self.obstacle_distance_threshold = 30.0  # meters

        # Threading and buffering
        self.processing_lock = threading.Lock()
        self.latest_pointcloud = None
        self.processing_thread = threading.Thread(target=self.process_pointcloud_loop, daemon=True)
        self.processing_thread.start()

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        # Visualization parameters
        self.marker_lifetime = 0.5  # seconds

        self.get_logger().info('LiDAR Perception Node initialized')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data."""
        with self.processing_lock:
            self.latest_pointcloud = msg

    def process_pointcloud_loop(self):
        """Process point clouds in background thread."""
        while rclpy.ok():
            try:
                with self.processing_lock:
                    if self.latest_pointcloud is None:
                        time.sleep(0.01)
                        continue

                    # Process the latest point cloud
                    pointcloud_msg = self.latest_pointcloud
                    self.latest_pointcloud = None

                start_time = time.time()

                # Convert PointCloud2 to numpy array
                points = []
                for point in point_cloud2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                    points.append([point[0], point[1], point[2]])

                if not points:
                    continue

                points = np.array(points)

                # Remove ground plane
                if self.enable_ground_removal:
                    obstacle_points, ground_points = self.remove_ground_plane(points)
                else:
                    obstacle_points = points
                    ground_points = np.array([])

                # Publish ground points
                if len(ground_points) > 0 and self.enable_ground_removal:
                    ground_cloud = self.create_pointcloud_msg(ground_points, pointcloud_msg.header, 'green')
                    self.ground_pub.publish(ground_cloud)

                # Cluster points for object detection
                if self.enable_clustering and len(obstacle_points) > 0:
                    clustered_cloud = self.cluster_points(obstacle_points, pointcloud_msg.header)
                    self.clustered_pub.publish(clustered_cloud)

                # Detect obstacles
                if self.enable_obstacle_detection and len(obstacle_points) > 0:
                    obstacle_cloud = self.detect_obstacles(obstacle_points, pointcloud_msg.header)
                    self.obstacles_pub.publish(obstacle_cloud)

                    # Create visualization markers
                    if self.enable_visualization:
                        obstacle_markers = self.create_obstacle_markers(obstacle_points, pointcloud_msg.header)
                        self.obstacles_viz_pub.publish(obstacle_markers)

                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)

                # Log performance periodically
                self.frame_count += 1
                if self.frame_count % 20 == 0:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    self.get_logger().info(
                        f'LiDAR processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}'
                    )

            except Exception as e:
                self.get_logger().error(f'Point cloud processing error: {e}')
                time.sleep(0.01)

    def remove_ground_plane(self, points):
        """Remove ground plane using RANSAC algorithm."""
        if len(points) < 3:
            return points, np.array([])

        best_inliers = []
        best_model = None

        # RANSAC algorithm for ground plane detection
        for _ in range(self.ransac_max_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]

            # Fit plane to sample points
            # For simplicity, we'll use a basic plane fitting
            # In practice, you'd use proper plane equations
            p1, p2, p3 = sample_points

            # Calculate plane normal (cross product of two vectors on plane)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)  # Normalize

            # Plane equation: ax + by + cz + d = 0
            d = -np.dot(normal, p1)

            # Calculate distances to plane
            distances = np.abs(np.dot(points, normal) + d)
            inliers = points[distances < self.ransac_distance_threshold]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (normal, d)

        if best_model is not None:
            normal, d = best_model
            distances = np.abs(np.dot(points, normal) + d)
            ground_mask = distances < self.ransac_distance_threshold
            ground_points = points[ground_mask]
            obstacle_points = points[~ground_mask]
        else:
            # If RANSAC failed, use simple height-based filtering
            ground_mask = points[:, 2] < self.ground_height_threshold
            ground_points = points[ground_mask]
            obstacle_points = points[~ground_mask]

        return obstacle_points, ground_points

    def cluster_points(self, points, header):
        """Cluster points using DBSCAN."""
        if len(points) < self.min_cluster_points:
            # Return original point cloud if too few points for clustering
            return self.create_pointcloud_msg(points, header, 'white')

        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=self.cluster_epsilon,
            min_samples=self.min_cluster_points
        ).fit(points)

        labels = clustering.labels_

        # Create colored point cloud based on clusters
        colored_points = []
        for point, label in zip(points, labels):
            if label == -1:  # Noise points
                color = [0.5, 0.5, 0.5]  # Gray for noise
            else:
                # Use different colors for different clusters
                color_idx = label % 7  # Cycle through 7 different colors
                colors = [
                    [1.0, 0.0, 0.0],  # Red
                    [0.0, 1.0, 0.0],  # Green
                    [0.0, 0.0, 1.0],  # Blue
                    [1.0, 1.0, 0.0],  # Yellow
                    [1.0, 0.0, 1.0],  # Magenta
                    [0.0, 1.0, 1.0],  # Cyan
                    [1.0, 0.5, 0.0]   # Orange
                ]
                color = colors[color_idx]

            # Add point with color information
            colored_points.append([
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(color[0]),
                float(color[1]),
                float(color[2])
            ])

        # Create point cloud with color fields
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'lidar_frame'  # Replace with actual frame ID

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]

        return pc2.create_cloud(header, fields, colored_points)

    def detect_obstacles(self, points, header):
        """Detect obstacles from point cloud."""
        # Filter points based on height and distance
        height_mask = (points[:, 2] > self.obstacle_height_min) & (points[:, 2] < self.obstacle_height_max)
        distance_mask = np.sqrt(points[:, 0]**2 + points[:, 1]**2) < self.obstacle_distance_threshold

        obstacle_mask = height_mask & distance_mask
        obstacle_points = points[obstacle_mask]

        # Create red-colored point cloud for obstacles
        obstacle_colored = []
        for point in obstacle_points:
            obstacle_colored.append([
                float(point[0]),
                float(point[1]),
                float(point[2]),
                1.0,  # Red
                0.0,  # Green
                0.0   # Blue
            ])

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]

        return pc2.create_cloud(header, fields, obstacle_colored)

    def create_pointcloud_msg(self, points, header, color_type='white'):
        """Create PointCloud2 message from numpy array."""
        if len(points) == 0:
            # Return empty point cloud with proper header
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            return pc2.create_cloud(header, fields, [])

        # Prepare points with color based on type
        points_list = []
        for point in points:
            if color_type == 'white':
                points_list.append([float(point[0]), float(point[1]), float(point[2]), 1.0, 1.0, 1.0])
            elif color_type == 'green':
                points_list.append([float(point[0]), float(point[1]), float(point[2]), 0.0, 1.0, 0.0])
            elif color_type == 'red':
                points_list.append([float(point[0]), float(point[1]), float(point[2]), 1.0, 0.0, 0.0])
            else:  # default white
                points_list.append([float(point[0]), float(point[1]), float(point[2]), 1.0, 1.0, 1.0])

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]

        return pc2.create_cloud(header, fields, points_list)

    def create_obstacle_markers(self, points, header):
        """Create visualization markers for detected obstacles."""
        marker_array = MarkerArray()

        # Perform clustering to identify individual obstacles
        if len(points) >= self.min_cluster_points:
            clustering = DBSCAN(eps=self.cluster_epsilon, min_samples=self.min_cluster_points).fit(points)
            labels = clustering.labels_

            # Create markers for each cluster (obstacle)
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue

                cluster_points = points[labels == label]

                if len(cluster_points) >= self.min_cluster_points:
                    # Calculate bounding box for obstacle
                    min_vals = np.min(cluster_points, axis=0)
                    max_vals = np.max(cluster_points, axis=0)

                    # Create marker for obstacle bounding box
                    obstacle_marker = Marker()
                    obstacle_marker.header = header
                    obstacle_marker.ns = "lidar_obstacles"
                    obstacle_marker.id = label
                    obstacle_marker.type = Marker.CUBE
                    obstacle_marker.action = Marker.ADD

                    # Center of bounding box
                    center_x = (min_vals[0] + max_vals[0]) / 2
                    center_y = (min_vals[1] + max_vals[1]) / 2
                    center_z = (min_vals[2] + max_vals[2]) / 2

                    obstacle_marker.pose.position.x = center_x
                    obstacle_marker.pose.position.y = center_y
                    obstacle_marker.pose.position.z = center_z
                    obstacle_marker.pose.orientation.w = 1.0

                    # Size of bounding box
                    obstacle_marker.scale.x = max_vals[0] - min_vals[0]
                    obstacle_marker.scale.y = max_vals[1] - min_vals[1]
                    obstacle_marker.scale.z = max_vals[2] - min_vals[2]

                    # Color based on size (larger obstacles more opaque)
                    size_factor = min(1.0, len(cluster_points) / 100.0)
                    obstacle_marker.color.r = 1.0
                    obstacle_marker.color.g = 0.0
                    obstacle_marker.color.b = 0.0
                    obstacle_marker.color.a = 0.3 + 0.5 * size_factor  # Alpha between 0.3 and 0.8

                    obstacle_marker.lifetime.sec = int(self.marker_lifetime)
                    obstacle_marker.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)

                    marker_array.markers.append(obstacle_marker)

        return marker_array

def main(args=None):
    rclpy.init(args=args)
    lidar_node = LiDARPerceptionNode()

    try:
        rclpy.spin(lidar_node)
    except KeyboardInterrupt:
        lidar_node.get_logger().info('Shutting down LiDAR Perception Node')
    finally:
        lidar_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion

### Multi-Sensor Data Integration

```python
# sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan, Imu
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String, Header, Float32
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import threading
import time
from typing import Dict, Any, Optional, List

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Publishers
        self.fused_detections_pub = self.create_publisher(Detection2DArray, 'fused_detections', 10)
        self.fused_pose_pub = self.create_publisher(PoseStamped, 'fused_pose', 10)
        self.fused_scan_pub = self.create_publisher(LaserScan, 'fused_scan', 10)
        self.fusion_status_pub = self.create_publisher(String, 'fusion_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 'velodyne_points', self.pointcloud_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )
        self.camera_detections_sub = self.create_subscription(
            Detection2DArray, 'camera/detections', self.camera_detections_callback, 10
        )

        # TF listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Sensor data storage
        self.latest_image = None
        self.latest_pointcloud = None
        self.latest_scan = None
        self.latest_imu = None
        self.latest_camera_detections = None

        # Data synchronization buffers
        self.image_buffer = deque(maxlen=10)
        self.pointcloud_buffer = deque(maxlen=5)
        self.scan_buffer = deque(maxlen=5)
        self.imu_buffer = deque(maxlen=20)
        self.detection_buffer = deque(maxlen=10)

        # Time synchronization parameters
        self.time_sync_threshold = 0.1  # seconds
        self.enable_time_sync = True

        # Fusion parameters
        self.enable_visual_lidar_fusion = True
        self.enable_multi_sensor_fusion = True
        self.confidence_threshold = 0.6

        # Robot state estimation
        self.position_estimate = np.array([0.0, 0.0, 0.0])
        self.orientation_estimate = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.velocity_estimate = np.array([0.0, 0.0, 0.0])

        # Kalman filter parameters (simplified)
        self.process_noise = np.eye(6) * 0.1  # Process noise
        self.measurement_noise = np.eye(6) * 0.5  # Measurement noise
        self.state_covariance = np.eye(6) * 1.0  # Initial state covariance

        # Threading for fusion processing
        self.fusion_thread = threading.Thread(target=self.fusion_loop, daemon=True)
        self.fusion_lock = threading.Lock()
        self.new_data_available = threading.Event()

        # Performance tracking
        self.fusion_times = []
        self.fusion_count = 0

        self.get_logger().info('Sensor Fusion Node initialized')

    def image_callback(self, msg):
        """Process image data."""
        self.latest_image = msg
        self.image_buffer.append({
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'data': msg
        })
        self.new_data_available.set()

    def pointcloud_callback(self, msg):
        """Process point cloud data."""
        self.latest_pointcloud = msg
        self.pointcloud_buffer.append({
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'data': msg
        })
        self.new_data_available.set()

    def scan_callback(self, msg):
        """Process laser scan data."""
        self.latest_scan = msg
        self.scan_buffer.append({
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'data': msg
        })
        self.new_data_available.set()

    def imu_callback(self, msg):
        """Process IMU data."""
        self.latest_imu = msg
        self.imu_buffer.append({
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'data': msg
        })

        # Update state estimate from IMU
        self.update_state_from_imu(msg)

        self.new_data_available.set()

    def camera_detections_callback(self, msg):
        """Process camera detections."""
        self.latest_camera_detections = msg
        self.detection_buffer.append({
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'data': msg
        })
        self.new_data_available.set()

    def update_state_from_imu(self, imu_msg):
        """Update state estimate from IMU data."""
        # Extract angular velocity for orientation update
        angular_vel = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        # Extract linear acceleration
        linear_acc = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        # Update orientation using gyroscope integration (simplified)
        dt = 0.01  # Assuming 100Hz IMU rate
        angular_speed = np.linalg.norm(angular_vel)
        if angular_speed > 0:
            axis = angular_vel / angular_speed
            angle = angular_speed * dt

            # Create rotation quaternion
            s = np.sin(angle / 2)
            c = np.cos(angle / 2)
            dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

            # Apply rotation to current orientation
            self.orientation_estimate = self.quaternion_multiply(
                self.orientation_estimate, dq
            )

        # Normalize quaternion
        self.orientation_estimate = self.orientation_estimate / np.linalg.norm(self.orientation_estimate)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([x, y, z, w])

    def fusion_loop(self):
        """Main fusion processing loop."""
        while rclpy.ok():
            try:
                self.new_data_available.wait(timeout=0.1)
                self.new_data_available.clear()

                start_time = time.time()

                # Perform sensor fusion
                with self.fusion_lock:
                    if self.enable_time_sync:
                        synced_data = self.synchronize_sensor_data()
                    else:
                        # Use latest available data
                        synced_data = {
                            'image': self.latest_image,
                            'pointcloud': self.latest_pointcloud,
                            'scan': self.latest_scan,
                            'imu': self.latest_imu,
                            'detections': self.latest_camera_detections
                        }

                    if synced_data:
                        fused_result = self.perform_sensor_fusion(synced_data)

                        if fused_result:
                            self.publish_fused_data(fused_result)

                # Track performance
                fusion_time = time.time() - start_time
                self.fusion_times.append(fusion_time)

                if len(self.fusion_times) > 100:
                    self.fusion_times.pop(0)

                # Update statistics
                self.fusion_count += 1

                # Log performance periodically
                if self.fusion_count % 50 == 0:
                    avg_time = sum(self.fusion_times) / len(self.fusion_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    self.get_logger().info(
                        f'Sensor fusion - Avg: {avg_time*1000:.1f}ms, Rate: {fps:.1f}Hz'
                    )

            except Exception as e:
                self.get_logger().error(f'Sensor fusion error: {e}')

    def synchronize_sensor_data(self) -> Dict[str, Any]:
        """Synchronize sensor data based on timestamps."""
        synced_data = {}

        # Find closest matching timestamps
        if (len(self.image_buffer) > 0 and
            len(self.pointcloud_buffer) > 0 and
            len(self.imu_buffer) > 0):

            # Get most recent image
            image_data = self.image_buffer[-1]
            image_time = image_data['timestamp']

            # Find closest point cloud
            closest_pointcloud = min(
                self.pointcloud_buffer,
                key=lambda x: abs(x['timestamp'] - image_time),
                default=None
            )

            # Find closest IMU data
            closest_imu = min(
                self.imu_buffer,
                key=lambda x: abs(x['timestamp'] - image_time),
                default=None
            )

            # Check time synchronization
            if closest_pointcloud and closest_imu:
                time_diff_pc = abs(closest_pointcloud['timestamp'] - image_time)
                time_diff_imu = abs(closest_imu['timestamp'] - image_time)

                if (time_diff_pc < self.time_sync_threshold and
                    time_diff_imu < self.time_sync_threshold):

                    synced_data = {
                        'image': image_data['data'],
                        'pointcloud': closest_pointcloud['data'],
                        'imu': closest_imu['data'],
                        'image_timestamp': image_time,
                        'sync_error': max(time_diff_pc, time_diff_imu)
                    }

        return synced_data

    def perform_sensor_fusion(self, sensor_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform sensor fusion to create enhanced perception."""
        fused_result = {
            'timestamp': self.get_clock().now().to_msg(),
            'detections': [],
            'pose_estimate': None,
            'confidence': 0.0
        }

        # Fuse visual and LiDAR data for enhanced object detection
        if (self.enable_visual_lidar_fusion and
            'image' in sensor_data and
            'pointcloud' in sensor_data):

            camera_detections = self.get_latest_camera_detections()
            if camera_detections:
                fused_detections = self.fuse_visual_lidar_data(
                    camera_detections, sensor_data['pointcloud']
                )
                fused_result['detections'] = fused_detections

        # Update pose estimate using multi-sensor data
        if self.enable_multi_sensor_fusion:
            pose_estimate = self.estimate_pose_from_sensors(sensor_data)
            fused_result['pose_estimate'] = pose_estimate

        # Calculate overall confidence
        fused_result['confidence'] = self.calculate_fusion_confidence(sensor_data)

        return fused_result

    def fuse_visual_lidar_data(self, camera_detections, pointcloud_msg):
        """Fuse visual and LiDAR data to create 3D object detections."""
        if not camera_detections.detections:
            return []

        fused_detections = []

        # Convert point cloud to numpy array
        points = []
        for point in point_cloud2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        if not points:
            return []

        points = np.array(points)

        # For each camera detection, find corresponding LiDAR points
        for detection in camera_detections.detections:
            # Get bounding box center and size
            bbox_center_x = detection.bbox.center.x
            bbox_center_y = detection.bbox.center.y
            bbox_width = detection.bbox.size_x
            bbox_height = detection.bbox.size_y

            # Convert 2D bounding box to 3D space (simplified approach)
            # In a real implementation, you'd use camera-LiDAR calibration
            # and project the bounding box into 3D space

            # For now, find LiDAR points that correspond to the general area
            # This is a simplified approach - real implementation would use
            # proper camera-LiDAR calibration and projection

            # Find points that are likely to correspond to the detected object
            # This is a very simplified approach
            object_points = points[
                (points[:, 0] > -2) & (points[:, 0] < 2) &  # X range
                (points[:, 1] > -2) & (points[:, 1] < 2)    # Y range
            ]

            if len(object_points) > 0:
                # Calculate 3D bounding box from points
                min_coords = np.min(object_points, axis=0)
                max_coords = np.max(object_points, axis=0)

                # Create fused detection
                fused_detection = {
                    'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                    'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                    'position_3d': {
                        'x': float((min_coords[0] + max_coords[0]) / 2),
                        'y': float((min_coords[1] + max_coords[1]) / 2),
                        'z': float((min_coords[2] + max_coords[2]) / 2)
                    },
                    'size_3d': {
                        'width': float(max_coords[0] - min_coords[0]),
                        'depth': float(max_coords[1] - min_coords[1]),
                        'height': float(max_coords[2] - min_coords[2])
                    }
                }

                fused_detections.append(fused_detection)

        return fused_detections

    def estimate_pose_from_sensors(self, sensor_data: Dict) -> Optional[Dict[str, float]]:
        """Estimate robot pose using multiple sensors."""
        # This is a simplified pose estimation
        # In a real implementation, you'd use an Extended Kalman Filter
        # or Particle Filter for sensor fusion

        estimated_pose = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }

        # Use IMU for orientation (if available)
        if 'imu' in sensor_data:
            imu_data = sensor_data['imu']
            estimated_pose['orientation'] = {
                'x': imu_data.orientation.x,
                'y': imu_data.orientation.y,
                'z': imu_data.orientation.z,
                'w': imu_data.orientation.w
            }

        # Use odometry or other sensors for position (simplified)
        # In real implementation, integrate velocity from IMU or use visual odometry
        estimated_pose['position']['x'] = float(self.position_estimate[0])
        estimated_pose['position']['y'] = float(self.position_estimate[1])
        estimated_pose['position']['z'] = float(self.position_estimate[2])

        return estimated_pose

    def calculate_fusion_confidence(self, sensor_data: Dict) -> float:
        """Calculate confidence in fused data."""
        confidence_factors = []

        # Confidence based on sensor availability
        if 'image' in sensor_data:
            confidence_factors.append(0.3)  # Camera data
        if 'pointcloud' in sensor_data:
            confidence_factors.append(0.4)  # LiDAR data
        if 'imu' in sensor_data:
            confidence_factors.append(0.3)  # IMU data

        # Confidence based on synchronization quality
        if 'sync_error' in sensor_data:
            sync_quality = max(0.0, 1.0 - sensor_data['sync_error'] / self.time_sync_threshold)
            confidence_factors.append(sync_quality * 0.2)

        # Calculate weighted confidence
        if confidence_factors:
            confidence = min(1.0, sum(confidence_factors))
        else:
            confidence = 0.0

        return confidence

    def publish_fused_data(self, fused_result: Dict[str, Any]):
        """Publish fused sensor data."""
        # Publish fused detections
        if fused_result['detections']:
            detections_msg = Detection2DArray()
            detections_msg.header.stamp = fused_result['timestamp']
            detections_msg.header.frame_id = 'map'  # or appropriate frame

            for detection in fused_result['detections']:
                detection_2d = Detection2D()
                detection_2d.results.append(ObjectHypothesisWithPose())

                # Set position based on 3D position
                detection_2d.bbox.center.x = detection['position_3d']['x']
                detection_2d.bbox.center.y = detection['position_3d']['y']
                detection_2d.bbox.size_x = detection['size_3d']['width']
                detection_2d.bbox.size_y = detection['size_3d']['depth']

                # Set confidence
                detection_2d.results[0].hypothesis.score = detection['confidence']
                detection_2d.results[0].hypothesis.class_id = detection['class']

                detections_msg.detections.append(detection_2d)

            self.fused_detections_pub.publish(detections_msg)

        # Publish fused pose estimate
        if fused_result['pose_estimate']:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = fused_result['timestamp']
            pose_msg.header.frame_id = 'map'

            pose = fused_result['pose_estimate']
            pose_msg.pose.position.x = pose['position']['x']
            pose_msg.pose.position.y = pose['position']['y']
            pose_msg.pose.position.z = pose['position']['z']

            pose_msg.pose.orientation.x = pose['orientation']['x']
            pose_msg.pose.orientation.y = pose['orientation']['y']
            pose_msg.pose.orientation.z = pose['orientation']['z']
            pose_msg.pose.orientation.w = pose['orientation']['w']

            self.fused_pose_pub.publish(pose_msg)

        # Publish fusion confidence
        confidence_msg = Float32()
        confidence_msg.data = float(fused_result['confidence'])
        # Create a publisher for confidence if needed

        # Publish status
        status_msg = String()
        status_msg.data = f"Fusion completed with {len(fused_result['detections'])} detections, confidence: {fused_result['confidence']:.2f}"
        self.fusion_status_pub.publish(status_msg)

    def get_latest_camera_detections(self):
        """Get latest camera detections."""
        if self.detection_buffer:
            return self.detection_buffer[-1]['data']
        return None

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    # Start fusion processing thread
    fusion_node.fusion_thread.start()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Shutting down Sensor Fusion Node')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Optimizing Perception Pipelines

```python
# perception_optimization.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String, Float32
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import time
from collections import deque
import psutil
import GPUtil

class OptimizedPerceptionNode(Node):
    def __init__(self):
        super().__init__('optimized_perception')

        # Publishers
        self.optimized_detections_pub = self.create_publisher(String, 'optimized_detections', 10)
        self.performance_pub = self.create_publisher(Float32, 'perception_fps', 10)
        self.resource_usage_pub = self.create_publisher(String, 'resource_usage', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.optimized_image_callback, 10
        )

        # CV Bridge
        self.bridge = CvBridge()

        # Optimization parameters
        self.enable_downsampling = True
        self.downsample_factor = 2  # Process every 2nd pixel
        self.enable_roi_processing = True
        self.roi_regions = []  # Regions of interest
        self.enable_multi_threading = True
        self.enable_gpu_processing = True if self.check_gpu_availability() else False

        # Processing queues and threading
        self.processing_queue = deque(maxlen=5)
        self.result_queue = deque(maxlen=5)
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.last_process_time = time.time()
        self.frame_count = 0

        # Resource monitoring
        self.resource_monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.resource_monitor_thread.start()

        self.get_logger().info(f'Optimized Perception Node initialized with GPU: {self.enable_gpu_processing}')

    def check_gpu_availability(self) -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def optimized_image_callback(self, msg):
        """Optimized image processing with performance considerations."""
        current_time = time.time()

        # Calculate frame rate
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)

        self.last_frame_time = current_time

        if self.enable_multi_threading:
            # Add to processing queue for background processing
            self.processing_queue.append(msg)
        else:
            # Process immediately in main thread
            self.process_image_optimized(msg)

    def processing_worker(self):
        """Background worker for image processing."""
        while rclpy.ok():
            try:
                if self.processing_queue:
                    msg = self.processing_queue.popleft()
                    result = self.process_image_optimized(msg)

                    if result:
                        self.result_queue.append(result)

                        # Publish result
                        result_msg = String()
                        result_msg.data = result
                        self.optimized_detections_pub.publish(result_msg)

                time.sleep(0.001)  # Small delay to prevent busy waiting
            except Exception as e:
                self.get_logger().error(f'Processing worker error: {e}')

    def process_image_optimized(self, msg) -> str:
        """Optimized image processing function."""
        start_time = time.time()

        try:
            # Convert image with error handling
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return ""

        # Apply optimizations
        if self.enable_downsampling:
            cv_image = self.downsample_image(cv_image)

        if self.enable_roi_processing and self.roi_regions:
            cv_image = self.process_roi_regions(cv_image)

        # Perform detection (simplified for this example)
        detections = self.perform_optimized_detection(cv_image)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create result string
        result = f"Processed {len(detections)} objects in {processing_time*1000:.1f}ms"

        return result

    def downsample_image(self, image):
        """Downsample image for faster processing."""
        if self.downsample_factor > 1:
            new_size = (
                image.shape[1] // self.downsample_factor,
                image.shape[0] // self.downsample_factor
            )
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def process_roi_regions(self, image):
        """Process only regions of interest."""
        if not self.roi_regions:
            return image

        # For now, just return the original image
        # In a real implementation, you'd process only specified regions
        processed_image = image.copy()

        for roi in self.roi_regions:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w]
            # Process ROI image here
            processed_image[y:y+h, x:x+w] = roi_image

        return processed_image

    def perform_optimized_detection(self, image):
        """Perform optimized object detection."""
        # This would typically use a neural network
        # For this example, we'll simulate detection
        # In a real implementation, you'd use TensorRT, OpenVINO, or similar
        # for optimized inference

        height, width = image.shape[:2]

        # Simulate detection results
        detections = []
        for i in range(np.random.randint(1, 5)):  # 1-4 simulated detections
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)

            detections.append({
                'bbox': [x, y, w, h],
                'class': np.random.choice(['person', 'car', 'chair', 'bottle']),
                'confidence': float(np.random.uniform(0.6, 0.99))
            })

        return detections

    def monitor_resources(self):
        """Monitor system resources."""
        while rclpy.ok():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory_percent = psutil.virtual_memory().percent

                # GPU usage (if available)
                gpu_percent = 0
                gpu_memory_percent = 0
                if self.enable_gpu_processing:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_percent = gpus[0].load * 100
                            gpu_memory_percent = gpus[0].memoryUtil * 100
                    except:
                        pass

                # Calculate FPS
                if self.frame_times:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                else:
                    fps = 0

                # Publish FPS
                fps_msg = Float32()
                fps_msg.data = float(fps)
                self.performance_pub.publish(fps_msg)

                # Publish resource usage
                resource_msg = String()
                resource_msg.data = f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, GPU: {gpu_percent:.1f}%, FPS: {fps:.1f}"
                self.resource_usage_pub.publish(resource_msg)

                time.sleep(2.0)  # Monitor every 2 seconds

            except Exception as e:
                self.get_logger().error(f'Resource monitoring error: {e}')
                time.sleep(2.0)

    def set_downsample_factor(self, factor: int):
        """Set downsample factor for optimization."""
        if factor >= 1:
            self.downsample_factor = factor
            self.get_logger().info(f'Downsample factor set to: {factor}')

    def add_roi_region(self, x: int, y: int, width: int, height: int):
        """Add region of interest for processing."""
        self.roi_regions.append((x, y, width, height))
        self.get_logger().info(f'Added ROI region: ({x}, {y}, {width}, {height})')

    def enable_gpu_processing(self, enable: bool):
        """Enable or disable GPU processing."""
        if enable and self.check_gpu_availability():
            self.enable_gpu_processing = True
            self.get_logger().info('GPU processing enabled')
        else:
            self.enable_gpu_processing = False
            self.get_logger().info('GPU processing disabled')

def main(args=None):
    rclpy.init(args=args)
    opt_node = OptimizedPerceptionNode()

    try:
        rclpy.spin(opt_node)
    except KeyboardInterrupt:
        opt_node.get_logger().info('Shutting down Optimized Perception Node')
    finally:
        opt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive perception integration techniques:

- **Camera Integration**: RGB and stereo camera processing with object detection
- **LiDAR Processing**: Point cloud processing, ground plane removal, clustering
- **Sensor Fusion**: Multi-sensor data integration and synchronization
- **Performance Optimization**: Techniques for real-time perception processing
- **ROS 2 Integration**: Complete integration with ROS 2 messaging system

Proper perception integration is crucial for robotic autonomy, enabling robots to understand and interact with their environment effectively.

## Exercises

1. Implement camera perception with object detection for your robot
2. Set up LiDAR processing for obstacle detection and mapping
3. Create a sensor fusion system combining multiple sensors
4. Optimize your perception pipeline for real-time performance
5. Test your perception system with various environmental conditions

## Quiz

1. What is the main purpose of sensor fusion in robotics?
   a) Reduce sensor cost
   b) Combine multiple sensor inputs for better perception
   c) Increase sensor complexity
   d) Decrease sensor accuracy

2. What does RANSAC stand for in point cloud processing?
   a) Random Sample Consensus
   b) Rapid Sensor Calibration
   c) Robust Algorithm for Sensor Correction
   d) Real-time Adaptive Sensor Control

3. Which of these is NOT a benefit of Unity for robotics simulation?
   a) Photorealistic graphics
   b) Accurate physics simulation
   c) Native ROS integration
   d) Machine learning training environments

## Mini-Project: Complete Perception System

Create a complete perception system that:
1. Integrates camera and LiDAR sensors
2. Implements sensor fusion for enhanced object detection
3. Optimizes processing for real-time performance
4. Integrates with ROS 2 for robot control
5. Tests with various environmental conditions
6. Evaluates perception accuracy and performance