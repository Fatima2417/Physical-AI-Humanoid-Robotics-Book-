---
sidebar_position: 5
---

# Sensor Integration with ROS 2

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate various types of sensors with ROS 2 for robotic perception
- Implement sensor data processing and fusion pipelines
- Configure and calibrate sensors for optimal performance
- Handle sensor synchronization and timing issues
- Implement sensor-based feedback for robotic control systems
- Validate sensor data quality and reliability

## Introduction to Sensor Integration

Robotic perception relies heavily on sensor data to understand the environment and enable autonomous behavior. Proper sensor integration is crucial for creating intelligent robotic systems that can operate effectively in real-world environments.

### Sensor Categories in Robotics

```
Robot Sensor Categories:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Proprioceptive│    │   Exteroceptive │    │   Interoceptive │
│   Sensors       │    │   Sensors       │    │   Sensors       │
│   • IMU         │    │   • Cameras     │    │   • Temperature │
│   • Encoders    │    │   • LiDAR       │    │   • Current     │
│   • Joint       │    │   • Sonar       │    │   • Voltage     │
│   • Force/Torque│    │   • GPS         │    │   • Power       │
│   • Gyro        │    │   • Radar       │    │   • Internal    │
└─────────────────┘    │   • Depth Cam   │    │   State         │
                       └─────────────────┘    └─────────────────┘
```

### Sensor Integration Architecture

```
Sensor Integration Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Physical      │───→│   ROS 2         │───→│   Perception    │
│   Sensors       │    │   Drivers       │    │   Processing    │
│   (Hardware)    │    │   (Nodes)       │    │   (Algorithms)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Sensor        │
                         │   Fusion &      │
                         │   Calibration   │
                         └─────────────────┘
```

## Camera Sensor Integration

### RGB Camera Integration

```python
# camera_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import threading
import time

class CameraIntegrationNode(Node):
    def __init__(self):
        super().__init__('camera_integration')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_processed', 10)
        self.detections_pub = self.create_publisher(Detection2DArray, 'camera/detections', 10)
        self.feature_points_pub = self.create_publisher(Point, 'camera/feature_points', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.image_width = 640
        self.image_height = 480

        # Processing parameters
        self.enable_object_detection = True
        self.enable_feature_extraction = True
        self.enable_calibration = False

        # Threading for processing
        self.processing_lock = threading.Lock()
        self.latest_image = None

        # Performance monitoring
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('Camera Integration Node initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.image_width = msg.width
        self.image_height = msg.height

        self.get_logger().info(
            f'Camera calibration received: {self.image_width}x{self.image_height}'
        )

    def image_callback(self, msg):
        """Process incoming camera images."""
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Process image based on enabled features
        processed_image = cv_image.copy()

        if self.enable_object_detection:
            detections = self.perform_object_detection(cv_image)
            if detections:
                self.detections_pub.publish(detections)

        if self.enable_feature_extraction:
            processed_image = self.extract_features(processed_image)

        if self.enable_calibration:
            processed_image = self.apply_calibration_correction(processed_image)

        # Publish processed image
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.image_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f'Could not convert processed image: {e}')

        # Performance tracking
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Log performance periodically
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Every 30 frames
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'Camera processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}'
            )

    def perform_object_detection(self, image):
        """Perform object detection on the image (simulated)."""
        # In a real implementation, this would use a trained model
        # For simulation, we'll create mock detections
        detections = Detection2DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera_frame'

        # Simulate object detection
        height, width = image.shape[:2]

        # Generate some simulated detections
        num_detections = np.random.randint(1, 4)
        for i in range(num_detections):
            detection = Detection2D()

            # Random bounding box
            bbox_x = np.random.randint(0, width // 2)
            bbox_y = np.random.randint(0, height // 2)
            bbox_w = np.random.randint(50, 200)
            bbox_h = np.random.randint(50, 200)

            # Ensure bounding box doesn't exceed image dimensions
            bbox_w = min(bbox_w, width - bbox_x)
            bbox_h = min(bbox_h, height - bbox_y)

            detection.bbox.center.x = float(bbox_x + bbox_w / 2)
            detection.bbox.center.y = float(bbox_y + bbox_h / 2)
            detection.bbox.size_x = float(bbox_w)
            detection.bbox.size_y = float(bbox_h)

            # Random class and confidence
            classes = ['person', 'car', 'bicycle', 'traffic_sign', 'dog', 'cat']
            class_name = np.random.choice(classes)
            confidence = float(np.random.uniform(0.7, 0.99))

            # Create hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = confidence

            detection.results.append(hypothesis)
            detections.detections.append(detection)

        return detections

    def extract_features(self, image):
        """Extract features from the image."""
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect features using Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )

        if corners is not None:
            corners = np.int0(corners)

            # Draw features on image
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

                # Publish feature point
                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                point_msg.z = 0.0  # Z would require depth information
                self.feature_points_pub.publish(point_msg)

        return image

    def apply_calibration_correction(self, image):
        """Apply camera calibration correction."""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            # Apply undistortion
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.distortion_coeffs,
                (w, h),
                1,
                (w, h)
            )

            corrected_image = cv2.undistort(
                image,
                self.camera_matrix,
                self.distortion_coeffs,
                None,
                new_camera_matrix
            )

            # Crop the image based on ROI
            x, y, w, h = roi
            corrected_image = corrected_image[y:y+h, x:x+w]

            return corrected_image

        return image

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraIntegrationNode()

    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Shutting down Camera Integration Node')
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Depth Camera Integration

```python
# depth_camera_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import struct
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

class DepthCameraIntegration(Node):
    def __init__(self):
        super().__init__('depth_camera_integration')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Publishers
        self.depth_pub = self.create_publisher(Image, 'camera/depth_processed', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'camera/pointcloud', 10)

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image, 'camera/depth/image_raw', self.depth_callback, 10
        )
        self.rgb_sub = self.create_subscription(
            Image, 'camera/image_raw', self.rgb_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )

        # Camera parameters
        self.camera_matrix = None
        self.depth_image = None
        self.rgb_image = None

        # Processing parameters
        self.enable_pointcloud_generation = True
        self.enable_depth_filtering = True
        self.min_depth = 0.1  # meters
        self.max_depth = 10.0  # meters

        self.get_logger().info('Depth Camera Integration initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.get_logger().info(f'Depth camera calibration updated')

    def depth_callback(self, msg):
        """Process depth image."""
        try:
            # Convert depth image to numpy array
            if msg.encoding == '16UC1':
                # 16-bit unsigned integer depth image
                depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width
                ).astype(np.float32)
                # Convert millimeters to meters
                depth_array = depth_array / 1000.0
            elif msg.encoding == '32FC1':
                # 32-bit float depth image
                depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    msg.height, msg.width
                )
            else:
                self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')
                return

            # Store latest depth image
            self.depth_image = depth_array

            # Filter depth image if enabled
            if self.enable_depth_filtering:
                depth_array = self.filter_depth_image(depth_array)

            # Publish processed depth image
            processed_msg = self.bridge.cv2_to_imgmsg(depth_array, encoding='32FC1')
            processed_msg.header = msg.header
            self.depth_pub.publish(processed_msg)

            # Generate point cloud if RGB image is available
            if self.rgb_image is not None and self.enable_pointcloud_generation:
                pointcloud = self.generate_pointcloud(depth_array, self.rgb_image, msg.header)
                if pointcloud is not None:
                    self.pointcloud_pub.publish(pointcloud)

        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def rgb_callback(self, msg):
        """Store RGB image for point cloud generation."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB image conversion error: {e}')

    def filter_depth_image(self, depth_array):
        """Filter depth image to remove noise and invalid values."""
        # Create mask for valid depth values
        valid_mask = (depth_array >= self.min_depth) & (depth_array <= self.max_depth)

        # Apply median filter to reduce noise
        filtered_depth = depth_array.copy()

        # Apply filtering only to valid regions
        if np.any(valid_mask):
            import cv2
            # Use median filter to reduce noise while preserving edges
            filtered_region = cv2.medianBlur(
                (depth_array * valid_mask).astype(np.float32), 5
            )
            filtered_depth = np.where(valid_mask, filtered_region, depth_array)

        return filtered_depth

    def generate_pointcloud(self, depth_array, rgb_image, header):
        """Generate point cloud from depth image and RGB image."""
        if self.camera_matrix is None:
            return None

        height, width = depth_array.shape

        # Get camera intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Create point cloud
        points = []
        rgb_values = []

        for v in range(height):
            for u in range(width):
                depth = depth_array[v, u]

                # Skip invalid depth values
                if depth < self.min_depth or depth > self.max_depth:
                    continue

                # Convert pixel coordinates to 3D world coordinates
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth

                points.append([x, y, z])

                # Get RGB color for this point
                if rgb_image is not None and v < rgb_image.shape[0] and u < rgb_image.shape[1]:
                    bgr = rgb_image[v, u]
                    # Convert BGR to RGB and pack as single float
                    rgb = struct.unpack('I', struct.pack('BBBB', int(bgr[2]), int(bgr[1]), int(bgr[0]), 255))[0]
                    rgb_values.append(rgb)

        if not points:
            return None

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        if rgb_values:
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))

        # Pack point cloud data
        pointcloud_data = []
        for i, point in enumerate(points):
            if rgb_values:
                pointcloud_data.append([point[0], point[1], point[2], rgb_values[i]])
            else:
                pointcloud_data.append([point[0], point[1], point[2]])

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_depth_frame'

        pointcloud_msg = pc2.create_cloud(header, fields, pointcloud_data)
        return pointcloud_msg

def main(args=None):
    rclpy.init(args=args)
    depth_node = DepthCameraIntegration()

    try:
        rclpy.spin(depth_node)
    except KeyboardInterrupt:
        depth_node.get_logger().info('Shutting down Depth Camera Integration')
    finally:
        depth_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## LiDAR Sensor Integration

### 2D LiDAR Integration

```python
# lidar_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
import numpy as np
import math
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
import struct

class LiDARIntegration(Node):
    def __init__(self):
        super().__init__('lidar_integration')

        # Publishers
        self.scan_pub = self.create_publisher(LaserScan, 'scan_filtered', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'lidar_pointcloud', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Processing parameters
        self.enable_obstacle_detection = True
        self.enable_ground_removal = False
        self.enable_outlier_removal = True
        self.obstacle_distance_threshold = 1.0  # meters
        self.min_obstacle_points = 5  # minimum points to consider as obstacle

        # For obstacle tracking
        self.obstacle_clusters = []
        self.next_obstacle_id = 0

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('LiDAR Integration Node initialized')

    def scan_callback(self, msg):
        """Process incoming LiDAR scan data."""
        start_time = self.get_clock().now().nanoseconds / 1e9

        # Filter scan data
        filtered_scan = self.filter_scan_data(msg)

        # Publish filtered scan
        self.scan_pub.publish(filtered_scan)

        # Convert to point cloud
        pointcloud = self.scan_to_pointcloud(filtered_scan)
        if pointcloud:
            self.pointcloud_pub.publish(pointcloud)

        # Detect obstacles
        if self.enable_obstacle_detection:
            obstacles = self.detect_obstacles(filtered_scan)
            self.publish_obstacle_markers(obstacles, msg.header)

        # Performance tracking
        end_time = self.get_clock().now().nanoseconds / 1e9
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Log performance periodically
        self.frame_count += 1
        if self.frame_count % 50 == 0:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'LiDAR processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}'
            )

    def filter_scan_data(self, scan_msg):
        """Filter LiDAR scan data."""
        # Create filtered scan message
        filtered_scan = LaserScan()
        filtered_scan.header = scan_msg.header
        filtered_scan.angle_min = scan_msg.angle_min
        filtered_scan.angle_max = scan_msg.angle_max
        filtered_scan.angle_increment = scan_msg.angle_increment
        filtered_scan.time_increment = scan_msg.time_increment
        filtered_scan.scan_time = scan_msg.scan_time
        filtered_scan.range_min = scan_msg.range_min
        filtered_scan.range_max = scan_msg.range_max

        # Filter ranges
        filtered_ranges = []
        for i, range_val in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                # Apply outlier filtering
                if self.enable_outlier_removal:
                    # Check neighboring beams for consistency
                    prev_idx = max(0, i - 1)
                    next_idx = min(len(scan_msg.ranges) - 1, i + 1)

                    prev_range = scan_msg.ranges[prev_idx]
                    next_range = scan_msg.ranges[next_idx]

                    # If this range is significantly different from neighbors, consider it an outlier
                    if (abs(range_val - prev_range) < 2.0 and
                        abs(range_val - next_range) < 2.0):
                        filtered_ranges.append(range_val)
                    else:
                        # Keep the value if it's consistent with at least one neighbor
                        if (abs(range_val - prev_range) < 1.0 or
                            abs(range_val - next_range) < 1.0):
                            filtered_ranges.append(range_val)
                        else:
                            # Mark as invalid
                            filtered_ranges.append(float('inf'))
                else:
                    filtered_ranges.append(range_val)
            else:
                filtered_ranges.append(float('inf'))  # Invalid range

        filtered_scan.ranges = filtered_ranges

        # Filter intensities if available
        if scan_msg.intensities:
            filtered_intensities = []
            for i, intensity in enumerate(scan_msg.intensities):
                if scan_msg.ranges[i] != float('inf'):
                    filtered_intensities.append(intensity)
                else:
                    filtered_intensities.append(0.0)  # Invalid intensity

            filtered_scan.intensities = filtered_intensities

        return filtered_scan

    def scan_to_pointcloud(self, scan_msg):
        """Convert LaserScan to PointCloud2."""
        if not scan_msg.ranges:
            return None

        # Convert scan to point cloud
        points = []
        angle = scan_msg.angle_min

        for range_val in scan_msg.ranges:
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                z = 0.0  # 2D LiDAR, so z = 0

                points.append([x, y, z])

            angle += scan_msg.angle_increment

        if not points:
            return None

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = scan_msg.header.stamp
        header.frame_id = scan_msg.header.frame_id

        pointcloud_msg = pc2.create_cloud(header, fields, points)
        return pointcloud_msg

    def detect_obstacles(self, scan_msg):
        """Detect obstacles from scan data."""
        obstacles = []

        # Convert scan to Cartesian points for clustering
        points = []
        angle = scan_msg.angle_min

        for i, range_val in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                points.append((x, y, i))  # Include index for reference

            angle += scan_msg.angle_increment

        # Simple clustering: group nearby points
        clusters = self.cluster_points(points)

        # Filter clusters based on size and distance
        for cluster in clusters:
            if len(cluster) >= self.min_obstacle_points:
                # Calculate cluster center and bounding box
                xs = [p[0] for p in cluster]
                ys = [p[1] for p in cluster]

                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Only consider obstacles within threshold distance
                distance_to_robot = math.sqrt(center_x**2 + center_y**2)
                if distance_to_robot <= self.obstacle_distance_threshold:
                    obstacles.append({
                        'center': (center_x, center_y),
                        'bounding_box': ((min_x, min_y), (max_x, max_y)),
                        'size': len(cluster),
                        'distance': distance_to_robot
                    })

        return obstacles

    def cluster_points(self, points):
        """Simple clustering of points."""
        clusters = []
        visited = set()

        for i, (x1, y1, idx1) in enumerate(points):
            if i in visited:
                continue

            cluster = [(x1, y1, idx1)]
            visited.add(i)

            # Find nearby points
            for j, (x2, y2, idx2) in enumerate(points[i+1:], i+1):
                if j in visited:
                    continue

                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < 0.3:  # 30cm threshold
                    cluster.append((x2, y2, idx2))
                    visited.add(j)

            if len(cluster) > 1:  # Only consider clusters with multiple points
                clusters.append(cluster)

        return clusters

    def publish_obstacle_markers(self, obstacles, header):
        """Publish obstacle markers for visualization."""
        marker_array = MarkerArray()

        for i, obstacle in enumerate(obstacles):
            # Create marker for obstacle center
            center_marker = Marker()
            center_marker.header = header
            center_marker.ns = "obstacles"
            center_marker.id = i * 2
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD

            center_marker.pose.position.x = obstacle['center'][0]
            center_marker.pose.position.y = obstacle['center'][1]
            center_marker.pose.position.z = 0.1  # Slightly above ground
            center_marker.pose.orientation.w = 1.0

            center_marker.scale.x = 0.2
            center_marker.scale.y = 0.2
            center_marker.scale.z = 0.2

            center_marker.color.r = 1.0
            center_marker.color.g = 0.0
            center_marker.color.b = 0.0
            center_marker.color.a = 0.8

            # Create marker for bounding box
            bbox_marker = Marker()
            bbox_marker.header = header
            bbox_marker.ns = "obstacle_bboxes"
            bbox_marker.id = i * 2 + 1
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD

            # Define bounding box vertices
            min_pt, max_pt = obstacle['bounding_box']
            vertices = [
                Point32(x=min_pt[0], y=min_pt[1], z=0.1),
                Point32(x=max_pt[0], y=min_pt[1], z=0.1),
                Point32(x=max_pt[0], y=max_pt[1], z=0.1),
                Point32(x=min_pt[0], y=max_pt[1], z=0.1),
                Point32(x=min_pt[0], y=min_pt[1], z=0.1)  # Close the loop
            ]

            bbox_marker.points = vertices
            bbox_marker.scale.x = 0.05  # Line width

            bbox_marker.color.r = 1.0
            bbox_marker.color.g = 0.5
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 0.8

            marker_array.markers.extend([center_marker, bbox_marker])

        self.obstacle_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    lidar_node = LiDARIntegration()

    try:
        rclpy.spin(lidar_node)
    except KeyboardInterrupt:
        lidar_node.get_logger().info('Shutting down LiDAR Integration Node')
    finally:
        lidar_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3D LiDAR Integration

```python
# lidar_3d_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import math

class LiDAR3DIntegration(Node):
    def __init__(self):
        super().__init__('lidar_3d_integration')

        # Publishers
        self.filtered_pc_pub = self.create_publisher(PointCloud2, 'pointcloud_filtered', 10)
        self.obstacles_pub = self.create_publisher(MarkerArray, '3d_obstacles', 10)
        self.ground_pub = self.create_publisher(PointCloud2, 'ground_points', 10)

        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2, 'velodyne_points', self.pointcloud_callback, 10
        )

        # Processing parameters
        self.enable_ground_removal = True
        self.enable_obstacle_segmentation = True
        self.enable_clustering = True
        self.ground_height_threshold = 0.1  # meters above ground
        self.cluster_eps = 0.5  # clustering distance threshold
        self.min_cluster_points = 10  # minimum points for valid cluster

        # Ground plane parameters for RANSAC
        self.ransac_iterations = 100
        self.ransac_threshold = 0.1

        self.get_logger().info('3D LiDAR Integration Node initialized')

    def pointcloud_callback(self, msg):
        """Process incoming 3D point cloud data."""
        # Convert PointCloud2 to numpy array
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        if not points:
            return

        points = np.array(points)

        # Remove ground plane if enabled
        if self.enable_ground_removal:
            filtered_points, ground_points = self.remove_ground_plane(points)
        else:
            filtered_points = points
            ground_points = np.array([])

        # Publish filtered point cloud
        if len(filtered_points) > 0:
            filtered_pc_msg = self.create_pointcloud_msg(filtered_points, msg.header)
            self.filtered_pc_pub.publish(filtered_pc_msg)

        # Publish ground points separately
        if len(ground_points) > 0:
            ground_pc_msg = self.create_pointcloud_msg(ground_points, msg.header)
            self.ground_pub.publish(ground_pc_msg)

        # Segment obstacles if enabled
        if self.enable_obstacle_segmentation:
            obstacles = self.segment_obstacles(filtered_points)
            self.publish_obstacle_markers(obstacles, msg.header)

    def remove_ground_plane(self, points):
        """Remove ground plane using RANSAC algorithm."""
        if len(points) < 3:
            return points, np.array([])

        best_model = None
        best_inliers = []
        best_error = float('inf')

        # RANSAC algorithm
        for _ in range(self.ransac_iterations):
            # Randomly sample 3 points
            indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[indices]

            # Fit plane to sample points
            try:
                plane_model = self.fit_plane_to_points(sample_points)

                # Calculate distances to plane
                distances = self.distance_point_to_plane(points, plane_model)
                inliers = points[distances < self.ransac_threshold]

                # Calculate error (sum of squared distances for inliers)
                error = np.sum(distances[distances < self.ransac_threshold] ** 2)

                if len(inliers) > len(best_inliers) and error < best_error:
                    best_model = plane_model
                    best_inliers = inliers
                    best_error = error
            except:
                continue

        # Remove ground points (points close to the plane)
        if best_model is not None:
            distances = self.distance_point_to_plane(points, best_model)
            ground_mask = distances < self.ransac_threshold
            ground_points = points[ground_mask]
            obstacle_points = points[~ground_mask]
        else:
            # If RANSAC failed, assume Z < threshold is ground
            ground_mask = points[:, 2] < self.ground_height_threshold
            ground_points = points[ground_mask]
            obstacle_points = points[~ground_mask]

        return obstacle_points, ground_points

    def fit_plane_to_points(self, points):
        """Fit a plane to 3 points."""
        if len(points) != 3:
            raise ValueError("Need exactly 3 points to fit a plane")

        p1, p2, p3 = points[0], points[1], points[2]

        # Calculate plane normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # Plane equation: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, p1)

        return np.array([a, b, c, d])

    def distance_point_to_plane(self, points, plane_model):
        """Calculate distance from points to plane."""
        a, b, c, d = plane_model
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        distances = np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)
        return distances

    def segment_obstacles(self, points):
        """Segment obstacles using clustering."""
        if len(points) < self.min_cluster_points:
            return []

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_cluster_points).fit(points)
        labels = clustering.labels_

        # Group points by cluster
        clusters = {}
        for point, label in zip(points, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(point)

        # Remove noise cluster (-1)
        clusters.pop(-1, None)

        # Calculate cluster properties
        obstacles = []
        for label, cluster_points in clusters.items():
            cluster_points = np.array(cluster_points)

            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)

            # Calculate bounding box
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)

            # Calculate size
            size = max_bounds - min_bounds

            obstacles.append({
                'centroid': centroid,
                'points': cluster_points,
                'min_bounds': min_bounds,
                'max_bounds': max_bounds,
                'size': size,
                'num_points': len(cluster_points)
            })

        return obstacles

    def create_pointcloud_msg(self, points, header):
        """Create PointCloud2 message from numpy array."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        return pc2.create_cloud(header, fields, points)

    def publish_obstacle_markers(self, obstacles, header):
        """Publish obstacle markers for visualization."""
        marker_array = MarkerArray()

        for i, obstacle in enumerate(obstacles):
            # Create marker for obstacle centroid
            centroid_marker = Marker()
            centroid_marker.header = header
            centroid_marker.ns = "obstacles"
            centroid_marker.id = i * 2
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD

            centroid_marker.pose.position.x = float(obstacle['centroid'][0])
            centroid_marker.pose.position.y = float(obstacle['centroid'][1])
            centroid_marker.pose.position.z = float(obstacle['centroid'][2])
            centroid_marker.pose.orientation.w = 1.0

            # Scale based on obstacle size
            avg_size = np.mean(obstacle['size'])
            scale = max(0.2, min(avg_size, 1.0))  # Clamp between 0.2 and 1.0
            centroid_marker.scale.x = scale
            centroid_marker.scale.y = scale
            centroid_marker.scale.z = scale

            centroid_marker.color.r = 1.0
            centroid_marker.color.g = 0.0
            centroid_marker.color.b = 0.0
            centroid_marker.color.a = 0.8

            # Create marker for bounding box
            bbox_marker = Marker()
            bbox_marker.header = header
            bbox_marker.ns = "obstacle_bboxes"
            bbox_marker.id = i * 2 + 1
            bbox_marker.type = Marker.LINE_LIST
            bbox_marker.action = Marker.ADD

            # Define bounding box edges
            min_pt = obstacle['min_bounds']
            max_pt = obstacle['max_bounds']

            # Create line segments for bounding box wireframe
            bbox_marker.points = [
                Point(x=min_pt[0], y=min_pt[1], z=min_pt[2]),
                Point(x=max_pt[0], y=min_pt[1], z=min_pt[2]),
                Point(x=max_pt[0], y=min_pt[1], z=min_pt[2]),
                Point(x=max_pt[0], y=max_pt[1], z=min_pt[2]),
                Point(x=max_pt[0], y=max_pt[1], z=min_pt[2]),
                Point(x=min_pt[0], y=max_pt[1], z=min_pt[2]),
                Point(x=min_pt[0], y=max_pt[1], z=min_pt[2]),
                Point(x=min_pt[0], y=min_pt[1], z=min_pt[2]),

                Point(x=min_pt[0], y=min_pt[1], z=max_pt[2]),
                Point(x=max_pt[0], y=min_pt[1], z=max_pt[2]),
                Point(x=max_pt[0], y=min_pt[1], z=max_pt[2]),
                Point(x=max_pt[0], y=max_pt[1], z=max_pt[2]),
                Point(x=max_pt[0], y=max_pt[1], z=max_pt[2]),
                Point(x=min_pt[0], y=max_pt[1], z=max_pt[2]),
                Point(x=min_pt[0], y=max_pt[1], z=max_pt[2]),
                Point(x=min_pt[0], y=min_pt[1], z=max_pt[2]),

                Point(x=min_pt[0], y=min_pt[1], z=min_pt[2]),
                Point(x=min_pt[0], y=min_pt[1], z=max_pt[2]),
                Point(x=max_pt[0], y=min_pt[1], z=min_pt[2]),
                Point(x=max_pt[0], y=min_pt[1], z=max_pt[2]),
                Point(x=max_pt[0], y=max_pt[1], z=min_pt[2]),
                Point(x=max_pt[0], y=max_pt[1], z=max_pt[2]),
                Point(x=min_pt[0], y=max_pt[1], z=min_pt[2]),
                Point(x=min_pt[0], y=max_pt[1], z=max_pt[2])
            ]

            bbox_marker.scale.x = 0.02  # Line width

            bbox_marker.color.r = 1.0
            bbox_marker.color.g = 0.5
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 0.8

            marker_array.markers.extend([centroid_marker, bbox_marker])

        self.obstacles_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    lidar_3d_node = LiDAR3DIntegration()

    try:
        rclpy.spin(lidar_3d_node)
    except KeyboardInterrupt:
        lidar_3d_node.get_logger().info('Shutting down 3D LiDAR Integration Node')
    finally:
        lidar_3d_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## IMU and Inertial Sensor Integration

### IMU Integration

```python
# imu_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Header, Float64
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import threading
import time

class IMUIntegration(Node):
    def __init__(self):
        super().__init__('imu_integration')

        # Publishers
        self.imu_pub = self.create_publisher(Imu, 'imu/data_filtered', 10)
        self.orientation_pub = self.create_publisher(Quaternion, 'imu/orientation', 10)
        self.angular_velocity_pub = self.create_publisher(Vector3, 'imu/angular_velocity', 10)
        self.linear_acceleration_pub = self.create_publisher(Vector3, 'imu/linear_acceleration', 10)
        self.magnetic_field_pub = self.create_publisher(MagneticField, 'imu/magnetic_field', 10)

        # Subscribers
        self.raw_imu_sub = self.create_subscription(
            Imu, 'imu/data_raw', self.imu_callback, 10
        )

        # Processing parameters
        self.enable_orientation_filtering = True
        self.enable_calibration = True
        self.complementary_filter_alpha = 0.98  # For accelerometer-magnetometer fusion

        # IMU state tracking
        self.last_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.last_time = None
        self.bias_estimate = np.zeros(6)  # [angular_x, angular_y, angular_z, linear_x, linear_y, linear_z]

        # Threading for processing
        self.processing_lock = threading.Lock()

        # Performance tracking
        self.processing_times = []
        self.sample_count = 0

        self.get_logger().info('IMU Integration Node initialized')

    def imu_callback(self, msg):
        """Process incoming IMU data."""
        start_time = time.time()

        with self.processing_lock:
            # Get current time
            current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # Extract raw measurements
            angular_velocity = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            linear_acceleration = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Calibrate measurements if enabled
            if self.enable_calibration:
                angular_velocity = self.calibrate_angular_velocity(angular_velocity)
                linear_acceleration = self.calibrate_linear_acceleration(linear_acceleration)

            # Estimate orientation if we have previous data
            if self.last_time is not None:
                dt = current_time - self.last_time

                if self.enable_orientation_filtering:
                    # Use gyroscope integration with accelerometer/magnetometer correction
                    new_orientation = self.estimate_orientation(
                        angular_velocity, linear_acceleration, dt
                    )
                else:
                    # Simple gyroscope integration
                    new_orientation = self.integrate_gyroscope(
                        self.last_orientation, angular_velocity, dt
                    )
            else:
                # Initialize orientation from accelerometer
                new_orientation = self.estimate_initial_orientation(linear_acceleration)
                dt = 0.0

            # Update state
            self.last_orientation = new_orientation
            self.last_time = current_time

            # Create filtered IMU message
            filtered_msg = Imu()
            filtered_msg.header = msg.header
            filtered_msg.header.frame_id = 'imu_filtered'

            # Set orientation
            filtered_msg.orientation.x = float(new_orientation[0])
            filtered_msg.orientation.y = float(new_orientation[1])
            filtered_msg.orientation.z = float(new_orientation[2])
            filtered_msg.orientation.w = float(new_orientation[3])

            # Set angular velocity
            filtered_msg.angular_velocity.x = float(angular_velocity[0])
            filtered_msg.angular_velocity.y = float(angular_velocity[1])
            filtered_msg.angular_velocity.z = float(angular_velocity[2])

            # Set linear acceleration
            filtered_msg.linear_acceleration.x = float(linear_acceleration[0])
            filtered_msg.linear_acceleration.y = float(linear_acceleration[1])
            filtered_msg.linear_acceleration.z = float(linear_acceleration[2])

            # Set covariances (simplified)
            identity_cov = [0.0] * 9
            for i in range(0, 9, 4):  # Diagonal elements
                identity_cov[i] = 0.01

            filtered_msg.orientation_covariance = identity_cov
            filtered_msg.angular_velocity_covariance = identity_cov
            filtered_msg.linear_acceleration_covariance = identity_cov

            # Publish filtered data
            self.imu_pub.publish(filtered_msg)

            # Publish separate components for convenience
            orientation_msg = Quaternion()
            orientation_msg.x = float(new_orientation[0])
            orientation_msg.y = float(new_orientation[1])
            orientation_msg.z = float(new_orientation[2])
            orientation_msg.w = float(new_orientation[3])
            self.orientation_pub.publish(orientation_msg)

            angular_vel_msg = Vector3()
            angular_vel_msg.x = float(angular_velocity[0])
            angular_vel_msg.y = float(angular_velocity[1])
            angular_vel_msg.z = float(angular_velocity[2])
            self.angular_velocity_pub.publish(angular_vel_msg)

            linear_acc_msg = Vector3()
            linear_acc_msg.x = float(linear_acceleration[0])
            linear_acc_msg.y = float(linear_acceleration[1])
            linear_acc_msg.z = float(linear_acceleration[2])
            self.linear_acceleration_pub.publish(linear_acc_msg)

        # Performance tracking
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Log performance periodically
        self.sample_count += 1
        if self.sample_count % 100 == 0:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            freq = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'IMU processing - Avg: {avg_time*1000:.2f}ms, Freq: {freq:.1f}Hz'
            )

    def calibrate_angular_velocity(self, angular_vel):
        """Apply bias calibration to angular velocity."""
        # In a real system, this would use stored bias estimates
        # For simulation, we'll add some basic compensation
        return angular_vel - self.bias_estimate[:3]

    def calibrate_linear_acceleration(self, linear_acc):
        """Apply bias calibration to linear acceleration."""
        # In a real system, this would use stored bias estimates
        # For simulation, we'll add some basic compensation
        return linear_acc - self.bias_estimate[3:]

    def estimate_initial_orientation(self, linear_acc):
        """Estimate initial orientation from accelerometer reading."""
        # Normalize acceleration vector
        acc_norm = linear_acc / np.linalg.norm(linear_acc)

        # Calculate roll and pitch from accelerometer
        pitch = math.atan2(-acc_norm[0], math.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
        roll = math.atan2(acc_norm[1], acc_norm[2])

        # Yaw is unknown from accelerometer alone, set to 0
        yaw = 0.0

        # Convert to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([x, y, z, w])

    def estimate_orientation(self, angular_vel, linear_acc, dt):
        """Estimate orientation using complementary filter."""
        # Integrate gyroscope
        integrated_orientation = self.integrate_gyroscope(
            self.last_orientation, angular_vel, dt
        )

        # Estimate orientation from accelerometer
        accel_orientation = self.estimate_initial_orientation(linear_acc)

        # Use complementary filter to combine estimates
        # Higher alpha means trust gyroscope more, lower means trust accelerometer more
        alpha = self.complementary_filter_alpha

        # Convert to rotation vectors for interpolation
        rot_integrated = R.from_quat(integrated_orientation)
        rot_accel = R.from_quat(accel_orientation)

        # Interpolate between orientations
        rot_combined = R.from_quat(
            R.concat(rot_integrated, rot_accel.inv()).as_quat() * (1 - alpha) +
            np.array([0, 0, 0, alpha])
        ).as_quat()

        # Ensure positive scalar part for consistency
        if rot_combined[3] < 0:
            rot_combined = -rot_combined

        return rot_combined

    def integrate_gyroscope(self, current_orientation, angular_vel, dt):
        """Integrate gyroscope measurements."""
        # Convert angular velocity to rotation vector
        angle = np.linalg.norm(angular_vel) * dt

        if angle == 0:
            return current_orientation

        axis = angular_vel / np.linalg.norm(angular_vel)

        # Convert to quaternion
        s = math.sin(angle / 2)
        w = math.cos(angle / 2)
        x = axis[0] * s
        y = axis[1] * s
        z = axis[2] * s

        # Create rotation quaternion
        rotation_quat = np.array([x, y, z, w])

        # Apply rotation to current orientation
        # Quaternion multiplication: q_new = rotation_quat * current_orientation
        q_new = np.array([
            rotation_quat[3] * current_orientation[0] + rotation_quat[0] * current_orientation[3] +
            rotation_quat[1] * current_orientation[2] - rotation_quat[2] * current_orientation[1],
            rotation_quat[3] * current_orientation[1] - rotation_quat[0] * current_orientation[2] +
            rotation_quat[1] * current_orientation[3] + rotation_quat[2] * current_orientation[0],
            rotation_quat[3] * current_orientation[2] + rotation_quat[0] * current_orientation[1] -
            rotation_quat[1] * current_orientation[0] + rotation_quat[2] * current_orientation[3],
            rotation_quat[3] * current_orientation[3] - rotation_quat[0] * current_orientation[0] -
            rotation_quat[1] * current_orientation[1] - rotation_quat[2] * current_orientation[2]
        ])

        # Normalize quaternion
        q_new = q_new / np.linalg.norm(q_new)

        return q_new

    def update_bias_estimates(self, angular_vel, linear_acc):
        """Update bias estimates using simple averaging."""
        # This is a simplified bias estimation
        # In practice, you'd use more sophisticated methods
        measurements = np.concatenate([angular_vel, linear_acc])
        self.bias_estimate = 0.99 * self.bias_estimate + 0.01 * measurements

def main(args=None):
    rclpy.init(args=args)
    imu_node = IMUIntegration()

    try:
        rclpy.spin(imu_node)
    except KeyboardInterrupt:
        imu_node.get_logger().info('Shutting down IMU Integration Node')
    finally:
        imu_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion and Calibration

### Multi-Sensor Fusion

```python
# sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, Image
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Publishers
        self.fused_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'fused_pose', 10)
        self.fused_twist_pub = self.create_publisher(TwistWithCovarianceStamped, 'fused_twist', 10)
        self.odom_pub = self.create_publisher(Odometry, 'fused_odom', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, 'imu/data_filtered', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan_filtered', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'wheel_odom', self.odom_callback, 10)

        # State estimation
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.linear_velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # Covariance matrices (simplified)
        self.pose_covariance = np.eye(6) * 0.1
        self.twist_covariance = np.eye(6) * 0.1

        # Timestamps for synchronization
        self.imu_time = None
        self.odom_time = None
        self.scan_time = None

        # Fusion weights
        self.imu_weight = 0.7
        self.odom_weight = 0.8
        self.scan_weight = 0.3

        # For extended Kalman filter
        self.state_vector = np.zeros(12)  # [position, velocity, orientation, angular_velocity]
        self.state_covariance = np.eye(12) * 0.1

        self.get_logger().info('Sensor Fusion Node initialized')

    def imu_callback(self, msg):
        """Process IMU data for state estimation."""
        # Extract orientation from IMU
        self.orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Extract angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Update timestamps
        self.imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update state vector with IMU data
        self.update_state_with_imu(msg)

    def odom_callback(self, msg):
        """Process odometry data for state estimation."""
        # Extract position from odometry
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Extract linear velocity
        self.linear_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

        # Update timestamps
        self.odom_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update state vector with odometry data
        self.update_state_with_odom(msg)

    def scan_callback(self, msg):
        """Process scan data for position correction."""
        # Extract timestamp
        self.scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update state vector with scan data (for position correction)
        self.update_state_with_scan(msg)

    def update_state_with_imu(self, imu_msg):
        """Update state vector with IMU measurements."""
        # Convert quaternion to rotation matrix
        rotation = R.from_quat([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ]).as_matrix()

        # Update orientation in state vector
        self.state_vector[6:10] = [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ]

        # Update angular velocity in state vector
        self.state_vector[10:13] = [
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ]

    def update_state_with_odom(self, odom_msg):
        """Update state vector with odometry measurements."""
        # Update position in state vector
        self.state_vector[0:3] = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ]

        # Update linear velocity in state vector
        self.state_vector[3:6] = [
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z
        ]

    def update_state_with_scan(self, scan_msg):
        """Update state with scan-based position corrections."""
        # This is a simplified approach
        # In a real system, you'd use scan matching or landmark-based localization

        # For now, we'll just use the scan to verify position consistency
        # and potentially correct drift
        pass

    def predict_state(self, dt):
        """Predict state forward in time using motion model."""
        # Simple motion model prediction
        # Position update: x = x + v*dt
        self.state_vector[0:3] += self.state_vector[3:6] * dt

        # Orientation update using angular velocity
        angular_vel = self.state_vector[10:13]
        angle = np.linalg.norm(angular_vel) * dt

        if angle > 0:
            axis = angular_vel / angle
            s = math.sin(angle / 2)
            c = math.cos(angle / 2)

            dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

            # Apply rotation to current orientation
            current_quat = self.state_vector[6:10]
            new_quat = self.quaternion_multiply(dq, current_quat)
            self.state_vector[6:10] = new_quat / np.linalg.norm(new_quat)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([x, y, z, w])

    def fuse_sensor_data(self):
        """Fuse sensor data using weighted combination."""
        # This is a simplified fusion approach
        # In practice, you'd use a proper Kalman filter or particle filter

        # For now, we'll just publish the current state
        self.publish_fused_state()

    def publish_fused_state(self):
        """Publish the fused state estimate."""
        current_time = self.get_clock().now()

        # Create pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose.position.x = float(self.state_vector[0])
        pose_msg.pose.pose.position.y = float(self.state_vector[1])
        pose_msg.pose.pose.position.z = float(self.state_vector[2])

        pose_msg.pose.pose.orientation.x = float(self.state_vector[6])
        pose_msg.pose.pose.orientation.y = float(self.state_vector[7])
        pose_msg.pose.pose.orientation.z = float(self.state_vector[8])
        pose_msg.pose.pose.orientation.w = float(self.state_vector[9])

        pose_msg.pose.covariance = self.pose_covariance.flatten().tolist()

        self.fused_pose_pub.publish(pose_msg)

        # Create twist message
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = current_time.to_msg()
        twist_msg.header.frame_id = 'base_link'

        twist_msg.twist.twist.linear.x = float(self.state_vector[3])
        twist_msg.twist.twist.linear.y = float(self.state_vector[4])
        twist_msg.twist.twist.linear.z = float(self.state_vector[5])

        twist_msg.twist.twist.angular.x = float(self.state_vector[10])
        twist_msg.twist.twist.angular.y = float(self.state_vector[11])
        twist_msg.twist.twist.angular.z = float(self.state_vector[12])

        twist_msg.twist.covariance = self.twist_covariance.flatten().tolist()

        self.fused_twist_pub.publish(twist_msg)

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose = pose_msg.pose
        odom_msg.twist = twist_msg.twist

        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    # Timer for fusion updates
    fusion_timer = fusion_node.create_timer(0.05, fusion_node.fuse_sensor_data)  # 20 Hz

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

## Sensor Calibration

### Calibration Procedures

```python
# sensor_calibration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Imu, LaserScan
from std_msgs.msg import String
import numpy as np
import yaml
import os
from pathlib import Path

class SensorCalibrationNode(Node):
    def __init__(self):
        super().__init__('sensor_calibration')

        # Publishers for calibrated data
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/calibrated_info', 10)
        self.calibration_status_pub = self.create_publisher(String, 'calibration_status', 10)

        # Subscribers for raw sensor data
        self.camera_raw_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )
        self.imu_raw_sub = self.create_subscription(
            Imu, 'imu/data_raw', self.imu_callback, 10
        )
        self.scan_raw_sub = self.create_subscription(
            LaserScan, 'scan_raw', self.scan_callback, 10
        )

        # Calibration parameters
        self.camera_calibration_params = {
            'intrinsic_matrix': np.eye(3),
            'distortion_coeffs': np.zeros(5),
            'rectification_matrix': np.eye(3),
            'projection_matrix': np.zeros((3, 4))
        }

        self.imu_calibration_params = {
            'bias': np.zeros(6),  # [angular_x, angular_y, angular_z, linear_x, linear_y, linear_z]
            'scale_factor': np.ones(6),
            'alignment_matrix': np.eye(3)
        }

        self.lidar_calibration_params = {
            'extrinsic_transform': np.eye(4),  # 4x4 transformation matrix
            'range_offset': 0.0,
            'angle_offset': 0.0
        }

        # Calibration state
        self.camera_calibrated = False
        self.imu_calibrated = False
        self.lidar_calibrated = False

        # Calibration methods
        self.calibration_methods = {
            'camera': self.calibrate_camera,
            'imu': self.calibrate_imu,
            'lidar': self.calibrate_lidar
        }

        self.get_logger().info('Sensor Calibration Node initialized')

    def camera_info_callback(self, msg):
        """Process camera info for calibration."""
        if not self.camera_calibrated:
            # Check if we have calibration data available
            calib_file = self.get_parameter_or('camera_calibration_file', '').value
            if calib_file and os.path.exists(calib_file):
                self.load_camera_calibration(calib_file)
                self.camera_calibrated = True

                # Publish calibrated camera info
                calibrated_info = self.apply_camera_calibration(msg)
                self.camera_info_pub.publish(calibrated_info)

                self.get_logger().info('Camera calibration applied')

    def imu_callback(self, msg):
        """Process IMU data for calibration."""
        if not self.imu_calibrated:
            # Check if we have calibration data
            calib_file = self.get_parameter_or('imu_calibration_file', '').value
            if calib_file and os.path.exists(calib_file):
                self.load_imu_calibration(calib_file)
                self.imu_calibrated = True
                self.get_logger().info('IMU calibration loaded')

    def scan_callback(self, msg):
        """Process LiDAR scan for calibration."""
        if not self.lidar_calibrated:
            # Check if we have calibration data
            calib_file = self.get_parameter_or('lidar_calibration_file', '').value
            if calib_file and os.path.exists(calib_file):
                self.load_lidar_calibration(calib_file)
                self.lidar_calibrated = True
                self.get_logger().info('LiDAR calibration loaded')

    def calibrate_camera(self):
        """Perform camera calibration."""
        self.get_logger().info('Starting camera calibration...')

        # In a real implementation, this would:
        # 1. Collect images of calibration pattern (checkerboard)
        # 2. Detect calibration pattern in images
        # 3. Compute intrinsic parameters
        # 4. Compute distortion coefficients
        # 5. Save parameters to file

        # For simulation, we'll create mock calibration data
        self.camera_calibration_params['intrinsic_matrix'] = np.array([
            [616.27, 0.0, 320.0],
            [0.0, 616.27, 240.0],
            [0.0, 0.0, 1.0]
        ])

        self.camera_calibration_params['distortion_coeffs'] = np.array([-0.4, 0.1, 0.0, 0.0, 0.0])

        self.camera_calibrated = True
        self.get_logger().info('Camera calibration completed')

    def calibrate_imu(self):
        """Perform IMU calibration."""
        self.get_logger().info('Starting IMU calibration...')

        # IMU calibration typically involves:
        # 1. Static bias estimation
        # 2. Scale factor calibration
        # 3. Alignment calibration

        # For static bias estimation, collect data while IMU is stationary
        # This would typically run for several seconds to accumulate data

        self.imu_calibration_params['bias'] = np.array([0.01, -0.02, 0.005, 0.05, -0.03, 9.81])  # Example biases
        self.imu_calibrated = True
        self.get_logger().info('IMU calibration completed')

    def calibrate_lidar(self):
        """Perform LiDAR calibration."""
        self.get_logger().info('Starting LiDAR calibration...')

        # LiDAR calibration involves:
        # 1. Extrinsics: position and orientation relative to robot
        # 2. Intrinsics: if applicable (range accuracy, angular precision)

        self.lidar_calibration_params['extrinsic_transform'] = np.eye(4)
        self.lidar_calibrated = True
        self.get_logger().info('LiDAR calibration completed')

    def apply_camera_calibration(self, original_info):
        """Apply camera calibration parameters."""
        calibrated_info = CameraInfo()
        calibrated_info.header = original_info.header

        # Apply calibrated parameters
        calibrated_info.height = original_info.height
        calibrated_info.width = original_info.width

        # Use calibrated intrinsic matrix
        calibrated_info.k = self.camera_calibration_params['intrinsic_matrix'].flatten().tolist()

        # Use calibrated distortion coefficients
        calibrated_info.d = self.camera_calibration_params['distortion_coeffs'].tolist()

        # Compute rectification and projection matrices
        calibrated_info.r = self.camera_calibration_params['rectification_matrix'].flatten().tolist()
        calibrated_info.p = self.camera_calibration_params['projection_matrix'].flatten().tolist()

        calibrated_info.distortion_model = 'plumb_bob'

        return calibrated_info

    def apply_imu_calibration(self, original_imu):
        """Apply IMU calibration parameters."""
        calibrated_imu = Imu()
        calibrated_imu.header = original_imu.header

        # Apply bias correction and scale factor
        angular_vel = np.array([
            original_imu.angular_velocity.x,
            original_imu.angular_velocity.y,
            original_imu.angular_velocity.z
        ])

        linear_acc = np.array([
            original_imu.linear_acceleration.x,
            original_imu.linear_acceleration.y,
            original_imu.linear_acceleration.z
        ])

        # Apply calibration
        calibrated_angular = (angular_vel - self.imu_calibration_params['bias'][:3]) * self.imu_calibration_params['scale_factor'][:3]
        calibrated_linear = (linear_acc - self.imu_calibration_params['bias'][3:]) * self.imu_calibration_params['scale_factor'][3:]

        calibrated_imu.angular_velocity.x = calibrated_angular[0]
        calibrated_imu.angular_velocity.y = calibrated_angular[1]
        calibrated_imu.angular_velocity.z = calibrated_angular[2]

        calibrated_imu.linear_acceleration.x = calibrated_linear[0]
        calibrated_imu.linear_acceleration.y = calibrated_linear[1]
        calibrated_imu.linear_acceleration.z = calibrated_linear[2]

        # Copy orientation (assumes already calibrated or from magnetometer)
        calibrated_imu.orientation = original_imu.orientation
        calibrated_imu.orientation_covariance = original_imu.orientation_covariance

        return calibrated_imu

    def save_calibration_to_file(self, sensor_type, file_path):
        """Save calibration parameters to file."""
        calib_data = {}

        if sensor_type == 'camera':
            calib_data = {
                'camera_matrix': self.camera_calibration_params['intrinsic_matrix'].tolist(),
                'distortion_coefficients': self.camera_calibration_params['distortion_coeffs'].tolist(),
                'rectification_matrix': self.camera_calibration_params['rectification_matrix'].tolist(),
                'projection_matrix': self.camera_calibration_params['projection_matrix'].tolist()
            }
        elif sensor_type == 'imu':
            calib_data = {
                'bias': self.imu_calibration_params['bias'].tolist(),
                'scale_factor': self.imu_calibration_params['scale_factor'].tolist(),
                'alignment_matrix': self.imu_calibration_params['alignment_matrix'].tolist()
            }
        elif sensor_type == 'lidar':
            calib_data = {
                'extrinsic_transform': self.lidar_calibration_params['extrinsic_transform'].tolist(),
                'range_offset': self.lidar_calibration_params['range_offset'],
                'angle_offset': self.lidar_calibration_params['angle_offset']
            }

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.dump(calib_data, f)

        self.get_logger().info(f'Calibration saved to: {file_path}')

    def load_camera_calibration(self, file_path):
        """Load camera calibration from file."""
        try:
            with open(file_path, 'r') as f:
                calib_data = yaml.safe_load(f)

            self.camera_calibration_params['intrinsic_matrix'] = np.array(calib_data['camera_matrix']).reshape(3, 3)
            self.camera_calibration_params['distortion_coeffs'] = np.array(calib_data['distortion_coefficients'])
            self.camera_calibration_params['rectification_matrix'] = np.array(calib_data['rectification_matrix']).reshape(3, 3)
            self.camera_calibration_params['projection_matrix'] = np.array(calib_data['projection_matrix']).reshape(3, 4)

            self.get_logger().info(f'Camera calibration loaded from: {file_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load camera calibration: {e}')

    def load_imu_calibration(self, file_path):
        """Load IMU calibration from file."""
        try:
            with open(file_path, 'r') as f:
                calib_data = yaml.safe_load(f)

            self.imu_calibration_params['bias'] = np.array(calib_data['bias'])
            self.imu_calibration_params['scale_factor'] = np.array(calib_data['scale_factor'])
            self.imu_calibration_params['alignment_matrix'] = np.array(calib_data['alignment_matrix']).reshape(3, 3)

            self.get_logger().info(f'IMU calibration loaded from: {file_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load IMU calibration: {e}')

    def load_lidar_calibration(self, file_path):
        """Load LiDAR calibration from file."""
        try:
            with open(file_path, 'r') as f:
                calib_data = yaml.safe_load(f)

            self.lidar_calibration_params['extrinsic_transform'] = np.array(calib_data['extrinsic_transform']).reshape(4, 4)
            self.lidar_calibration_params['range_offset'] = calib_data['range_offset']
            self.lidar_calibration_params['angle_offset'] = calib_data['angle_offset']

            self.get_logger().info(f'LiDAR calibration loaded from: {file_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load LiDAR calibration: {e}')

    def start_calibration_procedure(self, sensor_type):
        """Start calibration procedure for specified sensor."""
        if sensor_type in self.calibration_methods:
            self.calibration_methods[sensor_type]()

            # Save calibration to default location
            default_path = f'calibration/{sensor_type}_calibration.yaml'
            self.save_calibration_to_file(sensor_type, default_path)

            # Publish status
            status_msg = String()
            status_msg.data = f'{sensor_type} calibration completed and saved'
            self.calibration_status_pub.publish(status_msg)
        else:
            self.get_logger().error(f'Unknown sensor type for calibration: {sensor_type}')

def main(args=None):
    rclpy.init(args=args)
    calib_node = SensorCalibrationNode()

    try:
        rclpy.spin(calib_node)
    except KeyboardInterrupt:
        calib_node.get_logger().info('Shutting down Sensor Calibration Node')
    finally:
        calib_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive sensor integration for robotics:

- **Camera Integration**: RGB and depth camera setup, processing, and calibration
- **LiDAR Integration**: 2D and 3D LiDAR processing, obstacle detection, and segmentation
- **IMU Integration**: Inertial measurement processing, orientation estimation, and filtering
- **Sensor Fusion**: Combining multiple sensor inputs for improved state estimation
- **Calibration**: Sensor calibration procedures and parameter management

Proper sensor integration is fundamental to robotic perception and enables robots to understand and interact with their environment effectively.

## Exercises

1. Set up camera integration with your robotic system
2. Implement LiDAR processing for obstacle detection
3. Create a sensor fusion system combining IMU and odometry
4. Perform calibration procedures for your sensors
5. Test sensor integration in various environmental conditions

## Quiz

1. What is the main purpose of sensor fusion in robotics?
   a) Reduce the number of sensors needed
   b) Combine multiple sensor inputs for better state estimation
   c) Increase sensor cost
   d) Make sensors heavier

2. Which sensor would be best for precise distance measurements in robotics?
   a) Camera only
   b) IMU only
   c) LiDAR or depth camera
   d) GPS only

3. What does IMU stand for?
   a) Intelligent Motion Unit
   b) Inertial Measurement Unit
   c) Integrated Motor Unit
   d) Internal Memory Unit

## Mini-Project: Complete Sensor Integration System

Create a complete sensor integration system with:
1. Camera, LiDAR, and IMU integration
2. Sensor fusion for state estimation
3. Calibration procedures for all sensors
4. Real-time sensor data processing and visualization
5. Testing with various environmental conditions
6. Performance evaluation and optimization