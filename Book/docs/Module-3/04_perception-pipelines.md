---
sidebar_position: 4
---

# Perception Pipelines

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement GPU-accelerated perception pipelines using Isaac tools
- Configure and optimize deep learning models for robotic perception tasks
- Integrate multiple sensors for robust perception in complex environments
- Implement sensor fusion techniques for enhanced perception accuracy
- Validate perception system performance in simulation and real-world scenarios

## Introduction to Perception in Robotics

Perception is the cornerstone of autonomous robotics, enabling robots to understand and interact with their environment. In the Isaac ecosystem, perception pipelines leverage GPU acceleration and deep learning to achieve real-time performance with high accuracy.

### Perception Pipeline Architecture

```
Robot Perception Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Sensors   │───→│   Processing    │───→│   Understanding │
│   (Cameras,     │    │   (Filtering,   │    │   (Object      │
│   LiDAR, IMU)   │    │   Enhancement)  │    │   Detection,   │
│                 │    │                 │    │   Classification│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Action        │
                         │   Planning      │
                         │   (Navigation,  │
                         │   Manipulation) │
                         └─────────────────┘
```

### Key Perception Tasks

1. **Object Detection**: Identifying and localizing objects in the environment
2. **Semantic Segmentation**: Pixel-level classification of scene elements
3. **Instance Segmentation**: Differentiating between individual object instances
4. **Pose Estimation**: Determining 3D position and orientation of objects
5. **Scene Understanding**: Interpreting complex scenes and relationships

## GPU-Accelerated Perception

### Leveraging CUDA and TensorRT

GPU acceleration is fundamental to Isaac's perception capabilities, providing the computational power needed for real-time deep learning inference.

```python
# gpu_perception_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

try:
    import torch
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU perception will use simulation.")

class GPUPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('gpu_perception_pipeline')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )

        # Create publishers
        self.detections_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.segmentation_pub = self.create_publisher(Image, 'segmentation', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Load pre-trained model (YOLOv5 or similar)
        self.load_detection_model()

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('GPU Perception Pipeline initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def load_detection_model(self):
        """Load and prepare GPU-accelerated detection model."""
        if not TORCH_AVAILABLE:
            self.get_logger().warn('PyTorch not available, using simulated processing')
            return

        try:
            # Load a pre-trained model (example with YOLOv5)
            # In real implementation, you'd load a TensorRT optimized model
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            self.model.eval()

            self.get_logger().info('Detection model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Could not load detection model: {e}')
            self.model = None

    def image_callback(self, msg):
        """Process incoming camera images using GPU acceleration."""
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Process image using GPU
        if TORCH_AVAILABLE and self.model is not None:
            detections = self.process_with_gpu(cv_image, msg.header)
        else:
            # Fallback to CPU or simulated processing
            detections = self.process_with_simulation(cv_image, msg.header)

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        # Limit stored times for performance
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Publish results
        if detections is not None:
            self.detections_pub.publish(detections)

        # Log performance metrics periodically
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(f'GPU Processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}')

    def process_with_gpu(self, image, header):
        """Process image using GPU-accelerated model."""
        try:
            # Preprocess image for model
            img_tensor = self.preprocess_image(image)

            # Move to GPU
            img_tensor = img_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                results = self.model(img_tensor)

            # Process results
            detections = self.postprocess_results(results, header, image.shape)

            return detections
        except Exception as e:
            self.get_logger().error(f'GPU processing error: {e}')
            return self.process_with_simulation(image, header)

    def preprocess_image(self, image):
        """Preprocess image for neural network."""
        # Resize image to model input size (640x640 for YOLOv5)
        img_resized = cv2.resize(image, (640, 640))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        img_tensor = transforms.ToTensor()(img_rgb)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def postprocess_results(self, results, header, original_shape):
        """Postprocess neural network results."""
        # In a real implementation, this would extract bounding boxes,
        # class labels, and confidence scores from the model output
        detections = Detection2DArray()
        detections.header = header

        # For simulation, create mock detections
        height, width = original_shape[:2]

        # Generate some simulated detections
        num_detections = np.random.randint(0, 5)

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

    def process_with_simulation(self, image, header):
        """Simulate processing when GPU is not available."""
        self.get_logger().warn('Using simulated processing - GPU acceleration unavailable')

        # Simulate some processing time
        time.sleep(0.1)  # Simulate 100ms processing time

        # Return mock detections
        return self.create_mock_detections(header)

    def create_mock_detections(self, header):
        """Create mock detection results for simulation."""
        detections = Detection2DArray()
        detections.header = header

        # Add some random detections
        for i in range(np.random.randint(1, 4)):
            detection = Detection2D()
            detection.bbox.center.x = float(np.random.uniform(100, 500))
            detection.bbox.center.y = float(np.random.uniform(100, 300))
            detection.bbox.size_x = float(np.random.uniform(50, 150))
            detection.bbox.size_y = float(np.random.uniform(50, 150))

            hypothesis = ObjectHypothesisWithPose()
            classes = ['person', 'car', 'chair']
            hypothesis.hypothesis.class_id = np.random.choice(classes)
            hypothesis.hypothesis.score = float(np.random.uniform(0.6, 0.95))

            detection.results.append(hypothesis)
            detections.detections.append(detection)

        return detections

def main(args=None):
    rclpy.init(args=args)
    pipeline = GPUPerceptionPipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pipeline.get_logger().info('Shutting down GPU Perception Pipeline')
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deep Learning Models for Perception

### Model Optimization with TensorRT

TensorRT optimization is crucial for achieving real-time performance in robotic perception systems.

```python
# tensorrt_optimization.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. Using simulated optimization.")

class TensorRTOptimizedPerception(Node):
    def __init__(self):
        super().__init__('tensorrt_optimized_perception')

        # Subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.detections_pub = self.create_publisher(Detection2DArray, 'tensorrt_detections', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # TensorRT components
        self.trt_engine = None
        self.context = None
        self.input_binding = None
        self.output_binding = None

        # Model parameters
        self.input_shape = (1, 3, 640, 640)  # Batch, Channels, Height, Width
        self.output_shape = (1, 25200, 85)   # For YOLOv5

        # Load TensorRT optimized model
        if TENSORRT_AVAILABLE:
            self.load_tensorrt_model()
        else:
            self.get_logger().warn('TensorRT not available, using simulated processing')

        # Performance tracking
        self.processing_times = []

        self.get_logger().info('TensorRT Optimized Perception initialized')

    def load_tensorrt_model(self):
        """Load TensorRT optimized model."""
        try:
            # Create TensorRT runtime
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.trt_logger)

            # Load pre-converted TensorRT engine file
            # In real implementation, you would have a .engine file
            # engine_path = "path/to/optimized_model.engine"
            # with open(engine_path, 'rb') as f:
            #     engine_data = f.read()
            # self.trt_engine = self.runtime.deserialize_cuda_engine(engine_data)

            # For simulation, we'll create a mock engine
            self.get_logger().info('TensorRT model loaded (simulated)')

        except Exception as e:
            self.get_logger().error(f'Could not load TensorRT model: {e}')
            self.trt_engine = None

    def image_callback(self, msg):
        """Process image with TensorRT optimized model."""
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Process with TensorRT if available, otherwise simulate
        if TENSORRT_AVAILABLE and self.trt_engine is not None:
            detections = self.process_with_tensorrt(cv_image, msg.header)
        else:
            detections = self.process_with_simulation(cv_image, msg.header)

        # Calculate and track processing time
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        if len(self.processing_times) > 50:
            self.processing_times.pop(0)

        # Publish results
        if detections is not None:
            self.detections_pub.publish(detections)

        # Log performance
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0

        if len(self.processing_times) % 10 == 0:  # Log every 10 frames
            self.get_logger().info(f'TensorRT Processing - Avg: {avg_time*1000:.1f}ms, FPS: {fps:.1f}')

    def process_with_tensorrt(self, image, header):
        """Process image using TensorRT optimized model."""
        try:
            # Preprocess image
            input_tensor = self.preprocess_for_tensorrt(image)

            # In real implementation, you would:
            # 1. Copy input to GPU memory
            # 2. Execute inference
            # 3. Copy output from GPU memory
            # 4. Postprocess results

            # Simulate TensorRT processing
            time.sleep(0.02)  # Simulate 20ms processing time for optimized model

            # Postprocess results
            detections = self.postprocess_tensorrt_results(header)

            return detections

        except Exception as e:
            self.get_logger().error(f'TensorRT processing error: {e}')
            return self.process_with_simulation(image, header)

    def preprocess_for_tensorrt(self, image):
        """Preprocess image for TensorRT model."""
        # Resize image to model input size
        img_resized = cv2.resize(image, (640, 640))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Transpose to CHW format (channels, height, width)
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch

    def postprocess_tensorrt_results(self, header):
        """Postprocess TensorRT results."""
        # In real implementation, this would parse the TensorRT output
        # For now, create mock detections
        detections = Detection2DArray()
        detections.header = header

        # Simulate detection results with high confidence (TensorRT advantage)
        for i in range(np.random.randint(1, 5)):
            detection = Detection2D()
            detection.bbox.center.x = float(np.random.uniform(100, 500))
            detection.bbox.center.y = float(np.random.uniform(100, 300))
            detection.bbox.size_x = float(np.random.uniform(50, 150))
            detection.bbox.size_y = float(np.random.uniform(50, 150))

            hypothesis = ObjectHypothesisWithPose()
            classes = ['person', 'car', 'bicycle', 'traffic_sign']
            hypothesis.hypothesis.class_id = np.random.choice(classes)
            # TensorRT models typically have higher confidence scores
            hypothesis.hypothesis.score = float(np.random.uniform(0.8, 0.99))

            detection.results.append(hypothesis)
            detections.detections.append(detection)

        return detections

    def process_with_simulation(self, image, header):
        """Simulate processing when TensorRT is not available."""
        self.get_logger().warn('Using simulated processing - TensorRT unavailable')

        # Simulate longer processing time for non-optimized approach
        time.sleep(0.05)  # Simulate 50ms processing time

        # Create mock detections
        detections = Detection2DArray()
        detections.header = header

        for i in range(np.random.randint(1, 4)):
            detection = Detection2D()
            detection.bbox.center.x = float(np.random.uniform(100, 500))
            detection.bbox.center.y = float(np.random.uniform(100, 300))
            detection.bbox.size_x = float(np.random.uniform(50, 150))
            detection.bbox.size_y = float(np.random.uniform(50, 150))

            hypothesis = ObjectHypothesisWithPose()
            classes = ['person', 'car', 'chair']
            hypothesis.hypothesis.class_id = np.random.choice(classes)
            hypothesis.hypothesis.score = float(np.random.uniform(0.6, 0.85))  # Lower confidence

            detection.results.append(hypothesis)
            detections.detections.append(detection)

        return detections

def main(args=None):
    rclpy.init(args=args)
    tensorrt_node = TensorRTOptimizedPerception()

    try:
        rclpy.spin(tensorrt_node)
    except KeyboardInterrupt:
        tensorrt_node.get_logger().info('Shutting down TensorRT Optimized Perception')
    finally:
        tensorrt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-Sensor Fusion

### Combining Camera and LiDAR Data

Multi-sensor fusion combines data from different sensors to create a more robust and accurate perception system.

```python
# multi_sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2, CameraInfo, Imu
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R

class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Subscribers for different sensors
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )

        # Publishers for fused data
        self.fused_detections_pub = self.create_publisher(Detection3DArray, 'fused_detections', 10)
        self.enhanced_detections_pub = self.create_publisher(Detection2DArray, 'enhanced_detections', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Sensor data storage
        self.latest_image = None
        self.latest_scan = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.imu_orientation = None

        # Sensor calibration (camera-LiDAR extrinsics)
        self.camera_to_lidar_rotation = np.eye(3)  # Identity as default
        self.camera_to_lidar_translation = np.array([0.0, 0.0, 0.0])

        # For tracking and association
        self.tracked_objects = {}
        self.next_object_id = 0

        self.get_logger().info('Multi-Sensor Fusion initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

        # Extract camera-LiDAR transformation from calibration
        # In real implementation, this would come from calibration file
        self.get_logger().info('Camera calibration received')

    def imu_callback(self, msg):
        """Process IMU data for orientation."""
        # Convert quaternion to rotation matrix
        q = msg.orientation
        self.imu_orientation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def image_callback(self, msg):
        """Process camera image and store for fusion."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_header = msg.header
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')

    def scan_callback(self, msg):
        """Process LiDAR scan data."""
        self.latest_scan = msg
        self.scan_header = msg.header

        # Convert scan to Cartesian points
        self.scan_points = self.scan_to_cartesian(msg)

    def scan_to_cartesian(self, scan_msg):
        """Convert polar scan data to Cartesian coordinates."""
        points = []
        angle = scan_msg.angle_min

        for i, range_val in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                points.append([x, y, 0.0])  # Assuming 2D scan at z=0

            angle += scan_msg.angle_increment

        return np.array(points)

    def project_point_to_image(self, point_3d, camera_matrix, dist_coeffs):
        """Project 3D point to 2D image coordinates."""
        if camera_matrix is None:
            return None

        # Apply camera-LiDAR transformation
        transformed_point = self.transform_point(point_3d)

        # Project to image
        point_2d, _ = cv2.projectPoints(
            transformed_point.reshape(1, 1, 3),
            np.zeros(3),  # rvec
            np.zeros(3),  # tvec
            camera_matrix,
            dist_coeffs if dist_coeffs is not None else np.zeros(5)
        )

        u, v = int(point_2d[0, 0, 0]), int(point_2d[0, 0, 1])
        return (u, v)

    def transform_point(self, point):
        """Apply camera-LiDAR transformation to a point."""
        # Rotate point
        rotated_point = self.camera_to_lidar_rotation @ point
        # Translate point
        transformed_point = rotated_point + self.camera_to_lidar_translation
        return transformed_point

    def associate_detections(self, image_detections, scan_points):
        """Associate 2D image detections with 3D LiDAR points."""
        if self.camera_matrix is None or scan_points.size == 0:
            return []

        associated_detections = []

        for detection in image_detections:
            # Get bounding box center
            center_2d = (int(detection.bbox.center.x), int(detection.bbox.center.y))

            # Find LiDAR points that project near this center
            nearby_points = []
            for point_3d in scan_points:
                projected_2d = self.project_point_to_image(point_3d, self.camera_matrix, self.distortion_coeffs)

                if projected_2d is not None:
                    dist = math.sqrt((projected_2d[0] - center_2d[0])**2 + (projected_2d[1] - center_2d[1])**2)

                    # If point projects within detection bounding box area
                    bbox_threshold = max(detection.bbox.size_x, detection.bbox.size_y) / 4  # Adjust as needed
                    if dist < bbox_threshold:
                        nearby_points.append(point_3d)

            if nearby_points:
                # Create 3D detection by combining 2D detection with 3D position
                detection_3d = self.create_3d_detection(detection, nearby_points)
                associated_detections.append(detection_3d)

        return associated_detections

    def create_3d_detection(self, detection_2d, nearby_points):
        """Create 3D detection from 2D detection and nearby LiDAR points."""
        detection_3d = Detection3D()
        detection_3d.header = self.image_header  # Use image timestamp

        # Calculate 3D position from LiDAR points
        avg_point = np.mean(nearby_points, axis=0)
        detection_3d.bbox.center.position.x = float(avg_point[0])
        detection_3d.bbox.center.position.y = float(avg_point[1])
        detection_3d.bbox.center.position.z = float(avg_point[2])

        # Estimate 3D size based on LiDAR points
        if len(nearby_points) > 1:
            point_cloud = np.array(nearby_points)
            size_x = float(np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0]))
            size_y = float(np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1]))
            size_z = float(np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2]))

            detection_3d.bbox.size.x = max(size_x, 0.5)  # Minimum size
            detection_3d.bbox.size.y = max(size_y, 0.5)
            detection_3d.bbox.size.z = max(size_z, 0.5)
        else:
            # Default size if only one point
            detection_3d.bbox.size.x = 1.0
            detection_3d.bbox.size.y = 1.0
            detection_3d.bbox.size.z = 1.0

        # Copy results from 2D detection
        detection_3d.results = detection_2d.results

        return detection_3d

    def run_fusion(self):
        """Run the fusion algorithm when both sensors have new data."""
        if self.latest_image is None or self.latest_scan is None:
            return

        # For this example, we'll create mock 2D detections from the image
        # In real implementation, you would run the actual detection algorithm
        mock_detections_2d = self.create_mock_image_detections()

        # Associate 2D detections with 3D LiDAR points
        fused_detections = self.associate_detections(mock_detections_2d, self.scan_points)

        # Create and publish 3D detection array
        detection_array = Detection3DArray()
        detection_array.header = self.image_header
        detection_array.detections = fused_detections

        self.fused_detections_pub.publish(detection_array)

        # Also publish enhanced 2D detections with additional info
        enhanced_detections = self.create_enhanced_detections(mock_detections_2d, fused_detections)
        self.enhanced_detections_pub.publish(enhanced_detections)

        self.get_logger().info(f'Fusion completed: {len(fused_detections)} 3D detections created')

    def create_mock_image_detections(self):
        """Create mock 2D detections for demonstration."""
        # In real implementation, this would come from the perception pipeline
        detections = []

        for i in range(np.random.randint(1, 4)):
            detection = Detection2D()
            detection.bbox.center.x = float(np.random.uniform(100, 500))
            detection.bbox.center.y = float(np.random.uniform(100, 300))
            detection.bbox.size_x = float(np.random.uniform(50, 150))
            detection.bbox.size_y = float(np.random.uniform(50, 150))

            hypothesis = ObjectHypothesisWithPose()
            classes = ['person', 'car', 'bicycle']
            hypothesis.hypothesis.class_id = np.random.choice(classes)
            hypothesis.hypothesis.score = float(np.random.uniform(0.7, 0.95))

            detection.results.append(hypothesis)
            detections.append(detection)

        return detections

    def create_enhanced_detections(self, detections_2d, detections_3d):
        """Create enhanced 2D detections with fused information."""
        # In a real system, this would add 3D information to 2D detections
        # For now, return the original 2D detections
        enhanced_array = Detection2DArray()
        enhanced_array.header = self.image_header
        enhanced_array.detections = detections_2d
        return enhanced_array

def multi_sensor_fusion_timer_callback():
    """Timer callback to run fusion periodically."""
    # This would be added as a timer in the actual implementation
    pass

def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultiSensorFusion()

    # In real implementation, you'd add a timer to run fusion periodically
    # fusion_timer = fusion_node.create_timer(0.1, fusion_node.run_fusion)

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Shutting down Multi-Sensor Fusion')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Semantic Segmentation Pipeline

Semantic segmentation provides pixel-level understanding of the environment.

```python
# semantic_segmentation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import numpy as np
import cv2

try:
    import torch
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simulated segmentation.")

class SemanticSegmentationPipeline(Node):
    def __init__(self):
        super().__init__('semantic_segmentation_pipeline')

        # Subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.segmentation_pub = self.create_publisher(Image, 'segmentation', 10)
        self.segmentation_viz_pub = self.create_publisher(Image, 'segmentation_visualization', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'segmentation_markers', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Load segmentation model
        self.load_segmentation_model()

        # Color palette for different classes
        self.color_palette = self.create_color_palette()

        self.get_logger().info('Semantic Segmentation Pipeline initialized')

    def load_segmentation_model(self):
        """Load semantic segmentation model."""
        if not TORCH_AVAILABLE:
            self.get_logger().warn('PyTorch not available, using simulated segmentation')
            return

        try:
            # Load a pre-trained segmentation model (example with DeepLabV3)
            # In real implementation, you'd use a TensorRT optimized model
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
            self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.model.eval()

            self.get_logger().info('Segmentation model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Could not load segmentation model: {e}')
            self.model = None

    def create_color_palette(self):
        """Create a color palette for different segmentation classes."""
        # Common classes in Cityscapes dataset
        classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle'
        ]

        # Assign random colors to classes (in practice, you'd use consistent colors)
        palette = {}
        for i, class_name in enumerate(classes):
            # Generate a random color
            color = np.random.randint(0, 255, size=3)
            palette[class_name] = color.astype(np.uint8)

        # Add a color for unknown/unlabeled areas
        palette['unknown'] = np.array([0, 0, 0], dtype=np.uint8)  # Black

        return palette

    def image_callback(self, msg):
        """Process image for semantic segmentation."""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Perform segmentation
        if TORCH_AVAILABLE and self.model is not None:
            segmentation_mask = self.perform_segmentation(cv_image)
        else:
            segmentation_mask = self.perform_segmentation_simulation(cv_image)

        # Create visualization
        visualization = self.create_segmentation_visualization(cv_image, segmentation_mask)

        # Publish results
        try:
            segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_mask, encoding='mono8')
            segmentation_msg.header = msg.header
            self.segmentation_pub.publish(segmentation_msg)

            viz_msg = self.bridge.cv2_to_imgmsg(visualization, encoding='bgr8')
            viz_msg.header = msg.header
            self.segmentation_viz_pub.publish(viz_msg)
        except Exception as e:
            self.get_logger().error(f'Could not convert/publish segmentation: {e}')

    def perform_segmentation(self, image):
        """Perform semantic segmentation using deep learning model."""
        try:
            # Preprocess image
            img_tensor = self.preprocess_segmentation_image(image)

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                predictions = outputs['out']

            # Get predicted classes
            predicted_mask = torch.argmax(predictions.squeeze(0), dim=0).cpu().numpy()

            # Convert to 8-bit format
            segmentation_mask = predicted_mask.astype(np.uint8)

            return segmentation_mask

        except Exception as e:
            self.get_logger().error(f'Segmentation inference error: {e}')
            return self.perform_segmentation_simulation(image)

    def preprocess_segmentation_image(self, image):
        """Preprocess image for segmentation model."""
        # Resize image to model input size (typically 512x1024 for Cityscapes)
        img_resized = cv2.resize(image, (1024, 512))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_tensor = transforms.ToTensor()(img_normalized)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = img_tensor.to(device)

        return img_tensor

    def perform_segmentation_simulation(self, image):
        """Simulate segmentation when deep learning model is not available."""
        self.get_logger().warn('Using simulated segmentation')

        # Create a simulated segmentation mask
        # In reality, this would be much more sophisticated
        height, width = image.shape[:2]

        # Create a simple geometric pattern to simulate segmentation
        segmentation_mask = np.zeros((height, width), dtype=np.uint8)

        # Add some simulated "objects"
        # Road (class 0)
        cv2.rectangle(segmentation_mask, (0, height//2), (width, height), 0, -1)

        # Sky (class 10)
        cv2.rectangle(segmentation_mask, (0, 0), (width, height//2), 10, -1)

        # Buildings (class 2)
        cv2.rectangle(segmentation_mask, (width//4, height//2), (width//2, height*3//4), 2, -1)
        cv2.rectangle(segmentation_mask, (width*3//4, height//3), (width, height*2//3), 2, -1)

        # Person (class 11)
        cv2.circle(segmentation_mask, (width//2, height*2//3), 20, 11, -1)

        # Car (class 13)
        cv2.rectangle(segmentation_mask, (width//3, height*3//4), (width//3 + 60, height*3//4 + 30), 13, -1)

        return segmentation_mask

    def create_segmentation_visualization(self, original_image, segmentation_mask):
        """Create a color visualization of the segmentation."""
        # Create a color image for visualization
        height, width = segmentation_mask.shape
        visualization = np.zeros((height, width, 3), dtype=np.uint8)

        # Assign colors based on segmentation mask
        for class_id in np.unique(segmentation_mask):
            mask = segmentation_mask == class_id

            # Use a color from the palette, or random color if not in palette
            if class_id < len(self.color_palette):
                color = list(self.color_palette.values())[class_id]
            else:
                # Generate a random color for unknown classes
                color = np.random.randint(0, 255, size=3)

            visualization[mask] = color

        # Blend with original image for better visualization
        alpha = 0.6
        blended = cv2.addWeighted(original_image, 1-alpha, visualization, alpha, 0)

        return blended

    def create_segmentation_markers(self, segmentation_mask, header):
        """Create visualization markers for segmentation results."""
        markers = MarkerArray()

        # In a real implementation, you'd create markers for segmented regions
        # For now, we'll create a simple example

        # Create a marker for the entire segmentation
        marker = Marker()
        marker.header = header
        marker.ns = "segmentation"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = f"Segmentation: {segmentation_mask.shape[1]}x{segmentation_mask.shape[0]}"

        markers.markers.append(marker)

        return markers

def main(args=None):
    rclpy.init(args=args)
    seg_node = SemanticSegmentationPipeline()

    try:
        rclpy.spin(seg_node)
    except KeyboardInterrupt:
        seg_node.get_logger().info('Shutting down Semantic Segmentation Pipeline')
    finally:
        seg_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3D Object Detection and Pose Estimation

### Detecting and Estimating Object Poses

```python
# object_pose_estimation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection3D
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import math

class ObjectPoseEstimation(Node):
    def __init__(self):
        super().__init__('object_pose_estimation')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 'points', self.pointcloud_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )

        # Publishers
        self.pose_estimates_pub = self.create_publisher(Detection3DArray, 'pose_estimates', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Latest data
        self.latest_image = None
        self.latest_pointcloud = None

        # Object models for pose estimation (simplified)
        self.object_models = self.initialize_object_models()

        self.get_logger().info('Object Pose Estimation initialized')

    def initialize_object_models(self):
        """Initialize 3D models for known objects."""
        # In a real implementation, this would load CAD models or point clouds
        # For simulation, we'll define simple geometric models
        models = {
            'car': {
                'dimensions': [4.0, 1.8, 1.5],  # length, width, height
                'keypoints': np.array([
                    [-2, -0.9, -0.75], [2, -0.9, -0.75],  # front corners
                    [-2, 0.9, -0.75], [2, 0.9, -0.75],    # back corners
                    [-2, -0.9, 0.75], [2, -0.9, 0.75],    # top front
                    [-2, 0.9, 0.75], [2, 0.9, 0.75]       # top back
                ])
            },
            'person': {
                'dimensions': [0.6, 0.6, 1.7],  # width, depth, height
                'keypoints': np.array([
                    [0, 0, 0],      # center bottom
                    [0, 0, 0.85],   # center middle
                    [0, 0, 1.7]     # center top
                ])
            },
            'chair': {
                'dimensions': [0.5, 0.5, 0.8],  # width, depth, height
                'keypoints': np.array([
                    [-0.25, -0.25, 0], [0.25, -0.25, 0],  # base corners
                    [-0.25, 0.25, 0], [0.25, 0.25, 0],
                    [0, 0, 0.8]  # top center
                ])
            }
        }
        return models

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process camera image for 2D object detection."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_header = msg.header
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')

    def pointcloud_callback(self, msg):
        """Process point cloud data."""
        # In a real implementation, you'd convert PointCloud2 to numpy array
        # For simulation, we'll generate mock point cloud data
        self.latest_pointcloud = msg
        self.pointcloud_header = msg.header

        # Process when we have both image and point cloud data
        if self.latest_image is not None:
            self.estimate_poses()

    def estimate_poses(self):
        """Estimate 3D poses by combining 2D detections with point cloud data."""
        if self.camera_matrix is None:
            return

        # For simulation, create mock 2D detections
        detections_2d = self.create_mock_2d_detections()

        # Estimate 3D poses for each detection
        pose_estimates = Detection3DArray()
        pose_estimates.header = self.image_header

        for detection in detections_2d:
            # Estimate 3D pose using point cloud data
            pose_3d = self.estimate_single_pose(detection)
            if pose_3d is not None:
                pose_estimates.detections.append(pose_3d)

        # Publish results
        self.pose_estimates_pub.publish(pose_estimates)

        self.get_logger().info(f'Pose estimation completed: {len(pose_estimates.detections)} objects')

    def create_mock_2d_detections(self):
        """Create mock 2D detections for demonstration."""
        # In real implementation, this would come from a 2D detector
        detections = []

        for i in range(np.random.randint(1, 4)):
            detection = Detection2D()
            detection.bbox.center.x = float(np.random.uniform(100, 500))
            detection.bbox.center.y = float(np.random.uniform(100, 300))
            detection.bbox.size_x = float(np.random.uniform(50, 150))
            detection.bbox.size_y = float(np.random.uniform(50, 150))

            hypothesis = ObjectHypothesisWithPose()
            classes = ['car', 'person', 'chair']
            hypothesis.hypothesis.class_id = np.random.choice(classes)
            hypothesis.hypothesis.score = float(np.random.uniform(0.7, 0.95))

            detection.results.append(hypothesis)
            detections.append(detection)

        return detections

    def estimate_single_pose(self, detection_2d):
        """Estimate 3D pose for a single 2D detection."""
        # Get the object class
        obj_class = detection_2d.results[0].hypothesis.class_id

        if obj_class not in self.object_models:
            return None

        # Get object model
        model = self.object_models[obj_class]
        dimensions = model['dimensions']

        # Estimate depth using point cloud data in the bounding box region
        # For simulation, we'll generate a plausible depth
        estimated_depth = self.estimate_depth_in_bbox(detection_2d)

        if estimated_depth is None:
            return None

        # Calculate 3D position
        u = int(detection_2d.bbox.center.x)
        v = int(detection_2d.bbox.center.y)

        # Convert image coordinates to 3D world coordinates
        x = (u - self.camera_matrix[0, 2]) * estimated_depth / self.camera_matrix[0, 0]
        y = (v - self.camera_matrix[1, 2]) * estimated_depth / self.camera_matrix[1, 1]
        z = estimated_depth

        # Create 3D detection
        detection_3d = Detection3D()
        detection_3d.header = self.image_header

        # Set position
        detection_3d.bbox.center.position.x = x
        detection_3d.bbox.center.position.y = y
        detection_3d.bbox.center.position.z = z

        # Set size based on model dimensions
        detection_3d.bbox.size.x = dimensions[0]
        detection_3d.bbox.size.y = dimensions[1]
        detection_3d.bbox.size.z = dimensions[2]

        # Set orientation (for simulation, assume upright)
        detection_3d.bbox.center.orientation.w = 1.0
        detection_3d.bbox.center.orientation.x = 0.0
        detection_3d.bbox.center.orientation.y = 0.0
        detection_3d.bbox.center.orientation.z = 0.0

        # Copy results from 2D detection
        detection_3d.results = detection_2d.results

        return detection_3d

    def estimate_depth_in_bbox(self, detection_2d):
        """Estimate depth of object using point cloud data in bounding box."""
        # In a real implementation, you'd:
        # 1. Project bounding box to 3D space
        # 2. Find corresponding points in point cloud
        # 3. Calculate average depth of points in the region

        # For simulation, return a plausible depth based on object size
        # Larger objects are likely closer
        bbox_size = math.sqrt(detection_2d.bbox.size_x * detection_2d.bbox.size_y)

        # Simulate depth estimation: larger bounding boxes = closer objects
        # Range from 1m (large) to 20m (small)
        estimated_depth = max(1.0, min(20.0, 20.0 - (bbox_size / 50.0)))

        return estimated_depth

def main(args=None):
    rclpy.init(args=args)
    pose_node = ObjectPoseEstimation()

    try:
        rclpy.spin(pose_node)
    except KeyboardInterrupt:
        pose_node.get_logger().info('Shutting down Object Pose Estimation')
    finally:
        pose_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Validation and Testing

### Validating Perception System Performance

```python
# perception_validation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import numpy as np
import time
from collections import deque

class PerceptionValidation(Node):
    def __init__(self):
        super().__init__('perception_validation')

        # Subscribers for perception outputs
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Publishers for validation metrics
        self.accuracy_pub = self.create_publisher(Float32, 'perception_accuracy', 10)
        self.precision_pub = self.create_publisher(Float32, 'perception_precision', 10)
        self.recall_pub = self.create_publisher(Float32, 'perception_recall', 10)
        self.fps_pub = self.create_publisher(Float32, 'perception_fps', 10)
        self.quality_pub = self.create_publisher(Float32, 'perception_quality', 10)
        self.status_pub = self.create_publisher(Bool, 'perception_status', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Performance tracking
        self.detection_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)

        # Ground truth for validation (in simulation scenario)
        self.ground_truth = self.generate_ground_truth()

        # Validation metrics
        self.total_detections = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        # Performance thresholds
        self.min_fps_threshold = 10.0
        self.min_accuracy_threshold = 0.85

        self.get_logger().info('Perception Validation initialized')

    def generate_ground_truth(self):
        """Generate ground truth data for validation (simulation scenario)."""
        # In a real validation system, this would come from:
        # - Manual annotations
        # - Simulation ground truth
        # - High-accuracy reference system
        # For simulation, we'll create mock ground truth
        ground_truth = {
            'timestamp': time.time(),
            'objects': [
                {'class': 'person', 'bbox': [100, 100, 200, 250], 'confidence': 0.95},
                {'class': 'car', 'bbox': [300, 150, 450, 300], 'confidence': 0.92},
                {'class': 'bicycle', 'bbox': [50, 200, 150, 300], 'confidence': 0.88}
            ]
        }
        return ground_truth

    def image_callback(self, msg):
        """Track image input for FPS calculation."""
        current_time = time.time()

        # Add to processing time tracking
        if hasattr(self, 'last_image_time'):
            processing_time = current_time - self.last_image_time
            self.processing_times.append(processing_time)

        self.last_image_time = current_time

    def scan_callback(self, msg):
        """Process LiDAR data for validation."""
        # In real validation, LiDAR data can provide additional ground truth
        pass

    def detections_callback(self, msg):
        """Validate incoming detections."""
        current_time = time.time()
        self.detection_times.append(current_time)
        self.detection_counts.append(len(msg.detections))

        # Validate detections against ground truth
        validation_results = self.validate_detections(msg)

        # Update metrics
        self.update_metrics(validation_results)

        # Calculate and publish validation metrics
        self.publish_validation_metrics()

        # Log validation results periodically
        if len(self.detection_times) % 10 == 0:
            accuracy = self.calculate_accuracy()
            fps = self.calculate_fps()

            status_msg = Bool()
            status_msg.data = (accuracy >= self.min_accuracy_threshold and
                              fps >= self.min_fps_threshold)
            self.status_pub.publish(status_msg)

            self.get_logger().info(
                f'Validation - Accuracy: {accuracy:.2f}, FPS: {fps:.1f}, '
                f'True Positives: {self.true_positives}, '
                f'False Positives: {self.false_positives}'
            )

    def validate_detections(self, detections_msg):
        """Validate detections against ground truth."""
        results = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'matches': []
        }

        # For simulation, we'll use IoU (Intersection over Union) to match detections
        ground_truth = self.ground_truth['objects']
        detections = []

        # Convert ROS detections to simple format
        for detection in detections_msg.detections:
            x1 = detection.bbox.center.x - detection.bbox.size_x / 2
            y1 = detection.bbox.center.y - detection.bbox.size_y / 2
            x2 = detection.bbox.center.x + detection.bbox.size_x / 2
            y2 = detection.bbox.center.y + detection.bbox.size_y / 2

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                'confidence': detection.results[0].hypothesis.score if detection.results else 0.0
            })

        # Match detections to ground truth using IoU
        matched_gt = set()
        matched_det = set()

        for i, det in enumerate(detections):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue

                iou = self.calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou and iou > 0.5:  # 0.5 IoU threshold
                    best_iou = iou
                    best_gt_idx = j

            if best_gt_idx != -1:
                # True positive - detection matches ground truth
                results['true_positives'] += 1
                matched_gt.add(best_gt_idx)
                matched_det.add(i)
                results['matches'].append((i, best_gt_idx))
            else:
                # False positive - detection doesn't match any ground truth
                results['false_positives'] += 1

        # False negatives - ground truth objects not detected
        results['false_negatives'] = len(ground_truth) - len(matched_gt)

        return results

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

    def update_metrics(self, validation_results):
        """Update validation metrics."""
        self.true_positives += validation_results['true_positives']
        self.false_positives += validation_results['false_positives']
        self.false_negatives += validation_results['false_negatives']
        self.total_detections += (validation_results['true_positives'] +
                                 validation_results['false_positives'])

    def calculate_accuracy(self):
        """Calculate overall accuracy."""
        if self.total_detections == 0:
            return 0.0
        return self.true_positives / self.total_detections

    def calculate_precision(self):
        """Calculate precision."""
        if (self.true_positives + self.false_positives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def calculate_recall(self):
        """Calculate recall."""
        if (self.true_positives + self.false_negatives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def calculate_fps(self):
        """Calculate frames per second."""
        if len(self.detection_times) < 2:
            return 0.0

        time_span = self.detection_times[-1] - self.detection_times[0]
        if time_span == 0:
            return 0.0

        return len(self.detection_times) / time_span

    def calculate_quality_score(self):
        """Calculate overall quality score."""
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        fps = self.calculate_fps()

        # Combine metrics into a single quality score
        # Weight precision and recall equally, adjust FPS contribution
        quality = (0.4 * precision + 0.4 * recall + 0.2 * min(fps / 30.0, 1.0))
        return min(quality, 1.0)  # Cap at 1.0

    def publish_validation_metrics(self):
        """Publish validation metrics."""
        # Accuracy
        accuracy_msg = Float32()
        accuracy_msg.data = float(self.calculate_accuracy())
        self.accuracy_pub.publish(accuracy_msg)

        # Precision
        precision_msg = Float32()
        precision_msg.data = float(self.calculate_precision())
        self.precision_pub.publish(precision_msg)

        # Recall
        recall_msg = Float32()
        recall_msg.data = float(self.calculate_recall())
        self.recall_pub.publish(recall_msg)

        # FPS
        fps_msg = Float32()
        fps_msg.data = float(self.calculate_fps())
        self.fps_pub.publish(fps_msg)

        # Quality score
        quality_msg = Float32()
        quality_msg.data = float(self.calculate_quality_score())
        self.quality_pub.publish(quality_msg)

def main(args=None):
    rclpy.init(args=args)
    validation_node = PerceptionValidation()

    try:
        rclpy.spin(validation_node)
    except KeyboardInterrupt:
        validation_node.get_logger().info('Shutting down Perception Validation')
    finally:
        validation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive perception pipelines in robotics:

- **GPU Acceleration**: Leveraging CUDA and TensorRT for real-time performance
- **Deep Learning Models**: Optimizing neural networks for robotic perception
- **Multi-Sensor Fusion**: Combining camera, LiDAR, and other sensors
- **Semantic Segmentation**: Pixel-level scene understanding
- **3D Pose Estimation**: Detecting and localizing objects in 3D space
- **Performance Validation**: Measuring and validating perception system quality

These perception capabilities are essential for autonomous robots to understand and navigate their environment effectively.

## Exercises

1. Implement a GPU-accelerated object detection pipeline
2. Create a multi-sensor fusion system combining camera and LiDAR
3. Develop a semantic segmentation system for scene understanding
4. Implement 3D object pose estimation from 2D detections
5. Validate your perception system with appropriate metrics

## Quiz

1. What is the primary advantage of using TensorRT for perception models?
   a) Simpler code
   b) GPU acceleration and optimized inference
   c) Better accuracy
   d) Lower memory usage

2. What does semantic segmentation provide that object detection does not?
   a) Bounding boxes
   b) Pixel-level classification
   c) 3D positions
   d) Velocities

3. Why is multi-sensor fusion important in robotics?
   a) Reduces computational requirements
   b) Provides more robust and accurate perception
   c) Simplifies sensor calibration
   d) Reduces sensor cost

## Mini-Project: Complete Perception System

Create a complete perception system with:
1. GPU-accelerated object detection pipeline
2. Multi-sensor fusion (camera + LiDAR)
3. Semantic segmentation capabilities
4. 3D pose estimation for detected objects
5. Performance validation and metrics reporting
6. Integration with navigation and planning systems