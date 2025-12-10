---
sidebar_position: 5
---

# Sensor Integration and Robotics

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate various types of sensors into robot models for simulation
- Implement realistic sensor simulation in both Gazebo and Unity
- Process and interpret sensor data for robotic applications
- Handle sensor noise and uncertainty in robotic systems
- Validate sensor performance in simulation environments

## Introduction to Robotic Sensors

Sensors are the eyes and ears of robotic systems, providing crucial information about the robot's environment and internal state. In simulation, accurate sensor modeling is essential for developing and testing perception algorithms that will eventually run on real robots.

### Sensor Classification

Robotic sensors can be broadly classified into:

```
Robotic Sensor Classification:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Proprioceptive│    │   Exteroceptive │    │   Interoceptive │
│   (Internal)    │    │   (External)    │    │   (Internal)    │
│   • IMU         │    │   • Cameras     │    │   • Temperature │
│   • Encoders    │    │   • LiDAR       │    │   • Current     │
│   • Joint       │    │   • Sonar       │    │   • Voltage     │
│   • Force/Torque│    │   • GPS         │    │   • Power       │
└─────────────────┘    │   • Radar       │    └─────────────────┘
                       └─────────────────┘
```

### Sensor Data Processing Pipeline

```
Sensor Data Pipeline:
Raw Sensor Data → Filtering → Calibration → Fusion → Perception → Action
```

## Camera Sensors

### Camera Sensor Fundamentals

Camera sensors provide visual information about the environment, essential for tasks like object recognition, navigation, and mapping.

```xml
<!-- Gazebo camera sensor configuration -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <topic_name>camera/image_raw</topic_name>
      <camera_info_topic_name>camera/camera_info</camera_info_topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Calibration and Parameters

Camera parameters are crucial for accurate perception:

```python
# camera_calibration.py
import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

        # Default camera matrix (will be calibrated)
        self.camera_matrix = np.array([
            [500, 0, image_width/2],   # fx, 0, cx
            [0, 500, image_height/2],  # 0, fy, cy
            [0, 0, 1]                 # 0, 0, 1
        ])

        # Distortion coefficients [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.zeros((5, 1))

    def calibrate_from_chessboard(self, images):
        """Calibrate camera using chessboard images."""
        # Prepare object points (3D points in real world space)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Calibrate camera
        if len(objpoints) > 0:
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            return ret
        return False

    def undistort_image(self, img):
        """Remove distortion from image using calibration parameters."""
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort
        dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx)

        # Crop the image
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]

# Example usage
calibrator = CameraCalibrator(640, 480)
```

### Camera Data Processing

```python
# camera_processing.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )

        # Create subscriber for camera info
        self.info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.info_callback, 10
        )

        # Create publisher for processed images
        self.processed_pub = self.create_publisher(Image, 'camera/processed', 10)

        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Camera parameters (to be filled from camera info)
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Processing parameters
        self.enable_edge_detection = True
        self.enable_object_detection = False

    def info_callback(self, msg):
        """Process camera info to extract intrinsic parameters."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Apply processing based on enabled features
        processed_image = cv_image.copy()

        if self.enable_edge_detection:
            processed_image = self.apply_edge_detection(processed_image)

        if self.enable_object_detection:
            processed_image = self.apply_object_detection(processed_image)

        # Convert back to ROS Image message
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f'Could not convert processed image: {e}')

    def apply_edge_detection(self, image):
        """Apply Canny edge detection to the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Convert back to 3-channel image for visualization
        edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edge_image

    def apply_object_detection(self, image):
        """Apply simple color-based object detection."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color (in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours of detected objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected objects
        result = image.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return result

def main(args=None):
    rclpy.init(args=args)
    processor = CameraProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## LiDAR Sensors

### LiDAR Fundamentals

LiDAR (Light Detection and Ranging) sensors provide accurate distance measurements by timing the return of laser pulses. They are essential for navigation, mapping, and obstacle detection.

```xml
<!-- Gazebo LiDAR sensor configuration -->
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -π radians -->
          <max_angle>3.14159</max_angle>   <!-- π radians -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Data Processing

```python
# lidar_processing.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
import numpy as np
import math

class LiDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Create subscriber for LiDAR data
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Create publisher for obstacle detection
        self.obstacle_pub = self.create_publisher(PointStamped, 'obstacle_point', 10)

        # Processing parameters
        self.safety_distance = 1.0  # meters
        self.min_obstacle_size = 0.2  # meters
        self.obstacle_threshold = 10  # number of consecutive points for obstacle detection

        # For obstacle clustering
        self.obstacle_clusters = []
        self.last_scan = None

    def scan_callback(self, msg):
        """Process incoming LiDAR scan data."""
        # Store the scan for processing
        self.last_scan = msg

        # Process scan data
        obstacles = self.detect_obstacles(msg)

        # Publish obstacle information
        if obstacles:
            for obstacle in obstacles:
                self.publish_obstacle(obstacle, msg.header)

        # Perform safety checks
        self.check_safety(msg)

    def detect_obstacles(self, scan_msg):
        """Detect obstacles from LiDAR scan data."""
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(
            scan_msg.angle_min,
            scan_msg.angle_max,
            len(ranges)
        )

        # Filter out invalid ranges
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # Convert to Cartesian coordinates
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)

        # Simple clustering to identify obstacles
        obstacles = []
        current_cluster = []

        for i in range(len(x_coords)):
            if i > 0:
                # Calculate distance to previous point
                dist_to_prev = math.sqrt(
                    (x_coords[i] - x_coords[i-1])**2 +
                    (y_coords[i] - y_coords[i-1])**2
                )

                if dist_to_prev < self.min_obstacle_size:
                    # Continue current cluster
                    current_cluster.append((x_coords[i], y_coords[i]))
                else:
                    # End current cluster if it's large enough
                    if len(current_cluster) >= self.obstacle_threshold:
                        obstacles.append(self.calculate_cluster_center(current_cluster))
                    current_cluster = [(x_coords[i], y_coords[i])]
            else:
                current_cluster = [(x_coords[i], y_coords[i])]

        # Don't forget the last cluster
        if len(current_cluster) >= self.obstacle_threshold:
            obstacles.append(self.calculate_cluster_center(current_cluster))

        return obstacles

    def calculate_cluster_center(self, cluster):
        """Calculate the center of an obstacle cluster."""
        x_coords = [point[0] for point in cluster]
        y_coords = [point[1] for point in cluster]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    def check_safety(self, scan_msg):
        """Check if there are obstacles within safety distance."""
        ranges = np.array(scan_msg.ranges)
        min_range = np.min(ranges[(ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)])

        if min_range < self.safety_distance:
            self.get_logger().warn(f'Obstacle detected at {min_range:.2f}m! Safety distance is {self.safety_distance}m')
        else:
            self.get_logger().info(f'Clear path: closest obstacle at {min_range:.2f}m')

    def publish_obstacle(self, obstacle, header):
        """Publish obstacle information as a point."""
        point_msg = PointStamped()
        point_msg.header = header
        point_msg.point.x = obstacle[0]
        point_msg.point.y = obstacle[1]
        point_msg.point.z = 0.0  # Assuming 2D navigation

        self.obstacle_pub.publish(point_msg)

def main(args=None):
    rclpy.init(args=args)
    processor = LiDARProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## IMU Sensors

### IMU Fundamentals

Inertial Measurement Units (IMUs) provide information about orientation, angular velocity, and linear acceleration. They are crucial for robot localization and stabilization.

```xml
<!-- Gazebo IMU sensor configuration -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Data Processing and Sensor Fusion

```python
# imu_processing.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Create subscriber for IMU data
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )

        # Create publisher for processed orientation
        self.orientation_pub = self.create_publisher(Vector3Stamped, 'imu/euler', 10)

        # Initialize orientation (using quaternion)
        self.orientation_q = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
        self.angular_velocity = [0.0, 0.0, 0.0]    # x, y, z
        self.linear_acceleration = [0.0, 0.0, 0.0] # x, y, z

        # For integration
        self.last_time = None

        # Covariance thresholds for data quality
        self.orientation_cov_threshold = 0.1
        self.angular_velocity_cov_threshold = 0.1

    def imu_callback(self, msg):
        """Process incoming IMU data."""
        # Update internal state
        self.orientation_q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

        self.angular_velocity = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]

        self.linear_acceleration = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]

        # Check data quality based on covariance
        orientation_ok = self.check_covariance_reliability(
            msg.orientation_covariance, self.orientation_cov_threshold
        )

        angular_velocity_ok = self.check_covariance_reliability(
            msg.angular_velocity_covariance, self.angular_velocity_cov_threshold
        )

        if orientation_ok:
            # Publish Euler angles
            euler = self.quaternion_to_euler(*self.orientation_q)
            self.publish_euler_angles(euler, msg.header)

        if angular_velocity_ok:
            # Could integrate to get orientation if needed
            self.integrate_angular_velocity(msg)

    def check_covariance_reliability(self, covariance, threshold):
        """Check if covariance values are below threshold (reliable data)."""
        # Check diagonal elements (variances) of covariance matrix
        for i in range(0, len(covariance), 4):  # Diagonal elements: 0, 4, 8
            if i < len(covariance) and covariance[i] > threshold:
                return False
        return True

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
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

        return x, y, z, w

    def integrate_angular_velocity(self, msg):
        """Integrate angular velocity to estimate orientation."""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_time is not None:
            dt = current_time - self.last_time

            # Integrate angular velocity to update orientation
            # This is a simple integration - in practice, you'd want to use
            # more sophisticated methods like quaternion integration
            delta_angle = [
                self.angular_velocity[0] * dt,
                self.angular_velocity[1] * dt,
                self.angular_velocity[2] * dt
            ]

            # Convert small angle to quaternion
            angle_magnitude = math.sqrt(sum(a*a for a in delta_angle))
            if angle_magnitude > 0:
                axis = [a / angle_magnitude for a in delta_angle]

                # Create rotation quaternion
                half_angle = angle_magnitude / 2.0
                sin_half = math.sin(half_angle)

                dq = [
                    axis[0] * sin_half,  # x
                    axis[1] * sin_half,  # y
                    axis[2] * sin_half,  # z
                    math.cos(half_angle) # w
                ]

                # Multiply with current orientation
                self.orientation_q = self.quaternion_multiply(
                    self.orientation_q, dq
                )

        self.last_time = current_time

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return [x, y, z, w]

    def publish_euler_angles(self, euler, header):
        """Publish Euler angles as a Vector3Stamped message."""
        roll, pitch, yaw = euler

        euler_msg = Vector3Stamped()
        euler_msg.header = header
        euler_msg.vector.x = roll
        euler_msg.vector.y = pitch
        euler_msg.vector.z = yaw

        self.orientation_pub.publish(euler_msg)

def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Unity Sensor Simulation

### Unity Camera Sensor Implementation

```csharp
// UnityCameraSensor.cs
using UnityEngine;
using Unity.Robotics.Sensors;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using System.Collections;
using System.Threading.Tasks;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fieldOfView = 60.0f;
    public float publishRate = 10.0f; // Hz

    [Header("ROS Settings")]
    public string imageTopic = "/camera/image_raw";
    public string cameraInfoTopic = "/camera/camera_info";

    private Camera unityCamera;
    private RenderTexture renderTexture;
    private Texture2D tempTexture;
    private float publishTimer;

    // ROS communication
    private ROSConnection ros;
    private string rosEndpoint = "127.0.0.1";
    private int rosPort = 10000;

    void Start()
    {
        SetupCamera();
        SetupROSConnection();
    }

    void SetupCamera()
    {
        unityCamera = GetComponent<Camera>();
        if (unityCamera == null)
        {
            unityCamera = gameObject.AddComponent<Camera>();
        }

        unityCamera.fieldOfView = fieldOfView;
        unityCamera.allowMSAA = false;
        unityCamera.backgroundColor = Color.black;
        unityCamera.clearFlags = CameraClearFlags.SolidColor;

        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        unityCamera.targetTexture = renderTexture;

        // Create temporary texture for reading
        tempTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    void SetupROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosEndpoint, rosPort);
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        if (publishTimer >= 1.0f / publishRate)
        {
            PublishCameraData();
            publishTimer = 0.0f;
        }
    }

    async void PublishCameraData()
    {
        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        tempTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tempTexture.Apply();

        // Convert to byte array
        byte[] imageBytes = tempTexture.EncodeToJPG();

        // Create ROS message
        var imageMsg = new sensor_msgs.msg.Image();
        imageMsg.header = new std_msgs.msg.Header();
        imageMsg.header.stamp = new builtin_interfaces.msg.Time();
        imageMsg.header.frame_id = "camera_optical_frame";
        imageMsg.height = (uint)imageHeight;
        imageMsg.width = (uint)imageWidth;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(imageWidth * 3); // 3 bytes per pixel
        imageMsg.data = imageBytes;

        // Publish image
        ros.Publish(imageTopic, imageMsg);

        // Publish camera info
        PublishCameraInfo();
    }

    void PublishCameraInfo()
    {
        var cameraInfoMsg = new sensor_msgs.msg.CameraInfo();
        cameraInfoMsg.header = new std_msgs.msg.Header();
        cameraInfoMsg.header.stamp = new builtin_interfaces.msg.Time();
        cameraInfoMsg.header.frame_id = "camera_optical_frame";
        cameraInfoMsg.height = (uint)imageHeight;
        cameraInfoMsg.width = (uint)imageWidth;

        // Camera matrix (intrinsic parameters)
        // fx, 0, cx, 0, fy, cy, 0, 0, 1
        float fx = (imageWidth / 2.0f) / Mathf.Tan(Mathf.Deg2Rad * fieldOfView / 2.0f);
        float fy = fx; // Assuming square pixels
        float cx = imageWidth / 2.0f;
        float cy = imageHeight / 2.0f;

        cameraInfoMsg.k = new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 };

        // No distortion (ideal camera)
        cameraInfoMsg.d = new double[] { 0, 0, 0, 0, 0 };

        // R (rectification matrix) - identity for monocular camera
        cameraInfoMsg.r = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

        // P (projection matrix)
        cameraInfoMsg.p = new double[] { fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0 };

        ros.Publish(cameraInfoTopic, cameraInfoMsg);
    }

    void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
        }
        if (tempTexture != null)
        {
            Destroy(tempTexture);
        }
    }
}
```

### Unity LiDAR Sensor Implementation

```csharp
// UnityLiDARSensor.cs
using UnityEngine;
using System.Collections.Generic;

public class UnityLiDARSensor : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public int numberOfRays = 360;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float maxDistance = 30.0f;
    public float updateRate = 10.0f; // Hz
    public LayerMask detectionLayers = -1; // All layers

    [Header("Noise Settings")]
    public float distanceNoiseStdDev = 0.01f;
    public float angleNoiseStdDev = 0.001f;

    [Header("ROS Settings")]
    public string scanTopic = "/scan";

    private float[] ranges;
    private float[] intensities;
    private float publishTimer;
    private ROSConnection ros;

    void Start()
    {
        ranges = new float[numberOfRays];
        intensities = new float[numberOfRays];

        SetupROSConnection();
    }

    void SetupROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize("127.0.0.1", 10000);
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        if (publishTimer >= 1.0f / updateRate)
        {
            SimulateLiDARScan();
            PublishScanData();
            publishTimer = 0.0f;
        }
    }

    void SimulateLiDARScan()
    {
        for (int i = 0; i < numberOfRays; i++)
        {
            // Calculate angle with noise
            float angle = Mathf.Lerp(minAngle, maxAngle, (float)i / (numberOfRays - 1));
            angle += RandomGaussian() * angleNoiseStdDev;

            // Create ray direction
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxDistance, detectionLayers))
            {
                float distance = hit.distance;

                // Add noise to distance
                distance += RandomGaussian() * distanceNoiseStdDev;
                distance = Mathf.Clamp(distance, 0, maxDistance);

                ranges[i] = distance;

                // Set intensity based on surface properties
                intensities[i] = CalculateIntensity(hit);
            }
            else
            {
                ranges[i] = maxDistance; // No obstacle detected
                intensities[i] = 0;
            }
        }
    }

    float CalculateIntensity(RaycastHit hit)
    {
        // Simple intensity calculation based on surface properties
        // In real LiDAR, this would depend on material reflectance
        float baseIntensity = 1000.0f;

        // Reduce intensity based on distance
        float distanceFactor = Mathf.Clamp01(1.0f - hit.distance / maxDistance);
        return baseIntensity * distanceFactor;
    }

    void PublishScanData()
    {
        var scanMsg = new sensor_msgs.msg.LaserScan();
        scanMsg.header = new std_msgs.msg.Header();
        scanMsg.header.stamp = new builtin_interfaces.msg.Time();
        scanMsg.header.frame_id = "laser_frame";
        scanMsg.angle_min = minAngle;
        scanMsg.angle_max = maxAngle;
        scanMsg.angle_increment = (maxAngle - minAngle) / (numberOfRays - 1);
        scanMsg.time_increment = 0.0f; // Set if you have rotating LiDAR
        scanMsg.scan_time = 1.0f / updateRate;
        scanMsg.range_min = 0.1f;
        scanMsg.range_max = maxDistance;
        scanMsg.ranges = new float[numberOfRays];

        // Copy ranges data
        for (int i = 0; i < numberOfRays; i++)
        {
            scanMsg.ranges[i] = ranges[i];
        }

        // Intensities are optional
        if (intensities != null && intensities.Length > 0)
        {
            scanMsg.intensities = new float[numberOfRays];
            for (int i = 0; i < numberOfRays; i++)
            {
                scanMsg.intensities[i] = intensities[i];
            }
        }

        ros.Publish(scanTopic, scanMsg);
    }

    // Generate Gaussian noise using Box-Muller transform
    float RandomGaussian()
    {
        float u1 = Random.value; // Uniform(0,1] random doubles
        float u2 = Random.value;
        if (u1 == 0) u1 = 0.0000001f; // Converting to (0,1]
        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }
}
```

## Sensor Fusion and Data Processing

### Multi-Sensor Data Integration

```python
# sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math
from collections import deque

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensor types
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # Publishers for fused data
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'fused_pose', 10)
        self.twist_pub = self.create_publisher(TwistWithCovarianceStamped, 'fused_twist', 10)

        # Storage for sensor data
        self.scan_data = None
        self.imu_data = None
        self.odom_data = None

        # For temporal alignment
        self.scan_buffer = deque(maxlen=10)
        self.imu_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)

        # For sensor fusion (simple Kalman filter implementation)
        self.state = np.array([0, 0, 0, 0, 0, 0])  # [x, y, theta, vx, vy, omega]
        self.covariance = np.eye(6) * 100  # Initial uncertainty

        # Process noise
        self.process_noise = np.diag([0.1, 0.1, 0.01, 0.1, 0.1, 0.01])

        # Timer for fusion update
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)  # 10Hz

    def scan_callback(self, msg):
        """Process LiDAR scan data."""
        self.scan_data = msg
        self.scan_buffer.append((msg.header.stamp, msg))

        # Extract useful information from scan
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[(ranges >= msg.range_min) & (ranges <= msg.range_max)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().debug(f'Closest obstacle: {min_distance:.2f}m')

    def imu_callback(self, msg):
        """Process IMU data."""
        self.imu_data = msg
        self.imu_buffer.append((msg.header.stamp, msg))

    def odom_callback(self, msg):
        """Process odometry data."""
        self.odom_data = msg
        self.odom_buffer.append((msg.header.stamp, msg))

    def fusion_callback(self):
        """Perform sensor fusion and publish results."""
        if self.odom_data is None or self.imu_data is None:
            return  # Need both sensors for fusion

        # Get current data
        odom = self.odom_data
        imu = self.imu_data

        # Prediction step (motion model)
        dt = 0.1  # 10Hz update rate
        self.predict(dt)

        # Update step (sensor measurements)
        self.update_from_odom(odom)
        self.update_from_imu(imu)

        # Publish fused state
        self.publish_fused_state()

    def predict(self, dt):
        """Prediction step of the filter."""
        # Simple motion model: constant velocity
        x, y, theta, vx, vy, omega = self.state

        # Update positions based on velocities
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_theta = theta + omega * dt

        # Linearized motion model Jacobian
        F = np.eye(6)
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        F[2, 5] = dt  # dtheta/domega

        # Update state and covariance
        self.state[0] = new_x
        self.state[1] = new_y
        self.state[2] = new_theta

        # Propagate covariance
        self.covariance = F @ self.covariance @ F.T + self.process_noise * dt

    def update_from_odom(self, odom_msg):
        """Update state using odometry measurements."""
        # Measurement model: directly observe position and linear velocity
        z = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.position.y,
            self.quaternion_to_yaw(odom_msg.pose.pose.orientation),
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y
        ])

        # Measurement matrix (which state variables are observed)
        H = np.zeros((5, 6))
        H[0, 0] = 1  # x position
        H[1, 1] = 1  # y position
        H[2, 2] = 1  # orientation
        H[3, 3] = 1  # x velocity
        H[4, 4] = 1  # y velocity

        # Measurement noise
        R = np.diag([0.1, 0.1, 0.05, 0.1, 0.1])  # [x, y, theta, vx, vy]

        # Kalman gain calculation
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Innovation (difference between measurement and prediction)
        h_x = np.array([
            self.state[0],  # predicted x
            self.state[1],  # predicted y
            self.state[2],  # predicted theta
            self.state[3],  # predicted vx
            self.state[4]   # predicted vy
        ])

        y = z - h_x  # innovation

        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance

    def update_from_imu(self, imu_msg):
        """Update state using IMU measurements."""
        # For simplicity, just update orientation and angular velocity
        measured_theta = self.quaternion_to_yaw(imu_msg.orientation)
        measured_omega = imu_msg.angular_velocity.z

        # Update orientation
        self.state[2] = measured_theta

        # Update angular velocity
        self.state[5] = measured_omega

        # Reduce uncertainty in these measurements
        self.covariance[2, 2] = 0.01  # Low uncertainty in orientation
        self.covariance[5, 5] = 0.05  # Low uncertainty in angular velocity

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_fused_state(self):
        """Publish the fused state estimate."""
        # Publish pose with covariance
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose.position.x = float(self.state[0])
        pose_msg.pose.pose.position.y = float(self.state[1])
        pose_msg.pose.pose.position.z = 0.0

        # Convert orientation back to quaternion
        yaw = float(self.state[2])
        qx = 0
        qy = 0
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)

        pose_msg.pose.pose.orientation.x = qx
        pose_msg.pose.pose.orientation.y = qy
        pose_msg.pose.pose.orientation.z = qz
        pose_msg.pose.pose.orientation.w = qw

        # Copy covariance matrix (flatten 6x6 to 36-element array)
        pose_msg.pose.covariance = np.zeros(36)
        for i in range(6):
            for j in range(6):
                pose_msg.pose.covariance[i*6 + j] = float(self.covariance[i, j])

        self.pose_pub.publish(pose_msg)

    def get_closest_obstacle_direction(self):
        """Get direction to closest obstacle from scan data."""
        if self.scan_data is None:
            return None, float('inf')

        ranges = np.array(self.scan_data.ranges)
        valid_indices = np.where((ranges >= self.scan_data.range_min) &
                                (ranges <= self.scan_data.range_max))[0]

        if len(valid_indices) == 0:
            return None, float('inf')

        closest_idx = valid_indices[np.argmin(ranges[valid_indices])]
        angle_to_closest = (self.scan_data.angle_min +
                           closest_idx * self.scan_data.angle_increment)

        return angle_to_closest, ranges[closest_idx]

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Validation and Testing

### Sensor Testing Framework

```python
# sensor_validation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from std_msgs.msg import Float32
import numpy as np
import time

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Subscribers for different sensors
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Publishers for validation results
        self.scan_quality_pub = self.create_publisher(Float32, 'sensor_validation/scan_quality', 10)
        self.imu_quality_pub = self.create_publisher(Float32, 'sensor_validation/imu_quality', 10)
        self.image_quality_pub = self.create_publisher(Float32, 'sensor_validation/image_quality', 10)

        # Validation statistics
        self.scan_stats = {'count': 0, 'avg_rate': 0, 'last_time': None}
        self.imu_stats = {'count': 0, 'avg_rate': 0, 'last_time': None}
        self.image_stats = {'count': 0, 'avg_rate': 0, 'last_time': None}

        # Validation timers
        self.validation_timer = self.create_timer(5.0, self.validate_sensors)

    def scan_callback(self, msg):
        """Process LiDAR scan and update statistics."""
        current_time = time.time()

        if self.scan_stats['last_time'] is not None:
            dt = current_time - self.scan_stats['last_time']
            if dt > 0:
                current_rate = 1.0 / dt
                self.scan_stats['avg_rate'] = 0.9 * self.scan_stats['avg_rate'] + 0.1 * current_rate

        self.scan_stats['last_time'] = current_time
        self.scan_stats['count'] += 1

    def imu_callback(self, msg):
        """Process IMU data and update statistics."""
        current_time = time.time()

        if self.imu_stats['last_time'] is not None:
            dt = current_time - self.imu_stats['last_time']
            if dt > 0:
                current_rate = 1.0 / dt
                self.imu_stats['avg_rate'] = 0.9 * self.imu_stats['avg_rate'] + 0.1 * current_rate

        self.imu_stats['last_time'] = current_time
        self.imu_stats['count'] += 1

    def image_callback(self, msg):
        """Process camera image and update statistics."""
        current_time = time.time()

        if self.image_stats['last_time'] is not None:
            dt = current_time - self.image_stats['last_time']
            if dt > 0:
                current_rate = 1.0 / dt
                self.image_stats['avg_rate'] = 0.9 * self.image_stats['avg_rate'] + 0.1 * current_rate

        self.image_stats['last_time'] = current_time
        self.image_stats['count'] += 1

    def validate_sensors(self):
        """Validate sensor performance and publish quality metrics."""
        # Validate LiDAR
        scan_quality = self.validate_scan_sensor()
        scan_quality_msg = Float32()
        scan_quality_msg.data = scan_quality
        self.scan_quality_pub.publish(scan_quality_msg)

        # Validate IMU
        imu_quality = self.validate_imu_sensor()
        imu_quality_msg = Float32()
        imu_quality_msg.data = imu_quality
        self.imu_quality_pub.publish(imu_quality_msg)

        # Validate Camera
        image_quality = self.validate_image_sensor()
        image_quality_msg = Float32()
        image_quality_msg.data = image_quality
        self.image_quality_pub.publish(image_quality_msg)

    def validate_scan_sensor(self):
        """Validate LiDAR sensor performance."""
        if self.scan_stats['count'] == 0:
            return 0.0  # No data received

        # Check if update rate is within expected range (e.g., 5-20 Hz)
        expected_rate = 10.0  # Hz
        rate_tolerance = 2.0  # Hz

        rate_ok = abs(self.scan_stats['avg_rate'] - expected_rate) <= rate_tolerance

        # Check if we have reasonable range data
        if self.scan_data is not None:
            ranges = np.array(self.scan_data.ranges)
            valid_ranges = ranges[(ranges >= self.scan_data.range_min) &
                                 (ranges <= self.scan_data.range_max)]
            range_data_ok = len(valid_ranges) > len(ranges) * 0.5  # At least 50% valid
        else:
            range_data_ok = False

        # Combine metrics
        quality = 0.0
        if rate_ok:
            quality += 0.4  # 40% for correct rate
        if range_data_ok:
            quality += 0.6  # 60% for valid range data

        return quality

    def validate_imu_sensor(self):
        """Validate IMU sensor performance."""
        if self.imu_stats['count'] == 0:
            return 0.0

        # Check update rate
        expected_rate = 100.0  # IMU typically runs at 100Hz
        rate_tolerance = 20.0  # Allow some tolerance

        rate_ok = abs(self.imu_stats['avg_rate'] - expected_rate) <= rate_tolerance

        # Check if orientation is reasonable (unit quaternion)
        orientation_ok = False
        if self.imu_data is not None:
            q = self.imu_data.orientation
            norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
            orientation_ok = abs(norm - 1.0) < 0.1

        # Combine metrics
        quality = 0.0
        if rate_ok:
            quality += 0.5
        if orientation_ok:
            quality += 0.5

        return quality

    def validate_image_sensor(self):
        """Validate camera sensor performance."""
        if self.image_stats['count'] == 0:
            return 0.0

        # Check update rate
        expected_rate = 10.0  # Camera typically runs at 10-30Hz
        rate_tolerance = 5.0

        rate_ok = abs(self.image_stats['avg_rate'] - expected_rate) <= rate_tolerance

        # Check image data validity (if we have access to image data)
        image_ok = False
        if self.image_data is not None:
            # Check if image has reasonable size and data
            image_ok = (self.image_data.width > 0 and
                       self.image_data.height > 0 and
                       len(self.image_data.data) > 0)

        # Combine metrics
        quality = 0.0
        if rate_ok:
            quality += 0.5
        if image_ok:
            quality += 0.5

        return quality

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive sensor integration for robotics:

- **Camera sensors**: Configuration, calibration, and image processing
- **LiDAR sensors**: Configuration, data processing, and obstacle detection
- **IMU sensors**: Configuration, orientation estimation, and sensor fusion
- **Unity sensor simulation**: Implementation of sensors in Unity environment
- **Sensor fusion**: Combining multiple sensor inputs for better state estimation
- **Validation**: Testing and validating sensor performance

Proper sensor integration is crucial for robotic systems, providing the necessary information for navigation, mapping, perception, and control tasks.

## Exercises

1. Implement a sensor fusion node that combines LiDAR and IMU data
2. Create a camera-based object detection system in simulation
3. Validate sensor performance using the testing framework
4. Implement noise models for different sensor types

## Quiz

1. What is the main purpose of sensor fusion in robotics?
   a) To reduce the number of sensors needed
   b) To combine multiple sensor inputs for better state estimation
   c) To increase sensor update rates
   d) To reduce sensor costs

2. Which sensor would be most appropriate for precise distance measurements in a structured environment?
   a) Camera
   b) IMU
   c) LiDAR
   d) GPS

3. What does the covariance matrix represent in sensor data?
   a) The sensor's physical size
   b) The uncertainty or reliability of sensor measurements
   c) The sensor's update rate
   d) The sensor's power consumption

## Mini-Project: Multi-Sensor Robot

Create a robot with multiple sensors (camera, LiDAR, IMU) that:
1. Integrates all sensor data using a fusion algorithm
2. Performs basic navigation using sensor information
3. Validates sensor performance in real-time
4. Demonstrates the benefits of multi-sensor integration
5. Includes error handling for sensor failures