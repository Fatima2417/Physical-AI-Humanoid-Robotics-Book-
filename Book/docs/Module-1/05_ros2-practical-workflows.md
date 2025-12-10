---
sidebar_position: 5
---

# ROS 2 Practical Workflows

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement complete ROS 2 development workflows for robotics projects
- Debug and profile ROS 2 applications effectively
- Apply testing strategies for ROS 2 systems
- Manage dependencies and packages in ROS 2 workspaces
- Deploy ROS 2 applications to different environments

## Development Workflow Overview

A typical ROS 2 development workflow involves several phases: setup, development, testing, debugging, and deployment. Each phase has specific tools and practices that ensure robust robot applications.

### Project Setup Workflow

```
ROS 2 Development Workflow:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │───→│   Development   │───→│    Testing      │
│   Setup         │    │   & Coding      │    │   & Debugging   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Deployment    │
                         │   & Operation   │
                         └─────────────────┘
```

### Creating a New ROS 2 Package

```bash
# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create a new package using ros2 pkg
ros2 pkg create --build-type ament_python my_robot_package --dependencies rclpy std_msgs geometry_msgs sensor_msgs

# Or for C++ package
ros2 pkg create --build-type ament_cmake my_robot_cpp_package --dependencies rclcpp std_msgs geometry_msgs sensor_msgs
```

### Package Structure

A well-structured ROS 2 package follows this layout:

```
my_robot_package/
├── CMakeLists.txt              # Build configuration (C++)
├── package.xml                 # Package metadata and dependencies
├── setup.py                    # Python build configuration
├── setup.cfg                   # Python installation configuration
├── my_robot_package/           # Python package directory
│   ├── __init__.py
│   ├── nodes/                  # Node implementations
│   │   ├── robot_controller.py
│   │   └── sensor_processor.py
│   ├── utils/                  # Utility functions
│   │   ├── transforms.py
│   │   └── conversions.py
│   └── msg/                    # Custom message definitions (if any)
├── launch/                     # Launch files
│   ├── robot.launch.py
│   └── simulation.launch.py
├── config/                     # Configuration files
│   ├── robot_params.yaml
│   └── sensors.yaml
├── test/                       # Test files
│   ├── test_robot_controller.py
│   └── test_sensor_processor.py
└── README.md                   # Package documentation
```

### Package.xml Example

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.1.0</version>
  <description>Package for my robot control and sensing</description>
  <maintainer email="maintainer@todo.todo">maintainer</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Testing Strategies for ROS 2

### Unit Testing

ROS 2 supports standard Python and C++ testing frameworks:

```python
# test/test_robot_controller.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_package.nodes.robot_controller import RobotController

class TestRobotController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = RobotController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_initial_state(self):
        """Test that robot starts in expected initial state."""
        self.assertEqual(self.node.current_state, 'idle')
        self.assertEqual(self.node.velocity, 0.0)

    def test_velocity_command(self):
        """Test velocity command processing."""
        initial_vel = self.node.velocity

        # Simulate receiving a velocity command
        self.node.velocity_cmd = 1.0
        self.node.process_commands()

        self.assertNotEqual(self.node.velocity, initial_vel)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
# test/test_integration.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class TestRobotIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = rclpy.create_node('integration_tester')

        # Create publishers and subscribers for testing
        self.cmd_vel_pub = self.node.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.node.create_subscription(
            Float32, 'odom_velocity', self.odom_callback, 10
        )

        self.received_velocity = None
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def odom_callback(self, msg):
        self.received_velocity = msg.data

    def test_command_response(self):
        """Test that velocity commands produce expected responses."""
        # Send a command
        cmd = Twist()
        cmd.linear.x = 1.0
        self.cmd_vel_pub.publish(cmd)

        # Wait for response
        rclpy.spin_once(self.node, timeout_sec=1.0)

        # Verify response
        self.assertIsNotNone(self.received_velocity)
        self.assertGreater(self.received_velocity, 0.0)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    unittest.main()
```

### Launch Testing

```python
# test/test_launch.py
import unittest
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import pytest

@pytest.mark.launch_test
def generate_test_description():
    """Launch the nodes to be tested."""
    test_node = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller_test'
    )

    return LaunchDescription([
        test_node,
        launch_testing.actions.ReadyToTest()
    ])

def test_node_running():
    """Test that the node is running."""
    # This test will run after the launch description is ready
    assert True  # Placeholder - actual test would check node status
```

## Debugging Techniques

### ROS 2 Command Line Tools

```bash
# Check node status
ros2 node list
ros2 node info /robot_controller

# Check topic communication
ros2 topic list
ros2 topic echo /cmd_vel
ros2 topic hz /sensor_data

# Check service availability
ros2 service list
ros2 service call /set_parameters example_interfaces/srv/SetParameters "{parameters: [{name: 'speed', value: {double_value: 1.0}}]}"

# Check action servers
ros2 action list
ros2 action info /navigate_to_pose

# Monitor system resources
ros2 run rqt_graph rqt_graph  # Visualize node graph
ros2 run rqt_console rqt_console  # View logs
ros2 run rqt_plot rqt_plot  # Plot numerical data
```

### Python Debugging

```python
# debug_robot.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pdb  # Python debugger

class DebugRobotNode(Node):
    def __init__(self):
        super().__init__('debug_robot')
        self.publisher = self.create_publisher(String, 'debug_topic', 10)
        self.subscription = self.create_subscription(
            String, 'input_topic', self.debug_callback, 10
        )
        self.timer = self.create_timer(1.0, self.debug_timer)

    def debug_callback(self, msg):
        # Set breakpoint for debugging
        pdb.set_trace()  # This will pause execution and start debugger
        self.get_logger().info(f'Received: {msg.data}')

        # Process message
        processed_msg = String()
        processed_msg.data = f'Processed: {msg.data}'
        self.publisher.publish(processed_msg)

    def debug_timer(self):
        self.get_logger().info('Timer callback executed')

def main(args=None):
    rclpy.init(args=args)
    node = DebugRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Logging and Diagnostics

```python
# diagnostic_robot.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from std_msgs.msg import String
import time

class DiagnosticRobotNode(Node):
    def __init__(self):
        super().__init__('diagnostic_robot')

        # Create diagnostic publisher
        qos = QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', qos)

        # Create regular publisher
        self.data_pub = self.create_publisher(String, 'robot_data', 10)

        # Timer for diagnostics
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)
        self.data_timer = self.create_timer(0.1, self.publish_data)

        self.heartbeat_counter = 0

    def publish_data(self):
        """Publish regular robot data."""
        msg = String()
        msg.data = f'Robot data: {time.time()}'
        self.data_pub.publish(msg)

    def publish_diagnostics(self):
        """Publish diagnostic information."""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Robot status diagnostic
        robot_status = DiagnosticStatus()
        robot_status.name = 'Robot Controller'
        robot_status.hardware_id = 'robot_controller_01'
        robot_status.level = DiagnosticStatus.OK
        robot_status.message = 'Robot operating normally'
        robot_status.values = [
            {'key': 'Uptime', 'value': str(self.heartbeat_counter)},
            {'key': 'Status', 'value': 'Running'},
            {'key': 'Mode', 'value': 'Autonomous'}
        ]

        # Sensor status diagnostic
        sensor_status = DiagnosticStatus()
        sensor_status.name = 'Laser Sensor'
        sensor_status.hardware_id = 'laser_01'
        sensor_status.level = DiagnosticStatus.WARN  # Simulate warning
        sensor_status.message = 'Sensor range limit approaching'
        sensor_status.values = [
            {'key': 'Range', 'value': '29.5m'},
            {'key': 'Status', 'value': 'Warning'},
            {'key': 'Calibration', 'value': 'OK'}
        ]

        diag_array.status = [robot_status, sensor_status]
        self.diag_pub.publish(diag_array)
        self.heartbeat_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Profiling

### CPU and Memory Profiling

```python
# performance_profiler.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import psutil  # pip install psutil
import time

class PerformanceProfiler(Node):
    def __init__(self):
        super().__init__('performance_profiler')

        # Publishers for performance metrics
        self.cpu_pub = self.create_publisher(Float32, 'cpu_usage', 10)
        self.memory_pub = self.create_publisher(Float32, 'memory_usage', 10)

        # Timer for profiling
        self.profile_timer = self.create_timer(0.5, self.profile_system)

        # Process information
        self.process = psutil.Process()

    def profile_system(self):
        """Profile system resources."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_msg = Float32()
        cpu_msg.data = float(cpu_percent)
        self.cpu_pub.publish(cpu_msg)

        # Memory usage
        memory_info = self.process.memory_info()
        memory_percent = (memory_info.rss / psutil.virtual_memory().total) * 100
        memory_msg = Float32()
        memory_msg.data = float(memory_percent)
        self.memory_pub.publish(memory_msg)

        # Log if resources are high
        if cpu_percent > 80:
            self.get_logger().warn(f'High CPU usage: {cpu_percent}%')
        if memory_percent > 80:
            self.get_logger().warn(f'High memory usage: {memory_percent}%')

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceProfiler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Message Timing Analysis

```python
# timing_analyzer.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time

class TimingAnalyzer(Node):
    def __init__(self):
        super().__init__('timing_analyzer')

        # Publisher for timing analysis
        self.pub = self.create_publisher(Header, 'timing_test', 10)
        self.sub = self.create_subscription(
            Header, 'timing_test', self.timing_callback, 10
        )

        # Timing analysis
        self.sent_times = {}
        self.message_id = 0

        # Timer for sending messages
        self.timer = self.create_timer(0.1, self.send_timing_message)

    def send_timing_message(self):
        """Send a message with timing information."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f'msg_{self.message_id}'

        # Store sent time for later analysis
        sent_time = time.time_ns()
        self.sent_times[self.message_id] = sent_time

        self.pub.publish(header)
        self.message_id += 1

    def timing_callback(self, msg):
        """Analyze timing when message is received."""
        if msg.frame_id.startswith('msg_'):
            msg_id = int(msg.frame_id.split('_')[1])

            if msg_id in self.sent_times:
                sent_time = self.sent_times[msg_id]
                received_time = time.time_ns()

                # Calculate round-trip time (or just processing time if same node)
                rtt_ns = received_time - sent_time
                rtt_ms = rtt_ns / 1_000_000

                self.get_logger().info(f'Message {msg_id} round-trip time: {rtt_ms:.2f}ms')

                # Remove from tracking
                del self.sent_times[msg_id]

def main(args=None):
    rclpy.init(args=args)
    node = TimingAnalyzer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Dependency Management

### Managing Dependencies in package.xml

```xml
<!-- Best practices for dependencies -->
<package format="3">
  <!-- Core dependencies -->
  <depend>rclpy</depend>  <!-- or rclcpp for C++ -->
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <!-- Conditional dependencies -->
  <exec_depend condition="$TARGET_OS == ubuntu and $TARGET_ARCH == amd64">libopencv-dev</exec_depend>

  <!-- Test dependencies -->
  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <!-- Build dependencies -->
  <build_depend>ament_cmake_python</build_depend>
  <buildtool_depend>ament_cmake</buildtool_depend>
</package>
```

### Using colcon for Building

```bash
# Build the entire workspace
colcon build

# Build only specific packages
colcon build --packages-select my_robot_package

# Build with specific CMake arguments
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build and run tests
colcon build --packages-select my_robot_package
colcon test --packages-select my_robot_package
colcon test-result --all

# Clean build artifacts
rm -rf build/ install/ log/
```

## Deployment Workflows

### Docker Deployment

Create a Dockerfile for your ROS 2 application:

```dockerfile
# Dockerfile
FROM ros:humble-ros-base

# Set environment variables
ENV ROS_DISTRO=humble
ENV COLCON_WS=/opt/ros_ws

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p $COLCON_WS/src

# Copy package source
COPY . $COLCON_WS/src/my_robot_package

# Install Python dependencies
RUN pip3 install -r $COLCON_WS/src/my_robot_package/requirements.txt

# Build the workspace
WORKDIR $COLCON_WS
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build --packages-select my_robot_package

# Source the workspace
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source $COLCON_WS/install/setup.bash" >> ~/.bashrc

# Set entrypoint
CMD ["bash", "-c", "source /opt/ros/$ROS_DISTRO/setup.bash && source $COLCON_WS/install/setup.bash && exec \"$@\"", "ros_entrypoint.sh"]
```

### Creating a requirements.txt for Python dependencies

```txt
# requirements.txt
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
opencv-python>=4.5.0
pyyaml>=5.4.0
```

## Practical Example: Complete Robot Control System

Here's a complete example combining all the concepts:

```python
# robot_control_system.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # QoS profiles
        sensor_qos = qos_profile_sensor_data
        cmd_qos = QoSProfile(depth=10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', cmd_qos)
        self.safety_pub = self.create_publisher(Bool, 'safety_status', cmd_qos)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, sensor_qos
        )
        self.velocity_sub = self.create_subscription(
            Float32, 'desired_velocity', self.velocity_callback, cmd_qos
        )

        # Parameters
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)

        # Internal state
        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value

        self.desired_velocity = 0.0
        self.obstacle_detected = False
        self.closest_obstacle = float('inf')

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Robot controller initialized')

    def scan_callback(self, msg):
        """Process laser scan data."""
        if len(msg.ranges) > 0:
            # Find closest obstacle
            valid_ranges = [r for r in msg.ranges if 0.1 < r < 30.0]  # Filter invalid ranges
            if valid_ranges:
                self.closest_obstacle = min(valid_ranges)
                self.obstacle_detected = self.closest_obstacle < self.safety_distance
            else:
                self.closest_obstacle = float('inf')
                self.obstacle_detected = False

    def velocity_callback(self, msg):
        """Update desired velocity."""
        self.desired_velocity = max(0.0, min(msg.data, self.max_linear_vel))

    def control_loop(self):
        """Main control loop."""
        cmd = Twist()

        if self.obstacle_detected:
            # Emergency stop if too close to obstacle
            cmd.linear = Vector3(x=0.0, y=0.0, z=0.0)
            cmd.angular = Vector3(x=0.0, y=0.0, z=0.0)
            self.get_logger().warn(f'Obstacle detected at {self.closest_obstacle:.2f}m, stopping!')
        else:
            # Normal operation
            cmd.linear = Vector3(x=self.desired_velocity, y=0.0, z=0.0)

            # Add some angular correction if needed (simplified)
            if self.closest_obstacle < 1.0:
                cmd.angular = Vector3(z=0.5)  # Turn slightly away from obstacles

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = not self.obstacle_detected
        self.safety_pub.publish(safety_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down robot controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered practical workflows for ROS 2 development:

- **Development workflow**: Complete process from setup to deployment
- **Testing strategies**: Unit, integration, and launch testing approaches
- **Debugging techniques**: Tools and methods for troubleshooting
- **Performance profiling**: Monitoring and optimization techniques
- **Dependency management**: Best practices for package management
- **Deployment workflows**: Strategies for different environments

These practical workflows are essential for developing robust, maintainable, and deployable robotic systems.

## Exercises

1. Create a complete ROS 2 package with proper structure, testing, and documentation.
2. Implement a diagnostic system for a robot that monitors various subsystems.
3. Create a Docker container for your ROS 2 application and test it.

## Quiz

1. What is the recommended way to build ROS 2 packages?
   a) Using catkin_make
   b) Using colcon
   c) Using cmake directly
   d) Using make

2. Which QoS profile is typically used for sensor data?
   a) qos_profile_services_default
   b) qos_profile_sensor_data
   c) qos_profile_parameters
   d) qos_profile_default

3. What is the purpose of launch testing?
   a) To test the launch system itself
   b) To test nodes within their launch context
   c) To measure launch performance
   d) To validate launch file syntax

## Mini-Project: Complete Robot System

Create a complete robot system with:
1. Proper package structure with nodes, launch files, and configuration
2. Unit and integration tests for all components
3. Diagnostic and safety systems
4. Performance monitoring capabilities
5. A launch file that starts the complete system
6. Documentation and deployment instructions

Test your system by running it with different configurations and verifying all components work together correctly.