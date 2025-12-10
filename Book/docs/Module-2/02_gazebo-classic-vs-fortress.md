---
sidebar_position: 2
---

# Gazebo Classic vs Gazebo Fortress/Harmonic

## Learning Objectives

By the end of this chapter, you will be able to:
- Compare the architectural differences between Gazebo Classic and Gazebo Fortress/Harmonic
- Understand the migration path from Gazebo Classic to the new Gazebo
- Choose the appropriate Gazebo version for your project requirements
- Set up and configure both Gazebo environments
- Implement best practices for each version

## Introduction to Gazebo Evolution

The Gazebo simulation environment has undergone a significant architectural transformation. What was traditionally known as "Gazebo" has been renamed to "Gazebo Classic" to distinguish it from the new "Gazebo" (formerly Ignition Gazebo). This change represents a complete re-architecture of the simulation platform.

### Gazebo Classic (Old Architecture)

```
Gazebo Classic Architecture:
┌─────────────────┐
│   GUI Client    │  ← Qt-based GUI
│   (gzclient)    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Server Core   │  ← Single monolithic process
│   (gzserver)    │     with embedded physics
└─────────────────┘
         │
    ┌─────────┐
    │ Physics │ ← ODE, Bullet, Simbody
    │ Engine  │
    └─────────┘
```

### New Gazebo (Fortress/Harmonic) Architecture

```
New Gazebo Architecture:
┌─────────────────┐    ┌─────────────────┐
│   GUI Client    │    │   Server Core   │
│   (gz-sim)      │    │   (gz-sim)      │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
              Gazebo Transport
              (DDS-based messaging)

┌─────────────────┐    ┌─────────────────┐
│   Physics       │    │   Rendering     │
│   System        │    │   System        │
└─────────────────┘    └─────────────────┘
         │                       │
    ┌─────────┐           ┌─────────┐
    │ Plugins │           │ Plugins │
    └─────────┘           └─────────┘
```

## Gazebo Classic Overview

Gazebo Classic (versions up to 11.x) was the traditional simulation environment that has been widely used in the robotics community for many years.

### Key Characteristics of Gazebo Classic

- **Monolithic Architecture**: Single server process managing physics, rendering, and plugins
- **Native ROS Integration**: Direct support for ROS and ROS 2 through plugins
- **SDF Format**: Uses Simulation Description Format for models and worlds
- **Qt GUI**: Built-in Qt-based graphical interface
- **Stable and Mature**: Well-tested with extensive documentation and examples

### Gazebo Classic Installation

```bash
# For ROS 2 Humble (includes Gazebo Classic)
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins

# Launch with classic Gazebo
gazebo
```

### Gazebo Classic World File Example

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Example robot -->
    <model name="simple_robot">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="chassis">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box><size>1 1 0.5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 0.5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## New Gazebo (Fortress/Harmonic) Overview

New Gazebo (formerly Ignition Gazebo) represents a complete re-architecture with a modular, distributed design.

### Key Characteristics of New Gazebo

- **Modular Architecture**: Separate processes for different functions
- **Plugin-Based**: Everything is implemented as plugins
- **Modern C++**: Uses modern C++ practices and libraries
- **Improved Performance**: Better parallelization and resource management
- **Enhanced Rendering**: More advanced rendering capabilities
- **Better Scalability**: Can handle larger and more complex simulations

### New Gazebo Installation

```bash
# Install Gazebo Fortress (or Harmonic)
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-dev

# Or install standalone Gazebo
# Follow instructions at: https://gazebosim.org/docs/harmonic/install_ubuntu

# Launch new Gazebo
gz sim
```

### New Gazebo World File Example

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics system -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Scene configuration -->
    <scene>
      <ambient>0.8 0.8 0.8 1</ambient>
      <background>0.2 0.2 0.8 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Light sources -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>

    <!-- Example model -->
    <model name="simple_robot" placement_frame="chassis">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="chassis">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box><size>1 1 0.5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 0.5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Comparison Table

| Feature | Gazebo Classic | New Gazebo (Fortress/Harmonic) |
|---------|----------------|--------------------------------|
| Architecture | Monolithic | Modular, distributed |
| Performance | Good | Better (parallelized) |
| Rendering | Good | Excellent (modern graphics) |
| Plugin System | Good | Superior (everything is plugin) |
| ROS Integration | Native | Through bridge |
| Development Status | Maintenance mode | Active development |
| Community Support | Extensive | Growing |
| Learning Resources | Abundant | Increasing |
| SDF Compatibility | Native | High compatibility |
| Physics Engines | ODE, Bullet, Simbody | Same engines, better integration |

## Migration Considerations

### When to Use Gazebo Classic

Choose Gazebo Classic when:
- You have existing projects with Classic dependencies
- You need mature ROS integration without additional bridges
- You have extensive existing models/worlds that are complex to migrate
- Your team is already familiar with Classic
- You're working on stable, long-term projects with minimal new features

### When to Use New Gazebo

Choose New Gazebo when:
- Starting new projects from scratch
- You need advanced rendering capabilities
- Performance is critical for your application
- You want to use the latest features and improvements
- You're building for future compatibility
- You need better scalability for complex simulations

## ROS 2 Integration Differences

### Gazebo Classic ROS Integration

```xml
<!-- In URDF for Gazebo Classic -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>
```

### New Gazebo ROS Integration

New Gazebo uses the Gazebo ROS packages for ROS integration:

```xml
<!-- In URDF for New Gazebo -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_description)/config/ros2_control.yaml</parameters>
  </plugin>
</gazebo>
```

### Launch File Differences

```python
# Gazebo Classic launch file
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo Classic
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_init.so'],
            output='screen'
        )
    ])
```

```python
# New Gazebo launch file
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world_file = PathJoinSubstitution([
        FindPackageShare('my_robot_gazebo'),
        'worlds',
        'my_world.sdf'
    ])

    return LaunchDescription([
        # Launch new Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={
                'world': world_file,
                'verbose': 'true'
            }.items()
        )
    ])
```

## Practical Example: Setting Up Both Versions

Here's how to set up a basic simulation that can work with both versions:

### Universal URDF (Compatible with Both)

```xml
<?xml version="1.0"?>
<robot name="universal_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Gazebo Classic plugin -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Gazebo ROS Control (compatible with both) -->
  <gazebo>
    <plugin name="object_controller" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>odom</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>map</frameName>
    </plugin>
  </gazebo>
</robot>
```

### ROS 2 Node for Both Versions

```python
# universal_sim_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math

class UniversalSimController(Node):
    def __init__(self):
        super().__init__('universal_sim_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.position_x = 0.0
        self.position_y = 0.0
        self.orientation_z = 0.0
        self.closest_obstacle = float('inf')

    def odom_callback(self, msg):
        """Handle odometry messages from either simulator."""
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        # Extract orientation (simplified)
        self.orientation_z = math.atan2(
            2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z),
            1 - 2 * (msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)
        )

    def scan_callback(self, msg):
        """Handle laser scan messages from either simulator."""
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if 0.1 < r < 30.0]
            if valid_ranges:
                self.closest_obstacle = min(valid_ranges)

    def control_loop(self):
        """Simple control algorithm that works with both simulators."""
        cmd = Twist()

        # Simple navigation - avoid obstacles and move forward
        if self.closest_obstacle < 1.0:
            # Turn to avoid obstacle
            cmd.angular.z = 0.5
            cmd.linear.x = 0.0
        else:
            # Move forward
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = UniversalSimController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Comparison

### Benchmarking Considerations

When comparing performance between the two versions, consider:

1. **Physics Update Rate**: Both can run at similar rates, but New Gazebo may have better parallelization
2. **Rendering Quality**: New Gazebo typically offers better graphics
3. **Memory Usage**: New Gazebo may use more memory due to modular architecture
4. **Plugin Loading**: New Gazebo has faster plugin loading times

### Example Benchmark Script

```python
# performance_benchmark.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import psutil

class PerformanceBenchmark(Node):
    def __init__(self):
        super().__init__('performance_benchmark')

        # Publishers for metrics
        self.cpu_pub = self.create_publisher(Float32, 'cpu_usage', 10)
        self.memory_pub = self.create_publisher(Float32, 'memory_usage', 10)

        # Timer for monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_performance)

        # Process information
        self.process = psutil.Process()

    def monitor_performance(self):
        """Monitor system performance."""
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

        self.get_logger().info(f'CPU: {cpu_percent}%, Memory: {memory_percent}%')

def main(args=None):
    rclpy.init(args=args)
    benchmark = PerformanceBenchmark()

    try:
        rclpy.spin(benchmark)
    except KeyboardInterrupt:
        pass
    finally:
        benchmark.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Migration Path

### From Classic to New Gazebo

1. **Evaluate Requirements**: Assess if new features justify migration
2. **Test Compatibility**: Verify your models work with New Gazebo
3. **Update Dependencies**: Change package dependencies and launch files
4. **Update Plugins**: Migrate custom plugins to new plugin system
5. **Test Thoroughly**: Validate all functionality in new environment
6. **Gradual Migration**: Consider running both during transition

### Common Migration Issues

- **SDF Format Changes**: Minor syntax differences between versions
- **Plugin Names**: Some plugin names have changed
- **Topic Names**: Default topic names may differ
- **Parameter Names**: Some parameters have been renamed

## Summary

This chapter compared Gazebo Classic and New Gazebo (Fortress/Harmonic), highlighting their architectural differences and use cases:

- **Gazebo Classic**: Stable, mature, with native ROS integration - suitable for existing projects
- **New Gazebo**: Modern, modular, with superior performance and rendering - ideal for new projects

The choice between them depends on your project's specific needs, timeline, and requirements for future compatibility. For new projects, New Gazebo is generally recommended, while existing projects may benefit from staying with Classic until a migration is planned.

## Exercises

1. Set up both Gazebo Classic and New Gazebo on your system
2. Create a simple robot model that works in both environments
3. Compare the performance of the same simulation in both versions
4. Migrate a simple existing simulation from Classic to New Gazebo

## Quiz

1. What is the main architectural difference between Gazebo Classic and New Gazebo?
   a) Different physics engines
   b) Classic is monolithic, New is modular/distributed
   c) Different rendering engines only
   d) Same architecture, different GUI

2. Which version has native ROS integration without additional bridges?
   a) New Gazebo only
   b) Gazebo Classic only
   c) Both versions
   d) Neither version

3. What is the recommended approach for new projects?
   a) Always use Gazebo Classic
   b) Always use New Gazebo
   c) Depends on specific requirements
   d) Use both simultaneously

## Mini-Project: Simulation Comparison

Create a complete robot simulation that works in both Gazebo Classic and New Gazebo:
1. Design a simple differential drive robot model
2. Create launch files for both versions
3. Implement a navigation algorithm that works with both
4. Compare performance metrics between the two versions
5. Document the differences and similarities you observe
6. Create a migration guide for moving from one to the other