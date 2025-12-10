---
sidebar_position: 4
---

# ROS 2 Launch Files and Composition

## Learning Objectives

By the end of this chapter, you will be able to:
- Create and manage complex launch files for multi-node systems
- Implement node composition for improved performance
- Use launch arguments and conditions for flexible configurations
- Understand the differences between standalone and composed nodes
- Apply best practices for system deployment and management

## ROS 2 Launch System

The ROS 2 launch system provides a powerful way to start and configure multiple nodes with specific parameters and dependencies. It replaces the XML-based launch system from ROS 1 with a Python-based system that offers greater flexibility.

### Basic Launch File Structure

```python
# launch/basic_example.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='publisher_node',
            output='screen'
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='subscriber_node',
            output='screen'
        )
    ])
```

### Launch Arguments

Launch arguments allow for flexible configurations:

```python
# launch/argument_example.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='robot1',
        description='Namespace for the robot nodes'
    )

    rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='Publish rate in Hz'
    )

    # Use launch configurations in node definitions
    namespace = LaunchConfiguration('namespace')
    publish_rate = LaunchConfiguration('publish_rate')

    return LaunchDescription([
        namespace_arg,
        rate_arg,

        Node(
            package='demo_nodes_py',
            executable='talker',
            name='talker_node',
            namespace=namespace,
            parameters=[{'publish_rate': publish_rate}],
            output='screen'
        )
    ])
```

### Conditional Launch

Launch files can include conditional logic:

```python
# launch/conditional_example.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    debug = LaunchConfiguration('debug')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        debug_arg,
        sim_time_arg,

        # Log message based on condition
        LogInfo(
            condition=IfCondition(debug),
            msg='Debug mode is enabled'
        ),

        # Conditional node
        Node(
            condition=IfCondition(debug),
            package='demo_nodes_py',
            executable='talker',
            name='debug_talker',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Node that runs unless debug is enabled
        Node(
            condition=UnlessCondition(debug),
            package='demo_nodes_py',
            executable='listener',
            name='normal_listener',
            output='screen'
        )
    ])
```

## Advanced Launch Features

### Remapping Topics

Launch files can remap topics between nodes:

```python
# launch/remapping_example.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='publisher',
            remappings=[
                ('chatter', 'custom_topic')  # Remap chatter to custom_topic
            ],
            output='screen'
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='subscriber',
            remappings=[
                ('chatter', 'custom_topic')  # Listen to the remapped topic
            ],
            output='screen'
        )
    ])
```

### Loading Parameters from Files

```python
# launch/param_file_example.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get path to parameter file
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[config],
            output='screen'
        )
    ])
```

Example parameter file (`config/robot_params.yaml`):
```yaml
/**:
  ros__parameters:
    robot_name: "my_robot"
    max_velocity: 0.5
    sensors:
      laser_scan:
        enabled: true
        range_min: 0.1
        range_max: 30.0
      imu:
        enabled: true
```

### Grouping Nodes

Nodes can be grouped with shared configurations:

```python
# launch/group_example.launch.py
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        # Group nodes under a common namespace
        GroupAction(
            actions=[
                PushRosNamespace('robot1'),
                Node(
                    package='demo_nodes_py',
                    executable='talker',
                    name='talker',
                    output='screen'
                ),
                Node(
                    package='demo_nodes_py',
                    executable='listener',
                    name='listener',
                    output='screen'
                )
            ]
        ),
        GroupAction(
            actions=[
                PushRosNamespace('robot2'),
                Node(
                    package='demo_nodes_py',
                    executable='talker',
                    name='talker',
                    output='screen'
                ),
                Node(
                    package='demo_nodes_py',
                    executable='listener',
                    name='listener',
                    output='screen'
                )
            ]
        )
    ])
```

## Node Composition

Node composition allows multiple nodes to run within a single process, reducing communication overhead and improving performance.

### Composition Concepts

In traditional ROS 2, each node runs as a separate process:

```
Traditional Approach:
┌─────────────────┐    ┌─────────────────┐
│   Node A        │    │   Node B        │
│   Process 1     │    │   Process 2     │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
              DDS Communication
```

With composition, multiple nodes run in the same process:

```
Composition Approach:
┌─────────────────────────────────────────┐
│           Single Process                │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Node A    │  │   Node B    │      │
│  │   (Component│  │   (Component│      │
│  │    Manager) │  │    Manager) │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

### Creating Composable Nodes

First, create a composable node (component):

```cpp
// include/my_robot_components/talker_component.hpp
#ifndef TALKER_COMPONENT_HPP_
#define TALKER_COMPONENT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace my_robot_components
{
class Talker : public rclcpp::Node
{
public:
    Talker(const rclcpp::NodeOptions & options);

private:
    void timer_callback();
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    size_t count_;
};

}  // namespace my_robot_components

#endif  // TALKER_COMPONENT_HPP_
```

```cpp
// src/talker_component.cpp
#include "my_robot_components/talker_component.hpp"

namespace my_robot_components
{

Talker::Talker(const rclcpp::NodeOptions & options)
: Node("talker", options), count_(0)
{
    pub_ = this->create_publisher<std_msgs::msg::String>("chatter", 10);
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&Talker::timer_callback, this)
    );
}

void Talker::timer_callback()
{
    auto msg = std_msgs::msg::String();
    msg.data = "Hello World: " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", msg.data.c_str());
    pub_->publish(msg);
}

}  // namespace my_robot_components

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with the ROS 2 node factory
RCLCPP_COMPONENTS_REGISTER_NODE(my_robot_components::Talker)
```

### Composition Launch File

```python
# launch/composition_example.launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    """Create a container and add composable nodes to it."""
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='my_robot_components',
                plugin='my_robot_components::Talker',
                name='talker',
                parameters=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='my_robot_components',
                plugin='my_robot_components::Listener',
                name='listener',
                parameters=[{'use_intra_process_comms': True}]
            )
        ],
        output='screen'
    )

    return LaunchDescription([container])
```

### Alternative: Manual Composition

You can also manually compose nodes within your own process:

```python
# manual_composition.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.count}'
        self.publisher.publish(msg)
        self.count += 1

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String, 'chatter', self.listener_callback, 10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main():
    rclpy.init()

    # Create multiple nodes in the same process
    talker = TalkerNode()
    listener = ListenerNode()

    # Spin all nodes together
    try:
        rclpy.spin(listener)  # Both nodes will be processed
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File Best Practices

### Organizing Complex Systems

For complex robotic systems, organize launch files hierarchically:

```python
# launch/robot_system.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Include other launch files
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('my_robot_description'),
                    'launch',
                    'robot_description.launch.py'
                ])
            ])
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('my_robot_hardware'),
                    'launch',
                    'hardware_interface.launch.py'
                ])
            ])
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('nav2_bringup'),
                    'launch',
                    'navigation_launch.py'
                ])
            ])
        )
    ])
```

### Error Handling in Launch Files

```python
# launch/error_handling_example.launch.py
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    talker_node = Node(
        package='demo_nodes_py',
        executable='talker',
        name='talker',
        output='screen'
    )

    listener_node = Node(
        package='demo_nodes_py',
        executable='listener',
        name='listener',
        output='screen'
    )

    # Event handler for process start
    on_talker_start = RegisterEventHandler(
        OnProcessStart(
            target_action=talker_node,
            on_start=[
                # Do something when talker starts
            ]
        )
    )

    # Event handler for process exit
    on_talker_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=talker_node,
            on_exit=[
                # Do something when talker exits
            ]
        )
    )

    return LaunchDescription([
        on_talker_start,
        on_talker_exit,
        talker_node,
        listener_node
    ])
```

## Practical Example: Complete Robot Launch System

Here's a comprehensive example of a robot launch system:

```python
# launch/complete_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='robot',
        description='Robot namespace'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    enable_composition_arg = DeclareLaunchArgument(
        'enable_composition',
        default_value='false',
        description='Enable node composition for better performance'
    )

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_composition = LaunchConfiguration('enable_composition')

    # Get parameter file path
    config_path = os.path.join(
        get_package_share_directory('my_robot_bringup'),
        'config',
        'robot_config.yaml'
    )

    # Define nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            config_path
        ],
        output='screen'
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # If composition is enabled, use container
    composition_container = ComposableNodeContainer(
        condition=IfCondition(enable_composition),
        name='robot_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='my_robot_components',
                plugin='my_robot_components::SensorProcessor',
                name='sensor_processor',
                parameters=[{'use_sim_time': use_sim_time}]
            ),
            ComposableNode(
                package='my_robot_components',
                plugin='my_robot_components::Controller',
                name='controller',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ],
        output='screen'
    )

    # If composition is disabled, use standalone nodes
    sensor_processor = Node(
        condition=IfCondition(enable_composition),
        package='my_robot_nodes',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    controller = Node(
        condition=IfCondition(enable_composition),
        package='my_robot_nodes',
        executable='controller',
        name='controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        use_sim_time_arg,
        enable_composition_arg,

        # Group nodes under namespace
        GroupAction(
            actions=[
                PushRosNamespace(namespace),
                robot_state_publisher,
                joint_state_publisher,
                composition_container,
                sensor_processor,
                controller
            ]
        )
    ])
```

## Performance Considerations

### When to Use Composition

**Use composition when:**
- Nodes communicate frequently with low latency requirements
- Running on resource-constrained hardware
- Need to reduce process overhead
- Nodes are tightly coupled functionally

**Use standalone nodes when:**
- Need process isolation for stability
- Nodes have different resource requirements
- Need independent restart capabilities
- Debugging is easier with separate processes

### Intra-Process Communication

When using composition, enable intra-process communication for better performance:

```python
# In your nodes
from rclpy.context import Context

# Enable intra-process communication in parameters
parameters = [
    {'use_intra_process_comms': True}
]
```

## Summary

This chapter covered the advanced ROS 2 launch system and node composition:

- **Launch files**: For starting and configuring complex multi-node systems
- **Launch arguments**: For flexible, configurable deployments
- **Node composition**: For improved performance through shared processes
- **Best practices**: For organizing and managing robotic systems

These tools are essential for deploying real-world robotic systems with proper configuration management and performance optimization.

## Exercises

1. Create a launch file that starts a robot simulation with multiple sensors and a controller.
2. Implement a composable node for sensor data processing and integrate it into a container.
3. Create a hierarchical launch system that includes multiple launch files.

## Quiz

1. What is the main advantage of node composition?
   a) Better security
   b) Reduced communication overhead and improved performance
   c) Easier debugging
   d) More features

2. How do you enable intra-process communication in a launch file?
   a) Set use_intra_process_comms parameter to True
   b) Use a special launch action
   c) It's enabled by default
   d) Through environment variables

3. What is the purpose of LaunchConfiguration?
   a) To store launch parameters permanently
   b) To access launch argument values in launch files
   c) To configure nodes at runtime
   d) To define node parameters

## Mini-Project: Robot Launch System

Create a complete robot launch system with:
1. A main launch file that includes multiple subsystems
2. Parameter files for different robot configurations
3. Both standalone and composable node implementations
4. Launch arguments for flexible deployment options
5. Proper error handling and event management

Test your system by launching it with different configurations and verifying all components start correctly.