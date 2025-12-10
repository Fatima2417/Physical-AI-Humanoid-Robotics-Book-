---
sidebar_position: 1
---

# ROS 2 Overview

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the architecture and components of ROS 2
- Explain the differences between ROS 1 and ROS 2
- Set up a basic ROS 2 development environment
- Create and run a simple ROS 2 node

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is the next-generation robotics middleware designed to provide a flexible framework for developing robot applications. Unlike the original ROS, ROS 2 is built on DDS (Data Distribution Service) for robust, scalable, and real-time communication between robot components.

### What is ROS 2?

ROS 2 is not actually an operating system, but rather a collection of software frameworks and tools that help develop robot applications. It provides:

- **Communication**: Message passing between processes and machines
- **Package Management**: Tools for organizing and building code
- **Hardware Abstraction**: Interfaces for various sensors and actuators
- **Device Drivers**: Support for various hardware components
- **Libraries**: Commonly used functions for robot development

### Key Improvements over ROS 1

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Communication | Custom TCP/UDP | DDS-based (Real-time QoS) |
| Multi-machine | Manual setup | Automatic discovery |
| Real-time support | Limited | First-class support |
| Security | None | Built-in security |
| Cross-platform | Linux-focused | Windows, macOS, Linux |
| Lifecycle management | Basic | Advanced state management |

## ROS 2 Architecture

ROS 2 uses a DDS (Data Distribution Service) implementation for communication between nodes. This provides better real-time performance, security, and multi-machine capabilities compared to ROS 1's custom communication layer.

```
ROS 2 Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node A        │    │   Node B        │    │   Node C        │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Publisher   │ │    │ │ Subscriber  │ │    │ │ Service     │ │
│ │ /topic1     │ │    │ │ /topic1     │ │    │ │ /service    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                        DDS (Data Distribution Service)
                        (Provides communication backbone)
```

### Core Components

1. **Nodes**: Processes that perform computation
2. **Topics**: Named buses over which nodes exchange messages
3. **Messages**: Data structures passed between nodes
4. **Services**: Synchronous request/response communication
5. **Actions**: Asynchronous goal-based communication
6. **Parameters**: Configuration values accessible to nodes
7. **Lifecycle nodes**: Nodes with well-defined state transitions

## Setting Up ROS 2

### Installation Requirements

ROS 2 Humble Hawksbill (Ubuntu 22.04 LTS) is recommended for this textbook:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
```

### Environment Setup

After installation, source the ROS 2 setup script:

```bash
# Add to ~/.bashrc to source automatically
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Or source for current terminal
source /opt/ros/humble/setup.bash
```

## Basic ROS 2 Concepts

### Creating a Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace (even if empty)
colcon build

# Source the workspace
source install/setup.bash
```

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Here's a simple Python node:

```python
# publisher_member_function.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Topics and Messages

Topics are named buses for data exchange. Messages are the data structures sent over topics:

```python
# subscriber_member_function.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Running ROS 2 Examples

### Terminal 1 - Start the publisher:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run demo_nodes_py talker
```

### Terminal 2 - Start the subscriber:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run demo_nodes_py listener
```

## ROS 2 Command Line Tools

ROS 2 provides several command-line tools for introspection and debugging:

- `ros2 node list` - List active nodes
- `ros2 topic list` - List active topics
- `ros2 service list` - List active services
- `ros2 param list` - List parameters for a node
- `ros2 run <package> <executable>` - Run a node
- `ros2 launch <package> <launch_file>` - Launch multiple nodes

### Example: Examining Topics

```bash
# List all topics
ros2 topic list

# Show message type for a topic
ros2 topic type /topic

# Echo messages on a topic
ros2 topic echo /topic

# Publish a message to a topic
ros2 topic pub /topic std_msgs/String "data: 'Hello'"
```

## Quality of Service (QoS)

ROS 2 introduces Quality of Service profiles to handle different communication requirements:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Configure QoS for reliability-critical data
qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)

# For real-time data where latest is most important
qos_profile = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST
)
```

## Package Structure

A typical ROS 2 package follows this structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── src/                    # Source code
│   ├── publisher.cpp
│   └── subscriber.cpp
├── launch/                 # Launch files
│   └── robot.launch.py
├── config/                 # Configuration files
│   └── params.yaml
└── test/                   # Test files
```

## Summary

ROS 2 provides a robust framework for developing robot applications with improved real-time capabilities, security, and cross-platform support. Understanding its architecture and basic concepts is fundamental to developing Physical AI systems. The next chapters will explore ROS 2 in greater depth, including nodes, topics, services, and practical workflows.

## Exercises

1. Create a simple ROS 2 workspace and run the talker/listener example.
2. Modify the publisher to publish a custom message type.
3. Use ROS 2 command-line tools to inspect the communication between nodes.

## Quiz

1. What does DDS stand for in ROS 2?
   a) Data Distribution Service
   b) Dynamic Discovery System
   c) Distributed Data System
   d) Digital Design Specification

2. Which of these is NOT a core component of ROS 2?
   a) Nodes
   b) Topics
   c) Classes
   d) Services

3. What command is used to run a ROS 2 node?
   a) ros2 run `<package>` `<executable>`
   b) ros2 start `<package>` `<executable>`
   c) ros2 launch `<package>` `<executable>`
   d) ros2 exec `<package>` `<executable>`

## Mini-Project: ROS 2 Hello World

Create a ROS 2 package with:
1. A publisher node that publishes "Hello from ROS 2" every 2 seconds
2. A subscriber node that receives and logs the messages
3. A launch file that starts both nodes simultaneously

Test your package by running it and verifying communication between the nodes.