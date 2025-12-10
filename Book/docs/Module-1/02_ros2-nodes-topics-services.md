---
sidebar_position: 2
---

# ROS 2 Nodes, Topics, and Services

## Learning Objectives

By the end of this chapter, you will be able to:
- Create and manage ROS 2 nodes in Python and C++
- Implement topic-based communication between nodes
- Develop service-based request/response communication
- Use ROS 2 tools for debugging and introspection
- Understand the differences between topics and services

## ROS 2 Nodes

A node is the fundamental building block of a ROS 2 program. Nodes are processes that perform computation and communicate with other nodes through topics, services, actions, and parameters.

### Node Structure in Python

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Node initialization code here
        self.get_logger().info('Node has been initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

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

### Node Structure in C++

```cpp
#include <rclcpp/rclcpp.hpp>

class MyNode : public rclcpp::Node
{
public:
    MyNode() : Node("node_name")
    {
        RCLCPP_INFO(this->get_logger(), "Node has been initialized");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Node Parameters

Nodes can accept parameters that can be configured at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('param_name', 'default_value')
        self.declare_parameter('frequency', 1.0)

        # Get parameter values
        param_value = self.get_parameter('param_name').value
        frequency = self.get_parameter('frequency').value

        self.get_logger().info(f'Parameter value: {param_value}')
        self.get_logger().info(f'Frequency: {frequency}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

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

## Topics and Publishers/Subscribers

Topics enable asynchronous, many-to-many communication through a publish/subscribe model.

### Creating Publishers

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')

        # Create publisher
        self.publisher = self.create_publisher(
            String,           # Message type
            'chatter',        # Topic name
            10                # Queue size
        )

        # Create timer for periodic publishing
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = TalkerNode()

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

### Creating Subscribers

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')

        # Create subscriber
        self.subscription = self.create_subscription(
            String,           # Message type
            'chatter',        # Topic name
            self.listener_callback,  # Callback function
            10                # Queue size
        )
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()

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

### Custom Message Types

You can create custom message types by defining `.msg` files:

```
# In msg/Num.msg
int64 num
```

```python
# In your Python code
from my_robot_package.msg import Num

class CustomMessageNode(Node):
    def __init__(self):
        super().__init__('custom_message_node')

        self.publisher = self.create_publisher(Num, 'custom_topic', 10)
        self.subscription = self.create_subscription(
            Num, 'custom_topic', self.custom_callback, 10
        )

        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.publish_custom_msg)
        self.counter = 0

    def publish_custom_msg(self):
        msg = Num()
        msg.num = self.counter
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.num}')
        self.counter += 1

    def custom_callback(self, msg):
        self.get_logger().info(f'Received: {msg.num}')
```

## Services

Services provide synchronous request/response communication between nodes.

### Creating Services

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')

        # Create service server
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    service_server = ServiceServer()

    try:
        rclpy.spin(service_server)
    except KeyboardInterrupt:
        pass
    finally:
        service_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating Service Clients

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')

        # Create client
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    client = ServiceClient()

    # Send request
    future = client.send_request(2, 3)

    try:
        rclpy.spin_until_future_complete(client, future)
        response = future.result()
        client.get_logger().info(f'Result: {response.sum}')
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Settings

QoS settings control how messages are delivered between nodes:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# Reliable communication (messages guaranteed to be delivered)
reliable_qos = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Best effort communication (faster but not guaranteed)
best_effort_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Publisher with specific QoS
publisher = node.create_publisher(String, 'topic', reliable_qos)
```

## Debugging and Introspection Tools

ROS 2 provides powerful tools for debugging communication:

### Topic Tools
```bash
# List all topics
ros2 topic list

# Show message type for a topic
ros2 topic type /chatter

# Echo messages on a topic
ros2 topic echo /chatter

# Show topic information
ros2 topic info /chatter

# Publish a message to a topic
ros2 topic pub /chatter std_msgs/String "data: 'Hello'"
```

### Service Tools
```bash
# List all services
ros2 service list

# Show service type
ros2 service type /add_two_ints

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

### Node Tools
```bash
# List all nodes
ros2 node list

# Show node information
ros2 node info /talker

# Show graph of nodes and topics
ros2 run rqt_graph rqt_graph
```

## Advanced Topic Features

### Latching (Transient Local Durability)

For important messages that late-joining subscribers should receive:

```python
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

latched_qos = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
)

publisher = node.create_publisher(String, 'important_topic', latched_qos)
```

### Message Filters

For processing messages from multiple topics simultaneously:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber

class MultiTopicNode(Node):
    def __init__(self):
        super().__init__('multi_topic_node')

        # Create subscribers for multiple topics
        sub1 = Subscriber(self, String, 'topic1')
        sub2 = Subscriber(self, Int32, 'topic2')

        # Synchronize messages from both topics
        ats = ApproximateTimeSynchronizer(
            [sub1, sub2],
            queue_size=10,
            slop=0.1
        )
        ats.registerCallback(self.sync_callback)

    def sync_callback(self, msg1, msg2):
        self.get_logger().info(f'Received synchronized: {msg1.data}, {msg2.data}')
```

## Practical Example: Sensor Data Publisher

Here's a practical example combining multiple concepts:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import random

class SensorSimulator(Node):
    def __init__(self):
        super().__init__('sensor_simulator')

        # Create publishers for different sensor types
        self.laser_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.distance_pub = self.create_publisher(Float32, 'distance', 10)

        # Create timer for periodic sensor simulation
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

    def publish_sensor_data(self):
        # Simulate laser scan data
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -1.57  # -90 degrees
        scan_msg.angle_max = 1.57   # 90 degrees
        scan_msg.angle_increment = 0.0174  # 1 degree
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        scan_msg.ranges = [random.uniform(0.5, 5.0) for _ in range(181)]

        self.laser_pub.publish(scan_msg)

        # Publish distance to closest obstacle
        min_distance = min(scan_msg.ranges) if scan_msg.ranges else 0.0
        distance_msg = Float32()
        distance_msg.data = min_distance
        self.distance_pub.publish(distance_msg)

        self.get_logger().info(f'Published sensor data. Min distance: {min_distance:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    sensor_node = SensorSimulator()

    try:
        rclpy.spin(sensor_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered the fundamental communication patterns in ROS 2:
- **Nodes**: The basic execution units that perform computation
- **Topics**: Asynchronous publish/subscribe communication
- **Services**: Synchronous request/response communication
- **QoS settings**: Control over message delivery characteristics
- **Debugging tools**: Essential tools for introspection and debugging

Understanding these concepts is crucial for building distributed robotic systems that can effectively coordinate between different components.

## Exercises

1. Create a node that publishes temperature data and another that subscribes to it and logs warnings when temperature exceeds a threshold.
2. Implement a service that converts temperatures between Celsius and Fahrenheit.
3. Use ROS 2 command-line tools to visualize the communication between your nodes.

## Quiz

1. What is the difference between topics and services in ROS 2?
   a) Topics are faster than services
   b) Topics are asynchronous, services are synchronous
   c) Topics use more memory than services
   d) There is no difference

2. What does QoS stand for?
   a) Quality of Service
   b) Quick Operating System
   c) Quantum Operation System
   d) Quality Operating Service

3. Which QoS policy ensures messages are delivered reliably?
   a) BEST_EFFORT
   b) RELIABLE
   c) VOLATILE
   d) TRANSIENT

## Mini-Project: Robot Health Monitor

Create a system with:
1. A sensor node that publishes random sensor readings (temperature, voltage, etc.)
2. A monitoring node that subscribes to sensor data and checks for out-of-range values
3. A service that allows external systems to request the current health status
4. Use appropriate QoS settings for different types of data

Test your system by running all nodes and using ROS 2 tools to verify communication.