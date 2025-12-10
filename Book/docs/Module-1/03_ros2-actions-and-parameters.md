---
sidebar_position: 3
---

# ROS 2 Actions and Parameters

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement action-based communication for long-running tasks
- Create and manage ROS 2 parameters in nodes
- Use launch files to configure and start multiple nodes
- Understand when to use actions vs services vs topics
- Implement lifecycle management for complex nodes

## ROS 2 Actions

Actions are a communication pattern for long-running tasks that provide feedback during execution. They are ideal for tasks like navigation, manipulation, or calibration where you need to know the progress and potentially cancel the operation.

### Action Structure

An action consists of three message types:
- **Goal**: Request sent to the action server
- **Result**: Response sent back when the action completes
- **Feedback**: Updates sent during execution

```
Action Communication Flow:
┌─────────────────┐    Goal     ┌─────────────────┐
│   Action Client │ ─────────→ │ Action Server   │
│                 │ ←───────── │                 │
│   Send Goal     │   Result   │   Execute Goal  │
│   Receive       │ ←───────── │   Send Feedback │
│   Feedback      │   Feedback │                 │
└─────────────────┘            └─────────────────┘
```

### Creating Action Messages

First, define an action in an `.action` file:

```
# In action/Fibonacci.action
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

This defines:
- Goal: `order` (the Fibonacci sequence order)
- Result: `sequence` (the complete sequence)
- Feedback: `partial_sequence` (partial sequence during computation)

### Action Server Implementation

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info('Received goal request')
        # Accept all goals
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request."""
        self.get_logger().info('Received cancel request')
        # Accept all cancel requests
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Create feedback message
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        # Simulate long-running task
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            # Update feedback
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1]
            )

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.partial_sequence}')

            # Simulate processing time
            from time import sleep
            sleep(0.5)

        # Populate result message
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence

        goal_handle.succeed()
        self.get_logger().info(f'Result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    action_server = FibonacciActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')

        # Create action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

    def send_goal(self, order):
        # Wait for action server
        self._action_client.wait_for_server()

        # Create goal message
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Send goal and get future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(
            f'Received feedback: {feedback_msg.feedback.partial_sequence}'
        )

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()

    # Send goal when node is ready
    action_client.send_goal(10)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## ROS 2 Parameters

Parameters provide a way to configure nodes at runtime and are stored in a parameter server.

### Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptors
        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Name of the robot'
            )
        )

        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum velocity of the robot',
                floating_point_range=[ParameterDescriptor(
                    from_value=0.0,
                    to_value=10.0
                )]
            )
        )

        self.declare_parameter('wheel_diameter', 0.1)  # Default value: 0.1

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.wheel_diameter = self.get_parameter('wheel_diameter').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Wheel diameter: {self.wheel_diameter}')

        # Set up parameter callback for dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Callback for parameter changes."""
        for param in params:
            if param.name == 'max_velocity' and param.type == ParameterType.PARAMETER_DOUBLE:
                if 0.0 <= param.value <= 10.0:
                    self.max_velocity = param.value
                    self.get_logger().info(f'Updated max velocity to: {self.max_velocity}')
                    return SetParametersResult(successful=True)
                else:
                    self.get_logger().info('Max velocity must be between 0.0 and 10.0')
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

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

### Parameter Files (YAML)

Create parameter files to configure nodes:

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    robot_name: "turtlebot3"
    max_velocity: 0.5
    wheel_diameter: 0.066
    sensors:
      laser_scan:
        enabled: true
        frame_id: "laser_frame"
        range_min: 0.1
        range_max: 30.0
      imu:
        enabled: true
        frame_id: "imu_frame"
```

### Using Parameters in Launch Files

```python
# launch/robot_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get config file path
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='parameter_node',
            name='robot_controller',
            parameters=[config],  # Load parameters from YAML
            output='screen'
        )
    ])
```

## Launch Files

Launch files allow you to start multiple nodes with specific configurations.

### Basic Launch File

```python
# launch/basic_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='publisher',
            parameters=[
                {'param_name': 'param_value'}
            ],
            remappings=[
                ('chatter', 'custom_topic')
            ],
            output='screen'
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='subscriber',
            output='screen'
        )
    ])
```

### Advanced Launch File with Conditions

```python
# launch/conditional_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    debug_mode = LaunchConfiguration('debug_mode')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'debug_mode',
            default_value='false',
            description='Enable debug mode'
        ),

        # Conditional node
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'debug_mode': debug_mode}
            ],
            condition=IfCondition(debug_mode)  # Only run if debug_mode is true
        ),

        # Always run node
        Node(
            package='my_robot_package',
            executable='robot_driver',
            name='robot_driver'
        )
    ])
```

## Lifecycle Nodes

Lifecycle nodes provide a structured way to manage the state of complex nodes.

### Lifecycle Node Implementation

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile

class LifecycleTalker(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_talker')
        self.pub = None

    def on_configure(self, state):
        """Called when node transitions to CONFIGURING state."""
        self.get_logger().info(f'Configuring node: {state}')
        self.pub = self.create_publisher(String, 'lifecycle_chatter', QoSProfile(depth=10))
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Called when node transitions to ACTIVATING state."""
        self.get_logger().info(f'Activating node: {state}')
        # Activate the publisher
        self.pub.on_activate()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Called when node transitions to DEACTIVATING state."""
        self.get_logger().info(f'Deactivating node: {state}')
        # Deactivate the publisher
        self.pub.on_deactivate()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Called when node transitions to CLEANINGUP state."""
        self.get_logger().info(f'Cleaning up node: {state}')
        # Destroy publisher
        self.destroy_publisher(self.pub)
        self.pub = None
        return TransitionCallbackReturn.SUCCESS

    def publish_message(self):
        """Publish a message if the node is active."""
        if self.pub is not None and self.pub.is_activated:
            msg = String()
            msg.data = f'Lifecycle message: {self.get_clock().now().nanoseconds}'
            self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleTalker()

    try:
        # The node needs to be manually triggered to transition through its states
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## When to Use Each Communication Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| Topics | Continuous data streams | Asynchronous, many-to-many, no guarantee of delivery |
| Services | Request/response tasks | Synchronous, one-to-one, request must be processed |
| Actions | Long-running tasks | Asynchronous, provides feedback, cancellable |
| Parameters | Configuration | Persistent, accessible at runtime, changeable |

### Decision Matrix

```
Communication Pattern Decision Tree:
Is it a continuous data stream?
├── YES → Use Topics
└── NO → Is it a simple request/response?
    ├── YES → Is it a long-running task?
    │   ├── YES → Use Actions
    │   └── NO → Use Services
    └── NO → Is it configuration?
        ├── YES → Use Parameters
        └── NO → Consider the specific requirements
```

## Practical Example: Navigation System

Here's a practical example combining actions and parameters:

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Pose
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import math

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')

        # Declare navigation parameters
        self.declare_parameter(
            'max_linear_velocity',
            0.5,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum linear velocity for navigation'
            )
        )

        self.declare_parameter(
            'max_angular_velocity',
            1.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum angular velocity for navigation'
            )
        )

        # Get parameter values
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value

        # Create action server for navigation
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_navigate_callback
        )

        # Current robot pose (in a real system, this would come from localization)
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0

    def distance_to_goal(self, goal_pose):
        """Calculate distance to goal."""
        dx = goal_pose.pose.position.x - self.current_pose.position.x
        dy = goal_pose.pose.position.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    async def execute_navigate_callback(self, goal_handle):
        """Execute navigation to pose."""
        self.get_logger().info('Executing navigation goal...')

        goal_pose = goal_handle.request.pose
        initial_distance = self.distance_to_goal(goal_pose)

        # Simulate navigation with feedback
        feedback_msg = NavigateToPose.Feedback()
        remaining_distance = initial_distance

        while remaining_distance > 0.1:  # Tolerance of 0.1m
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation canceled')
                return NavigateToPose.Result()

            # Update robot position (simulated)
            dx = goal_pose.pose.position.x - self.current_pose.position.x
            dy = goal_pose.pose.position.y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > 0.01:  # Avoid division by zero
                # Move robot toward goal
                scale = min(self.max_linear_vel * 0.1 / distance, 1.0)  # 0.1s time step
                self.current_pose.position.x += dx * scale
                self.current_pose.position.y += dy * scale

            remaining_distance = self.distance_to_goal(goal_pose)

            # Update feedback
            feedback_msg.current_pose = self.current_pose
            feedback_msg.distance_remaining = remaining_distance

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Distance remaining: {remaining_distance:.2f}m')

            # Simulate processing time
            from time import sleep
            sleep(0.1)

        # Navigation completed
        result = NavigateToPose.Result()
        result.result = True  # Success

        goal_handle.succeed()
        self.get_logger().info('Navigation completed successfully')

        return result

def main(args=None):
    rclpy.init(args=args)
    nav_server = NavigationActionServer()

    try:
        rclpy.spin(nav_server)
    except KeyboardInterrupt:
        pass
    finally:
        nav_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered advanced ROS 2 communication patterns:

- **Actions**: For long-running tasks with feedback and cancellation
- **Parameters**: For runtime configuration of nodes
- **Launch files**: For starting multiple nodes with specific configurations
- **Lifecycle nodes**: For managing complex node states

These patterns provide the tools needed to build sophisticated robotic systems with proper configuration management and complex task execution.

## Exercises

1. Create an action server that simulates a robot arm moving to different positions with feedback.
2. Implement a parameter server that allows changing robot speeds at runtime.
3. Create a launch file that starts multiple nodes with different parameter configurations.

## Quiz

1. What is the main difference between services and actions in ROS 2?
   a) Services are faster than actions
   b) Actions provide feedback and can be canceled during execution
   c) Services use more memory than actions
   d) There is no difference

2. How do you declare a parameter in a ROS 2 node?
   a) Using declare_parameter() method
   b) Using create_parameter() method
   c) Using set_parameter() method
   d) Using add_parameter() method

3. What are the three message types in an action?
   a) Request, Response, Update
   b) Goal, Result, Feedback
   c) Start, Stop, Continue
   d) Input, Output, Status

## Mini-Project: Robot Task Manager

Create a system with:
1. An action server that manages robot tasks (navigation, manipulation, etc.)
2. Parameters to configure task execution (speeds, tolerances, etc.)
3. A launch file that starts the system with appropriate configurations
4. A client that sends tasks to the server and monitors progress

Test your system by sending different types of tasks and verifying proper execution with feedback.