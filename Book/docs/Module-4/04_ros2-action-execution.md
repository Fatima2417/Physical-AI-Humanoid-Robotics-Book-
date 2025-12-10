---
sidebar_position: 4
---

# ROS 2 Action Execution

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement and use ROS 2 actions for long-running robotic tasks
- Create custom action definitions for robotics applications
- Integrate action execution with LLM cognitive planning
- Handle action feedback, goals, and results in robotic systems
- Implement action servers and clients for robotic behaviors
- Design action-based architectures for complex robotic tasks

## Introduction to ROS 2 Actions

ROS 2 actions are a communication pattern designed for long-running tasks that provide feedback during execution and can be preempted or canceled. Unlike services (which are synchronous) or topics (which are asynchronous), actions are perfect for robotic tasks that take time and need to provide ongoing status updates.

### Action vs Other Communication Patterns

```
ROS 2 Communication Patterns Comparison:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Topics        │    │   Services      │    │   Actions       │
│   (Async)       │    │   (Sync)        │    │   (Long-running)│
│   • Publishers  │    │   • Request/    │    │   • Goals       │
│   • Subscribers │    │   • Response    │    │   • Feedback    │
│   • Fire & Forget│    │   • One-shot    │    │   • Results     │
│   • No Status   │    │   • No Progress │    │   • Cancelable  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Action Architecture

```
Action Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Action Client │────│   Action Server │────│   Robot         │
│   (Goal Sender) │    │   (Goal Executor│    │   (Hardware     │
│                 │    │   , Feedback    │    │   Interface)    │
│   • Send Goal   │    │   , Result)     │    │                 │
│   • Receive     │    │                 │    │                 │
│   • Feedback    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Action        │
                         │   Interface     │
                         │   (Goal,        │
                         │   Feedback,     │
                         │   Result)       │
                         └─────────────────┘
```

## Creating Custom Actions

### Defining Action Messages

Action definitions consist of three message types:
- **Goal**: Request sent to the action server
- **Result**: Response sent back when the action completes
- **Feedback**: Updates sent during execution

Create an action definition file `RobotTask.action`:

```
# Goal: Request for the robot to perform a task
string task_type  # Type of task (navigation, manipulation, etc.)
geometry_msgs/Pose target_pose  # Target position/orientation
string description  # Human-readable description of the task

---
# Result: Final outcome of the task
bool success  # Whether the task was completed successfully
string message  # Human-readable status message
float64 execution_time  # Time taken to complete the task in seconds

---
# Feedback: Ongoing status during task execution
string status  # Current status (e.g., "navigating", "grasping", "completed")
float64 progress  # Progress percentage (0.0 to 100.0)
geometry_msgs/Pose current_pose  # Current robot pose
```

### Building Action Messages

```bash
# Create the action definition file
mkdir -p my_robot_msgs/action
echo '# Goal: Request for the robot to perform a task
string task_type  # Type of task (navigation, manipulation, etc.)
geometry_msgs/Pose target_pose  # Target position/orientation
string description  # Human-readable description of the task

---
# Result: Final outcome of the task
bool success  # Whether the task was completed successfully
string message  # Human-readable status message
float64 execution_time  # Time taken to complete the task in seconds

---
# Feedback: Ongoing status during task execution
string status  # Current status (e.g., "navigating", "grasping", "completed")
float64 progress  # Progress percentage (0.0 to 100.0)
geometry_msgs/Pose current_pose  # Current robot pose' > my_robot_msgs/action/RobotTask.action

# Add to package.xml
# <depend>action_msgs</depend>
# <member_of_group>rosidl_interface_packages</member_of_group>

# Add to CMakeLists.txt
# find_package(rosidl_default_generators REQUIRED)
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "action/RobotTask.action"
#   DEPENDENCIES geometry_msgs
# )
```

## Action Server Implementation

### Basic Action Server

```python
# robot_task_server.py
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import math
import time
from my_robot_msgs.action import RobotTask

class RobotTaskServer(Node):
    def __init__(self):
        super().__init__('robot_task_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            RobotTask,
            'robot_task',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Robot state
        self.current_pose = Pose()
        self.is_moving = False
        self.is_executing = False

        # Subscriptions for robot state
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )

        self.get_logger().info('Robot Task Server initialized')

    def odom_callback(self, msg):
        """Update current robot pose from odometry."""
        self.current_pose = msg.pose.pose

    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info(f'Received goal: {goal_request.description}')

        # Accept all goals for this example
        # In real applications, you might check feasibility
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request."""
        self.get_logger().info('Received cancel request')

        # Accept all cancel requests
        # In real applications, you might have specific cancel policies
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Get goal parameters
        task_type = goal_handle.request.task_type
        target_pose = goal_handle.request.target_pose
        description = goal_handle.request.description

        # Create feedback message
        feedback_msg = RobotTask.Feedback()
        feedback_msg.status = 'initialized'
        feedback_msg.progress = 0.0
        feedback_msg.current_pose = self.current_pose

        # Start execution
        start_time = time.time()

        # Check if task type is supported
        if task_type == 'navigation':
            result = await self.execute_navigation_task(goal_handle, target_pose, feedback_msg)
        elif task_type == 'manipulation':
            result = await self.execute_manipulation_task(goal_handle, target_pose, feedback_msg)
        elif task_type == 'inspection':
            result = await self.execute_inspection_task(goal_handle, target_pose, feedback_msg)
        else:
            # Unknown task type
            result = RobotTask.Result()
            result.success = False
            result.message = f'Unknown task type: {task_type}'
            result.execution_time = time.time() - start_time

        return result

    async def execute_navigation_task(self, goal_handle, target_pose, feedback_msg):
        """Execute navigation task."""
        self.get_logger().info('Starting navigation task')

        # Calculate distance to target
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        dz = target_pose.position.z - self.current_pose.position.z
        distance_to_target = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Simple navigation approach (in real applications, use navigation2)
        steps = 50
        for i in range(steps):
            # Check if goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation task canceled')

                result = RobotTask.Result()
                result.success = False
                result.message = 'Navigation task canceled'
                result.execution_time = time.time() - time.time()  # Placeholder
                return result

            # Update feedback
            progress = (i / steps) * 100.0
            feedback_msg.status = f'navigating ({progress:.1f}%)'
            feedback_msg.progress = progress
            feedback_msg.current_pose = self.current_pose

            goal_handle.publish_feedback(feedback_msg)

            # Simulate movement toward target
            # In real implementation, this would send commands to navigation stack
            self.get_logger().info(f'Navigation progress: {progress:.1f}%')

            # Sleep to simulate processing time
            time.sleep(0.1)

        # Task completed
        feedback_msg.status = 'completed'
        feedback_msg.progress = 100.0
        feedback_msg.current_pose = target_pose
        goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        self.get_logger().info('Navigation task completed successfully')

        result = RobotTask.Result()
        result.success = True
        result.message = 'Navigation task completed successfully'
        result.execution_time = time.time() - time.time()  # Placeholder
        return result

    async def execute_manipulation_task(self, goal_handle, target_pose, feedback_msg):
        """Execute manipulation task."""
        self.get_logger().info('Starting manipulation task')

        # Simulate manipulation steps
        manipulation_steps = [
            'approaching_object',
            'aligning_gripper',
            'grasping_object',
            'lifting_object',
            'transporting_object',
            'placing_object'
        ]

        for i, step in enumerate(manipulation_steps):
            # Check if goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Manipulation task canceled')

                result = RobotTask.Result()
                result.success = False
                result.message = 'Manipulation task canceled'
                result.execution_time = time.time() - time.time()  # Placeholder
                return result

            # Update feedback
            progress = ((i + 1) / len(manipulation_steps)) * 100.0
            feedback_msg.status = step
            feedback_msg.progress = progress
            feedback_msg.current_pose = self.current_pose

            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f'Manipulation step: {step} ({progress:.1f}%)')

            # Sleep to simulate processing time
            time.sleep(0.2)

        # Task completed
        feedback_msg.status = 'completed'
        feedback_msg.progress = 100.0
        feedback_msg.current_pose = target_pose
        goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        self.get_logger().info('Manipulation task completed successfully')

        result = RobotTask.Result()
        result.success = True
        result.message = 'Manipulation task completed successfully'
        result.execution_time = time.time() - time.time()  # Placeholder
        return result

    async def execute_inspection_task(self, goal_handle, target_pose, feedback_msg):
        """Execute inspection task."""
        self.get_logger().info('Starting inspection task')

        # Simulate inspection steps
        inspection_steps = [
            'scanning_area',
            'identifying_objects',
            'analyzing_environment',
            'generating_report'
        ]

        for i, step in enumerate(inspection_steps):
            # Check if goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Inspection task canceled')

                result = RobotTask.Result()
                result.success = False
                result.message = 'Inspection task canceled'
                result.execution_time = time.time() - time.time()  # Placeholder
                return result

            # Update feedback
            progress = ((i + 1) / len(inspection_steps)) * 100.0
            feedback_msg.status = step
            feedback_msg.progress = progress
            feedback_msg.current_pose = self.current_pose

            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f'Inspection step: {step} ({progress:.1f}%)')

            # Sleep to simulate processing time
            time.sleep(0.15)

        # Task completed
        feedback_msg.status = 'completed'
        feedback_msg.progress = 100.0
        feedback_msg.current_pose = target_pose
        goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        self.get_logger().info('Inspection task completed successfully')

        result = RobotTask.Result()
        result.success = True
        result.message = 'Inspection task completed successfully'
        result.execution_time = time.time() - time.time()  # Placeholder
        return result

def main(args=None):
    rclpy.init(args=args)
    server = RobotTaskServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Shutting down Robot Task Server')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Action Client Implementation

### Basic Action Client

```python
# robot_task_client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from my_robot_msgs.action import RobotTask

class RobotTaskClient(Node):
    def __init__(self):
        super().__init__('robot_task_client')

        # Create action client
        self._action_client = ActionClient(
            self,
            RobotTask,
            'robot_task'
        )

        self.get_logger().info('Robot Task Client initialized')

    def send_goal(self, task_type, target_pose, description):
        """Send a goal to the robot task server."""
        # Wait for action server
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create goal message
        goal_msg = RobotTask.Goal()
        goal_msg.task_type = task_type
        goal_msg.target_pose = target_pose
        goal_msg.description = description

        # Send goal and get future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        return self._send_goal_future

    def goal_response_callback(self, future):
        """Handle goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from action server."""
        self.get_logger().info(
            f'Feedback: {feedback_msg.status}, '
            f'Progress: {feedback_msg.progress:.1f}%, '
            f'Position: ({feedback_msg.current_pose.position.x:.2f}, '
            f'{feedback_msg.current_pose.position.y:.2f})'
        )

    def get_result_callback(self, future):
        """Handle result from action server."""
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}, Success: {result.success}')

def main(args=None):
    rclpy.init(args=args)
    client = RobotTaskClient()

    # Example: Send a navigation goal
    target_pose = Pose()
    target_pose.position.x = 2.0
    target_pose.position.y = 1.0
    target_pose.position.z = 0.0
    target_pose.orientation.w = 1.0  # No rotation

    # Send the goal
    client.send_goal('navigation', target_pose, 'Navigate to position (2, 1)')

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Shutting down Robot Task Client')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Action Patterns

### Multi-Step Action Server

```python
# advanced_robot_task_server.py
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from my_robot_msgs.action import RobotTask
import asyncio
import threading
from enum import Enum
from typing import Dict, Any, Optional

class TaskState(Enum):
    """States for multi-step task execution."""
    INITIALIZED = 1
    PLANNING = 2
    EXECUTING = 3
    MONITORING = 4
    COMPLETING = 5
    CANCELING = 6
    FAILED = 7

class AdvancedRobotTaskServer(Node):
    def __init__(self):
        super().__init__('advanced_robot_task_server')

        # Create action server with custom callback group
        self._action_server = ActionServer(
            self,
            RobotTask,
            'advanced_robot_task',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Robot state
        self.current_pose = Pose()
        self.is_busy = False
        self.active_task_state = TaskState.INITIALIZED

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Pose, 'current_pose', self.pose_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )

        # Task-specific state
        self.current_task = None
        self.task_progress = 0.0
        self.task_feedback = RobotTask.Feedback()

        # Task execution lock
        self.task_lock = threading.Lock()

        self.get_logger().info('Advanced Robot Task Server initialized')

    def pose_callback(self, msg):
        """Update current pose."""
        self.current_pose = msg

    def scan_callback(self, msg):
        """Process laser scan data."""
        # Store scan data for obstacle detection during tasks
        pass

    def detections_callback(self, msg):
        """Process object detections."""
        # Store detection data for perception tasks
        pass

    def goal_callback(self, goal_request):
        """Accept or reject goals based on current state."""
        self.get_logger().info(f'Received goal: {goal_request.description}')

        # Check if robot is currently busy
        with self.task_lock:
            if self.is_busy:
                self.get_logger().warn('Robot is busy, rejecting new goal')
                return GoalResponse.REJECT

        # Accept the goal
        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancel requests."""
        self.get_logger().info('Cancel request received')

        # Check if task is in a cancelable state
        with self.task_lock:
            if self.active_task_state in [TaskState.PLANNING, TaskState.EXECUTING, TaskState.MONITORING]:
                self.active_task_state = TaskState.CANCELING
                return CancelResponse.ACCEPT
            else:
                return CancelResponse.REJECT

    async def execute_callback(self, goal_handle):
        """Execute the goal with advanced state management."""
        self.get_logger().info('Starting advanced task execution')

        # Initialize task state
        with self.task_lock:
            self.current_task = goal_handle
            self.is_busy = True
            self.active_task_state = TaskState.INITIALIZED
            self.task_progress = 0.0

        # Create feedback message
        self.task_feedback = RobotTask.Feedback()
        self.task_feedback.status = 'initialized'
        self.task_feedback.progress = 0.0
        self.task_feedback.current_pose = self.current_pose

        # Start execution
        start_time = self.get_clock().now().nanoseconds / 1e9

        try:
            # Execute task based on type
            if goal_handle.request.task_type == 'complex_navigation':
                result = await self.execute_complex_navigation_task(goal_handle)
            elif goal_handle.request.task_type == 'perception_guided_manipulation':
                result = await self.execute_perception_guided_manipulation_task(goal_handle)
            elif goal_handle.request.task_type == 'multi_location_inspection':
                result = await self.execute_multi_location_inspection_task(goal_handle)
            else:
                result = await self.execute_generic_task(goal_handle)

        except Exception as e:
            self.get_logger().error(f'Task execution error: {e}')

            with self.task_lock:
                self.active_task_state = TaskState.FAILED

            result = RobotTask.Result()
            result.success = False
            result.message = f'Task execution failed: {str(e)}'
            result.execution_time = self.get_clock().now().nanoseconds / 1e9 - start_time

        finally:
            # Cleanup task state
            with self.task_lock:
                self.current_task = None
                self.is_busy = False

        return result

    async def execute_complex_navigation_task(self, goal_handle):
        """Execute complex navigation with obstacle avoidance."""
        self.get_logger().info('Executing complex navigation task')

        # Task steps
        steps = [
            self.plan_navigation,
            self.execute_navigation,
            self.avoid_obstacles,
            self.verify_arrival
        ]

        total_steps = len(steps)
        for i, step_func in enumerate(steps):
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                return await self.handle_task_cancellation(goal_handle)

            # Update state
            with self.task_lock:
                self.active_task_state = TaskState.EXECUTING

            # Execute step
            self.task_feedback.status = f'executing_step_{i+1}_of_{total_steps}'
            self.task_feedback.progress = (i / total_steps) * 100.0
            self.task_feedback.current_pose = self.current_pose
            goal_handle.publish_feedback(self.task_feedback)

            try:
                success = await step_func(goal_handle)
                if not success:
                    result = RobotTask.Result()
                    result.success = False
                    result.message = f'Navigation step {i+1} failed'
                    result.execution_time = 0.0  # Will be calculated properly
                    return result
            except Exception as e:
                self.get_logger().error(f'Navigation step {i+1} failed: {e}')
                result = RobotTask.Result()
                result.success = False
                result.message = f'Navigation step {i+1} error: {str(e)}'
                result.execution_time = 0.0  # Will be calculated properly
                return result

        # Task completed successfully
        with self.task_lock:
            self.active_task_state = TaskState.COMPLETING

        self.task_feedback.status = 'completed'
        self.task_feedback.progress = 100.0
        self.task_feedback.current_pose = goal_handle.request.target_pose
        goal_handle.publish_feedback(self.task_feedback)

        goal_handle.succeed()

        result = RobotTask.Result()
        result.success = True
        result.message = 'Complex navigation completed successfully'
        result.execution_time = 0.0  # Will be calculated properly
        return result

    async def plan_navigation(self, goal_handle):
        """Plan navigation path."""
        self.get_logger().info('Planning navigation path')

        # Simulate path planning
        await asyncio.sleep(1.0)  # Simulate planning time

        return True

    async def execute_navigation(self, goal_handle):
        """Execute navigation."""
        self.get_logger().info('Executing navigation')

        # Simulate navigation execution
        for step in range(10):
            if goal_handle.is_cancel_requested:
                return False

            await asyncio.sleep(0.2)  # Simulate movement

            # Update feedback
            self.task_feedback.current_pose = self.interpolate_pose(
                self.current_pose,
                goal_handle.request.target_pose,
                (step + 1) / 10.0
            )
            goal_handle.publish_feedback(self.task_feedback)

        return True

    async def avoid_obstacles(self, goal_handle):
        """Handle obstacle avoidance during navigation."""
        self.get_logger().info('Checking for obstacles')

        # Simulate obstacle detection and avoidance
        await asyncio.sleep(0.5)

        return True

    async def verify_arrival(self, goal_handle):
        """Verify robot reached destination."""
        self.get_logger().info('Verifying arrival at destination')

        # Simulate arrival verification
        await asyncio.sleep(0.5)

        return True

    async def execute_perception_guided_manipulation_task(self, goal_handle):
        """Execute manipulation guided by perception."""
        self.get_logger().info('Executing perception-guided manipulation')

        # Steps for perception-guided manipulation
        steps = [
            self.perceive_environment,
            self.identify_target_object,
            self.plan_manipulation,
            self.execute_approach,
            self.grasp_object,
            self.verify_grasp
        ]

        total_steps = len(steps)
        for i, step_func in enumerate(steps):
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                return await self.handle_task_cancellation(goal_handle)

            # Update state
            with self.task_lock:
                self.active_task_state = TaskState.EXECUTING

            # Execute step
            self.task_feedback.status = f'perception_manipulation_step_{i+1}_of_{total_steps}'
            self.task_feedback.progress = (i / total_steps) * 100.0
            self.task_feedback.current_pose = self.current_pose
            goal_handle.publish_feedback(self.task_feedback)

            try:
                success = await step_func(goal_handle)
                if not success:
                    result = RobotTask.Result()
                    result.success = False
                    result.message = f'Manipulation step {i+1} failed'
                    result.execution_time = 0.0
                    return result
            except Exception as e:
                self.get_logger().error(f'Manipulation step {i+1} failed: {e}')
                result = RobotTask.Result()
                result.success = False
                result.message = f'Manipulation step {i+1} error: {str(e)}'
                result.execution_time = 0.0
                return result

        # Task completed successfully
        with self.task_lock:
            self.active_task_state = TaskState.COMPLETING

        self.task_feedback.status = 'completed'
        self.task_feedback.progress = 100.0
        self.task_feedback.current_pose = goal_handle.request.target_pose
        goal_handle.publish_feedback(self.task_feedback)

        goal_handle.succeed()

        result = RobotTask.Result()
        result.success = True
        result.message = 'Perception-guided manipulation completed successfully'
        result.execution_time = 0.0
        return result

    async def perceive_environment(self, goal_handle):
        """Perceive the environment."""
        self.get_logger().info('Perceiving environment')
        await asyncio.sleep(1.0)  # Simulate perception time
        return True

    async def identify_target_object(self, goal_handle):
        """Identify target object for manipulation."""
        self.get_logger().info('Identifying target object')
        await asyncio.sleep(0.8)  # Simulate object identification
        return True

    async def plan_manipulation(self, goal_handle):
        """Plan manipulation trajectory."""
        self.get_logger().info('Planning manipulation trajectory')
        await asyncio.sleep(0.5)  # Simulate planning time
        return True

    async def execute_approach(self, goal_handle):
        """Execute approach to target object."""
        self.get_logger().info('Executing approach to target')
        await asyncio.sleep(1.5)  # Simulate approach
        return True

    async def grasp_object(self, goal_handle):
        """Execute grasping action."""
        self.get_logger().info('Executing grasping action')
        await asyncio.sleep(1.0)  # Simulate grasping
        return True

    async def verify_grasp(self, goal_handle):
        """Verify successful grasp."""
        self.get_logger().info('Verifying grasp success')
        await asyncio.sleep(0.5)  # Simulate verification
        return True

    async def execute_multi_location_inspection_task(self, goal_handle):
        """Execute inspection at multiple locations."""
        self.get_logger().info('Executing multi-location inspection')

        # For this example, we'll simulate visiting 3 inspection points
        inspection_points = [
            self.create_inspection_point(1.0, 0.0, 0.0),
            self.create_inspection_point(2.0, 1.0, 0.0),
            self.create_inspection_point(0.5, 2.0, 0.0)
        ]

        total_points = len(inspection_points)
        for i, point in enumerate(inspection_points):
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                return await self.handle_task_cancellation(goal_handle)

            # Navigate to inspection point
            success = await self.navigate_to_inspection_point(goal_handle, point, i+1, total_points)
            if not success:
                result = RobotTask.Result()
                result.success = False
                result.message = f'Failed to reach inspection point {i+1}'
                result.execution_time = 0.0
                return result

            # Perform inspection at point
            success = await self.inspect_at_location(goal_handle, point, i+1, total_points)
            if not success:
                result = RobotTask.Result()
                result.success = False
                result.message = f'Inspection failed at point {i+1}'
                result.execution_time = 0.0
                return result

        # Task completed successfully
        with self.task_lock:
            self.active_task_state = TaskState.COMPLETING

        self.task_feedback.status = 'completed'
        self.task_feedback.progress = 100.0
        self.task_feedback.current_pose = self.current_pose
        goal_handle.publish_feedback(self.task_feedback)

        goal_handle.succeed()

        result = RobotTask.Result()
        result.success = True
        result.message = f'Multi-location inspection completed ({total_points} points)'
        result.execution_time = 0.0
        return result

    def create_inspection_point(self, x, y, z):
        """Create an inspection point."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0  # No rotation
        return pose

    async def navigate_to_inspection_point(self, goal_handle, point, current, total):
        """Navigate to an inspection point."""
        self.get_logger().info(f'Navigating to inspection point {current}/{total}')

        # Simulate navigation to point
        for step in range(5):
            if goal_handle.is_cancel_requested:
                return False

            await asyncio.sleep(0.3)  # Simulate movement

            # Update feedback
            self.task_feedback.status = f'navigating_to_point_{current}_of_{total}'
            self.task_feedback.progress = ((current-1) / total + (step+1)/(total*5)) * 100.0
            goal_handle.publish_feedback(self.task_feedback)

        return True

    async def inspect_at_location(self, goal_handle, point, current, total):
        """Perform inspection at location."""
        self.get_logger().info(f'Inspecting at point {current}/{total}')

        # Simulate inspection
        for step in range(3):
            if goal_handle.is_cancel_requested:
                return False

            await asyncio.sleep(0.4)  # Simulate inspection

            # Update feedback
            self.task_feedback.status = f'inspecting_point_{current}_of_{total}'
            self.task_feedback.progress = ((current-1) / total + (step+1)/(total*3)) * 100.0
            goal_handle.publish_feedback(self.task_feedback)

        return True

    async def execute_generic_task(self, goal_handle):
        """Execute generic task."""
        self.get_logger().info('Executing generic task')

        # Simulate generic task execution
        for i in range(10):
            if goal_handle.is_cancel_requested:
                return await self.handle_task_cancellation(goal_handle)

            await asyncio.sleep(0.1)

            # Update feedback
            self.task_feedback.status = f'executing_generic_task ({i+1}/10)'
            self.task_feedback.progress = ((i+1) / 10.0) * 100.0
            goal_handle.publish_feedback(self.task_feedback)

        # Complete task
        self.task_feedback.status = 'completed'
        self.task_feedback.progress = 100.0
        goal_handle.publish_feedback(self.task_feedback)

        goal_handle.succeed()

        result = RobotTask.Result()
        result.success = True
        result.message = 'Generic task completed successfully'
        result.execution_time = 0.0
        return result

    async def handle_task_cancellation(self, goal_handle):
        """Handle task cancellation."""
        self.get_logger().info('Handling task cancellation')

        with self.task_lock:
            self.active_task_state = TaskState.CANCELING

        # Perform cancellation steps
        await asyncio.sleep(0.5)  # Simulate cleanup

        goal_handle.canceled()

        result = RobotTask.Result()
        result.success = False
        result.message = 'Task was canceled'
        result.execution_time = 0.0
        return result

    def interpolate_pose(self, start_pose, end_pose, ratio):
        """Interpolate between two poses."""
        interpolated = Pose()
        interpolated.position.x = start_pose.position.x + ratio * (end_pose.position.x - start_pose.position.x)
        interpolated.position.y = start_pose.position.y + ratio * (end_pose.position.y - start_pose.position.y)
        interpolated.position.z = start_pose.position.z + ratio * (end_pose.position.z - start_pose.position.z)

        # Simple interpolation for orientation (in a real system, use quaternion slerp)
        interpolated.orientation.w = start_pose.orientation.w + ratio * (end_pose.orientation.w - start_pose.orientation.w)

        return interpolated

def main(args=None):
    rclpy.init(args=args)
    server = AdvancedRobotTaskServer()

    # Use multi-threaded executor to handle multiple tasks
    executor = MultiThreadedExecutor(num_threads=4)

    try:
        rclpy.spin(server, executor=executor)
    except KeyboardInterrupt:
        server.get_logger().info('Shutting down Advanced Robot Task Server')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## LLM Integration with Actions

### LLM-Driven Action Execution

```python
# llm_action_integration.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from my_robot_msgs.action import RobotTask
import json
import asyncio
from typing import Dict, Any, Optional

class LLMActionIntegration(Node):
    def __init__(self):
        super().__init__('llm_action_integration')

        # Subscribers
        self.plan_sub = self.create_subscription(
            String, 'llm_execution_plan', self.plan_callback, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, 'action_execution_status', 10)

        # Action client
        self.action_client = ActionClient(
            self,
            RobotTask,
            'robot_task'
        )

        # State tracking
        self.current_plan = None
        self.is_executing = False
        self.plan_queue = []

        self.get_logger().info('LLM Action Integration initialized')

    def plan_callback(self, msg):
        """Process LLM-generated plan."""
        try:
            plan_data = json.loads(msg.data)
            self.get_logger().info(f'Received plan: {plan_data.get("intent", "unknown")}')

            # Add plan to queue
            self.plan_queue.append(plan_data)

            # Execute plans if not already executing
            if not self.is_executing:
                self.execute_plans()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in plan: {e}')
        except Exception as e:
            self.get_logger().error(f'Plan processing error: {e}')

    def execute_plans(self):
        """Execute queued plans."""
        if not self.plan_queue:
            return

        self.is_executing = True

        # Process plans in queue
        while self.plan_queue:
            plan = self.plan_queue.pop(0)

            # Execute the plan
            success = self.execute_plan(plan)

            if not success:
                self.get_logger().error(f'Plan execution failed: {plan.get("intent", "unknown")}')
                break

        self.is_executing = False

    def execute_plan(self, plan: Dict[str, Any]) -> bool:
        """Execute a single plan."""
        try:
            intent = plan.get('intent', 'unknown')
            steps = plan.get('steps', [])

            self.get_logger().info(f'Executing plan intent: {intent}')

            # Execute each step in the plan
            for step in steps:
                success = self.execute_plan_step(step)
                if not success:
                    self.get_logger().error(f'Step execution failed: {step.get("action", "unknown")}')
                    return False

            # Plan completed successfully
            self.publish_status(f'Plan completed: {intent}')
            return True

        except Exception as e:
            self.get_logger().error(f'Plan execution error: {e}')
            return False

    def execute_plan_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single step from the plan."""
        try:
            action_type = step.get('action', 'unknown')
            parameters = step.get('parameters', {})

            self.get_logger().info(f'Executing step: {action_type}')

            if action_type == 'navigate':
                return self.execute_navigation_step(parameters)
            elif action_type == 'grasp':
                return self.execute_grasp_step(parameters)
            elif action_type == 'inspect':
                return self.execute_inspection_step(parameters)
            elif action_type == 'communicate':
                return self.execute_communication_step(parameters)
            else:
                self.get_logger().warn(f'Unknown action type: {action_type}')
                return False

        except Exception as e:
            self.get_logger().error(f'Step execution error: {e}')
            return False

    def execute_navigation_step(self, params: Dict[str, Any]) -> bool:
        """Execute navigation step using action."""
        try:
            # Wait for action server
            if not self.action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Action server not available')
                return False

            # Create target pose
            target_pose = Pose()
            target_pose.position.x = params.get('target_x', 0.0)
            target_pose.position.y = params.get('target_y', 0.0)
            target_pose.position.z = params.get('target_z', 0.0)

            # Set orientation (simplified)
            target_pose.orientation.w = 1.0

            # Create goal
            goal_msg = RobotTask.Goal()
            goal_msg.task_type = 'navigation'
            goal_msg.target_pose = target_pose
            goal_msg.description = f'Navigate to ({target_pose.position.x}, {target_pose.position.y})'

            # Send goal
            future = self.action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.navigation_feedback_callback
            )

            # Wait for result
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if goal_handle is None:
                self.get_logger().error('Navigation goal was rejected')
                return False

            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result

            self.get_logger().info(f'Navigation result: {result.message}, Success: {result.success}')
            return result.success

        except Exception as e:
            self.get_logger().error(f'Navigation step error: {e}')
            return False

    def execute_grasp_step(self, params: Dict[str, Any]) -> bool:
        """Execute grasp step using action."""
        try:
            # Wait for action server
            if not self.action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Action server not available')
                return False

            # Create target pose for grasping
            target_pose = Pose()
            target_pose.position.x = params.get('object_x', 0.0)
            target_pose.position.y = params.get('object_y', 0.0)
            target_pose.position.z = params.get('object_z', 0.0)
            target_pose.orientation.w = 1.0

            # Create goal
            goal_msg = RobotTask.Goal()
            goal_msg.task_type = 'manipulation'
            goal_msg.target_pose = target_pose
            goal_msg.description = f'Grasp object at ({target_pose.position.x}, {target_pose.position.y})'

            # Send goal
            future = self.action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.manipulation_feedback_callback
            )

            # Wait for result
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if goal_handle is None:
                self.get_logger().error('Grasp goal was rejected')
                return False

            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result

            self.get_logger().info(f'Grasp result: {result.message}, Success: {result.success}')
            return result.success

        except Exception as e:
            self.get_logger().error(f'Grasp step error: {e}')
            return False

    def execute_inspection_step(self, params: Dict[str, Any]) -> bool:
        """Execute inspection step using action."""
        try:
            # Wait for action server
            if not self.action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Action server not available')
                return False

            # Create target pose for inspection
            target_pose = Pose()
            target_pose.position.x = params.get('inspect_x', 0.0)
            target_pose.position.y = params.get('inspect_y', 0.0)
            target_pose.position.z = params.get('inspect_z', 0.0)
            target_pose.orientation.w = 1.0

            # Create goal
            goal_msg = RobotTask.Goal()
            goal_msg.task_type = 'inspection'
            goal_msg.target_pose = target_pose
            goal_msg.description = f'Inspect area at ({target_pose.position.x}, {target_pose.position.y})'

            # Send goal
            future = self.action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.inspection_feedback_callback
            )

            # Wait for result
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if goal_handle is None:
                self.get_logger().error('Inspection goal was rejected')
                return False

            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result

            self.get_logger().info(f'Inspection result: {result.message}, Success: {result.success}')
            return result.success

        except Exception as e:
            self.get_logger().error(f'Inspection step error: {e}')
            return False

    def execute_communication_step(self, params: Dict[str, Any]) -> bool:
        """Execute communication step."""
        try:
            message = params.get('message', 'Hello')
            self.get_logger().info(f'Communicating: {message}')

            # In a real system, this might trigger speech synthesis or other communication
            # For now, just log the communication

            return True
        except Exception as e:
            self.get_logger().error(f'Communication step error: {e}')
            return False

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        self.get_logger().info(
            f'Navigation feedback: {feedback_msg.status}, '
            f'Progress: {feedback_msg.progress:.1f}%'
        )

    def manipulation_feedback_callback(self, feedback_msg):
        """Handle manipulation feedback."""
        self.get_logger().info(
            f'Manipulation feedback: {feedback_msg.status}, '
            f'Progress: {feedback_msg.progress:.1f}%'
        )

    def inspection_feedback_callback(self, feedback_msg):
        """Handle inspection feedback."""
        self.get_logger().info(
            f'Inspection feedback: {feedback_msg.status}, '
            f'Progress: {feedback_msg.progress:.1f}%'
        )

    def publish_status(self, message: str):
        """Publish execution status."""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    integration = LLMActionIntegration()

    try:
        rclpy.spin(integration)
    except KeyboardInterrupt:
        integration.get_logger().info('Shutting down LLM Action Integration')
    finally:
        integration.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Action-Based Robot Behaviors

### Complex Robot Behavior Trees

```python
# action_behavior_trees.py
import rclpy
from rclpy.action import ActionServer, ActionClient
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from my_robot_msgs.action import RobotTask
from enum import Enum
from typing import Dict, List, Any, Callable
import asyncio
import threading
import time

class BehaviorStatus(Enum):
    """Status of behavior execution."""
    RUNNING = 1
    SUCCESS = 2
    FAILURE = 3
    CANCELLED = 4

class BehaviorNode:
    """Base class for behavior tree nodes."""
    def __init__(self, name: str):
        self.name = name
        self.status = BehaviorStatus.RUNNING

    async def tick(self) -> BehaviorStatus:
        """Execute one tick of the behavior."""
        raise NotImplementedError

class ActionNode(BehaviorNode):
    """Node that executes an action."""
    def __init__(self, name: str, action_client, action_type: str,
                 target_pose: Pose, description: str):
        super().__init__(name)
        self.action_client = action_client
        self.action_type = action_type
        self.target_pose = target_pose
        self.description = description
        self.is_executing = False

    async def tick(self) -> BehaviorStatus:
        """Execute the action."""
        if self.is_executing:
            # Check if action is still running
            return BehaviorStatus.RUNNING

        # Send action goal
        try:
            if not self.action_client.wait_for_server(timeout_sec=5.0):
                return BehaviorStatus.FAILURE

            goal_msg = RobotTask.Goal()
            goal_msg.task_type = self.action_type
            goal_msg.target_pose = self.target_pose
            goal_msg.description = self.description

            future = self.action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )

            # Wait for result with timeout
            rclpy.spin_until_future_complete(self.action_client, future, timeout_sec=30.0)
            goal_handle = future.result()

            if goal_handle is None:
                return BehaviorStatus.FAILURE

            # Get result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.action_client, result_future, timeout_sec=30.0)
            result = result_future.result().result

            return BehaviorStatus.SUCCESS if result.success else BehaviorStatus.FAILURE

        except Exception as e:
            print(f"Action execution error: {e}")
            return BehaviorStatus.FAILURE

    def feedback_callback(self, feedback_msg):
        """Handle action feedback."""
        print(f"Action feedback: {feedback_msg.status}, {feedback_msg.progress:.1f}%")

class SequenceNode(BehaviorNode):
    """Node that executes children in sequence."""
    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children
        self.current_child_index = 0

    async def tick(self) -> BehaviorStatus:
        """Execute children in sequence."""
        while self.current_child_index < len(self.children):
            child = self.children[self.current_child_index]
            status = await child.tick()

            if status == BehaviorStatus.FAILURE:
                # Reset for next time
                self.current_child_index = 0
                return BehaviorStatus.FAILURE
            elif status == BehaviorStatus.RUNNING:
                return BehaviorStatus.RUNNING
            elif status == BehaviorStatus.SUCCESS:
                self.current_child_index += 1
            elif status == BehaviorStatus.CANCELLED:
                self.current_child_index = 0
                return BehaviorStatus.CANCELLED

        # All children succeeded
        self.current_child_index = 0
        return BehaviorStatus.SUCCESS

class SelectorNode(BehaviorNode):
    """Node that tries children until one succeeds."""
    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children
        self.current_child_index = 0

    async def tick(self) -> BehaviorStatus:
        """Try children until one succeeds."""
        while self.current_child_index < len(self.children):
            child = self.children[self.current_child_index]
            status = await child.tick()

            if status == BehaviorStatus.SUCCESS:
                # Reset for next time
                self.current_child_index = 0
                return BehaviorStatus.SUCCESS
            elif status == BehaviorStatus.RUNNING:
                return BehaviorStatus.RUNNING
            elif status == BehaviorStatus.FAILURE:
                self.current_child_index += 1
            elif status == BehaviorStatus.CANCELLED:
                self.current_child_index = 0
                return BehaviorStatus.CANCELLED

        # All children failed
        self.current_child_index = 0
        return BehaviorStatus.FAILURE

class DecoratorNode(BehaviorNode):
    """Base class for decorator nodes."""
    def __init__(self, name: str, child: BehaviorNode):
        super().__init__(name)
        self.child = child

class InverterNode(DecoratorNode):
    """Decorator that inverts the result of its child."""
    async def tick(self) -> BehaviorStatus:
        """Invert the child's result."""
        status = await self.child.tick()

        if status == BehaviorStatus.SUCCESS:
            return BehaviorStatus.FAILURE
        elif status == BehaviorStatus.FAILURE:
            return BehaviorStatus.SUCCESS
        else:
            return status

class ActionBehaviorTreeServer(Node):
    def __init__(self):
        super().__init__('action_behavior_tree_server')

        # Action server
        self._action_server = ActionServer(
            self,
            RobotTask,
            'behavior_tree_task',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Action client for internal action execution
        self.action_client = ActionClient(
            self,
            RobotTask,
            'robot_task'
        )

        # Robot state
        self.current_pose = Pose()
        self.behavior_tree = None
        self.is_executing = False

        # Subscriptions
        self.pose_sub = self.create_subscription(
            Pose, 'current_pose', self.pose_callback, 10
        )

        self.get_logger().info('Action Behavior Tree Server initialized')

    def pose_callback(self, msg):
        """Update current pose."""
        self.current_pose = msg

    def goal_callback(self, goal_request):
        """Accept or reject goals."""
        self.get_logger().info(f'Received behavior tree goal: {goal_request.description}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancel requests."""
        self.get_logger().info('Behavior tree cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute behavior tree goal."""
        self.get_logger().info('Executing behavior tree goal')

        # Create a behavior tree based on the goal
        self.behavior_tree = self.create_behavior_tree(goal_handle.request)

        if self.behavior_tree is None:
            result = RobotTask.Result()
            result.success = False
            result.message = 'Could not create behavior tree'
            result.execution_time = 0.0
            return result

        # Execute the behavior tree
        start_time = time.time()
        self.is_executing = True

        try:
            while self.is_executing and not goal_handle.is_cancel_requested:
                status = await self.behavior_tree.tick()

                if status == BehaviorStatus.SUCCESS:
                    self.get_logger().info('Behavior tree completed successfully')
                    break
                elif status == BehaviorStatus.FAILURE:
                    self.get_logger().info('Behavior tree failed')
                    break
                elif status == BehaviorStatus.CANCELLED:
                    self.get_logger().info('Behavior tree cancelled')
                    break
                elif status == BehaviorStatus.RUNNING:
                    # Continue execution
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            # Prepare result
            result = RobotTask.Result()
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'Behavior tree cancelled'
            elif status == BehaviorStatus.SUCCESS:
                result.success = True
                result.message = 'Behavior tree completed successfully'
            else:
                result.success = False
                result.message = f'Behavior tree failed with status: {status}'

            result.execution_time = time.time() - start_time

        except Exception as e:
            self.get_logger().error(f'Behavior tree execution error: {e}')
            result = RobotTask.Result()
            result.success = False
            result.message = f'Execution error: {str(e)}'
            result.execution_time = time.time() - start_time

        finally:
            self.is_executing = False

        return result

    def create_behavior_tree(self, goal_request) -> Optional[BehaviorNode]:
        """Create behavior tree based on goal request."""
        try:
            if goal_request.task_type == 'patrol':
                return self.create_patrol_behavior_tree(goal_request)
            elif goal_request.task_type == 'delivery':
                return self.create_delivery_behavior_tree(goal_request)
            elif goal_request.task_type == 'inspection':
                return self.create_inspection_behavior_tree(goal_request)
            else:
                return self.create_generic_behavior_tree(goal_request)
        except Exception as e:
            self.get_logger().error(f'Error creating behavior tree: {e}')
            return None

    def create_patrol_behavior_tree(self, goal_request) -> BehaviorNode:
        """Create patrol behavior tree."""
        # Define patrol locations
        patrol_locations = [
            self.create_location_pose(1.0, 0.0, 0.0),
            self.create_location_pose(2.0, 1.0, 0.0),
            self.create_location_pose(0.5, 2.0, 0.0),
            self.create_location_pose(0.0, 0.0, 0.0)  # Return to start
        ]

        # Create navigation actions for each location
        navigation_actions = []
        for i, location in enumerate(patrol_locations):
            action_node = ActionNode(
                f'navigate_to_patrol_point_{i}',
                self.action_client,
                'navigation',
                location,
                f'Patrol point {i+1}'
            )
            navigation_actions.append(action_node)

        # Create sequence: navigate to each patrol point
        patrol_sequence = SequenceNode('patrol_sequence', navigation_actions)

        # Create selector with patrol sequence
        root_selector = SelectorNode('patrol_root', [patrol_sequence])

        return root_selector

    def create_delivery_behavior_tree(self, goal_request) -> BehaviorNode:
        """Create delivery behavior tree."""
        # Delivery steps: navigate to pickup, pick up, navigate to dropoff, drop off
        pickup_action = ActionNode(
            'navigate_to_pickup',
            self.action_client,
            'navigation',
            self.create_location_pose(1.0, 0.0, 0.0),
            'Navigate to pickup location'
        )

        pickup_object = ActionNode(
            'pickup_object',
            self.action_client,
            'manipulation',
            self.create_location_pose(1.0, 0.0, 0.0),
            'Pick up object'
        )

        navigate_to_dropoff = ActionNode(
            'navigate_to_dropoff',
            self.action_client,
            'navigation',
            self.create_location_pose(3.0, 2.0, 0.0),
            'Navigate to dropoff location'
        )

        dropoff_object = ActionNode(
            'dropoff_object',
            self.action_client,
            'manipulation',
            self.create_location_pose(3.0, 2.0, 0.0),
            'Drop off object'
        )

        # Create sequence: pickup -> navigate -> dropoff
        delivery_sequence = SequenceNode('delivery_sequence', [
            pickup_action,
            pickup_object,
            navigate_to_dropoff,
            dropoff_object
        ])

        return delivery_sequence

    def create_inspection_behavior_tree(self, goal_request) -> BehaviorNode:
        """Create inspection behavior tree."""
        # Inspection steps: navigate to inspection point, inspect, repeat
        inspection_points = [
            self.create_location_pose(1.0, 0.0, 0.0),
            self.create_location_pose(2.0, 1.0, 0.0),
            self.create_location_pose(0.5, 2.0, 0.0)
        ]

        inspection_actions = []
        for i, point in enumerate(inspection_points):
            navigate_action = ActionNode(
                f'navigate_to_inspection_point_{i}',
                self.action_client,
                'navigation',
                point,
                f'Navigate to inspection point {i+1}'
            )

            inspect_action = ActionNode(
                f'inspect_point_{i}',
                self.action_client,
                'inspection',
                point,
                f'Inspect area at point {i+1}'
            )

            # Sequence: navigate -> inspect
            point_sequence = SequenceNode(f'inspection_point_{i}_sequence', [
                navigate_action,
                inspect_action
            ])

            inspection_actions.append(point_sequence)

        # Create sequence of inspection points
        inspection_sequence = SequenceNode('inspection_sequence', inspection_actions)

        return inspection_sequence

    def create_generic_behavior_tree(self, goal_request) -> BehaviorNode:
        """Create generic behavior tree."""
        # Generic action based on goal request
        action_node = ActionNode(
            'generic_action',
            self.action_client,
            goal_request.task_type,
            goal_request.target_pose,
            goal_request.description
        )

        return action_node

    def create_location_pose(self, x: float, y: float, z: float) -> Pose:
        """Create a pose for a location."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0  # No rotation
        return pose

def main(args=None):
    rclpy.init(args=args)
    server = ActionBehaviorTreeServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Shutting down Action Behavior Tree Server')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive ROS 2 action execution for robotics:

- **Action Fundamentals**: Understanding the action communication pattern
- **Custom Action Definitions**: Creating action message types for robotics
- **Action Servers**: Implementing servers for long-running robotic tasks
- **Action Clients**: Creating clients to send goals and receive feedback
- **Advanced Action Patterns**: Multi-step and complex action execution
- **LLM Integration**: Connecting LLM cognitive planning with action execution
- **Behavior Trees**: Using actions within behavior tree architectures

Actions provide a powerful mechanism for implementing complex, long-running robotic behaviors with proper feedback and cancellation handling.

## Exercises

1. Create a custom action for a specific robotic task (e.g., docking, charging)
2. Implement an action server for your robot's main capabilities
3. Create an action client that sends goals based on sensor input
4. Integrate LLM planning with action execution
5. Build a behavior tree using actions as primitive behaviors

## Quiz

1. What makes ROS 2 actions different from services?
   a) Actions are faster
   b) Actions provide feedback and can be canceled during execution
   c) Actions use less memory
   d) Actions are simpler to implement

2. What are the three components of an action definition?
   a) Request, Response, Update
   b) Goal, Result, Feedback
   c) Start, Middle, End
   d) Input, Process, Output

3. When should you use actions instead of topics or services?
   a) For fire-and-forget messages
   b) For long-running tasks that need feedback
   c) For simple queries
   d) For high-frequency data streaming

## Mini-Project: Action-Based Robot System

Create a complete action-based robot system with:
1. Custom action definitions for navigation, manipulation, and inspection
2. Action servers that implement these behaviors
3. Action clients that can send goals and handle feedback
4. Integration with LLM cognitive planning
5. Behavior trees using actions as primitives
6. Testing with various complex tasks
7. Performance monitoring and error handling