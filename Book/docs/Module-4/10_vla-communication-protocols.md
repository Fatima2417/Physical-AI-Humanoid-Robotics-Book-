---
sidebar_position: 10
---

# VLA Communication Protocols

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement communication protocols for Vision-Language-Action systems
- Understand different communication patterns for multi-modal robotics
- Implement real-time communication with appropriate QoS settings
- Optimize message formats for efficient VLA communication
- Handle communication failures and implement resilience strategies
- Validate communication performance and reliability

## Introduction to VLA Communication Protocols

Communication protocols in Vision-Language-Action (VLA) systems are critical for ensuring that information flows effectively between perception, cognition, and action components. These protocols must handle different data types, timing requirements, and reliability needs while maintaining real-time performance.

### Communication Protocol Architecture

```
VLA Communication Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision        │◄───┤   Communication │───►│   Language      │
│   (Sensors,     │    │   Protocol      │    │   (Natural      │
│   Perception)   │    │   Layer         │    │   Language,     │
│                 │    │                 │    │   Planning)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Action        │
                         │   (Execution,   │
                         │   Control)      │
                         └─────────────────┘
```

### Key Communication Requirements

1. **Real-time Performance**: Low-latency communication for responsive systems
2. **Data Integrity**: Reliable delivery of critical information
3. **Bandwidth Efficiency**: Optimized message formats for different data types
4. **Synchronization**: Proper timing alignment between components
5. **Scalability**: Support for multiple concurrent communications
6. **Fault Tolerance**: Graceful handling of communication failures

## ROS 2 Communication Patterns for VLA

### Publisher-Subscriber Pattern

The publisher-subscriber pattern is fundamental to ROS 2 communication, enabling asynchronous communication between VLA components.

```python
# vla_publisher_subscriber.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, Twist
from builtin_interfaces.msg import Time
import time
from typing import Dict, Any, Optional
import threading
import queue

class VLAPublisherSubscriberNode(Node):
    def __init__(self):
        super().__init__('vla_publisher_subscriber')

        # Publishers for different data types
        self.vision_data_pub = self.create_publisher(String, 'vla/vision_data', 10)
        self.language_result_pub = self.create_publisher(String, 'vla/language_result', 10)
        self.action_command_pub = self.create_publisher(Twist, 'vla/action_command', 10)
        self.vla_status_pub = self.create_publisher(String, 'vla/system_status', 10)

        # Subscribers for different data types
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )
        self.language_command_sub = self.create_subscription(
            String, 'natural_language_command', self.language_command_callback, 10
        )
        self.action_feedback_sub = self.create_subscription(
            String, 'action_feedback', self.action_feedback_callback, 10
        )

        # Quality of Service profiles for different data types
        self.vision_qos = rclpy.qos.QoSProfile(
            depth=5,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST
        )

        self.language_qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST
        )

        self.action_qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST
        )

        # Message queues for processing
        self.vision_queue = queue.Queue(maxsize=10)
        self.language_queue = queue.Queue(maxsize=10)
        self.action_queue = queue.Queue(maxsize=5)

        # Processing threads
        self.vision_processing_thread = threading.Thread(
            target=self.vision_processing_loop, daemon=True
        )
        self.language_processing_thread = threading.Thread(
            target=self.language_processing_loop, daemon=True
        )
        self.action_processing_thread = threading.Thread(
            target=self.action_processing_loop, daemon=True
        )

        # Start processing threads
        self.vision_processing_thread.start()
        self.language_processing_thread.start()
        self.action_processing_thread.start()

        # Performance tracking
        self.message_counts = {'vision': 0, 'language': 0, 'action': 0}
        self.latency_measurements = []

        self.get_logger().info('VLA Publisher-Subscriber Node initialized')

    def image_callback(self, msg):
        """Process image data with appropriate QoS."""
        try:
            # Add to processing queue
            if not self.vision_queue.full():
                self.vision_queue.put({
                    'data': msg,
                    'timestamp': time.time(),
                    'header': msg.header
                }, timeout=0.01)

                self.message_counts['vision'] += 1
            else:
                self.get_logger().warn('Vision queue full, dropping message')
        except queue.Full:
            self.get_logger().warn('Vision message dropped due to full queue')

    def detections_callback(self, msg):
        """Process detection data."""
        try:
            if not self.vision_queue.full():
                self.vision_queue.put({
                    'type': 'detections',
                    'data': msg,
                    'timestamp': time.time(),
                    'header': msg.header
                }, timeout=0.01)
        except queue.Full:
            self.get_logger().warn('Detections message dropped due to full queue')

    def language_command_callback(self, msg):
        """Process language command."""
        try:
            if not self.language_queue.full():
                self.language_queue.put({
                    'data': msg.data,
                    'timestamp': time.time()
                }, timeout=0.01)

                self.message_counts['language'] += 1
            else:
                self.get_logger().warn('Language queue full, dropping command')
        except queue.Full:
            self.get_logger().warn('Language command dropped due to full queue')

    def action_feedback_callback(self, msg):
        """Process action feedback."""
        try:
            if not self.action_queue.full():
                self.action_queue.put({
                    'data': msg.data,
                    'timestamp': time.time()
                }, timeout=0.01)

                self.message_counts['action'] += 1
            else:
                self.get_logger().warn('Action queue full, dropping feedback')
        except queue.Full:
            self.get_logger().warn('Action feedback dropped due to full queue')

    def vision_processing_loop(self):
        """Process vision data in separate thread."""
        while rclpy.ok():
            try:
                # Get vision data
                vision_data = self.vision_queue.get(timeout=0.1)

                start_time = time.time()

                # Process vision data
                processed_result = self.process_vision_data(vision_data['data'])

                # Calculate processing latency
                processing_time = time.time() - start_time
                latency = time.time() - vision_data['timestamp']

                self.latency_measurements.append(latency)
                if len(self.latency_measurements) > 100:
                    self.latency_measurements.pop(0)

                # Publish processed result
                result_msg = String()
                result_msg.data = processed_result
                self.vision_data_pub.publish(result_msg)

                # Log performance periodically
                if self.message_counts['vision'] % 50 == 0:
                    avg_latency = sum(self.latency_measurements) / len(self.latency_measurements)
                    self.get_logger().info(
                        f'Vision processing - Latency: {avg_latency*1000:.1f}ms, '
                        f'Rate: {self.message_counts["vision"]/self.get_time_since_start():.1f} Hz'
                    )

                self.vision_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Vision processing error: {e}')

    def language_processing_loop(self):
        """Process language data in separate thread."""
        while rclpy.ok():
            try:
                # Get language data
                language_data = self.language_queue.get(timeout=0.1)

                start_time = time.time()

                # Process language command
                processed_result = self.process_language_command(language_data['data'])

                # Calculate processing time
                processing_time = time.time() - start_time
                latency = time.time() - language_data['timestamp']

                # Publish result
                result_msg = String()
                result_msg.data = processed_result
                self.language_result_pub.publish(result_msg)

                # Log performance periodically
                if self.message_counts['language'] % 20 == 0:
                    self.get_logger().info(
                        f'Language processing - Time: {processing_time*1000:.1f}ms, '
                        f'Rate: {self.message_counts["language"]/self.get_time_since_start():.1f} Hz'
                    )

                self.language_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Language processing error: {e}')

    def action_processing_loop(self):
        """Process action commands in separate thread."""
        while rclpy.ok():
            try:
                # Get action data
                action_data = self.action_queue.get(timeout=0.1)

                start_time = time.time()

                # Process action command
                processed_result = self.process_action_command(action_data['data'])

                # Calculate processing time
                processing_time = time.time() - start_time

                # Publish action command if needed
                if 'cmd_vel' in processed_result:
                    cmd_msg = Twist()
                    # Parse command from result
                    self.action_cmd_pub.publish(cmd_msg)

                # Log performance periodically
                if self.message_counts['action'] % 10 == 0:
                    self.get_logger().info(
                        f'Action processing - Time: {processing_time*1000:.1f}ms, '
                        f'Rate: {self.message_counts["action"]/self.get_time_since_start():.1f} Hz'
                    )

                self.action_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Action processing error: {e}')

    def process_vision_data(self, vision_msg):
        """Process vision data and extract relevant information."""
        # In a real implementation, this would process images/detections
        # For simulation, return mock result
        return f"Processed vision data with {len(getattr(vision_msg, 'detections', []))} detections"

    def process_language_command(self, command):
        """Process natural language command."""
        # In a real implementation, this would use NLP
        # For simulation, return mock result
        return f"Processed command: {command[:50]}..."

    def process_action_command(self, action_feedback):
        """Process action feedback."""
        # In a real implementation, this would handle feedback
        # For simulation, return mock result
        return f"Processed action feedback: {action_feedback[:30]}..."

    def get_time_since_start(self):
        """Get time elapsed since node started."""
        return self.get_clock().now().nanoseconds / 1e9 - self.start_time if hasattr(self, 'start_time') else 1.0

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLAPublisherSubscriberNode()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        vla_node.get_logger().info('Shutting down VLA Publisher-Subscriber Node')
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Service-Based Communication

### Synchronous Request-Response Pattern

Services provide synchronous communication for operations that require immediate responses:

```python
# vla_services.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Trigger, SetBool
from vision_msgs.srv import DetectObjects
from geometry_msgs.srv import GetPlan
from std_msgs.msg import String
import time
from typing import Dict, Any

class VLAServicesNode(Node):
    def __init__(self):
        super().__init__('vla_services')

        # Service servers
        self.vision_analysis_srv = self.create_service(
            Trigger, 'vla/analyze_vision', self.vision_analysis_callback
        )
        self.language_understanding_srv = self.create_service(
            String, 'vla/understand_language', self.language_understanding_callback
        )
        self.action_verification_srv = self.create_service(
            String, 'vla/verify_action', self.action_verification_callback
        )
        self.vla_plan_srv = self.create_service(
            String, 'vla/generate_plan', self.vla_plan_callback
        )

        # Service clients
        self.vision_client = self.create_client(DetectObjects, 'detect_objects')
        self.navigation_client = self.create_client(GetPlan, 'global_plan')
        self.perception_client = self.create_client(Trigger, 'run_perception')

        # Wait for services to be available
        while not self.vision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Vision service not available, waiting again...')

        while not self.navigation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Navigation service not available, waiting again...')

        # System state
        self.service_stats = {
            'vision_calls': 0,
            'language_calls': 0,
            'action_calls': 0,
            'plan_calls': 0
        }

        # Service performance tracking
        self.service_response_times = {
            'vision': [],
            'language': [],
            'action': [],
            'plan': []
        }

        self.get_logger().info('VLA Services Node initialized')

    def vision_analysis_callback(self, request, response):
        """Handle vision analysis service request."""
        start_time = time.time()

        self.service_stats['vision_calls'] += 1

        try:
            # Perform vision analysis (simplified)
            analysis_result = self.perform_vision_analysis()

            response.success = True
            response.message = analysis_result

            # Track response time
            response_time = time.time() - start_time
            self.service_response_times['vision'].append(response_time)

            if len(self.service_response_times['vision']) > 100:
                self.service_response_times['vision'].pop(0)

            self.get_logger().info(f'Vision analysis completed in {response_time:.3f}s')

        except Exception as e:
            response.success = False
            response.message = f'Vision analysis failed: {str(e)}'
            self.get_logger().error(f'Vision analysis error: {e}')

        return response

    def language_understanding_callback(self, request, response):
        """Handle language understanding service request."""
        start_time = time.time()

        self.service_stats['language_calls'] += 1

        try:
            # Process natural language command
            understanding_result = self.process_language_request(request.data)

            response.success = True
            response.message = understanding_result

            # Track response time
            response_time = time.time() - start_time
            self.service_response_times['language'].append(response_time)

            if len(self.service_response_times['language']) > 100:
                self.service_response_times['language'].pop(0)

            self.get_logger().info(f'Language understanding completed in {response_time:.3f}s')

        except Exception as e:
            response.success = False
            response.message = f'Language understanding failed: {str(e)}'
            self.get_logger().error(f'Language understanding error: {e}')

        return response

    def action_verification_callback(self, request, response):
        """Handle action verification service request."""
        start_time = time.time()

        self.service_stats['action_calls'] += 1

        try:
            # Verify action feasibility
            verification_result = self.verify_action_feasibility(request.data)

            response.success = True
            response.message = verification_result

            # Track response time
            response_time = time.time() - start_time
            self.service_response_times['action'].append(response_time)

            if len(self.service_response_times['action']) > 100:
                self.service_response_times['action'].pop(0)

            self.get_logger().info(f'Action verification completed in {response_time:.3f}s')

        except Exception as e:
            response.success = False
            response.message = f'Action verification failed: {str(e)}'
            self.get_logger().error(f'Action verification error: {e}')

        return response

    def vla_plan_callback(self, request, response):
        """Handle VLA planning service request."""
        start_time = time.time()

        self.service_stats['plan_calls'] += 1

        try:
            # Generate VLA plan
            plan_result = self.generate_vla_plan(request.data)

            response.success = True
            response.message = plan_result

            # Track response time
            response_time = time.time() - start_time
            self.service_response_times['plan'].append(response_time)

            if len(self.service_response_times['plan']) > 100:
                self.service_response_times['plan'].pop(0)

            self.get_logger().info(f'VLA planning completed in {response_time:.3f}s')

        except Exception as e:
            response.success = False
            response.message = f'VLA planning failed: {str(e)}'
            self.get_logger().error(f'VLA planning error: {e}')

        return response

    def perform_vision_analysis(self) -> str:
        """Perform vision analysis (simulated)."""
        # In a real implementation, this would analyze current scene
        # For simulation, return mock analysis
        return "Scene analysis complete: detected 3 objects, 2 obstacles, clear path forward"

    def process_language_request(self, command: str) -> str:
        """Process language request (simulated)."""
        # In a real implementation, this would use NLP
        # For simulation, return mock result
        return f"Command '{command}' interpreted as navigation task with obstacle avoidance"

    def verify_action_feasibility(self, action_description: str) -> str:
        """Verify action feasibility (simulated)."""
        # In a real implementation, this would check robot capabilities and environment
        # For simulation, return mock result
        return f"Action '{action_description}' is feasible with current configuration"

    def generate_vla_plan(self, task_description: str) -> str:
        """Generate VLA plan (simulated)."""
        # In a real implementation, this would create detailed execution plan
        # For simulation, return mock plan
        return f"VLA plan generated for: {task_description} with 5 steps and safety checks"

    def get_service_statistics(self):
        """Get service performance statistics."""
        stats = {}
        for service_type, times in self.service_response_times.items():
            if times:
                avg_time = sum(times) / len(times)
                stats[f'{service_type}_avg_time'] = avg_time
                stats[f'{service_type}_count'] = len(times)
            else:
                stats[f'{service_type}_avg_time'] = 0.0
                stats[f'{service_type}_count'] = 0

        return stats

    def call_vision_analysis_service(self):
        """Call vision analysis service."""
        if not self.vision_client.service_is_ready():
            self.get_logger().warn('Vision service not ready')
            return None

        request = Trigger.Request()
        future = self.vision_client.call_async(request)

        # Wait for response (with timeout)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.done():
            response = future.result()
            self.get_logger().info(f'Vision analysis response: {response.message}')
            return response
        else:
            self.get_logger().error('Vision analysis service call timed out')
            return None

    def call_language_understanding_service(self, command: str):
        """Call language understanding service."""
        if not self.language_client.service_is_ready():
            self.get_logger().warn('Language service not ready')
            return None

        request = String.Request()
        request.data = command
        future = self.language_client.call_async(request)

        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.done():
            response = future.result()
            self.get_logger().info(f'Language understanding response: {response.message}')
            return response
        else:
            self.get_logger().error('Language understanding service call timed out')
            return None

def main(args=None):
    rclpy.init(args=args)
    services_node = VLAServicesNode()

    try:
        rclpy.spin(services_node)
    except KeyboardInterrupt:
        services_node.get_logger().info('Shutting down VLA Services Node')
    finally:
        services_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Action-Based Communication

### Asynchronous Goal-Response Pattern

Actions provide a goal-response communication pattern with feedback for long-running operations:

```python
# vla_actions.py
import rclpy
from rclpy.action import ActionServer, ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from action_msgs.msg import GoalStatus
from vla_msgs.action import VisionTask, LanguageTask, ActionTask
import time
from typing import Optional
import threading

class VLAActionServer(Node):
    def __init__(self):
        super().__init__('vla_action_server')

        # Action servers
        self.vision_action_server = ActionServer(
            self,
            VisionTask,
            'vla/vision_task',
            self.execute_vision_task
        )
        self.language_action_server = ActionServer(
            self,
            LanguageTask,
            'vla/language_task',
            self.execute_language_task
        )
        self.action_action_server = ActionServer(
            self,
            ActionTask,
            'vla/action_task',
            self.execute_action_task
        )

        # Action clients for internal coordination
        self.vision_action_client = ActionClient(self, VisionTask, 'vla/vision_task')
        self.language_action_client = ActionClient(self, LanguageTask, 'vla/language_task')

        # System state
        self.active_vision_goals = {}
        self.active_language_goals = {}
        self.active_action_goals = {}

        # Performance tracking
        self.action_execution_times = []
        self.action_success_rates = {'vision': 0.0, 'language': 0.0, 'action': 0.0}

        self.get_logger().info('VLA Action Server initialized')

    def execute_vision_task(self, goal_handle):
        """Execute vision task with feedback."""
        self.get_logger().info(f'Executing vision task: {goal_handle.request.task_type}')

        task_type = goal_handle.request.task_type
        parameters = goal_handle.request.parameters

        # Track active goal
        goal_id = goal_handle.goal_id.uuid
        self.active_vision_goals[goal_id] = {
            'start_time': time.time(),
            'task_type': task_type
        }

        feedback_msg = VisionTask.Feedback()
        result = VisionTask.Result()

        try:
            # Simulate vision processing with feedback
            total_steps = 10
            for step in range(total_steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Vision task canceled'

                    # Remove from active goals
                    self.active_vision_goals.pop(goal_id, None)
                    return result

                # Update feedback
                feedback_msg.progress = float(step) / total_steps * 100.0
                feedback_msg.status = f'Processing step {step+1}/{total_steps}'
                feedback_msg.remaining_objects = total_steps - step - 1

                goal_handle.publish_feedback(feedback_msg)

                # Simulate processing time
                time.sleep(0.1)

            # Task completed successfully
            result.success = True
            result.message = f'Vision task {task_type} completed successfully'
            result.processing_time = time.time() - self.active_vision_goals[goal_id]['start_time']
            result.objects_detected = 3  # Simulated result

            goal_handle.succeed()

            # Update performance metrics
            self.action_execution_times.append(result.processing_time)
            if len(self.action_execution_times) > 100:
                self.action_execution_times.pop(0)

            # Remove from active goals
            self.active_vision_goals.pop(goal_id, None)

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f'Vision task failed: {str(e)}'

            # Remove from active goals
            self.active_vision_goals.pop(goal_id, None)

        return result

    def execute_language_task(self, goal_handle):
        """Execute language task with feedback."""
        self.get_logger().info(f'Executing language task: {goal_handle.request.task_type}')

        task_type = goal_handle.request.task_type
        command = goal_handle.request.command

        goal_id = goal_handle.goal_id.uuid
        self.active_language_goals[goal_id] = {
            'start_time': time.time(),
            'task_type': task_type
        }

        feedback_msg = LanguageTask.Feedback()
        result = LanguageTask.Result()

        try:
            # Simulate language processing with feedback
            total_processing_steps = 5
            for step in range(total_processing_steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Language task canceled'

                    self.active_language_goals.pop(goal_id, None)
                    return result

                # Update feedback
                feedback_msg.progress = float(step) / total_processing_steps * 100.0
                feedback_msg.status = f'Understanding command step {step+1}/{total_processing_steps}'
                feedback_msg.interpretation_confidence = 0.7 + (step * 0.06)  # Increase confidence with each step

                goal_handle.publish_feedback(feedback_msg)

                # Simulate processing
                time.sleep(0.2)

            # Task completed
            result.success = True
            result.message = f'Language task {task_type} completed: {command[:30]}...'
            result.processing_time = time.time() - self.active_language_goals[goal_id]['start_time']
            result.understood_command = command
            result.intent = self.determine_intent(command)

            goal_handle.succeed()

            # Update success rate
            successful_tasks = sum(1 for r in self.action_results if r.success)
            total_tasks = len(self.action_results)
            self.action_success_rates['language'] = successful_tasks / total_tasks if total_tasks > 0 else 0.0

            self.active_language_goals.pop(goal_id, None)

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f'Language task failed: {str(e)}'
            self.active_language_goals.pop(goal_id, None)

        return result

    def execute_action_task(self, goal_handle):
        """Execute action task with feedback."""
        self.get_logger().info(f'Executing action task: {goal_handle.request.task_type}')

        task_type = goal_handle.request.task_type
        parameters = goal_handle.request.parameters

        goal_id = goal_handle.goal_id.uuid
        self.active_action_goals[goal_id] = {
            'start_time': time.time(),
            'task_type': task_type
        }

        feedback_msg = ActionTask.Feedback()
        result = ActionTask.Result()

        try:
            # Simulate action execution with feedback
            total_execution_steps = 8
            for step in range(total_execution_steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Action task canceled'
                    self.active_action_goals.pop(goal_id, None)
                    return result

                # Update feedback
                feedback_msg.progress = float(step) / total_execution_steps * 100.0
                feedback_msg.status = f'Executing action step {step+1}/{total_execution_steps}'
                feedback_msg.executing_action = f'step_{step+1}'

                goal_handle.publish_feedback(feedback_msg)

                # Simulate action execution
                time.sleep(0.15)

            # Task completed
            result.success = True
            result.message = f'Action task {task_type} completed successfully'
            result.execution_time = time.time() - self.active_action_goals[goal_id]['start_time']
            result.completed_steps = total_execution_steps

            goal_handle.succeed()
            self.active_action_goals.pop(goal_id, None)

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f'Action task failed: {str(e)}'
            self.active_action_goals.pop(goal_id, None)

        return result

    def determine_intent(self, command: str) -> str:
        """Determine intent from command (simplified)."""
        command_lower = command.lower()

        if any(word in command_lower for word in ['go', 'move', 'navigate', 'go to']):
            return 'navigation'
        elif any(word in command_lower for word in ['grasp', 'pick', 'grasp', 'take']):
            return 'manipulation'
        elif any(word in command_lower for word in ['describe', 'what do you see', 'look']):
            return 'perception'
        else:
            return 'unknown'

    def get_active_goals_status(self) -> Dict:
        """Get status of all active goals."""
        return {
            'vision_goals': len(self.active_vision_goals),
            'language_goals': len(self.active_language_goals),
            'action_goals': len(self.active_action_goals),
            'average_execution_time': sum(self.action_execution_times) / len(self.action_execution_times) if self.action_execution_times else 0.0
        }

    def cancel_all_goals(self):
        """Cancel all active goals."""
        # In a real implementation, you'd cancel all active goals
        # This is a simplified version
        self.active_vision_goals.clear()
        self.active_language_goals.clear()
        self.active_action_goals.clear()
        self.get_logger().info('All active goals canceled')

class VLAActionClient(Node):
    def __init__(self):
        super().__init__('vla_action_client')

        # Action clients
        self.vision_action_client = ActionClient(self, VisionTask, 'vla/vision_task')
        self.language_action_client = ActionClient(self, LanguageTask, 'vla/language_task')
        self.action_action_client = ActionClient(self, ActionTask, 'vla/action_task')

        # Result tracking
        self.action_results = []
        self.action_callbacks = {}

        self.get_logger().info('VLA Action Client initialized')

    def send_vision_task(self, task_type: str, parameters: Optional[Dict] = None) -> Optional[Future]:
        """Send vision task to server."""
        if not self.vision_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Vision action server not available')
            return None

        goal_msg = VisionTask.Goal()
        goal_msg.task_type = task_type
        goal_msg.parameters = parameters or {}

        send_goal_future = self.vision_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.vision_feedback_callback
        )

        # Store callback for this goal
        goal_uuid = send_goal_future.goal_id  # This would be the actual goal ID
        self.action_callbacks[goal_uuid] = {
            'type': 'vision',
            'start_time': time.time(),
            'task_type': task_type
        }

        send_goal_future.add_done_callback(self.vision_goal_response_callback)

        return send_goal_future

    def vision_feedback_callback(self, feedback_msg):
        """Handle vision task feedback."""
        self.get_logger().info(
            f'Vision task feedback: {feedback_msg.status}, '
            f'Progress: {feedback_msg.progress:.1f}%'
        )

    def vision_goal_response_callback(self, future):
        """Handle vision goal response."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Vision goal rejected')
            return

        self.get_logger().info('Vision goal accepted')

        # Get result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.vision_result_callback)

    def vision_result_callback(self, future):
        """Handle vision result."""
        result = future.result().result
        self.get_logger().info(f'Vision task result: {result.message}')

        # Store result
        self.action_results.append({
            'type': 'vision',
            'success': result.success,
            'message': result.message,
            'processing_time': result.processing_time,
            'timestamp': time.time()
        })

        if len(self.action_results) > 100:
            self.action_results.pop(0)

    def send_language_task(self, task_type: str, command: str) -> Optional[Future]:
        """Send language task to server."""
        if not self.language_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Language action server not available')
            return None

        goal_msg = LanguageTask.Goal()
        goal_msg.task_type = task_type
        goal_msg.command = command

        send_goal_future = self.language_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.language_feedback_callback
        )

        send_goal_future.add_done_callback(self.language_goal_response_callback)

        return send_goal_future

    def language_feedback_callback(self, feedback_msg):
        """Handle language task feedback."""
        self.get_logger().info(
            f'Language task feedback: {feedback_msg.status}, '
            f'Confidence: {feedback_msg.interpretation_confidence:.2f}'
        )

    def language_goal_response_callback(self, future):
        """Handle language goal response."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Language goal rejected')
            return

        self.get_logger().info('Language goal accepted')

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.language_result_callback)

    def language_result_callback(self, future):
        """Handle language result."""
        result = future.result().result
        self.get_logger().info(f'Language task result: {result.message}')

        self.action_results.append({
            'type': 'language',
            'success': result.success,
            'message': result.message,
            'processing_time': result.processing_time,
            'timestamp': time.time()
        })

        if len(self.action_results) > 100:
            self.action_results.pop(0)

    def send_action_task(self, task_type: str, parameters: Optional[Dict] = None) -> Optional[Future]:
        """Send action task to server."""
        if not self.action_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action action server not available')
            return None

        goal_msg = ActionTask.Goal()
        goal_msg.task_type = task_type
        goal_msg.parameters = parameters or {}

        send_goal_future = self.action_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.action_feedback_callback
        )

        send_goal_future.add_done_callback(self.action_goal_response_callback)

        return send_goal_future

    def action_feedback_callback(self, feedback_msg):
        """Handle action task feedback."""
        self.get_logger().info(
            f'Action task feedback: {feedback_msg.status}, '
            f'Progress: {feedback_msg.progress:.1f}%'
        )

    def action_goal_response_callback(self, future):
        """Handle action goal response."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Action goal rejected')
            return

        self.get_logger().info('Action goal accepted')

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.action_result_callback)

    def action_result_callback(self, future):
        """Handle action result."""
        result = future.result().result
        self.get_logger().info(f'Action task result: {result.message}')

        self.action_results.append({
            'type': 'action',
            'success': result.success,
            'message': result.message,
            'execution_time': result.execution_time,
            'timestamp': time.time()
        })

        if len(self.action_results) > 100:
            self.action_results.pop(0)

    def get_action_performance_metrics(self) -> Dict:
        """Get action performance metrics."""
        if not self.action_results:
            return {
                'success_rate': 0.0,
                'avg_vision_time': 0.0,
                'avg_language_time': 0.0,
                'avg_action_time': 0.0
            }

        total_tasks = len(self.action_results)
        successful_tasks = sum(1 for r in self.action_results if r['success'])

        success_rate = successful_tasks / total_tasks

        # Calculate average times by type
        vision_times = [r['processing_time'] for r in self.action_results if r['type'] == 'vision']
        language_times = [r['processing_time'] for r in self.action_results if r['type'] == 'language']
        action_times = [r['execution_time'] for r in self.action_results if r['type'] == 'action']

        return {
            'success_rate': success_rate,
            'avg_vision_time': sum(vision_times) / len(vision_times) if vision_times else 0.0,
            'avg_language_time': sum(language_times) / len(language_times) if language_times else 0.0,
            'avg_action_time': sum(action_times) / len(action_times) if action_times else 0.0
        }

def main(args=None):
    rclpy.init(args=args)

    # Create both server and client for demonstration
    action_server = VLAActionServer()
    action_client = VLAActionClient()

    try:
        # Example: Send a vision task
        import threading
        def send_test_tasks():
            time.sleep(2)  # Wait for servers to be ready

            # Send a vision task
            vision_future = action_client.send_vision_task('object_detection', {'confidence_threshold': 0.7})

            # Send a language task
            language_future = action_client.send_language_task('command_interpretation', 'Move forward slowly')

            # Send an action task
            action_future = action_client.send_action_task('navigation', {'target_x': 1.0, 'target_y': 1.0})

        # Start test tasks in background
        test_thread = threading.Thread(target=send_test_tasks, daemon=True)
        test_thread.start()

        # Run both nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(action_server)
        executor.add_node(action_client)

        executor.spin()

    except KeyboardInterrupt:
        action_server.get_logger().info('Shutting down VLA Action Server')
        action_client.get_logger().info('Shutting down VLA Action Client')
    finally:
        action_server.destroy_node()
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Communication Optimization Strategies

### Bandwidth Optimization

```python
# communication_optimization.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import time
from typing import Dict, Any, Optional

class VLACommunicationOptimizer(Node):
    def __init__(self):
        super().__init__('vla_communication_optimizer')

        # Publishers with optimized settings
        self.optimized_image_pub = self.create_publisher(CompressedImage, 'camera/image_optimized', 5)  # Lower depth for bandwidth
        self.optimized_detections_pub = self.create_publisher(Detection2DArray, 'detections_optimized', 5)
        self.optimized_command_pub = self.create_publisher(Twist, 'cmd_vel_optimized', 1)  # Minimal depth for commands

        # Subscribers with optimized QoS
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.optimized_image_callback,
            rclpy.qos.QoSProfile(
                depth=1,  # Minimal queue depth
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,  # Best effort for images
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST
            )
        )

        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback,
            rclpy.qos.QoSProfile(
                depth=10,  # Higher depth for commands
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,  # Reliable for commands
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST
            )
        )

        # CV Bridge for image compression
        self.cv_bridge = CvBridge()

        # Optimization parameters
        self.enable_image_compression = True
        self.image_compression_quality = 85  # JPEG quality (1-100)
        self.enable_detections_filtering = True
        self.detection_confidence_threshold = 0.7
        self.enable_command_batching = True
        self.command_batch_size = 3

        # Command batching
        self.command_batch = []
        self.batch_lock = threading.Lock()

        # Performance tracking
        self.message_sizes = {'original': [], 'optimized': []}
        self.compression_ratios = []
        self.transmission_rates = []

        # Timer for optimization monitoring
        self.optimization_timer = self.create_timer(1.0, self.optimization_monitor)

        self.get_logger().info('VLA Communication Optimizer initialized')

    def image_callback_optimized(self, msg):
        """Process image with optimization."""
        start_time = time.time()

        if self.enable_image_compression:
            # Compress image for transmission
            compressed_msg = self.compress_image(msg)
            self.optimized_image_pub.publish(compressed_msg)

            # Track compression effectiveness
            original_size = len(msg.data)
            compressed_size = len(compressed_msg.data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

            self.message_sizes['original'].append(original_size)
            self.message_sizes['optimized'].append(compressed_size)
            self.compression_ratios.append(compression_ratio)

            if len(self.compression_ratios) > 100:
                self.compression_ratios.pop(0)
                self.message_sizes['original'].pop(0)
                self.message_sizes['optimized'].pop(0)

        else:
            # Publish uncompressed image
            self.optimized_image_pub.publish(msg)

        processing_time = time.time() - start_time
        self.get_logger().debug(f'Image optimization took {processing_time*1000:.1f}ms')

    def compress_image(self, image_msg):
        """Compress image message using JPEG compression."""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Encode as JPEG with specified quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_compression_quality]
            result, encoded_image = cv2.imencode('.jpg', cv_image, encode_param)

            if result:
                # Create compressed image message
                compressed_msg = CompressedImage()
                compressed_msg.header = image_msg.header
                compressed_msg.format = "jpeg"
                compressed_msg.data = encoded_image.tobytes()

                return compressed_msg
            else:
                self.get_logger().error('Image compression failed')
                return image_msg  # Return original if compression fails

        except Exception as e:
            self.get_logger().error(f'Image compression error: {e}')
            return image_msg  # Return original if error occurs

    def detections_callback_optimized(self, msg):
        """Process detections with optimization."""
        if self.enable_detections_filtering:
            # Filter detections based on confidence
            filtered_detections = self.filter_detections_by_confidence(msg)

            # Publish filtered detections
            self.optimized_detections_pub.publish(filtered_detections)
        else:
            # Publish all detections
            self.optimized_detections_pub.publish(msg)

    def filter_detections_by_confidence(self, detections_msg):
        """Filter detections based on confidence threshold."""
        filtered_msg = Detection2DArray()
        filtered_msg.header = detections_msg.header

        for detection in detections_msg.detections:
            if detection.results:
                confidence = detection.results[0].hypothesis.score
                if confidence >= self.detection_confidence_threshold:
                    filtered_msg.detections.append(detection)

        return filtered_msg

    def command_callback(self, msg):
        """Process command with optimization."""
        command = msg.data

        if self.enable_command_batching:
            # Add to batch
            with self.batch_lock:
                self.command_batch.append({
                    'command': command,
                    'timestamp': time.time()
                })

                # Publish batch when it reaches the desired size
                if len(self.command_batch) >= self.command_batch_size:
                    self.publish_command_batch()
        else:
            # Process command immediately
            self.process_single_command(command)

    def publish_command_batch(self):
        """Publish batched commands."""
        if not self.command_batch:
            return

        # Combine commands into a single message (simplified approach)
        combined_commands = "; ".join([item['command'] for item in self.command_batch])

        batch_msg = String()
        batch_msg.data = f"BATCH: {combined_commands}"

        self.optimized_command_pub.publish(batch_msg)

        # Clear batch
        with self.batch_lock:
            self.command_batch.clear()

    def process_single_command(self, command):
        """Process single command (placeholder for actual processing)."""
        # In a real implementation, this would convert command to action
        # For simulation, just log the command
        self.get_logger().info(f'Processing command: {command}')

    def optimization_monitor(self):
        """Monitor and report optimization metrics."""
        if self.compression_ratios:
            avg_compression = sum(self.compression_ratios) / len(self.compression_ratios)
            self.get_logger().info(
                f'Compression - Ratio: {avg_compression:.2f}x, '
                f'Saved: {(1-1/avg_compression)*100:.1f}% bandwidth'
            )

    def adjust_compression_quality(self, new_quality: int):
        """Adjust image compression quality."""
        self.image_compression_quality = max(1, min(100, new_quality))
        self.get_logger().info(f'Image compression quality set to: {self.image_compression_quality}')

    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        self.detection_confidence_threshold = max(0.0, min(1.0, threshold))
        self.get_logger().info(f'Detection confidence threshold set to: {self.detection_confidence_threshold}')

    def enable_compression(self, enable: bool):
        """Enable or disable image compression."""
        self.enable_image_compression = enable
        self.get_logger().info(f'Image compression {"enabled" if enable else "disabled"}')

    def enable_filtering(self, enable: bool):
        """Enable or disable detection filtering."""
        self.enable_detections_filtering = enable
        self.get_logger().info(f'Detection filtering {"enabled" if enable else "disabled"}')

    def enable_batching(self, enable: bool):
        """Enable or disable command batching."""
        self.enable_command_batching = enable
        self.get_logger().info(f'Command batching {"enabled" if enable else "disabled"}')

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'compression_enabled': self.enable_image_compression,
            'filtering_enabled': self.enable_detections_filtering,
            'batching_enabled': self.enable_command_batching,
            'compression_quality': self.image_compression_quality,
            'confidence_threshold': self.detection_confidence_threshold
        }

        if self.compression_ratios:
            stats['avg_compression_ratio'] = sum(self.compression_ratios) / len(self.compression_ratios)
            stats['total_messages_saved'] = len(self.compression_ratios)

        if self.message_sizes['original']:
            original_total = sum(self.message_sizes['original'])
            optimized_total = sum(self.message_sizes['optimized'])
            bandwidth_saved = ((original_total - optimized_total) / original_total) * 100 if original_total > 0 else 0
            stats['bandwidth_saved_percent'] = bandwidth_saved

        return stats

def main(args=None):
    rclpy.init(args=args)
    optimizer = VLACommunicationOptimizer()

    try:
        rclpy.spin(optimizer)
    except KeyboardInterrupt:
        optimizer.get_logger().info('Shutting down VLA Communication Optimizer')
    finally:
        optimizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Fault Tolerance and Communication Reliability

### Implementing Resilient Communication

```python
# resilient_communication.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time
from typing import Dict, Any, Optional, Callable
import threading
import queue
from dataclasses import dataclass
from enum import Enum

class MessageStatus(Enum):
    SENT = 1
    ACKNOWLEDGED = 2
    FAILED = 3
    RETRIED = 4

@dataclass
class MessageInfo:
    """Information about a published message."""
    message_id: str
    timestamp: float
    content: Any
    status: MessageStatus
    retries: int = 0
    last_retry: Optional[float] = None

class VLACommunicationReliability(Node):
    def __init__(self):
        super().__init__('vla_communication_reliability')

        # Publishers with different reliability requirements
        self.critical_cmd_pub = self.create_publisher(
            Twist, 'cmd_vel_critical',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.status_pub = self.create_publisher(
            String, 'system_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self.debug_pub = self.create_publisher(
            String, 'debug_info',
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'critical_command', self.critical_command_callback,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.status_ack_sub = self.create_subscription(
            String, 'status_ack', self.status_ack_callback,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Message tracking
        self.sent_messages = {}  # message_id -> MessageInfo
        self.message_queue = queue.Queue()
        self.acknowledged_messages = set()
        self.failed_messages = []

        # Reliability parameters
        self.message_timeout = 5.0  # seconds
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds
        self.enable_acknowledgments = True

        # Performance tracking
        self.successful_transmissions = 0
        self.failed_transmissions = 0
        self.retried_transmissions = 0

        # Acknowledgment timeout timer
        self.ack_timer = self.create_timer(0.1, self.check_message_acknowledgments)

        # Message cleanup timer
        self.cleanup_timer = self.create_timer(10.0, self.cleanup_old_messages)

        self.get_logger().info('VLA Communication Reliability Node initialized')

    def critical_command_callback(self, msg):
        """Handle critical commands that require reliability."""
        command = msg.data
        self.get_logger().info(f'Received critical command: {command}')

        # Process critical command with reliability
        success = self.process_critical_command_with_reliability(msg)

        if success:
            self.successful_transmissions += 1
            self.get_logger().info('Critical command processed successfully')
        else:
            self.failed_transmissions += 1
            self.get_logger().error('Critical command processing failed')

        # Send acknowledgment
        if self.enable_acknowledgments:
            ack_msg = String()
            ack_msg.data = f"ACK:{msg.data[:20]}..."  # Acknowledge the command
            self.status_ack_pub.publish(ack_msg)

    def process_critical_command_with_reliability(self, msg) -> bool:
        """Process critical command with reliability mechanisms."""
        try:
            # Add message to tracking
            message_id = f"cmd_{int(time.time()*1000)}"  # Simple ID generation
            message_info = MessageInfo(
                message_id=message_id,
                timestamp=time.time(),
                content=msg,
                status=MessageStatus.SENT
            )
            self.sent_messages[message_id] = message_info

            # Process the command (in a real system, this would involve actual processing)
            # For simulation, we'll just return success
            success = self.execute_command(msg.data)

            if success:
                message_info.status = MessageStatus.ACKNOWLEDGED
                self.acknowledged_messages.add(message_id)
            else:
                message_info.status = MessageStatus.FAILED
                self.failed_messages.append(message_info)

            return success

        except Exception as e:
            self.get_logger().error(f'Critical command processing error: {e}')
            return False

    def execute_command(self, command: str) -> bool:
        """Execute the command (simulated)."""
        # In a real implementation, this would execute the actual command
        # For simulation, return success for most commands
        return True

    def status_ack_callback(self, msg):
        """Handle status acknowledgments."""
        ack_content = msg.data

        if ack_content.startswith('ACK:'):
            # Extract message ID from acknowledgment
            # In a real system, you'd have proper message ID tracking
            # For simulation, we'll just mark the latest message as acknowledged
            if self.sent_messages:
                latest_msg_id = max(self.sent_messages.keys(), key=lambda x: self.sent_messages[x].timestamp)
                if latest_msg_id in self.sent_messages:
                    self.sent_messages[latest_msg_id].status = MessageStatus.ACKNOWLEDGED
                    self.acknowledged_messages.add(latest_msg_id)
                    self.get_logger().info(f'Message {latest_msg_id} acknowledged')

    def check_message_acknowledgments(self):
        """Check for unacknowledged messages and retry if necessary."""
        current_time = time.time()

        for msg_id, msg_info in list(self.sent_messages.items()):
            if msg_info.status == MessageStatus.SENT:
                # Check if message has timed out
                if current_time - msg_info.timestamp > self.message_timeout:
                    if msg_info.retries < self.max_retries:
                        # Retry the message
                        self.retry_message(msg_id, msg_info)
                    else:
                        # Mark as failed after max retries
                        msg_info.status = MessageStatus.FAILED
                        self.failed_messages.append(msg_info)
                        self.sent_messages.pop(msg_id, None)
                        self.get_logger().error(f'Message {msg_id} failed after {self.max_retries} retries')

    def retry_message(self, msg_id: str, msg_info: MessageInfo):
        """Retry sending a message."""
        self.get_logger().warn(f'Retrying message {msg_id}, attempt {msg_info.retries + 1}/{self.max_retries}')

        # Update retry information
        msg_info.retries += 1
        msg_info.last_retry = time.time()
        msg_info.status = MessageStatus.RETRIED
        self.retried_transmissions += 1

        # In a real system, you'd republish the message
        # For simulation, we'll just update the timestamp
        msg_info.timestamp = time.time()

    def cleanup_old_messages(self):
        """Clean up old message tracking information."""
        current_time = time.time()
        cleanup_threshold = 30.0  # seconds

        for msg_id in list(self.sent_messages.keys()):
            msg_info = self.sent_messages[msg_id]
            if current_time - msg_info.timestamp > cleanup_threshold:
                if msg_info.status in [MessageStatus.ACKNOWLEDGED, MessageStatus.FAILED]:
                    del self.sent_messages[msg_id]

        # Clean up acknowledged messages set
        for msg_id in list(self.acknowledged_messages):
            if msg_id not in self.sent_messages:
                self.acknowledged_messages.discard(msg_id)

        # Log statistics periodically
        total_messages = self.successful_transmissions + self.failed_transmissions
        success_rate = (self.successful_transmissions / total_messages * 100) if total_messages > 0 else 0

        self.get_logger().info(
            f'Communication stats - Success: {self.successful_transmissions}, '
            f'Failed: {self.failed_transmissions}, '
            f'Retried: {self.retried_transmissions}, '
            f'Success Rate: {success_rate:.1f}%'
        )

    def send_reliable_message(self, publisher, message, message_type: str = "generic"):
        """Send a message with reliability tracking."""
        message_id = f"{message_type}_{int(time.time()*1000)}"

        # Add to tracking
        msg_info = MessageInfo(
            message_id=message_id,
            timestamp=time.time(),
            content=message,
            status=MessageStatus.SENT
        )
        self.sent_messages[message_id] = msg_info

        # Publish message
        publisher.publish(message)

        return message_id

    def enable_acknowledgments(self, enable: bool):
        """Enable or disable acknowledgment system."""
        self.enable_acknowledgments = enable
        self.get_logger().info(f'Acknowledgment system {"enabled" if enable else "disabled"}')

    def set_timeout(self, timeout_seconds: float):
        """Set message timeout."""
        self.message_timeout = timeout_seconds
        self.get_logger().info(f'Message timeout set to {timeout_seconds}s')

    def set_max_retries(self, max_retries: int):
        """Set maximum number of retries."""
        self.max_retries = max_retries
        self.get_logger().info(f'Maximum retries set to {max_retries}')

    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics."""
        total_messages = self.successful_transmissions + self.failed_transmissions
        success_rate = (self.successful_transmissions / total_messages * 100) if total_messages > 0 else 0

        return {
            'successful_transmissions': self.successful_transmissions,
            'failed_transmissions': self.failed_transmissions,
            'retried_transmissions': self.retried_transmissions,
            'total_transmissions': total_messages,
            'success_rate_percent': success_rate,
            'pending_messages': len(self.sent_messages) - len(self.acknowledged_messages),
            'tracked_messages': len(self.sent_messages)
        }

    def reset_statistics(self):
        """Reset reliability statistics."""
        self.successful_transmissions = 0
        self.failed_transmissions = 0
        self.retried_transmissions = 0
        self.sent_messages.clear()
        self.acknowledged_messages.clear()
        self.failed_messages.clear()
        self.get_logger().info('Reliability statistics reset')

def main(args=None):
    rclpy.init(args=args)
    reliability_node = VLACommunicationReliability()

    try:
        rclpy.spin(reliability_node)
    except KeyboardInterrupt:
        reliability_node.get_logger().info('Shutting down VLA Communication Reliability Node')
    finally:
        reliability_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive communication protocols for Vision-Language-Action systems:

- **ROS 2 Communication Patterns**: Publisher-subscriber, service, and action-based communication
- **Message Optimization**: Techniques for reducing bandwidth and improving performance
- **Reliability Mechanisms**: Fault tolerance and message acknowledgment systems
- **Performance Monitoring**: Tracking and optimizing communication efficiency

Effective communication protocols are essential for ensuring that VLA systems operate reliably and efficiently, especially in real-time robotic applications.

## Exercises

1. Implement publisher-subscriber communication for VLA components
2. Create service-based communication for synchronous operations
3. Design action-based communication for long-running tasks
4. Optimize message formats for your specific application
5. Implement reliability mechanisms for critical communications

## Quiz

1. What is the main advantage of using actions over services in ROS 2?
   a) Lower bandwidth usage
   b) Support for long-running operations with feedback
   c) Faster execution
   d) Simpler implementation

2. Which QoS policy is best for real-time sensor data?
   a) RELIABLE
   b) BEST_EFFORT
   c) PERSISTENT
   d) TRANSIENT_LOCAL

3. What does message compression primarily optimize?
   a) Processing time
   b) Network bandwidth
   c) Storage space
   d) Memory usage

## Mini-Project: Complete VLA Communication System

Create a complete VLA communication system with:
1. Publisher-subscriber communication for real-time data
2. Service-based communication for synchronous requests
3. Action-based communication for complex tasks
4. Message optimization techniques
5. Reliability and fault tolerance mechanisms
6. Performance monitoring and metrics
7. Testing with various communication patterns
8. Evaluation of communication efficiency and reliability