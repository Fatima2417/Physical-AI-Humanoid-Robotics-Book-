---
sidebar_position: 6
---

# VLA Architectures

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand different architectural patterns for Vision-Language-Action systems
- Design and implement modular VLA architectures for robotics applications
- Evaluate trade-offs between centralized vs distributed VLA architectures
- Implement real-time processing pipelines for VLA systems
- Integrate multiple VLA components into cohesive robotic systems
- Optimize VLA architectures for specific robotics tasks and constraints

## Introduction to VLA Architectures

Vision-Language-Action (VLA) architectures define how visual perception, natural language understanding, and robotic action components are organized and interact within a robotic system. The architecture choice significantly impacts system performance, scalability, modularity, and real-time capabilities.

### VLA System Architecture Patterns

```
VLA Architecture Patterns:

Centralized Architecture:
┌─────────────────────────────────────────────────────────┐
│                    Centralized VLA System               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Vision    │  │   Language  │  │     Action      │ │
│  │  Component  │  │  Component  │  │   Component     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         │               │                   │          │
│         └───────────────┼───────────────────┘          │
│                         │                              │
│                 ┌─────────────────┐                    │
│                 │  Centralized    │                    │
│                 │  Coordinator    │                    │
│                 │  (Decision &    │                    │
│                 │   Planning)     │                    │
│                 └─────────────────┘                    │
└─────────────────────────────────────────────────────────┘

Distributed Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision        │    │   Language      │    │   Action        │
│   Component     │    │   Component     │    │   Component     │
│   (Perception)  │    │   (Cognition)   │    │   (Execution)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   ROS 2         │
                         │   Middleware    │
                         │   (Topics,      │
                         │   Services,     │
                         │   Actions)      │
                         └─────────────────┘
```

## Centralized VLA Architecture

### Monolithic Approach

The centralized approach consolidates all VLA processing in a single system component:

```python
# centralized_vla.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import numpy as np
import json
import threading
import time
from typing import Dict, Any, Optional

class CentralizedVLANode(Node):
    def __init__(self):
        super().__init__('centralized_vla')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.response_pub = self.create_publisher(String, 'vla_response', 10)
        self.status_pub = self.create_publisher(String, 'vla_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )

        # Internal state
        self.latest_image = None
        self.latest_scan = None
        self.latest_detections = None
        self.command_queue = []
        self.is_processing = False

        # Processing parameters
        self.enable_vision_processing = True
        self.enable_language_understanding = True
        self.enable_action_generation = True

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        # Threading for processing
        self.processing_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

        self.get_logger().info('Centralized VLA Node initialized')

    def command_callback(self, msg):
        """Queue incoming commands for processing."""
        command = msg.data
        self.get_logger().info(f'Command received: {command}')

        with self.processing_lock:
            self.command_queue.append({
                'command': command,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })

    def image_callback(self, msg):
        """Update latest image."""
        self.latest_image = msg

    def scan_callback(self, msg):
        """Update latest scan data."""
        self.latest_scan = msg

    def detections_callback(self, msg):
        """Update latest detections."""
        self.latest_detections = msg

    def process_commands(self):
        """Process commands in a separate thread."""
        while rclpy.ok():
            try:
                with self.processing_lock:
                    if self.command_queue:
                        command_item = self.command_queue.pop(0)
                        command = command_item['command']

                if command:
                    start_time = time.time()

                    # Process the complete VLA pipeline
                    result = self.process_vla_pipeline(command)

                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)

                    if len(self.processing_times) > 100:
                        self.processing_times.pop(0)

                    # Log performance periodically
                    self.frame_count += 1
                    if self.frame_count % 20 == 0:  # Every 20 commands
                        avg_time = sum(self.processing_times) / len(self.processing_times)
                        self.get_logger().info(
                            f'VLA processing - Avg: {avg_time*1000:.1f}ms, '
                            f'Commands processed: {self.frame_count}'
                        )

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                self.get_logger().error(f'Command processing error: {e}')
                time.sleep(0.1)

    def process_vla_pipeline(self, command: str) -> Dict[str, Any]:
        """Complete VLA pipeline: Vision → Language → Action."""
        try:
            # 1. VISION: Process current visual information
            vision_context = self.process_vision_context()

            # 2. LANGUAGE: Interpret command with context
            intent = self.interpret_command_with_context(command, vision_context)

            # 3. ACTION: Generate appropriate action plan
            action_plan = self.generate_action_plan(intent, vision_context)

            # 4. EXECUTE: Execute the action plan
            success = self.execute_action_plan(action_plan)

            # 5. RESPOND: Generate response
            response = self.generate_response(command, success, action_plan)

            # Publish response
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            # Update status
            status_msg = String()
            status_msg.data = f'VLA processed: {command[:50]}...' if len(command) > 50 else command
            self.status_pub.publish(status_msg)

            return {
                'success': success,
                'command': command,
                'intent': intent,
                'action_plan': action_plan,
                'response': response
            }

        except Exception as e:
            self.get_logger().error(f'VLA pipeline error: {e}')
            error_response = f"I'm sorry, I encountered an error processing your command: {str(e)}"

            error_msg = String()
            error_msg.data = error_response
            self.response_pub.publish(error_msg)

            return {
                'success': False,
                'command': command,
                'error': str(e)
            }

    def process_vision_context(self) -> Dict[str, Any]:
        """Process visual information to create context."""
        context = {
            'objects': [],
            'obstacles': [],
            'environment': 'indoor',  # Default assumption
            'timestamp': time.time()
        }

        # Process detections if available
        if self.latest_detections:
            for detection in self.latest_detections.detections:
                if detection.results:
                    class_name = detection.results[0].hypothesis.class_id
                    confidence = detection.results[0].hypothesis.score
                    position = detection.bbox.center

                    if confidence > 0.5:  # Confidence threshold
                        context['objects'].append({
                            'class': class_name,
                            'confidence': confidence,
                            'position': {'x': position.x, 'y': position.y}
                        })

        # Process laser scan for obstacles
        if self.latest_scan:
            obstacles = []
            for i, range_val in enumerate(self.latest_scan.ranges):
                if self.latest_scan.range_min < range_val < self.latest_scan.range_max:
                    angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
                    x = range_val * math.cos(angle)
                    y = range_val * math.sin(angle)

                    if range_val < 1.0:  # Obstacle threshold
                        obstacles.append({'x': x, 'y': y, 'distance': range_val})

            context['obstacles'] = obstacles

        return context

    def interpret_command_with_context(self, command: str, vision_context: Dict) -> str:
        """Interpret command using visual context."""
        command_lower = command.lower()

        # Enhanced command interpretation with context
        if 'go to' in command_lower or 'move to' in command_lower:
            # Extract target from command
            target = self.extract_target_from_command(command_lower, vision_context)
            return f'navigate_to_{target}'

        elif 'pick up' in command_lower or 'grasp' in command_lower:
            target = self.extract_target_from_command(command_lower, vision_context)
            return f'grasp_{target}'

        elif 'avoid' in command_lower or 'navigate around' in command_lower:
            return 'avoid_obstacles'

        elif 'describe' in command_lower or 'what do you see' in command_lower:
            return 'describe_scene'

        else:
            # Use vision context to determine appropriate action
            if vision_context['objects']:
                # If there are objects, default to navigation toward nearest
                return 'navigate_to_nearest_object'
            else:
                return 'explore_environment'

    def extract_target_from_command(self, command: str, vision_context: Dict) -> str:
        """Extract target object from command using visual context."""
        # In a real implementation, this would use NLP to extract the target
        # For now, we'll match with detected objects
        for obj in vision_context.get('objects', []):
            if obj['class'] in command:
                return obj['class']

        # If no match, return a generic target
        return 'target_location'

    def generate_action_plan(self, intent: str, vision_context: Dict) -> Dict[str, Any]:
        """Generate action plan based on intent and context."""
        plan = {
            'intent': intent,
            'steps': [],
            'context_used': len(vision_context.get('objects', [])) > 0,
            'estimated_duration': 0.0
        }

        if intent.startswith('navigate_to_'):
            target = intent.split('_', 2)[2]  # Extract target from intent

            # Find target in vision context
            target_obj = None
            for obj in vision_context.get('objects', []):
                if obj['class'] == target:
                    target_obj = obj
                    break

            if target_obj:
                # Navigate to detected object
                steps = [
                    {
                        'action': 'navigate',
                        'parameters': {
                            'target_x': target_obj['position']['x'],
                            'target_y': target_obj['position']['y'],
                            'approach_distance': 0.5
                        },
                        'description': f'Navigate to {target}'
                    }
                ]
            else:
                # Default navigation (in a real system, this might use semantic navigation)
                steps = [
                    {
                        'action': 'explore',
                        'parameters': {'direction': 'forward', 'distance': 1.0},
                        'description': f'Explore to find {target}'
                    }
                ]

        elif intent.startswith('grasp_'):
            target = intent.split('_', 1)[1]  # Extract target from intent

            # Find target object in vision context
            target_obj = None
            for obj in vision_context.get('objects', []):
                if obj['class'] == target:
                    target_obj = obj
                    break

            if target_obj:
                steps = [
                    {
                        'action': 'approach',
                        'parameters': {
                            'target_x': target_obj['position']['x'],
                            'target_y': target_obj['position']['y'],
                            'approach_distance': 0.3
                        },
                        'description': f'Approach {target}'
                    },
                    {
                        'action': 'grasp',
                        'parameters': {'object': target},
                        'description': f'Grasp {target}'
                    }
                ]
            else:
                steps = [
                    {
                        'action': 'request_clarification',
                        'parameters': {'question': f'I cannot see {target} in the current view. Can you guide me?'},
                        'description': f'Cannot find {target}, request clarification'
                    }
                ]

        elif intent == 'avoid_obstacles':
            steps = [
                {
                    'action': 'obstacle_avoidance',
                    'parameters': {'obstacles': vision_context.get('obstacles', [])},
                    'description': 'Execute obstacle avoidance maneuver'
                }
            ]

        elif intent == 'describe_scene':
            description = self.generate_scene_description(vision_context)
            steps = [
                {
                    'action': 'speak',
                    'parameters': {'text': description},
                    'description': f'Describe scene: {description}'
                }
            ]

        elif intent == 'navigate_to_nearest_object':
            if vision_context['objects']:
                nearest_obj = min(vision_context['objects'], key=lambda o: o['position']['x']**2 + o['position']['y']**2)
                steps = [
                    {
                        'action': 'navigate',
                        'parameters': {
                            'target_x': nearest_obj['position']['x'],
                            'target_y': nearest_obj['position']['y'],
                            'approach_distance': 0.5
                        },
                        'description': f'Navigate to nearest object: {nearest_obj["class"]}'
                    }
                ]
            else:
                steps = [
                    {
                        'action': 'explore',
                        'parameters': {'direction': 'random', 'distance': 1.0},
                        'description': 'Explore environment as no objects detected'
                    }
                ]

        else:  # explore_environment
            steps = [
                {
                    'action': 'explore',
                    'parameters': {'direction': 'forward', 'distance': 1.0},
                    'description': 'Explore environment'
                }
            ]

        plan['steps'] = steps
        plan['estimated_duration'] = len(steps) * 2.0  # 2 seconds per step estimate

        return plan

    def execute_action_plan(self, plan: Dict[str, Any]) -> bool:
        """Execute the generated action plan."""
        success = True

        for step in plan['steps']:
            action_type = step['action']
            parameters = step['parameters']
            description = step['description']

            self.get_logger().info(f'Executing: {description}')

            try:
                if action_type == 'navigate':
                    self.execute_navigation(parameters)
                elif action_type == 'approach':
                    self.execute_approach(parameters)
                elif action_type == 'grasp':
                    self.execute_grasp(parameters)
                elif action_type == 'obstacle_avoidance':
                    self.execute_obstacle_avoidance(parameters)
                elif action_type == 'speak':
                    self.execute_speak(parameters)
                elif action_type == 'explore':
                    self.execute_explore(parameters)
                elif action_type == 'request_clarification':
                    self.execute_request_clarification(parameters)
                else:
                    self.get_logger().warn(f'Unknown action type: {action_type}')
                    success = False
                    break

                # Small delay between steps
                time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f'Action execution error: {e}')
                success = False
                break

        return success

    def execute_navigation(self, params: Dict[str, Any]):
        """Execute navigation action."""
        target_x = params.get('target_x', 0.0)
        target_y = params.get('target_y', 0.0)
        approach_distance = params.get('approach_distance', 0.5)

        # In a real implementation, this would use navigation stack
        # For simulation, we'll just move in the general direction
        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward at 0.3 m/s
        cmd.angular.z = 0.0  # No rotation initially

        # Simple proportional controller
        dx = target_x - 0.0  # Assuming current position is (0,0) for simulation
        dy = target_y - 0.0

        distance = math.sqrt(dx*dx + dy*dy)

        if distance > approach_distance:
            # Calculate angle to target
            target_angle = math.atan2(dy, dx)
            # Simple PD controller for angular velocity
            cmd.angular.z = max(min(target_angle * 2.0, 1.0), -1.0)  # Limit angular velocity
            cmd.linear.x = max(min(distance * 0.5, 0.5), 0.1)  # Limit linear velocity

        self.cmd_vel_pub.publish(cmd)

    def execute_approach(self, params: Dict[str, Any]):
        """Execute approach action."""
        self.execute_navigation(params)  # For now, approach is similar to navigation

    def execute_grasp(self, params: Dict[str, Any]):
        """Execute grasp action (simulated)."""
        object_name = params.get('object', 'unknown')
        self.get_logger().info(f'Attempting to grasp: {object_name}')
        # In a real system, this would send commands to manipulator

    def execute_obstacle_avoidance(self, params: Dict[str, Any]):
        """Execute obstacle avoidance."""
        obstacles = params.get('obstacles', [])
        self.get_logger().info(f'Avoiding {len(obstacles)} obstacles')

        if obstacles:
            # Simple obstacle avoidance: turn away from closest obstacle
            closest_obstacle = min(obstacles, key=lambda o: o['distance'])

            cmd = Twist()
            if closest_obstacle['distance'] < 0.5:  # Too close
                cmd.angular.z = 0.5 if closest_obstacle['y'] > 0 else -0.5  # Turn away
            else:
                cmd.linear.x = 0.2  # Continue forward slowly

            self.cmd_vel_pub.publish(cmd)

    def execute_speak(self, params: Dict[str, Any]):
        """Execute speak action."""
        text = params.get('text', '')
        self.get_logger().info(f'Speaking: {text}')
        # In a real system, this would interface with text-to-speech

    def execute_explore(self, params: Dict[str, Any]):
        """Execute exploration action."""
        direction = params.get('direction', 'forward')
        distance = params.get('distance', 1.0)

        cmd = Twist()
        if direction == 'forward':
            cmd.linear.x = 0.3
        elif direction == 'left':
            cmd.angular.z = 0.5
        elif direction == 'right':
            cmd.angular.z = -0.5
        elif direction == 'random':
            import random
            cmd.linear.x = 0.2
            cmd.angular.z = random.uniform(-0.5, 0.5)

        self.cmd_vel_pub.publish(cmd)

    def execute_request_clarification(self, params: Dict[str, Any]):
        """Execute clarification request."""
        question = params.get('question', 'Could you please clarify?')
        self.get_logger().info(f'Requesting clarification: {question}')

        # Publish as response
        response_msg = String()
        response_msg.data = question
        self.response_pub.publish(response_msg)

    def generate_scene_description(self, vision_context: Dict) -> str:
        """Generate description of current scene."""
        objects = vision_context.get('objects', [])
        obstacles = vision_context.get('obstacles', [])

        if not objects and not obstacles:
            return "I don't see any objects or obstacles in the current view."

        description_parts = []

        if objects:
            object_names = [obj['class'] for obj in objects]
            unique_objects = list(set(object_names))

            if len(unique_objects) == 1:
                description_parts.append(f"I see a {unique_objects[0]}.")
            else:
                object_counts = {name: object_names.count(name) for name in unique_objects}
                objects_desc = ", ".join([f"{count} {name}" if count > 1 else f"{name}"
                                        for name, count in object_counts.items()])
                description_parts.append(f"I see {objects_desc}.")

        if obstacles:
            description_parts.append(f"There are {len(obstacles)} obstacles detected nearby.")

        return " ".join(description_parts)

    def generate_response(self, command: str, success: bool, action_plan: Dict) -> str:
        """Generate natural language response."""
        if success:
            intent = action_plan.get('intent', 'unknown')
            if 'navigate_to' in intent:
                target = intent.split('_', 2)[2] if len(intent.split('_')) > 2 else 'location'
                return f"I'm navigating to the {target} as requested."
            elif 'grasp' in intent:
                target = intent.split('_', 1)[1] if len(intent.split('_')) > 1 else 'object'
                return f"I'm attempting to grasp the {target}."
            elif intent == 'describe_scene':
                return self.generate_scene_description(self.process_vision_context())
            elif intent == 'avoid_obstacles':
                return "I'm navigating while avoiding obstacles."
            else:
                return f"I've completed the requested action: {command}"
        else:
            return f"I'm sorry, I couldn't complete the action: {command}. Something went wrong."

def main(args=None):
    rclpy.init(args=args)
    vla_node = CentralizedVLANode()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        vla_node.get_logger().info('Shutting down Centralized VLA Node')
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Distributed VLA Architecture

### Component-Based Approach

The distributed approach separates VLA components into independent, communicating modules:

```python
# distributed_vla_components.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, ActionClient
from my_robot_msgs.action import VisionTask, LanguageTask, ActionTask
import json
import threading
from typing import Dict, Any, Optional

class VisionComponent(Node):
    def __init__(self):
        super().__init__('vision_component')

        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, 'vision/detections', 10)
        self.scene_description_pub = self.create_publisher(String, 'vision/scene_description', 10)
        self.ready_pub = self.create_publisher(Bool, 'vision/ready', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Action server for vision tasks
        self.vision_action_server = ActionServer(
            self,
            VisionTask,
            'vision_task',
            self.execute_vision_task
        )

        # Internal state
        self.latest_image = None
        self.latest_scan = None
        self.is_ready = True

        # Performance tracking
        self.frame_count = 0
        self.processing_times = []

        self.get_logger().info('Vision Component initialized')

    def image_callback(self, msg):
        """Process incoming images."""
        self.latest_image = msg
        self.frame_count += 1

        # Process image if needed
        if self.frame_count % 5 == 0:  # Process every 5th frame to reduce load
            self.process_image_for_detections(msg)

    def scan_callback(self, msg):
        """Process incoming scan data."""
        self.latest_scan = msg

    def process_image_for_detections(self, image_msg):
        """Process image to generate detections (simulated)."""
        # In a real implementation, this would run object detection
        # For simulation, create mock detections
        detections = Detection2DArray()
        detections.header = image_msg.header

        # Simulate some detections
        import random
        for i in range(random.randint(0, 3)):
            detection = Detection2D()
            detection.bbox.center.x = float(random.uniform(100, 500))
            detection.bbox.center.y = float(random.uniform(100, 300))
            detection.bbox.size_x = float(random.uniform(50, 150))
            detection.bbox.size_y = float(random.uniform(50, 150))

            # Random class
            classes = ['person', 'chair', 'table', 'bottle', 'cup']
            class_name = random.choice(classes)

            from vision_msgs.msg import ObjectHypothesisWithPose
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = float(random.uniform(0.6, 0.95))

            detection.results.append(hypothesis)
            detections.detections.append(detection)

        self.detections_pub.publish(detections)

    async def execute_vision_task(self, goal_handle):
        """Execute vision task request."""
        self.get_logger().info(f'Executing vision task: {goal_handle.request.task_type}')

        task_type = goal_handle.request.task_type
        result = VisionTask.Result()

        if task_type == 'object_detection':
            # Return latest detections
            if self.latest_detections:
                result.detections = self.latest_detections.detections
                result.success = True
                result.message = 'Object detection completed'
            else:
                result.success = False
                result.message = 'No detections available'

        elif task_type == 'scene_analysis':
            # Generate scene description
            description = self.analyze_scene()
            description_msg = String()
            description_msg.data = description
            self.scene_description_pub.publish(description_msg)

            result.success = True
            result.message = description

        elif task_type == 'depth_estimation':
            # Process depth information from scan
            if self.latest_scan:
                # Extract depth information from laser scan
                valid_ranges = [r for r in self.latest_scan.ranges
                              if self.latest_scan.range_min < r < self.latest_scan.range_max]

                if valid_ranges:
                    avg_depth = sum(valid_ranges) / len(valid_ranges)
                    result.depth_info = f'Average depth: {avg_depth:.2f}m, Min: {min(valid_ranges):.2f}m'
                    result.success = True
                    result.message = 'Depth estimation completed'
                else:
                    result.success = False
                    result.message = 'No valid depth measurements'
            else:
                result.success = False
                result.message = 'No scan data available'

        else:
            result.success = False
            result.message = f'Unknown vision task: {task_type}'

        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    def analyze_scene(self) -> str:
        """Analyze current scene and generate description."""
        if self.latest_detections and self.latest_detections.detections:
            objects = []
            for detection in self.latest_detections.detections:
                if detection.results:
                    class_name = detection.results[0].hypothesis.class_id
                    confidence = detection.results[0].hypothesis.score
                    if confidence > 0.6:  # Confidence threshold
                        objects.append(class_name)

            if objects:
                unique_objects = list(set(objects))
                if len(unique_objects) == 1:
                    return f"I see a {unique_objects[0]} in the scene."
                else:
                    return f"I see {len(unique_objects)} different objects: {', '.join(unique_objects)}."
            else:
                return "I see objects but they're not confidently identified."
        else:
            return "I don't see any objects in the current view."

class LanguageComponent(Node):
    def __init__(self):
        super().__init__('language_component')

        # Publishers
        self.intent_pub = self.create_publisher(String, 'language/intent', 10)
        self.response_pub = self.create_publisher(String, 'language/response', 10)
        self.ready_pub = self.create_publisher(Bool, 'language/ready', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.scene_description_sub = self.create_subscription(
            String, 'vision/scene_description', self.scene_description_callback, 10
        )

        # Action server for language tasks
        self.language_action_server = ActionServer(
            self,
            LanguageTask,
            'language_task',
            self.execute_language_task
        )

        # Internal state
        self.current_scene_description = ""
        self.is_ready = True

        # Context for language understanding
        self.conversation_context = []

        self.get_logger().info('Language Component initialized')

    def command_callback(self, msg):
        """Process natural language command."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Process command with current context
        intent = self.interpret_command(command, self.current_scene_description)

        # Publish intent
        intent_msg = String()
        intent_msg.data = json.dumps({
            'command': command,
            'intent': intent,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        })
        self.intent_pub.publish(intent_msg)

        # Generate response
        response = self.generate_response(command, intent)
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

    def scene_description_callback(self, msg):
        """Update scene context from vision component."""
        self.current_scene_description = msg.data
        self.get_logger().debug(f'Updated scene context: {msg.data[:50]}...')

    async def execute_language_task(self, goal_handle):
        """Execute language processing task."""
        self.get_logger().info(f'Executing language task: {goal_handle.request.task_type}')

        task_type = goal_handle.request.task_type
        input_text = goal_handle.request.input_text
        context = goal_handle.request.context

        result = LanguageTask.Result()

        if task_type == 'command_interpretation':
            intent = self.interpret_command(input_text, context)
            result.intent = intent
            result.success = True
            result.message = f'Interpreted intent: {intent}'

        elif task_type == 'question_answering':
            answer = self.answer_question(input_text, context)
            result.response = answer
            result.success = True
            result.message = f'Answered: {answer}'

        elif task_type == 'dialogue_management':
            response = self.manage_dialogue(input_text, context)
            result.response = response
            result.success = True
            result.message = f'Responded: {response}'

        else:
            result.success = False
            result.message = f'Unknown language task: {task_type}'

        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    def interpret_command(self, command: str, context: str = "") -> str:
        """Interpret natural language command."""
        command_lower = command.lower()

        # Enhanced command interpretation with context awareness
        if any(word in command_lower for word in ['go to', 'navigate to', 'move to', 'go toward']):
            # Extract target from command or context
            target = self.extract_target(command_lower, context)
            return f'navigate_to_{target}'

        elif any(word in command_lower for word in ['pick up', 'grasp', 'get', 'take']):
            target = self.extract_target(command_lower, context)
            return f'grasp_{target}'

        elif any(word in command_lower for word in ['describe', 'what do you see', 'tell me about']):
            return 'describe_scene'

        elif any(word in command_lower for word in ['avoid', 'go around', 'navigate around']):
            return 'avoid_obstacles'

        elif any(word in command_lower for word in ['follow', 'track']):
            target = self.extract_target(command_lower, context)
            return f'follow_{target}'

        elif any(word in command_lower for word in ['stop', 'halt', 'freeze']):
            return 'stop_robot'

        else:
            # Use context to determine appropriate action
            if 'person' in context.lower():
                return 'approach_person'
            elif 'object' in context.lower():
                return 'investigate_object'
            else:
                return 'explore_environment'

    def extract_target(self, command: str, context: str) -> str:
        """Extract target from command using context."""
        # Simple extraction - in real system use NLP
        import re

        # Look for object names in command
        object_patterns = [
            r'go to the (\w+)',
            r'move to the (\w+)',
            r'pick up the (\w+)',
            r'grasp the (\w+)',
            r'follow the (\w+)'
        ]

        for pattern in object_patterns:
            match = re.search(pattern, command)
            if match:
                return match.group(1)

        # If not found in command, try to infer from context
        context_lower = context.lower()
        possible_targets = ['person', 'object', 'obstacle', 'target', 'location']
        for target in possible_targets:
            if target in context_lower:
                return target

        return 'target_location'

    def answer_question(self, question: str, context: str) -> str:
        """Answer natural language questions."""
        question_lower = question.lower()

        if 'where' in question_lower and 'robot' in question_lower:
            return "The robot is currently in the environment and ready to help."
        elif 'what' in question_lower and ('see' in question_lower or 'objects' in question_lower):
            if context:
                return f"The robot sees: {context}"
            else:
                return "The robot doesn't have current scene information."
        elif 'how' in question_lower and 'help' in question_lower:
            return "The robot can navigate, manipulate objects, and interact with the environment based on your commands."
        else:
            return "I can help with navigation, object manipulation, and environmental interaction. How can I assist you?"

    def manage_dialogue(self, input_text: str, context: str) -> str:
        """Manage natural dialogue."""
        # Simple dialogue management
        if any(greeting in input_text.lower() for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        elif any(affirmation in input_text.lower() for affirmation in ['thanks', 'thank you', 'great']):
            return "You're welcome! Is there anything else I can help with?"
        else:
            # Default response - defer to command interpretation
            intent = self.interpret_command(input_text, context)
            return f"I understand you want me to: {intent.replace('_', ' ')}"

class ActionComponent(Node):
    def __init__(self):
        super().__init__('action_component')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.action_status_pub = self.create_publisher(String, 'action/status', 10)
        self.ready_pub = self.create_publisher(Bool, 'action/ready', 10)

        # Subscribers
        self.intent_sub = self.create_subscription(
            String, 'language/intent', self.intent_callback, 10
        )

        # Action server for action execution
        self.action_action_server = ActionServer(
            self,
            ActionTask,
            'action_task',
            self.execute_action_task
        )

        # Internal state
        self.is_ready = True
        self.current_task = None
        self.is_executing = False

        self.get_logger().info('Action Component initialized')

    def intent_callback(self, msg):
        """Process intent from language component."""
        try:
            intent_data = json.loads(msg.data)
            intent = intent_data.get('intent', 'unknown')
            command = intent_data.get('command', '')

            self.get_logger().info(f'Processing intent: {intent}')

            # Execute action based on intent
            success = self.execute_intent(intent)

            # Publish status
            status_msg = String()
            status_msg.data = f'Intent {intent} {"succeeded" if success else "failed"}: {command}'
            self.action_status_pub.publish(status_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in intent message')

    def execute_intent(self, intent: str) -> bool:
        """Execute action based on intent."""
        try:
            if intent.startswith('navigate_to_'):
                target = intent.split('_', 2)[2] if len(intent.split('_')) > 2 else 'location'
                return self.execute_navigation(target)

            elif intent.startswith('grasp_'):
                target = intent.split('_', 1)[1] if len(intent.split('_')) > 1 else 'object'
                return self.execute_grasp(target)

            elif intent == 'describe_scene':
                # This is handled by the vision component's description
                return True

            elif intent == 'avoid_obstacles':
                return self.execute_obstacle_avoidance()

            elif intent.startswith('follow_'):
                target = intent.split('_', 1)[1] if len(intent.split('_')) > 1 else 'object'
                return self.execute_follow(target)

            elif intent == 'stop_robot':
                return self.execute_stop()

            elif intent in ['approach_person', 'investigate_object']:
                return self.execute_approach_object()

            elif intent == 'explore_environment':
                return self.execute_explore()

            else:
                self.get_logger().warn(f'Unknown intent: {intent}')
                return False

        except Exception as e:
            self.get_logger().error(f'Intent execution error: {e}')
            return False

    async def execute_action_task(self, goal_handle):
        """Execute action task request."""
        self.get_logger().info(f'Executing action task: {goal_handle.request.task_type}')

        task_type = goal_handle.request.task_type
        parameters = goal_handle.request.parameters

        result = ActionTask.Result()

        self.is_executing = True
        self.current_task = task_type

        try:
            if task_type == 'navigation':
                success = self.execute_navigation_to_pose(parameters)
                result.success = success
                result.message = 'Navigation task completed' if success else 'Navigation failed'

            elif task_type == 'manipulation':
                success = self.execute_manipulation(parameters)
                result.success = success
                result.message = 'Manipulation task completed' if success else 'Manipulation failed'

            elif task_type == 'locomotion':
                success = self.execute_locomotion(parameters)
                result.success = success
                result.message = 'Locomotion task completed' if success else 'Locomotion failed'

            elif task_type == 'inspection':
                success = self.execute_inspection(parameters)
                result.success = success
                result.message = 'Inspection task completed' if success else 'Inspection failed'

            else:
                result.success = False
                result.message = f'Unknown action task: {task_type}'

        except Exception as e:
            result.success = False
            result.message = f'Action execution error: {str(e)}'

        finally:
            self.is_executing = False
            self.current_task = None

        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    def execute_navigation_to_pose(self, params: Dict[str, Any]) -> bool:
        """Execute navigation to specific pose."""
        try:
            target_x = params.get('target_x', 0.0)
            target_y = params.get('target_y', 0.0)
            target_theta = params.get('target_theta', 0.0)
            approach_distance = params.get('approach_distance', 0.5)

            # Simple proportional navigation
            cmd = Twist()

            # Calculate distance and angle to target
            # (In real implementation, you'd get current pose from localization)
            current_x, current_y = 0.0, 0.0  # Simulated current position
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > approach_distance:
                # Calculate angle to target
                target_angle = math.atan2(dy, dx)

                # Simple PD controller
                cmd.linear.x = max(min(distance * 0.5, 0.5), 0.1)  # Limit speed
                cmd.angular.z = max(min(target_angle * 2.0, 1.0), -1.0)  # Limit angular velocity

            self.cmd_vel_pub.publish(cmd)
            return True

        except Exception as e:
            self.get_logger().error(f'Navigation execution error: {e}')
            return False

    def execute_manipulation(self, params: Dict[str, Any]) -> bool:
        """Execute manipulation task."""
        try:
            action = params.get('action', 'unknown')
            target = params.get('target', 'object')

            self.get_logger().info(f'Executing manipulation: {action} {target}')

            # In a real system, this would control robotic arms/manipulators
            # For simulation, just log the action
            if action in ['grasp', 'pick', 'take']:
                self.get_logger().info(f'Attempting to grasp {target}')
            elif action in ['place', 'release', 'drop']:
                self.get_logger().info(f'Attempting to release {target}')
            else:
                self.get_logger().info(f'Attempting manipulation action: {action}')

            return True

        except Exception as e:
            self.get_logger().error(f'Manipulation execution error: {e}')
            return False

    def execute_obstacle_avoidance(self) -> bool:
        """Execute obstacle avoidance behavior."""
        # This would typically subscribe to scan data and implement避障algorithm
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward slowly
        cmd.angular.z = 0.0  # No rotation initially

        self.cmd_vel_pub.publish(cmd)
        return True

    def execute_follow(self, target: str) -> bool:
        """Execute follow behavior."""
        # In a real system, this would use tracking algorithms
        cmd = Twist()
        cmd.linear.x = 0.3  # Follow at moderate speed
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)
        return True

    def execute_stop(self) -> bool:
        """Execute stop behavior."""
        cmd = Twist()
        # Zero all velocities
        self.cmd_vel_pub.publish(cmd)
        return True

    def execute_approach_object(self) -> bool:
        """Execute approach object behavior."""
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward slowly to approach
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)
        return True

    def execute_explore(self) -> bool:
        """Execute exploration behavior."""
        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward
        cmd.angular.z = 0.1  # Slight turn to explore

        self.cmd_vel_pub.publish(cmd)
        return True

def main(args=None):
    rclpy.init(args=args)

    # Create all components
    vision_comp = VisionComponent()
    language_comp = LanguageComponent()
    action_comp = ActionComponent()

    # Use MultiThreadedExecutor to handle multiple nodes
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
    executor.add_node(vision_comp)
    executor.add_node(language_comp)
    executor.add_node(action_comp)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('Shutting down Distributed VLA Components')
    finally:
        executor.shutdown()
        vision_comp.destroy_node()
        language_comp.destroy_node()
        action_comp.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Hybrid VLA Architecture

### Combining Centralized and Distributed Approaches

```python
# hybrid_vla_architecture.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from rclpy.action import ActionServer, ActionClient
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from my_robot_msgs.action import VLATask
import json
import threading
import time
from typing import Dict, Any, Optional, List

class HybridVLANode(Node):
    def __init__(self):
        super().__init__('hybrid_vla')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.response_pub = self.create_publisher(String, 'vla_response', 10)
        self.status_pub = self.create_publisher(String, 'vla_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )

        # Action server for complex VLA tasks
        self.vla_action_server = ActionServer(
            self,
            VLATask,
            'vla_task',
            self.execute_vla_task
        )

        # Action clients for distributed components (when needed)
        self.vision_action_client = ActionClient(self, VisionTask, 'vision_task')
        self.language_action_client = ActionClient(self, LanguageTask, 'language_task')
        self.action_action_client = ActionClient(self, ActionTask, 'action_task')

        # Internal state management
        self.latest_image = None
        self.latest_scan = None
        self.latest_detections = None
        self.command_queue = []
        self.is_processing = False

        # Architecture configuration
        self.use_centralized_processing = True  # Can be toggled
        self.fallback_to_distributed = True    # Use distributed as backup

        # Performance optimization
        self.enable_caching = True
        self.cache_timeout = 2.0  # seconds
        self.vision_cache = {}
        self.language_cache = {}

        # Threading for background processing
        self.processing_thread = threading.Thread(target=self.background_processing, daemon=True)
        self.processing_thread.start()

        # Performance metrics
        self.centralized_times = []
        self.distributed_times = []
        self.failure_count = 0

        self.get_logger().info('Hybrid VLA Architecture initialized')

    def command_callback(self, msg):
        """Process incoming commands using hybrid architecture."""
        command = msg.data
        self.get_logger().info(f'Hybrid VLA received command: {command}')

        # Add to processing queue
        command_item = {
            'command': command,
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'source': 'direct'
        }

        with self.processing_lock:
            self.command_queue.append(command_item)

    def image_callback(self, msg):
        """Update latest image."""
        self.latest_image = msg

    def scan_callback(self, msg):
        """Update latest scan data."""
        self.latest_scan = msg

    def detections_callback(self, msg):
        """Update latest detections."""
        self.latest_detections = msg

    def background_processing(self):
        """Background thread for processing commands."""
        while rclpy.ok():
            try:
                with self.processing_lock:
                    if self.command_queue:
                        command_item = self.command_queue.pop(0)
                        command = command_item['command']

                if command:
                    start_time = time.time()

                    # Choose processing method based on command complexity
                    if self.should_use_centralized(command):
                        result = self.process_with_centralized_vla(command)
                        processing_time = time.time() - start_time
                        self.centralized_times.append(processing_time)

                        if len(self.centralized_times) > 50:
                            self.centralized_times.pop(0)
                    else:
                        result = self.process_with_distributed_vla(command)
                        processing_time = time.time() - start_time
                        self.distributed_times.append(processing_time)

                        if len(self.distributed_times) > 50:
                            self.distributed_times.pop(0)

                    # Publish results
                    if result:
                        response_msg = String()
                        response_msg.data = result.get('response', 'Processing completed')
                        self.response_pub.publish(response_msg)

                        status_msg = String()
                        status_msg.data = f'Processed: {command[:30]}...' if len(command) > 30 else command
                        self.status_pub.publish(status_msg)

                time.sleep(0.01)

            except Exception as e:
                self.get_logger().error(f'Background processing error: {e}')
                time.sleep(0.1)

    def should_use_centralized(self, command: str) -> bool:
        """Determine whether to use centralized or distributed processing."""
        # Use centralized for simple, direct commands
        simple_keywords = ['go', 'stop', 'turn', 'move', 'forward', 'backward', 'left', 'right']

        # Use distributed for complex, multi-step commands
        complex_keywords = ['describe', 'analyze', 'plan', 'strategy', 'multiple', 'sequence']

        has_simple = any(keyword in command.lower() for keyword in simple_keywords)
        has_complex = any(keyword in command.lower() for keyword in complex_keywords)

        # Prefer centralized for simple commands, distributed for complex ones
        return has_simple and not has_complex

    def process_with_centralized_vla(self, command: str) -> Optional[Dict[str, Any]]:
        """Process command using centralized VLA pipeline."""
        try:
            # Check cache first
            cache_key = f"centralized_{hash(command)}"
            if self.enable_caching and cache_key in self.vision_cache:
                cached_time, cached_result = self.vision_cache[cache_key]
                if time.time() - cached_time < self.cache_timeout:
                    return cached_result

            # Process with centralized pipeline
            vision_context = self.process_vision_context()
            intent = self.interpret_command_centralized(command, vision_context)
            action_plan = self.generate_action_plan_centralized(intent, vision_context)
            success = self.execute_action_plan_centralized(action_plan)
            response = self.generate_response_centralized(command, success, action_plan)

            result = {
                'command': command,
                'intent': intent,
                'action_plan': action_plan,
                'response': response,
                'success': success,
                'architecture_used': 'centralized'
            }

            # Cache result
            if self.enable_caching:
                self.vision_cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            self.get_logger().error(f'Centralized VLA processing error: {e}')

            # Fallback to distributed if enabled
            if self.fallback_to_distributed:
                self.get_logger().info('Falling back to distributed processing')
                return self.process_with_distributed_vla(command)
            else:
                return {
                    'command': command,
                    'success': False,
                    'error': str(e),
                    'architecture_used': 'centralized_failed'
                }

    def process_with_distributed_vla(self, command: str) -> Optional[Dict[str, Any]]:
        """Process command using distributed VLA components."""
        try:
            # Check cache first
            cache_key = f"distributed_{hash(command)}"
            if self.enable_caching and cache_key in self.language_cache:
                cached_time, cached_result = self.language_cache[cache_key]
                if time.time() - cached_time < self.cache_timeout:
                    return cached_result

            # Use action clients to coordinate with distributed components
            # This simulates the distributed approach
            vision_result = self.request_vision_processing(command)
            language_result = self.request_language_processing(command, vision_result)
            action_result = self.request_action_execution(language_result)

            result = {
                'command': command,
                'vision_result': vision_result,
                'language_result': language_result,
                'action_result': action_result,
                'success': action_result.get('success', False),
                'architecture_used': 'distributed'
            }

            # Cache result
            if self.enable_caching:
                self.language_cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            self.get_logger().error(f'Distributed VLA processing error: {e}')

            # Fallback to centralized if enabled
            if self.fallback_to_distributed:
                self.get_logger().info('Falling back to centralized processing')
                return self.process_with_centralized_vla(command)
            else:
                return {
                    'command': command,
                    'success': False,
                    'error': str(e),
                    'architecture_used': 'distributed_failed'
                }

    def request_vision_processing(self, command: str) -> Dict[str, Any]:
        """Request vision processing from distributed component."""
        # Simulate vision processing request
        # In real implementation, this would use action client
        vision_context = self.process_vision_context()

        return {
            'objects_detected': len(vision_context.get('objects', [])),
            'obstacles_detected': len(vision_context.get('obstacles', [])),
            'scene_description': self.generate_scene_description(vision_context),
            'timestamp': time.time()
        }

    def request_language_processing(self, command: str, vision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Request language processing from distributed component."""
        # Simulate language processing request
        intent = self.interpret_command_centralized(command, vision_context)

        return {
            'intent': intent,
            'confidence': 0.9,  # Simulated confidence
            'parsed_command': command,
            'timestamp': time.time()
        }

    def request_action_execution(self, language_result: Dict[str, Any]) -> Dict[str, Any]:
        """Request action execution from distributed component."""
        # Simulate action execution request
        intent = language_result['intent']

        # Execute the action
        success = self.execute_intent_from_distributed(intent)

        return {
            'intent_executed': intent,
            'success': success,
            'execution_time': 1.0,  # Simulated time
            'timestamp': time.time()
        }

    def process_vision_context(self) -> Dict[str, Any]:
        """Process current visual information (shared between architectures)."""
        context = {
            'objects': [],
            'obstacles': [],
            'environment': 'indoor',
            'timestamp': time.time()
        }

        # Process detections if available
        if self.latest_detections:
            for detection in self.latest_detections.detections:
                if detection.results:
                    class_name = detection.results[0].hypothesis.class_id
                    confidence = detection.results[0].hypothesis.score
                    position = detection.bbox.center

                    if confidence > 0.5:  # Confidence threshold
                        context['objects'].append({
                            'class': class_name,
                            'confidence': confidence,
                            'position': {'x': position.x, 'y': position.y}
                        })

        # Process laser scan for obstacles
        if self.latest_scan:
            obstacles = []
            for i, range_val in enumerate(self.latest_scan.ranges):
                if self.latest_scan.range_min < range_val < self.latest_scan.range_max:
                    angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
                    x = range_val * math.cos(angle)
                    y = range_val * math.sin(angle)

                    if range_val < 1.0:  # Obstacle threshold
                        obstacles.append({'x': x, 'y': y, 'distance': range_val})

            context['obstacles'] = obstacles

        return context

    def interpret_command_centralized(self, command: str, vision_context: Dict) -> str:
        """Interpret command using centralized approach."""
        command_lower = command.lower()

        if 'go to' in command_lower or 'navigate to' in command_lower:
            target = self.extract_target_from_command(command_lower, vision_context)
            return f'navigate_to_{target}'

        elif 'pick up' in command_lower or 'grasp' in command_lower:
            target = self.extract_target_from_command(command_lower, vision_context)
            return f'grasp_{target}'

        elif 'avoid' in command_lower or 'navigate around' in command_lower:
            return 'avoid_obstacles'

        elif 'describe' in command_lower or 'what do you see' in command_lower:
            return 'describe_scene'

        else:
            if vision_context['objects']:
                return 'navigate_to_nearest_object'
            else:
                return 'explore_environment'

    def generate_action_plan_centralized(self, intent: str, vision_context: Dict) -> Dict[str, Any]:
        """Generate action plan using centralized approach."""
        plan = {
            'intent': intent,
            'steps': [],
            'context_used': len(vision_context.get('objects', [])) > 0,
            'estimated_duration': 0.0
        }

        # Implementation similar to centralized approach
        if intent.startswith('navigate_to_'):
            target = intent.split('_', 2)[2] if len(intent.split('_')) > 2 else 'location'

            target_obj = None
            for obj in vision_context.get('objects', []):
                if obj['class'] == target:
                    target_obj = obj
                    break

            if target_obj:
                steps = [{
                    'action': 'navigate',
                    'parameters': {
                        'target_x': target_obj['position']['x'],
                        'target_y': target_obj['position']['y'],
                        'approach_distance': 0.5
                    },
                    'description': f'Navigate to {target}'
                }]
            else:
                steps = [{
                    'action': 'explore',
                    'parameters': {'direction': 'forward', 'distance': 1.0},
                    'description': f'Explore to find {target}'
                }]

        elif intent.startswith('grasp_'):
            target = intent.split('_', 1)[1] if len(intent.split('_')) > 1 else 'object'

            target_obj = None
            for obj in vision_context.get('objects', []):
                if obj['class'] == target:
                    target_obj = obj
                    break

            if target_obj:
                steps = [
                    {
                        'action': 'approach',
                        'parameters': {
                            'target_x': target_obj['position']['x'],
                            'target_y': target_obj['position']['y'],
                            'approach_distance': 0.3
                        },
                        'description': f'Approach {target}'
                    },
                    {
                        'action': 'grasp',
                        'parameters': {'object': target},
                        'description': f'Grasp {target}'
                    }
                ]
            else:
                steps = [{
                    'action': 'request_clarification',
                    'parameters': {'question': f'I cannot see {target} in the current view. Can you guide me?'},
                    'description': f'Cannot find {target}, request clarification'
                }]

        elif intent == 'avoid_obstacles':
            steps = [{
                'action': 'obstacle_avoidance',
                'parameters': {'obstacles': vision_context.get('obstacles', [])},
                'description': 'Execute obstacle avoidance maneuver'
            }]

        elif intent == 'describe_scene':
            description = self.generate_scene_description(vision_context)
            steps = [{
                'action': 'speak',
                'parameters': {'text': description},
                'description': f'Describe scene: {description}'
            }]

        elif intent == 'navigate_to_nearest_object':
            if vision_context['objects']:
                nearest_obj = min(vision_context['objects'], key=lambda o: o['position']['x']**2 + o['position']['y']**2)
                steps = [{
                    'action': 'navigate',
                    'parameters': {
                        'target_x': nearest_obj['position']['x'],
                        'target_y': nearest_obj['position']['y'],
                        'approach_distance': 0.5
                    },
                    'description': f'Navigate to nearest object: {nearest_obj["class"]}'
                }]
            else:
                steps = [{
                    'action': 'explore',
                    'parameters': {'direction': 'random', 'distance': 1.0},
                    'description': 'Explore environment as no objects detected'
                }]

        else:  # explore_environment
            steps = [{
                'action': 'explore',
                'parameters': {'direction': 'forward', 'distance': 1.0},
                'description': 'Explore environment'
            }]

        plan['steps'] = steps
        plan['estimated_duration'] = len(steps) * 2.0  # 2 seconds per step estimate

        return plan

    def execute_action_plan_centralized(self, plan: Dict[str, Any]) -> bool:
        """Execute action plan using centralized approach."""
        success = True

        for step in plan['steps']:
            action_type = step['action']
            parameters = step['parameters']
            description = step['description']

            self.get_logger().info(f'Executing: {description}')

            try:
                if action_type == 'navigate':
                    self.execute_navigation_action(parameters)
                elif action_type == 'approach':
                    self.execute_approach_action(parameters)
                elif action_type == 'grasp':
                    self.execute_grasp_action(parameters)
                elif action_type == 'obstacle_avoidance':
                    self.execute_obstacle_avoidance_action(parameters)
                elif action_type == 'speak':
                    self.execute_speak_action(parameters)
                elif action_type == 'explore':
                    self.execute_explore_action(parameters)
                elif action_type == 'request_clarification':
                    self.execute_request_clarification_action(parameters)
                else:
                    self.get_logger().warn(f'Unknown action type: {action_type}')
                    success = False
                    break

                time.sleep(0.1)  # Small delay between steps

            except Exception as e:
                self.get_logger().error(f'Action execution error: {e}')
                success = False
                break

        return success

    def execute_intent_from_distributed(self, intent: str) -> bool:
        """Execute intent received from distributed processing."""
        # Map distributed intent to centralized execution
        if intent.startswith('navigate_to_'):
            target = intent.split('_', 2)[2] if len(intent.split('_')) > 2 else 'location'
            return self.execute_navigation_to_target(target)
        elif intent.startswith('grasp_'):
            target = intent.split('_', 1)[1] if len(intent.split('_')) > 1 else 'object'
            return self.execute_grasp_target(target)
        elif intent == 'avoid_obstacles':
            return self.execute_obstacle_avoidance_centralized()
        else:
            return self.execute_generic_intent(intent)

    def execute_navigation_to_target(self, target: str) -> bool:
        """Execute navigation to specific target."""
        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward
        cmd.angular.z = 0.0  # No rotation initially
        self.cmd_vel_pub.publish(cmd)
        time.sleep(2.0)  # Simulate navigation time
        return True

    def execute_grasp_target(self, target: str) -> bool:
        """Execute grasp of specific target."""
        self.get_logger().info(f'Attempting to grasp: {target}')
        time.sleep(1.0)  # Simulate grasp time
        return True

    def execute_obstacle_avoidance_centralized(self) -> bool:
        """Execute obstacle avoidance."""
        cmd = Twist()
        cmd.linear.x = 0.2  # Move slowly
        cmd.angular.z = 0.1  # Slight turn
        self.cmd_vel_pub.publish(cmd)
        time.sleep(1.0)
        return True

    def execute_generic_intent(self, intent: str) -> bool:
        """Execute generic intent."""
        self.get_logger().info(f'Executing generic intent: {intent}')
        return True

    def generate_response_centralized(self, command: str, success: bool, action_plan: Dict) -> str:
        """Generate response for centralized processing."""
        if success:
            intent = action_plan.get('intent', 'unknown')
            if 'navigate_to' in intent:
                target = intent.split('_', 2)[2] if len(intent.split('_')) > 2 else 'location'
                return f"I'm navigating to the {target} as requested."
            elif 'grasp' in intent:
                target = intent.split('_', 1)[1] if len(intent.split('_')) > 1 else 'object'
                return f"I'm attempting to grasp the {target}."
            elif intent == 'describe_scene':
                return self.generate_scene_description(self.process_vision_context())
            elif intent == 'avoid_obstacles':
                return "I'm navigating while avoiding obstacles."
            else:
                return f"I've completed the requested action: {command}"
        else:
            return f"I'm sorry, I couldn't complete the action: {command}. Something went wrong."

    async def execute_vla_task(self, goal_handle):
        """Execute complex VLA task using hybrid approach."""
        self.get_logger().info('Executing VLA task with hybrid architecture')

        task_type = goal_handle.request.task_type
        command = goal_handle.request.command
        context = goal_handle.request.context

        result = VLATask.Result()

        try:
            # Determine best architecture for this task
            if task_type == 'simple_navigation':
                # Use centralized for simple tasks
                process_result = self.process_with_centralized_vla(command)
            elif task_type == 'complex_reasoning':
                # Use distributed for complex reasoning
                process_result = self.process_with_distributed_vla(command)
            else:
                # Use hybrid decision based on command content
                if self.should_use_centralized(command):
                    process_result = self.process_with_centralized_vla(command)
                else:
                    process_result = self.process_with_distributed_vla(command)

            if process_result and process_result.get('success', False):
                result.success = True
                result.response = process_result.get('response', 'Task completed')
                result.architecture_used = process_result.get('architecture_used', 'unknown')
                goal_handle.succeed()
            else:
                result.success = False
                result.response = f'Task failed: {process_result.get("error", "Unknown error")}' if process_result else 'Task failed'
                result.architecture_used = 'unknown'
                goal_handle.abort()

        except Exception as e:
            result.success = False
            result.response = f'VLA task execution error: {str(e)}'
            result.architecture_used = 'error'
            goal_handle.abort()

        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for both architectures."""
        return {
            'centralized_avg_time': sum(self.centralized_times) / len(self.centralized_times) if self.centralized_times else 0,
            'distributed_avg_time': sum(self.distributed_times) / len(self.distributed_times) if self.distributed_times else 0,
            'centralized_call_count': len(self.centralized_times),
            'distributed_call_count': len(self.distributed_times),
            'failure_count': self.failure_count
        }

def main(args=None):
    rclpy.init(args=args)
    hybrid_node = HybridVLANode()

    try:
        rclpy.spin(hybrid_node)
    except KeyboardInterrupt:
        hybrid_node.get_logger().info('Shutting down Hybrid VLA Architecture')
    finally:
        hybrid_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Architecture Selection Guidelines

### When to Use Each Architecture

| Scenario | Recommended Architecture | Reason |
|----------|-------------------------|---------|
| Simple navigation tasks | Centralized | Faster response, fewer communication overhead |
| Complex multi-step tasks | Distributed | Better modularity, easier debugging |
| Real-time applications | Centralized | Lower latency, direct processing |
| Research/development | Distributed | Better for experimentation and testing |
| Resource-constrained | Centralized | Less process overhead |
| Large-scale systems | Distributed | Better scalability and fault tolerance |
| Safety-critical | Distributed | Better isolation and redundancy |
| Prototyping | Centralized | Faster development and testing |

### Performance Optimization Strategies

```python
# architecture_optimization.py
class VLAArchitectureOptimizer:
    def __init__(self):
        self.performance_history = []
        self.architecture_preferences = {}
        self.load_balancer = None

    def adaptive_architecture_selection(self, command: str, current_load: float) -> str:
        """Select architecture based on current conditions."""
        # Consider command complexity, system load, and historical performance
        command_complexity = self.estimate_command_complexity(command)

        if current_load > 0.8:  # High system load
            return 'centralized'  # Less communication overhead
        elif command_complexity > 0.7:  # Complex command
            return 'distributed'  # Better for complex reasoning
        else:
            return 'centralized'  # Default to centralized for simple tasks

    def estimate_command_complexity(self, command: str) -> float:
        """Estimate command complexity (0.0 to 1.0)."""
        # Count number of concepts, conjunctions, and context requirements
        complexity_factors = [
            len(command.split()) / 10.0,  # Length factor
            command.lower().count('and') * 0.1,  # Conjunctions
            command.lower().count('then') * 0.1,  # Sequences
            command.lower().count('while') * 0.1,  # Concurrent actions
        ]

        return min(sum(complexity_factors), 1.0)

    def optimize_for_latency(self) -> str:
        """Select architecture for minimum latency."""
        return 'centralized'

    def optimize_for_accuracy(self) -> str:
        """Select architecture for maximum accuracy."""
        return 'distributed'  # More processing power available

    def optimize_for_reliability(self) -> str:
        """Select architecture for maximum reliability."""
        return 'distributed'  # Better fault tolerance
```

## Summary

This chapter explored different VLA architectural patterns:

- **Centralized Architecture**: Single monolithic system with integrated components
- **Distributed Architecture**: Separate, communicating components for modularity
- **Hybrid Architecture**: Combination of both approaches for flexibility
- **Architecture Selection**: Guidelines for choosing the right approach
- **Performance Optimization**: Strategies for maximizing system performance

The choice of architecture depends on specific application requirements including real-time constraints, system complexity, development resources, and operational needs.

## Exercises

1. Implement both centralized and distributed VLA architectures
2. Compare performance between different architectural approaches
3. Create a hybrid system that can switch between architectures
4. Optimize your chosen architecture for your specific robotics application
5. Test scalability with multiple concurrent tasks

## Quiz

1. What is the main advantage of a distributed VLA architecture?
   a) Lower latency
   b) Better modularity and fault tolerance
   c) Simpler implementation
   d) Lower cost

2. When should you use a centralized architecture?
   a) For complex multi-step tasks
   b) For simple, real-time tasks
   c) For research applications only
   d) For safety-critical systems only

3. What does a hybrid architecture combine?
   a) Centralized and distributed approaches
   b) Different programming languages
   c) Multiple robots
   d) Different sensors

## Mini-Project: Architecture Comparison Study

Create and compare three different VLA architectures:
1. A centralized implementation for simple tasks
2. A distributed implementation for complex tasks
3. A hybrid implementation that can adaptively choose architecture
4. Performance benchmarking of all three approaches
5. Analysis of trade-offs and recommendations for different use cases
6. Documentation of your findings and architectural decisions