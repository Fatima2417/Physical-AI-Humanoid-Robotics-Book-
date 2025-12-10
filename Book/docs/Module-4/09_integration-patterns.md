---
sidebar_position: 9
---

# Integration Patterns

## Learning Objectives

By the end of this chapter, you will be able to:
- Identify and implement various integration patterns for Vision-Language-Action systems
- Design scalable architectures for multi-modal robotic systems
- Apply design patterns for robust VLA system integration
- Implement communication protocols between VLA components
- Optimize system performance through effective integration strategies
- Validate and test integrated VLA systems

## Introduction to Integration Patterns

Integration patterns are fundamental architectural approaches for connecting Vision, Language, and Action components in robotic systems. These patterns determine how information flows between perception, cognition, and action systems, impacting system performance, scalability, and maintainability.

### VLA Integration Architecture Patterns

```
VLA Integration Patterns:

Pattern 1: Sequential Pipeline
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Vision    │───→│   Language  │───→│   Action    │
│   (Perceive)│    │   (Understand)│    │   (Execute) │
└─────────────┘    └─────────────┘    └─────────────┘

Pattern 2: Parallel Processing
┌─────────────┐
│   Input     │
└─────┬───────┘
      │
┌─────▼───────┐    ┌─────────────┐
│   Vision    │───→│             │
│   (Parallel) │    │   Fusion    │───→ Action
└─────────────┘    │   (Combine) │
┌─────────────┐    │             │
│   Language  │───→│             │
│   (Parallel) │    └─────────────┘
└─────────────┘

Pattern 3: Feedback Loop
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Vision    │───→│   Language  │───→│   Action    │
│   (Perceive)│    │   (Plan)    │    │   (Execute) │
└─────┬───────┘    └─────────────┘    └─────┬───────┘
      │                                       │
      └───────────────────────────────────────┘
                   (Feedback)
```

### Key Integration Challenges

1. **Timing Synchronization**: Aligning different processing rates
2. **Data Format Conversion**: Standardizing data between components
3. **Latency Management**: Minimizing delays in real-time systems
4. **Resource Sharing**: Managing computational resources effectively
5. **Error Propagation**: Handling errors in multi-stage systems
6. **Scalability**: Supporting multiple concurrent tasks

## Sequential Pipeline Pattern

The sequential pipeline pattern processes information in a linear sequence: Vision → Language → Action.

### Implementation

```python
# sequential_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
import time
import threading
from queue import Queue
from typing import Dict, Any, Optional

class SequentialVLAPipeline(Node):
    def __init__(self):
        super().__init__('sequential_vla_pipeline')

        # Publishers
        self.vision_result_pub = self.create_publisher(String, 'vision_result', 10)
        self.language_result_pub = self.create_publisher(String, 'language_result', 10)
        self.action_result_pub = self.create_publisher(String, 'action_result', 10)
        self.pipeline_status_pub = self.create_publisher(Bool, 'pipeline_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )

        # Services
        self.pipeline_control_srv = self.create_service(
            Trigger, 'control_pipeline', self.pipeline_control_callback
        )

        # Pipeline state
        self.pipeline_active = True
        self.vision_result = None
        self.language_result = None
        self.action_result = None

        # Pipeline queues
        self.vision_queue = Queue(maxsize=5)
        self.language_queue = Queue(maxsize=5)
        self.action_queue = Queue(maxsize=5)

        # Pipeline threads
        self.vision_thread = threading.Thread(target=self.vision_processing_loop, daemon=True)
        self.language_thread = threading.Thread(target=self.language_processing_loop, daemon=True)
        self.action_thread = threading.Thread(target=self.action_execution_loop, daemon=True)

        # Start processing threads
        self.vision_thread.start()
        self.language_thread.start()
        self.action_thread.start()

        # Performance tracking
        self.pipeline_times = []
        self.pipeline_count = 0

        self.get_logger().info('Sequential VLA Pipeline initialized')

    def image_callback(self, msg):
        """Add image to vision processing queue."""
        if self.pipeline_active and not self.vision_queue.full():
            try:
                self.vision_queue.put({
                    'image': msg,
                    'timestamp': time.time()
                }, timeout=0.01)
            except:
                self.get_logger().warn('Vision queue full, dropping frame')

    def detections_callback(self, msg):
        """Process detections (used for context)."""
        self.vision_result = msg

    def command_callback(self, msg):
        """Process natural language command."""
        command = msg.data

        # Add to language processing queue
        if self.pipeline_active and not self.language_queue.full():
            try:
                self.language_queue.put({
                    'command': command,
                    'timestamp': time.time()
                }, timeout=0.01)
            except:
                self.get_logger().warn('Language queue full, dropping command')

    def vision_processing_loop(self):
        """Process vision data sequentially."""
        while rclpy.ok():
            try:
                # Get vision data
                vision_data = self.vision_queue.get(timeout=0.1)

                start_time = time.time()

                # Process vision (simulated)
                processed_result = self.process_vision_data(vision_data['image'])

                # Publish vision result
                vision_result_msg = String()
                vision_result_msg.data = f"Vision processed: {len(processed_result['objects'])} objects detected"
                self.vision_result_pub.publish(vision_result_msg)

                # Pass to language processing
                language_input = {
                    'vision_data': processed_result,
                    'command': getattr(self, 'current_command', ''),
                    'timestamp': vision_data['timestamp']
                }

                if not self.language_queue.full():
                    self.language_queue.put(language_input, timeout=0.01)

                # Track performance
                processing_time = time.time() - start_time
                self.vision_times.append(processing_time)

                if len(self.vision_times) > 100:
                    self.vision_times.pop(0)

            except Exception as e:
                self.get_logger().error(f'Vision processing error: {e}')

    def language_processing_loop(self):
        """Process language data sequentially."""
        while rclpy.ok():
            try:
                # Get language data
                language_data = self.language_queue.get(timeout=0.1)

                start_time = time.time()

                # Process language (simulated)
                if 'command' in language_data:
                    processed_result = self.process_language_command(language_data['command'])
                    self.current_command = language_data['command']
                else:
                    # This is vision-enhanced language processing
                    processed_result = self.process_enhanced_command(
                        language_data['vision_data'],
                        language_data['command']
                    )

                # Publish language result
                language_result_msg = String()
                language_result_msg.data = f"Language processed: {processed_result['intent']}"
                self.language_result_pub.publish(language_result_msg)

                # Pass to action processing
                action_input = {
                    'intent': processed_result['intent'],
                    'parameters': processed_result['parameters'],
                    'vision_context': language_data.get('vision_data', {}),
                    'timestamp': language_data['timestamp']
                }

                if not self.action_queue.full():
                    self.action_queue.put(action_input, timeout=0.01)

                # Track performance
                processing_time = time.time() - start_time
                self.language_times.append(processing_time)

                if len(self.language_times) > 100:
                    self.language_times.pop(0)

            except Exception as e:
                self.get_logger().error(f'Language processing error: {e}')

    def action_execution_loop(self):
        """Execute actions sequentially."""
        while rclpy.ok():
            try:
                # Get action data
                action_data = self.action_queue.get(timeout=0.1)

                start_time = time.time()

                # Execute action (simulated)
                execution_result = self.execute_action(action_data)

                # Publish action result
                action_result_msg = String()
                action_result_msg.data = f"Action executed: {execution_result['status']}"
                self.action_result_pub.publish(action_result_msg)

                # Track performance
                processing_time = time.time() - start_time
                self.action_times.append(processing_time)

                if len(self.action_times) > 100:
                    self.action_times.pop(0)

            except Exception as e:
                self.get_logger().error(f'Action execution error: {e}')

    def process_vision_data(self, image_msg):
        """Process image data and return structured result."""
        # Simulate vision processing
        # In a real system, this would perform object detection, scene analysis, etc.
        result = {
            'objects': [
                {'class': 'person', 'confidence': 0.85, 'position': [1.2, 0.5, 0.0]},
                {'class': 'chair', 'confidence': 0.92, 'position': [0.8, -1.2, 0.0]},
                {'class': 'table', 'confidence': 0.78, 'position': [-0.5, 0.3, 0.0]}
            ],
            'scene_description': 'Room with furniture and people',
            'timestamp': time.time()
        }
        return result

    def process_language_command(self, command):
        """Process natural language command and extract intent."""
        # Simulate language processing
        command_lower = command.lower()

        if 'go to' in command_lower or 'move to' in command_lower:
            intent = 'navigate_to_object'
            target = self.extract_target_object(command_lower)
        elif 'grasp' in command_lower or 'pick up' in command_lower:
            intent = 'grasp_object'
            target = self.extract_target_object(command_lower)
        elif 'describe' in command_lower:
            intent = 'describe_scene'
            target = 'scene'
        else:
            intent = 'unknown'
            target = 'unknown'

        result = {
            'intent': intent,
            'parameters': {'target': target},
            'confidence': 0.85,
            'timestamp': time.time()
        }

        return result

    def process_enhanced_command(self, vision_data, command):
        """Process command with vision context."""
        # Use vision data to enhance language understanding
        vision_objects = [obj['class'] for obj in vision_data.get('objects', [])]

        result = self.process_language_command(command)

        # Enhance with vision context
        if result['parameters']['target'] == 'unknown' and vision_objects:
            # If target is unknown, use the most confident object
            most_confident_obj = max(
                vision_data['objects'],
                key=lambda x: x['confidence']
            )
            result['parameters']['target'] = most_confident_obj['class']

        return result

    def execute_action(self, action_data):
        """Execute the planned action."""
        intent = action_data['intent']
        params = action_data['parameters']

        # Simulate action execution
        if intent == 'navigate_to_object':
            status = f"Navigating to {params['target']}"
        elif intent == 'grasp_object':
            status = f"Attempting to grasp {params['target']}"
        elif intent == 'describe_scene':
            status = f"Describing scene with {len(action_data['vision_context'].get('objects', []))} objects"
        else:
            status = f"Unknown action: {intent}"

        result = {
            'status': status,
            'success': True,
            'execution_time': 1.0,  # Simulated time
            'timestamp': time.time()
        }

        return result

    def extract_target_object(self, command):
        """Extract target object from command."""
        # Simple extraction - in reality use NLP
        known_objects = ['person', 'chair', 'table', 'bottle', 'cup', 'robot', 'ball']
        for obj in known_objects:
            if obj in command:
                return obj
        return 'unknown'

    def pipeline_control_callback(self, request, response):
        """Control pipeline activation."""
        if request.srv_name == 'start':
            self.pipeline_active = True
            response.success = True
            response.message = 'Pipeline started'
        elif request.srv_name == 'stop':
            self.pipeline_active = False
            response.success = True
            response.message = 'Pipeline stopped'
        else:
            response.success = False
            response.message = 'Unknown command'

        # Publish status
        status_msg = Bool()
        status_msg.data = self.pipeline_active
        self.pipeline_status_pub.publish(status_msg)

        return response

    def get_pipeline_performance(self):
        """Get pipeline performance metrics."""
        return {
            'vision_avg_time': sum(self.vision_times) / len(self.vision_times) if self.vision_times else 0,
            'language_avg_time': sum(self.language_times) / len(self.language_times) if self.language_times else 0,
            'action_avg_time': sum(self.action_times) / len(self.action_times) if self.action_times else 0,
            'vision_fps': 1.0 / (sum(self.vision_times) / len(self.vision_times)) if self.vision_times else 0,
            'language_rate': 1.0 / (sum(self.language_times) / len(self.language_times)) if self.language_times else 0,
            'action_rate': 1.0 / (sum(self.action_times) / len(self.action_times)) if self.action_times else 0
        }

def main(args=None):
    rclpy.init(args=args)
    pipeline = SequentialVLAPipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pipeline.get_logger().info('Shutting down Sequential VLA Pipeline')
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parallel Processing Pattern

The parallel processing pattern allows multiple components to process data simultaneously, with fusion happening at specific points.

### Implementation

```python
# parallel_processing.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
import threading
import time
from queue import Queue, Empty
from typing import Dict, Any, Optional

class ParallelVLAPipeline(Node):
    def __init__(self):
        super().__init__('parallel_vla_pipeline')

        # Publishers
        self.fused_result_pub = self.create_publisher(String, 'fused_result', 10)
        self.pipeline_status_pub = self.create_publisher(Bool, 'parallel_pipeline_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )

        # Data storage
        self.latest_image = None
        self.latest_command = None
        self.latest_detections = None

        # Processing queues
        self.vision_results = {}
        self.language_results = {}

        # Processing locks
        self.vision_lock = threading.Lock()
        self.language_lock = threading.Lock()
        self.fusion_lock = threading.Lock()

        # Processing threads
        self.vision_thread = threading.Thread(target=self.parallel_vision_processing, daemon=True)
        self.language_thread = threading.Thread(target=self.parallel_language_processing, daemon=True)
        self.fusion_thread = threading.Thread(target=self.fusion_processing, daemon=True)

        # Start threads
        self.vision_thread.start()
        self.language_thread.start()
        self.fusion_thread.start()

        # Synchronization
        self.new_vision_data = threading.Event()
        self.new_language_data = threading.Event()

        # Performance tracking
        self.vision_times = []
        self.language_times = []
        self.fusion_times = []

        self.get_logger().info('Parallel VLA Pipeline initialized')

    def image_callback(self, msg):
        """Process image in parallel thread."""
        self.latest_image = msg
        self.new_vision_data.set()

    def command_callback(self, msg):
        """Process command in parallel thread."""
        self.latest_command = msg.data
        self.new_language_data.set()

    def detections_callback(self, msg):
        """Update detections."""
        self.latest_detections = msg

    def parallel_vision_processing(self):
        """Process vision data in parallel."""
        while rclpy.ok():
            try:
                self.new_vision_data.wait(timeout=0.1)
                self.new_vision_data.clear()

                if self.latest_image is not None:
                    start_time = time.time()

                    # Process vision data
                    result = self.process_vision_data_parallel(self.latest_image)

                    with self.vision_lock:
                        self.vision_results[result['timestamp']] = result

                    # Track performance
                    processing_time = time.time() - start_time
                    self.vision_times.append(processing_time)

                    if len(self.vision_times) > 100:
                        self.vision_times.pop(0)

            except Exception as e:
                self.get_logger().error(f'Parallel vision processing error: {e}')

    def parallel_language_processing(self):
        """Process language data in parallel."""
        while rclpy.ok():
            try:
                self.new_language_data.wait(timeout=0.1)
                self.new_language_data.clear()

                if self.latest_command is not None:
                    start_time = time.time()

                    # Process language data
                    result = self.process_language_data_parallel(self.latest_command)

                    with self.language_lock:
                        self.language_results[result['timestamp']] = result

                    # Track performance
                    processing_time = time.time() - start_time
                    self.language_times.append(processing_time)

                    if len(self.language_times) > 100:
                        self.language_times.pop(0)

            except Exception as e:
                self.get_logger().error(f'Parallel language processing error: {e}')

    def fusion_processing(self):
        """Fuse vision and language results."""
        while rclpy.ok():
            try:
                # Wait briefly to allow both systems to process
                time.sleep(0.05)

                start_time = time.time()

                # Get most recent results from both systems
                with self.vision_lock:
                    latest_vision = max(self.vision_results.keys()) if self.vision_results else None
                    if latest_vision:
                        vision_result = self.vision_results[latest_vision]

                with self.language_lock:
                    latest_language = max(self.language_results.keys()) if self.language_results else None
                    if latest_language:
                        language_result = self.language_results[latest_language]

                # Fuse results if both are available
                if latest_vision and latest_language:
                    fused_result = self.fuse_vision_language_results(
                        vision_result, language_result
                    )

                    # Publish fused result
                    fused_msg = String()
                    fused_msg.data = fused_result['description']
                    self.fused_result_pub.publish(fused_msg)

                    # Track performance
                    fusion_time = time.time() - start_time
                    self.fusion_times.append(fusion_time)

                    if len(self.fusion_times) > 100:
                        self.fusion_times.pop(0)

            except Exception as e:
                self.get_logger().error(f'Fusion processing error: {e}')

    def process_vision_data_parallel(self, image_msg):
        """Process vision data with parallel computation."""
        # Simulate parallel vision processing
        # In reality, this might use multiple threads for different vision tasks
        result = {
            'objects': [
                {'class': 'person', 'confidence': 0.85, 'bbox': [100, 100, 200, 250]},
                {'class': 'chair', 'confidence': 0.92, 'bbox': [300, 150, 450, 300]}
            ],
            'scene_analysis': 'Indoor environment with furniture',
            'timestamp': time.time(),
            'processing_time': 0.05  # Simulated processing time
        }
        return result

    def process_language_data_parallel(self, command):
        """Process language data with parallel computation."""
        # Simulate parallel language processing
        # In reality, this might use multiple NLP models in parallel
        result = {
            'intent': 'navigate_to_object',
            'entities': [{'type': 'object', 'value': 'person'}],
            'confidence': 0.88,
            'timestamp': time.time(),
            'processing_time': 0.03  # Simulated processing time
        }
        return result

    def fuse_vision_language_results(self, vision_result, language_result):
        """Fuse vision and language results."""
        # Find matching objects based on command intent
        target_entity = None
        for entity in language_result['entities']:
            if entity['type'] == 'object':
                target_entity = entity['value']
                break

        # Find matching object in vision results
        target_object = None
        if target_entity:
            for obj in vision_result['objects']:
                if obj['class'] == target_entity:
                    target_object = obj
                    break

        # Create fused result
        fused_description = f"Command '{language_result['intent']}' for {target_entity if target_entity else 'unknown target'}"
        if target_object:
            fused_description += f", found object with {target_object['confidence']:.2f} confidence"

        result = {
            'description': fused_description,
            'intent': language_result['intent'],
            'target_object': target_object,
            'vision_context': vision_result,
            'language_context': language_result,
            'timestamp': time.time()
        }

        return result

    def get_parallel_performance(self):
        """Get performance metrics for parallel processing."""
        return {
            'vision_avg_time': sum(self.vision_times) / len(self.vision_times) if self.vision_times else 0,
            'language_avg_time': sum(self.language_times) / len(self.language_times) if self.language_times else 0,
            'fusion_avg_time': sum(self.fusion_times) / len(self.fusion_times) if self.fusion_times else 0,
            'vision_concurrent_rate': len(self.vision_times) / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0,
            'language_concurrent_rate': len(self.language_times) / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0
        }

def main(args=None):
    rclpy.init(args=args)
    parallel_pipeline = ParallelVLAPipeline()

    try:
        rclpy.spin(parallel_pipeline)
    except KeyboardInterrupt:
        parallel_pipeline.get_logger().info('Shutting down Parallel VLA Pipeline')
    finally:
        parallel_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Feedback Loop Integration Pattern

The feedback loop pattern incorporates continuous feedback from action execution to refine vision and language processing.

### Implementation

```python
# feedback_loop_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
import threading
import time
from typing import Dict, Any, Optional
import numpy as np

class FeedbackLoopVLAPipeline(Node):
    def __init__(self):
        super().__init__('feedback_loop_vla_pipeline')

        # Publishers
        self.action_cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.refined_vision_pub = self.create_publisher(String, 'refined_vision_result', 10)
        self.refined_language_pub = self.create_publisher(String, 'refined_language_result', 10)
        self.feedback_status_pub = self.create_publisher(Bool, 'feedback_active', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.action_feedback_sub = self.create_subscription(
            String, 'action_feedback', self.action_feedback_callback, 10
        )

        # Feedback loop components
        self.feedback_active = True
        self.action_history = []
        self.vision_context = {}
        self.language_context = {}

        # Feedback processing
        self.feedback_processing_thread = threading.Thread(
            target=self.feedback_processing_loop, daemon=True
        )
        self.feedback_processing_thread.start()

        # Action execution tracking
        self.current_action = None
        self.action_start_time = None
        self.action_completion_threshold = 0.95  # 95% confidence for completion

        # Performance tracking
        self.feedback_times = []
        self.refinement_iterations = 0

        # Feedback weights for refinement
        self.vision_refinement_weight = 0.8
        self.language_refinement_weight = 0.9
        self.action_refinement_weight = 0.7

        self.get_logger().info('Feedback Loop VLA Pipeline initialized')

    def image_callback(self, msg):
        """Process image and incorporate feedback."""
        if not self.feedback_active:
            return

        # Process image with current context
        vision_result = self.process_vision_with_context(msg)

        # Refine based on action feedback
        refined_result = self.refine_vision_result_with_feedback(vision_result)

        # Publish refined result
        refined_msg = String()
        refined_msg.data = f"Refined vision: {refined_result['description']}"
        self.refined_vision_pub.publish(refined_msg)

        # Update vision context
        self.vision_context = refined_result

    def command_callback(self, msg):
        """Process command and incorporate feedback."""
        if not self.feedback_active:
            return

        command = msg.data

        # Process command with current context
        language_result = self.process_command_with_context(command)

        # Refine based on action feedback
        refined_result = self.refine_language_result_with_feedback(language_result)

        # Publish refined result
        refined_msg = String()
        refined_msg.data = f"Refined language: {refined_result['intent']}"
        self.refined_language_pub.publish(refined_msg)

        # Update language context
        self.language_context = refined_result

    def odom_callback(self, msg):
        """Process odometry for action feedback."""
        if self.current_action:
            # Calculate how close we are to action completion
            action_progress = self.calculate_action_progress(msg)

            if action_progress > self.action_completion_threshold:
                # Action completed, update feedback
                self.update_action_feedback('completed', action_progress)

    def joint_state_callback(self, msg):
        """Process joint states for manipulation feedback."""
        if self.current_action and 'manipulation' in self.current_action.get('type', ''):
            # Check if manipulation action is progressing
            manipulation_progress = self.calculate_manipulation_progress(msg)
            self.update_action_feedback('manipulation_progress', manipulation_progress)

    def action_feedback_callback(self, msg):
        """Process action feedback for refinement."""
        try:
            feedback_data = eval(msg.data)  # In real system, use proper serialization
            self.process_action_feedback(feedback_data)
        except Exception as e:
            self.get_logger().error(f'Could not process action feedback: {e}')

    def process_vision_with_context(self, image_msg):
        """Process vision with current context."""
        # Basic vision processing
        vision_result = self.basic_vision_processing(image_msg)

        # Incorporate context from previous actions
        if self.action_history:
            recent_action = self.action_history[-1]
            vision_result['context'] = recent_action.get('result', {})

        # Incorporate language context
        if self.language_context:
            vision_result['intent_guidance'] = self.language_context.get('intent', 'explore')

        return vision_result

    def process_command_with_context(self, command):
        """Process command with current context."""
        # Basic command processing
        language_result = self.basic_language_processing(command)

        # Incorporate vision context
        if self.vision_context:
            language_result['perception_context'] = self.vision_context.get('objects', [])

        # Incorporate action history
        if self.action_history:
            language_result['action_context'] = self.action_history[-1].get('type', 'unknown')

        return language_result

    def refine_vision_result_with_feedback(self, vision_result):
        """Refine vision result based on action feedback."""
        if not self.action_history:
            return vision_result

        # Get the most recent action
        recent_action = self.action_history[-1]

        # Refine based on action outcome
        if recent_action.get('success', False):
            # Action succeeded, reinforce successful patterns
            vision_result['confidence'] = min(1.0, vision_result.get('confidence', 0.8) * 1.1)
        else:
            # Action failed, adjust vision processing
            vision_result['confidence'] = max(0.1, vision_result.get('confidence', 0.8) * 0.9)

        # Update objects based on action feedback
        if recent_action.get('type') == 'grasp' and recent_action.get('success', False):
            # Object was successfully grasped, remove from scene
            target_obj = recent_action.get('target', '')
            if target_obj:
                vision_result['objects'] = [
                    obj for obj in vision_result.get('objects', [])
                    if obj.get('class') != target_obj
                ]

        # Add refinement metadata
        vision_result['refined'] = True
        vision_result['refinement_iteration'] = self.refinement_iterations
        vision_result['feedback_incorporated'] = True

        self.refinement_iterations += 1
        return vision_result

    def refine_language_result_with_feedback(self, language_result):
        """Refine language result based on action feedback."""
        if not self.action_history:
            return language_result

        # Get the most recent action
        recent_action = self.action_history[-1]

        # Refine based on action outcome
        if recent_action.get('success', False):
            # Action succeeded, increase confidence in similar commands
            language_result['confidence'] = min(1.0, language_result.get('confidence', 0.8) * 1.1)
        else:
            # Action failed, reconsider command interpretation
            language_result['confidence'] = max(0.1, language_result.get('confidence', 0.8) * 0.9)

        # Update intent based on feedback
        if recent_action.get('type') == 'navigation' and not recent_action.get('success', False):
            # Navigation failed, perhaps we need to navigate to a different location
            if language_result['intent'] == 'navigate_to_object':
                language_result['alternative_intent'] = 'find_alternative_path'

        # Add refinement metadata
        language_result['refined'] = True
        language_result['refinement_iteration'] = self.refinement_iterations
        language_result['feedback_incorporated'] = True

        return language_result

    def calculate_action_progress(self, odom_msg):
        """Calculate action progress based on odometry."""
        if not self.current_action or not self.action_start_time:
            return 0.0

        # Calculate progress based on movement
        current_x = odom_msg.pose.pose.position.x
        current_y = odom_msg.pose.pose.position.y

        if 'target_position' in self.current_action:
            target_x = self.current_action['target_position']['x']
            target_y = self.current_action['target_position']['y']

            distance_traveled = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
            total_distance = self.current_action.get('total_distance', 1.0)

            progress = min(1.0, distance_traveled / total_distance)
            return progress

        return 0.0

    def calculate_manipulation_progress(self, joint_state_msg):
        """Calculate manipulation progress based on joint states."""
        if not self.current_action or not self.action_start_time:
            return 0.0

        # Calculate progress based on joint positions
        # This is a simplified example - real implementation would be more complex
        if 'target_joints' in self.current_action:
            target_joints = self.current_action['target_joints']
            current_joints = dict(zip(joint_state_msg.name, joint_state_msg.position))

            total_error = 0.0
            for joint_name, target_pos in target_joints.items():
                if joint_name in current_joints:
                    error = abs(current_joints[joint_name] - target_pos)
                    total_error += error

            # Convert to progress (0 to 1)
            avg_error = total_error / len(target_joints) if target_joints else 0
            progress = max(0.0, min(1.0, 1.0 - avg_error))  # Lower error = higher progress

            return progress

        return 0.0

    def process_action_feedback(self, feedback_data):
        """Process action feedback for system refinement."""
        action_type = feedback_data.get('type', 'unknown')
        success = feedback_data.get('success', False)
        confidence = feedback_data.get('confidence', 0.5)

        # Update action history
        self.action_history.append({
            'type': action_type,
            'success': success,
            'confidence': confidence,
            'timestamp': time.time(),
            'result': feedback_data.get('result', {})
        })

        # Limit history size
        if len(self.action_history) > 50:
            self.action_history.pop(0)

        # Refine system based on feedback
        if success:
            self.get_logger().info(f'Action {action_type} succeeded with confidence {confidence:.2f}')
        else:
            self.get_logger().warn(f'Action {action_type} failed with confidence {confidence:.2f}')
            # In a real system, you might adjust parameters based on failures

    def feedback_processing_loop(self):
        """Continuous feedback processing loop."""
        while rclpy.ok():
            try:
                # Process feedback and refine system
                self.refine_system_based_on_feedback()

                # Publish feedback status
                status_msg = Bool()
                status_msg.data = self.feedback_active
                self.feedback_status_pub.publish(status_msg)

                # Sleep to control processing rate
                time.sleep(0.1)  # 10 Hz feedback processing

            except Exception as e:
                self.get_logger().error(f'Feedback processing error: {e}')
                time.sleep(0.1)

    def refine_system_based_on_feedback(self):
        """Refine system parameters based on feedback."""
        if not self.action_history:
            return

        # Calculate success rate
        recent_actions = self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
        success_count = sum(1 for action in recent_actions if action.get('success', False))
        success_rate = success_count / len(recent_actions) if recent_actions else 0

        # Adjust parameters based on success rate
        if success_rate < 0.6:  # Low success rate
            # Be more conservative
            self.vision_refinement_weight *= 0.95
            self.language_refinement_weight *= 0.95
            self.get_logger().warn(f'Low success rate ({success_rate:.2f}), being more conservative')
        elif success_rate > 0.8:  # High success rate
            # Be more aggressive
            self.vision_refinement_weight *= 1.05
            self.language_refinement_weight *= 1.05
            self.get_logger().info(f'High success rate ({success_rate:.2f}), being more aggressive')

        # Limit weights to reasonable range
        self.vision_refinement_weight = max(0.5, min(1.5, self.vision_refinement_weight))
        self.language_refinement_weight = max(0.5, min(1.5, self.language_refinement_weight))

    def execute_action_with_feedback(self, action_plan):
        """Execute action with feedback incorporation."""
        self.current_action = action_plan
        self.action_start_time = time.time()

        # Execute the action
        success = self.execute_basic_action(action_plan)

        # Record action result for feedback
        action_result = {
            'type': action_plan.get('type', 'unknown'),
            'success': success,
            'confidence': action_plan.get('confidence', 0.8),
            'timestamp': time.time(),
            'plan': action_plan
        }

        self.action_history.append(action_result)

        # Limit history size
        if len(self.action_history) > 50:
            self.action_history.pop(0)

        return success

    def basic_vision_processing(self, image_msg):
        """Basic vision processing (simulated)."""
        # Simulate vision processing
        result = {
            'objects': [
                {'class': 'person', 'confidence': 0.85, 'position': [1.2, 0.5, 0.0]},
                {'class': 'chair', 'confidence': 0.92, 'position': [0.8, -1.2, 0.0]}
            ],
            'scene_analysis': 'Indoor environment with furniture',
            'confidence': 0.88,
            'timestamp': time.time()
        }
        return result

    def basic_language_processing(self, command):
        """Basic language processing (simulated)."""
        # Simulate language processing
        result = {
            'intent': 'navigate_to_object',
            'entities': [{'type': 'object', 'value': 'person'}],
            'confidence': 0.85,
            'timestamp': time.time()
        }
        return result

    def execute_basic_action(self, action_plan):
        """Execute basic action (simulated)."""
        # Simulate action execution
        action_type = action_plan.get('type', 'unknown')

        if action_type == 'navigation':
            # Simulate navigation
            cmd = Twist()
            cmd.linear.x = 0.5  # Move forward
            self.action_cmd_pub.publish(cmd)
            time.sleep(1.0)  # Simulate action duration
            return True  # Simulate success

        elif action_type == 'manipulation':
            # Simulate manipulation
            # In a real system, this would control robot arms/hands
            time.sleep(2.0)  # Simulate manipulation duration
            return True  # Simulate success

        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def enable_feedback_loop(self, enable=True):
        """Enable or disable the feedback loop."""
        self.feedback_active = enable
        status_msg = Bool()
        status_msg.data = enable
        self.feedback_status_pub.publish(status_msg)
        self.get_logger().info(f'Feedback loop {"enabled" if enable else "disabled"}')

def main(args=None):
    rclpy.init(args=args)
    feedback_pipeline = FeedbackLoopVLAPipeline()

    try:
        rclpy.spin(feedback_pipeline)
    except KeyboardInterrupt:
        feedback_pipeline.get_logger().info('Shutting down Feedback Loop VLA Pipeline')
    finally:
        feedback_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Hierarchical Integration Pattern

The hierarchical pattern organizes VLA components in layers of abstraction, with higher-level components coordinating lower-level ones.

### Implementation

```python
# hierarchical_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool, UInt8
from geometry_msgs.msg import Twist, Pose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, ActionClient
import threading
import time
from typing import Dict, Any, Optional, List
from enum import Enum

class TaskPriority(Enum):
    EMERGENCY = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class TaskStatus(Enum):
    PENDING = 1
    EXECUTING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

class HierarchicalVLAPipeline(Node):
    def __init__(self):
        super().__init__('hierarchical_vla_pipeline')

        # Publishers
        self.high_level_status_pub = self.create_publisher(String, 'high_level_status', 10)
        self.mid_level_status_pub = self.create_publisher(String, 'mid_level_status', 10)
        self.low_level_status_pub = self.create_publisher(String, 'low_level_status', 10)

        # Subscribers
        self.high_level_command_sub = self.create_subscription(
            String, 'high_level_command', self.high_level_command_callback, 10
        )
        self.mid_level_command_sub = self.create_subscription(
            String, 'mid_level_command', self.mid_level_command_callback, 10
        )

        # Action servers
        self.high_level_action_server = ActionServer(
            self,
            HighLevelTask,
            'high_level_task',
            self.execute_high_level_task
        )
        self.mid_level_action_server = ActionServer(
            self,
            MidLevelTask,
            'mid_level_task',
            self.execute_mid_level_task
        )

        # Action clients
        self.low_level_action_client = ActionClient(
            self,
            LowLevelTask,
            'low_level_task'
        )

        # Task management
        self.task_queue = []
        self.active_tasks = {}
        self.task_priorities = {}

        # System state
        self.current_behavior = 'idle'
        self.system_mode = 'autonomous'
        self.emergency_active = False

        # Hierarchical components
        self.high_level_planner = HighLevelPlanner(self)
        self.mid_level_coordinator = MidLevelCoordinator(self)
        self.low_level_executor = LowLevelExecutor(self)

        # Threading for task management
        self.task_manager_thread = threading.Thread(
            target=self.task_management_loop, daemon=True
        )
        self.task_manager_thread.start()

        # Performance tracking
        self.task_execution_times = []
        self.hierarchy_levels = 3  # High, Mid, Low

        self.get_logger().info('Hierarchical VLA Pipeline initialized')

    def high_level_command_callback(self, msg):
        """Process high-level commands."""
        command = msg.data
        self.get_logger().info(f'Received high-level command: {command}')

        # Parse command and create high-level task
        task = self.parse_high_level_command(command)

        if task:
            # Add to task queue with priority
            self.add_task(task, TaskPriority.NORMAL)

    def mid_level_command_callback(self, msg):
        """Process mid-level commands."""
        command = msg.data
        self.get_logger().info(f'Received mid-level command: {command}')

        # Create mid-level task
        task = self.parse_mid_level_command(command)

        if task:
            self.add_task(task, TaskPriority.HIGH)

    def add_task(self, task: Dict, priority: TaskPriority):
        """Add task to queue with priority."""
        task['priority'] = priority
        task['status'] = TaskStatus.PENDING
        task['timestamp'] = time.time()

        # Insert task based on priority
        insert_index = 0
        for i, existing_task in enumerate(self.task_queue):
            if existing_task['priority'].value > priority.value:
                insert_index = i
                break

        self.task_queue.insert(insert_index, task)
        self.get_logger().info(f'Task added with priority {priority}: {task["type"]}')

    def task_management_loop(self):
        """Manage task execution across hierarchy levels."""
        while rclpy.ok():
            try:
                # Check for emergency tasks first
                if self.emergency_active:
                    self.handle_emergency_tasks()
                    continue

                # Process tasks based on priority and system mode
                if self.task_queue and not self.is_system_busy():
                    task = self.task_queue.pop(0)
                    self.execute_task(task)

                time.sleep(0.05)  # 20 Hz task management

            except Exception as e:
                self.get_logger().error(f'Task management error: {e}')
                time.sleep(0.1)

    def execute_task(self, task: Dict):
        """Execute task based on its level."""
        task_type = task.get('type', 'unknown')
        task_level = task.get('level', 'high')

        self.get_logger().info(f'Executing task: {task_type} at level {task_level}')

        if task_level == 'high':
            result = self.high_level_planner.execute_task(task)
        elif task_level == 'mid':
            result = self.mid_level_coordinator.execute_task(task)
        elif task_level == 'low':
            result = self.low_level_executor.execute_task(task)
        else:
            result = {'success': False, 'message': f'Unknown task level: {task_level}'}

        # Update task status
        task['status'] = TaskStatus.COMPLETED if result.get('success', False) else TaskStatus.FAILED
        task['result'] = result

        # Publish status
        status_msg = String()
        status_msg.data = f'Task {task_type} {task["status"].name}: {result.get("message", "")}'
        self.high_level_status_pub.publish(status_msg)

    def is_system_busy(self) -> bool:
        """Check if system is busy with active tasks."""
        active_count = sum(1 for task in self.active_tasks.values()
                          if task['status'] in [TaskStatus.EXECUTING, TaskStatus.PENDING])
        return active_count > 0

    def handle_emergency_tasks(self):
        """Handle emergency tasks with highest priority."""
        # In emergency mode, cancel all ongoing tasks
        for task_id, task in self.active_tasks.items():
            if task['status'] == TaskStatus.EXECUTING:
                self.cancel_task(task_id)

        # Execute emergency behavior
        emergency_task = {
            'type': 'emergency_stop',
            'level': 'high',
            'priority': TaskPriority.EMERGENCY,
            'parameters': {}
        }
        self.execute_task(emergency_task)

    def parse_high_level_command(self, command: str) -> Optional[Dict]:
        """Parse high-level natural language command."""
        command_lower = command.lower()

        if 'navigate' in command_lower or 'go to' in command_lower:
            return {
                'type': 'navigation_task',
                'level': 'high',
                'target': self.extract_target_location(command_lower),
                'parameters': {'speed': 'normal', 'avoid_obstacles': True}
            }
        elif 'grasp' in command_lower or 'pick up' in command_lower:
            return {
                'type': 'manipulation_task',
                'level': 'high',
                'target': self.extract_target_object(command_lower),
                'parameters': {'precision': 'normal', 'force': 'medium'}
            }
        elif 'explore' in command_lower:
            return {
                'type': 'exploration_task',
                'level': 'high',
                'parameters': {'area_size': 'medium', 'coverage': 'thorough'}
            }
        elif 'stop' in command_lower:
            return {
                'type': 'stop_task',
                'level': 'high',
                'parameters': {}
            }
        else:
            return {
                'type': 'unknown_task',
                'level': 'high',
                'command': command,
                'parameters': {}
            }

    def parse_mid_level_command(self, command: str) -> Optional[Dict]:
        """Parse mid-level command for specific behaviors."""
        command_lower = command.lower()

        if 'move_forward' in command_lower:
            return {
                'type': 'move_forward',
                'level': 'mid',
                'parameters': {'distance': 1.0, 'speed': 0.5}
            }
        elif 'turn_left' in command_lower:
            return {
                'type': 'turn_left',
                'level': 'mid',
                'parameters': {'angle': 90, 'speed': 0.3}
            }
        elif 'approach_object' in command_lower:
            return {
                'type': 'approach_object',
                'level': 'mid',
                'target': self.extract_target_object(command_lower),
                'parameters': {'distance': 0.5, 'approach_method': 'safe'}
            }
        else:
            return {
                'type': 'low_level_command',
                'level': 'mid',
                'command': command,
                'parameters': {}
            }

    def extract_target_location(self, command: str) -> str:
        """Extract target location from command."""
        locations = ['kitchen', 'living_room', 'bedroom', 'office', 'hallway', 'entrance']
        for loc in locations:
            if loc in command:
                return loc
        return 'default_location'

    def extract_target_object(self, command: str) -> str:
        """Extract target object from command."""
        objects = ['person', 'chair', 'table', 'bottle', 'cup', 'ball', 'box']
        for obj in objects:
            if obj in command:
                return obj
        return 'unknown_object'

    def execute_high_level_task(self, goal_handle):
        """Execute high-level task via action server."""
        self.get_logger().info(f'Executing high-level task: {goal_handle.request.task_type}')

        task = {
            'type': goal_handle.request.task_type,
            'level': 'high',
            'parameters': goal_handle.request.parameters,
            'goal_handle': goal_handle
        }

        result = HighLevelTask.Result()

        try:
            # Execute through high-level planner
            execution_result = self.high_level_planner.execute_task(task)

            if execution_result.get('success', False):
                result.success = True
                result.message = execution_result.get('message', 'Task completed successfully')
                goal_handle.succeed()
            else:
                result.success = False
                result.message = execution_result.get('message', 'Task failed')
                goal_handle.abort()

        except Exception as e:
            result.success = False
            result.message = f'High-level task execution error: {str(e)}'
            goal_handle.abort()

        return result

    def execute_mid_level_task(self, goal_handle):
        """Execute mid-level task via action server."""
        self.get_logger().info(f'Executing mid-level task: {goal_handle.request.task_type}')

        task = {
            'type': goal_handle.request.task_type,
            'level': 'mid',
            'parameters': goal_handle.request.parameters,
            'goal_handle': goal_handle
        }

        result = MidLevelTask.Result()

        try:
            # Execute through mid-level coordinator
            execution_result = self.mid_level_coordinator.execute_task(task)

            if execution_result.get('success', False):
                result.success = True
                result.message = execution_result.get('message', 'Mid-level task completed')
                goal_handle.succeed()
            else:
                result.success = False
                result.message = execution_result.get('message', 'Mid-level task failed')
                goal_handle.abort()

        except Exception as e:
            result.success = False
            result.message = f'Mid-level task execution error: {str(e)}'
            goal_handle.abort()

        return result

class HighLevelPlanner:
    def __init__(self, node):
        self.node = node
        self.behavior_registry = {
            'navigation_task': self.execute_navigation_task,
            'manipulation_task': self.execute_manipulation_task,
            'exploration_task': self.execute_exploration_task,
            'stop_task': self.execute_stop_task
        }

    def execute_task(self, task: Dict) -> Dict:
        """Execute high-level task."""
        task_type = task.get('type', 'unknown')
        parameters = task.get('parameters', {})

        if task_type in self.behavior_registry:
            return self.behavior_registry[task_type](task)
        else:
            return {
                'success': False,
                'message': f'Unknown high-level task: {task_type}'
            }

    def execute_navigation_task(self, task: Dict) -> Dict:
        """Execute navigation task."""
        target = task.get('target', 'unknown_location')
        parameters = task.get('parameters', {})

        self.node.get_logger().info(f'Planning navigation to {target}')

        # Break down into mid-level tasks
        mid_level_tasks = [
            {'type': 'localize', 'level': 'mid', 'parameters': {}},
            {'type': 'plan_path', 'level': 'mid', 'parameters': {'target': target}},
            {'type': 'execute_navigation', 'level': 'mid', 'parameters': parameters}
        ]

        # Execute mid-level tasks
        for mid_task in mid_level_tasks:
            result = self.node.mid_level_coordinator.execute_task(mid_task)
            if not result.get('success', False):
                return {
                    'success': False,
                    'message': f'Mid-level task failed: {mid_task["type"]}'
                }

        return {
            'success': True,
            'message': f'Navigation to {target} completed successfully'
        }

    def execute_manipulation_task(self, task: Dict) -> Dict:
        """Execute manipulation task."""
        target = task.get('target', 'unknown_object')
        parameters = task.get('parameters', {})

        self.node.get_logger().info(f'Planning manipulation of {target}')

        # Break down into mid-level tasks
        mid_level_tasks = [
            {'type': 'locate_object', 'level': 'mid', 'parameters': {'object': target}},
            {'type': 'approach_object', 'level': 'mid', 'parameters': {'object': target}},
            {'type': 'grasp_object', 'level': 'mid', 'parameters': {**parameters, 'object': target}}
        ]

        # Execute mid-level tasks
        for mid_task in mid_level_tasks:
            result = self.node.mid_level_coordinator.execute_task(mid_task)
            if not result.get('success', False):
                return {
                    'success': False,
                    'message': f'Mid-level task failed: {mid_task["type"]}'
                }

        return {
            'success': True,
            'message': f'Manipulation of {target} completed successfully'
        }

    def execute_exploration_task(self, task: Dict) -> Dict:
        """Execute exploration task."""
        parameters = task.get('parameters', {})

        self.node.get_logger().info('Planning exploration task')

        # Break down into mid-level tasks
        mid_level_tasks = [
            {'type': 'initialize_exploration', 'level': 'mid', 'parameters': parameters},
            {'type': 'explore_area', 'level': 'mid', 'parameters': parameters},
            {'type': 'return_to_base', 'level': 'mid', 'parameters': {}}
        ]

        # Execute mid-level tasks
        for mid_task in mid_level_tasks:
            result = self.node.mid_level_coordinator.execute_task(mid_task)
            if not result.get('success', False):
                return {
                    'success': False,
                    'message': f'Mid-level task failed: {mid_task["type"]}'
                }

        return {
            'success': True,
            'message': 'Exploration task completed successfully'
        }

    def execute_stop_task(self, task: Dict) -> Dict:
        """Execute stop task."""
        self.node.get_logger().info('Executing stop task')

        # Send stop command to all subsystems
        stop_cmd = Twist()
        self.node.action_cmd_pub.publish(stop_cmd)

        return {
            'success': True,
            'message': 'Stop command executed successfully'
        }

class MidLevelCoordinator:
    def __init__(self, node):
        self.node = node
        self.primitive_registry = {
            'localize': self.execute_localize,
            'plan_path': self.execute_plan_path,
            'execute_navigation': self.execute_navigation_primitive,
            'locate_object': self.execute_locate_object,
            'approach_object': self.execute_approach_object,
            'grasp_object': self.execute_grasp_object,
            'explore_area': self.execute_explore_area,
            'initialize_exploration': self.execute_initialize_exploration,
            'return_to_base': self.execute_return_to_base
        }

    def execute_task(self, task: Dict) -> Dict:
        """Execute mid-level task."""
        task_type = task.get('type', 'unknown')
        parameters = task.get('parameters', {})

        if task_type in self.primitive_registry:
            return self.primitive_registry[task_type](task)
        else:
            return {
                'success': False,
                'message': f'Unknown mid-level task: {task_type}'
            }

    def execute_localize(self, task: Dict) -> Dict:
        """Execute localization primitive."""
        self.node.get_logger().info('Executing localization')

        # In a real system, this would use localization algorithms
        # For simulation, assume localization succeeds
        return {
            'success': True,
            'message': 'Localization completed',
            'position': {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        }

    def execute_plan_path(self, task: Dict) -> Dict:
        """Execute path planning primitive."""
        target = task.get('parameters', {}).get('target', 'unknown')

        self.node.get_logger().info(f'Planning path to {target}')

        # In a real system, this would use path planning algorithms
        # For simulation, return mock path
        return {
            'success': True,
            'message': f'Path to {target} planned successfully',
            'waypoints': [{'x': 1.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}, {'x': target_x, 'y': target_y}]
        }

    def execute_navigation_primitive(self, task: Dict) -> Dict:
        """Execute navigation primitive."""
        parameters = task.get('parameters', {})

        self.node.get_logger().info('Executing navigation primitive')

        # In a real system, this would send commands to navigation stack
        # For simulation, return success
        return {
            'success': True,
            'message': 'Navigation primitive executed successfully'
        }

    def execute_locate_object(self, task: Dict) -> Dict:
        """Execute object location primitive."""
        target_obj = task.get('parameters', {}).get('object', 'unknown')

        self.node.get_logger().info(f'Locating object: {target_obj}')

        # In a real system, this would use perception system
        # For simulation, return mock location
        return {
            'success': True,
            'message': f'Object {target_obj} located',
            'position': {'x': 1.2, 'y': 0.5, 'z': 0.0}
        }

    def execute_approach_object(self, task: Dict) -> Dict:
        """Execute approach object primitive."""
        target_obj = task.get('parameters', {}).get('object', 'unknown')

        self.node.get_logger().info(f'Approaching object: {target_obj}')

        # In a real system, this would use navigation to approach object
        # For simulation, return success
        return {
            'success': True,
            'message': f'Approached object {target_obj} successfully'
        }

    def execute_grasp_object(self, task: Dict) -> Dict:
        """Execute grasp object primitive."""
        target_obj = task.get('parameters', {}).get('object', 'unknown')
        parameters = task.get('parameters', {})

        self.node.get_logger().info(f'Grasping object: {target_obj}')

        # In a real system, this would use manipulation system
        # For simulation, return success
        return {
            'success': True,
            'message': f'Grasped object {target_obj} successfully',
            'grasp_force': parameters.get('force', 'medium')
        }

    def execute_explore_area(self, task: Dict) -> Dict:
        """Execute area exploration primitive."""
        parameters = task.get('parameters', {})

        self.node.get_logger().info('Exploring area')

        # In a real system, this would use exploration algorithms
        # For simulation, return success
        return {
            'success': True,
            'message': 'Area exploration completed successfully',
            'coverage_percentage': 95.0
        }

    def execute_initialize_exploration(self, task: Dict) -> Dict:
        """Execute exploration initialization."""
        self.node.get_logger().info('Initializing exploration')

        return {
            'success': True,
            'message': 'Exploration initialized successfully'
        }

    def execute_return_to_base(self, task: Dict) -> Dict:
        """Execute return to base primitive."""
        self.node.get_logger().info('Returning to base')

        # In a real system, this would navigate to base position
        # For simulation, return success
        return {
            'success': True,
            'message': 'Returned to base successfully'
        }

class LowLevelExecutor:
    def __init__(self, node):
        self.node = node

    def execute_task(self, task: Dict) -> Dict:
        """Execute low-level task (direct actuator control)."""
        task_type = task.get('type', 'unknown')
        parameters = task.get('parameters', {})

        # In a real system, this would directly control motors/actuators
        # For simulation, return success for all tasks
        self.node.get_logger().info(f'Executing low-level task: {task_type}')

        return {
            'success': True,
            'message': f'Low-level task {task_type} executed successfully'
        }

def main(args=None):
    rclpy.init(args=args)
    hierarchical_pipeline = HierarchicalVLAPipeline()

    try:
        rclpy.spin(hierarchical_pipeline)
    except KeyboardInterrupt:
        hierarchical_pipeline.get_logger().info('Shutting down Hierarchical VLA Pipeline')
    finally:
        hierarchical_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered various integration patterns for Vision-Language-Action systems:

- **Sequential Pipeline**: Linear processing from vision to language to action
- **Parallel Processing**: Simultaneous processing with fusion
- **Feedback Loop**: Continuous refinement through action feedback
- **Hierarchical Integration**: Layered architecture with different abstraction levels

Each pattern has specific use cases and trade-offs depending on the requirements for real-time performance, system complexity, and feedback needs.

## Exercises

1. Implement a sequential VLA pipeline for your robot
2. Create a parallel processing system for improved performance
3. Add feedback loops to refine your system's behavior
4. Design a hierarchical system for complex task management
5. Compare performance between different integration patterns

## Quiz

1. What is the main advantage of the feedback loop integration pattern?
   a) Faster processing
   b) Continuous system refinement based on action outcomes
   c) Lower memory usage
   d) Simpler implementation

2. Which pattern is best for real-time applications with strict timing requirements?
   a) Sequential pipeline
   b) Parallel processing
   c) Feedback loop
   d) Hierarchical

3. What does the hierarchical pattern organize VLA components by?
   a) Color
   b) Abstraction level
   c) Size
   d) Age

## Mini-Project: Integration Pattern Comparison

Create and compare all four integration patterns:
1. Implement each pattern for the same robotic task
2. Measure performance metrics for each approach
3. Analyze trade-offs between different patterns
4. Document when each pattern is most appropriate
5. Create a recommendation system for pattern selection