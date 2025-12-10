---
sidebar_position: 3
---

# LLM Cognitive Planning

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate Large Language Models (LLMs) with robotic systems for cognitive planning
- Implement reasoning and decision-making capabilities using LLMs
- Design prompting strategies for effective robot behavior planning
- Connect LLMs with ROS 2 for real-time cognitive planning
- Optimize LLM inference for robotics applications
- Validate and test cognitive planning systems

## Introduction to LLM Cognitive Planning

Large Language Models (LLMs) have revolutionized the field of artificial intelligence, and their integration with robotics opens new possibilities for cognitive planning and reasoning. In robotics applications, LLMs serve as high-level cognitive planners that can interpret complex natural language commands, reason about the environment, and generate detailed action plans for robotic execution.

### Cognitive Planning Architecture

```
LLM Cognitive Planning Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Natural       │───→│   LLM Cognitive │───→│   Action        │
│   Language      │    │   Planner       │    │   Execution     │
│   Commands      │    │   (Reasoning,   │    │   (Low-level    │
│   (What to do)  │    │   Planning,     │    │   Commands)     │
│                 │    │   Context)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Environment   │
                         │   Context       │
                         │   (Sensors,     │
                         │   Memory,       │
                         │   Perception)   │
                         └─────────────────┘
```

### Key Components of LLM Cognitive Planning

1. **Context Understanding**: Interpreting environmental and task context
2. **Command Interpretation**: Parsing natural language commands
3. **Reasoning and Planning**: Generating step-by-step action plans
4. **Knowledge Integration**: Incorporating world knowledge and experience
5. **Action Sequencing**: Creating executable action sequences

## LLM Integration Fundamentals

### Selecting Appropriate LLMs for Robotics

When selecting LLMs for robotic applications, consider these factors:

1. **Response Time**: Real-time robotics requires fast inference
2. **Model Size**: Larger models may be too slow for real-time applications
3. **Domain Knowledge**: Models should understand robotics concepts
4. **Safety**: Models should produce safe and predictable outputs
5. **Resource Requirements**: Computational constraints of robotic platforms

### Popular LLM Options for Robotics

| Model Family | Size | Speed | Robotics Suitability | Notes |
|--------------|------|-------|---------------------|-------|
| GPT-4 | Large | Medium | High | Excellent reasoning, expensive |
| Claude | Large | Medium | High | Good safety, reasoning |
| Llama 2/3 | Medium | Fast | Medium | Open-source, customizable |
| Mistral | Small | Fast | Medium | Efficient, good performance |
| Phi-2 | Small | Very Fast | Low-Medium | Lightweight, limited |

For robotics applications, smaller models with good speed-performance trade-offs are often preferred.

### Basic LLM Integration

```python
# basic_llm_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json
import time
from typing import Dict, List, Optional

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

class BasicLLMPlanner(Node):
    def __init__(self):
        super().__init__('basic_llm_planner')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.plan_pub = self.create_publisher(String, 'execution_plan', 10)
        self.response_pub = self.create_publisher(String, 'llm_response', 10)
        self.motion_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # LLM components
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Initialize LLM
        if TRANSFORMERS_AVAILABLE:
            self.initialize_llm()
        else:
            self.get_logger().warn('Transformers not available, using simulated processing')

        # Context storage
        self.environment_context = {}
        self.robot_capabilities = {
            'movement': ['forward', 'backward', 'left', 'right', 'turn'],
            'actions': ['grasp', 'release', 'navigate', 'inspect'],
            'sensors': ['camera', 'lidar', 'imu']
        }

        self.get_logger().info('Basic LLM Planner initialized')

    def initialize_llm(self):
        """Initialize the LLM for robotic planning."""
        try:
            # Using a smaller, faster model for robotics
            model_name = "microsoft/DialoGPT-medium"  # Adjust based on requirements

            self.get_logger().info(f'Loading model: {model_name}')

            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.get_logger().info('LLM loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Could not load LLM: {e}')
            # Fallback to simpler approach
            self.model = None

    def command_callback(self, msg):
        """Process natural language commands."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Generate plan using LLM
        plan = self.generate_plan_with_llm(command)

        if plan:
            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.plan_pub.publish(plan_msg)

            # Execute the plan
            self.execute_plan(plan)

            # Publish response
            response_msg = String()
            response_msg.data = plan.get('summary', 'Plan executed successfully')
            self.response_pub.publish(response_msg)
        else:
            # Handle case where plan could not be generated
            response_msg = String()
            response_msg.data = f"Could not understand command: {command}"
            self.response_pub.publish(response_msg)

    def generate_plan_with_llm(self, command: str) -> Optional[Dict]:
        """Generate execution plan using LLM."""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            # Simulate plan generation when LLM is not available
            return self.simulate_plan_generation(command)

        try:
            # Create a prompt for the LLM
            context = self.get_environment_context()
            prompt = self.create_planning_prompt(command, context)

            # In a real implementation, you would:
            # 1. Tokenize the prompt
            # 2. Generate response using the model
            # 3. Parse the JSON response
            # 4. Validate the plan

            # For now, return simulated result
            return self.simulate_plan_generation(command)

        except Exception as e:
            self.get_logger().error(f'LLM planning error: {e}')
            return None

    def create_planning_prompt(self, command: str, context: str) -> str:
        """Create a prompt for the LLM planner."""
        prompt = f"""
You are a helpful robot assistant. Based on the current environment and the user's command,
generate a step-by-step plan for the robot to execute.

Current environment: {context}
User command: {command}

Respond with a JSON object containing the plan with these keys:
- "intent": The main goal of the command
- "steps": An array of steps to execute
- "confidence": Your confidence in the plan (0-1)
- "summary": Brief summary of the plan

Each step should have:
- "action": The specific action to take
- "description": What the action accomplishes
- "parameters": Any parameters needed for the action
- "expected_outcome": What should happen after the action

Example response format:
{{
    "intent": "move_to_object",
    "steps": [
        {{
            "action": "perceive",
            "description": "Look for the target object",
            "parameters": {{"object_type": "red_ball"}},
            "expected_outcome": "Locate the red ball in the environment"
        }},
        {{
            "action": "navigate",
            "description": "Move toward the object",
            "parameters": {{"target_position": [1.0, 2.0, 0.0]}},
            "expected_outcome": "Robot is close to the object"
        }}
    ],
    "confidence": 0.85,
    "summary": "Robot will locate and approach the red ball"
}}

Your response (JSON only):
"""
        return prompt

    def get_environment_context(self) -> str:
        """Get current environment context."""
        # In a real system, this would come from perception system
        # For simulation, return mock context
        return """
        Robot is in a room with the following objects:
        - Red ball at position (2.0, 1.0, 0.0)
        - Blue box at position (0.5, 3.0, 0.0)
        - Chair at position (-1.0, 2.0, 0.0)
        Robot position: (0.0, 0.0, 0.0)
        Robot capabilities: movement, grasping, navigation
        """

    def simulate_plan_generation(self, command: str) -> Dict:
        """Simulate plan generation when LLM is not available."""
        import random

        # Parse command and generate appropriate plan
        command_lower = command.lower()

        if 'go to' in command_lower or 'move to' in command_lower:
            target = 'target location'
            if 'ball' in command_lower:
                target = 'red ball'
            elif 'box' in command_lower:
                target = 'blue box'
            elif 'chair' in command_lower:
                target = 'chair'

            return {
                "intent": "navigate_to_object",
                "steps": [
                    {
                        "action": "locate",
                        "description": f"Find the {target}",
                        "parameters": {"target": target},
                        "expected_outcome": f"Located the {target}"
                    },
                    {
                        "action": "navigate",
                        "description": f"Move toward the {target}",
                        "parameters": {"target": target, "speed": 0.5},
                        "expected_outcome": f"Reached position near the {target}"
                    }
                ],
                "confidence": 0.85,
                "summary": f"Navigating to {target}"
            }

        elif 'grasp' in command_lower or 'pick up' in command_lower:
            object_name = 'object'
            if 'ball' in command_lower:
                object_name = 'red ball'
            elif 'box' in command_lower:
                object_name = 'blue box'

            return {
                "intent": "grasp_object",
                "steps": [
                    {
                        "action": "approach",
                        "description": f"Move close to the {object_name}",
                        "parameters": {"target": object_name, "distance": 0.3},
                        "expected_outcome": f"Within grasping distance of {object_name}"
                    },
                    {
                        "action": "grasp",
                        "description": f"Grasp the {object_name}",
                        "parameters": {"object": object_name, "force": 10},
                        "expected_outcome": f"{object_name} is grasped"
                    }
                ],
                "confidence": 0.75,
                "summary": f"Grasping {object_name}"
            }

        elif 'describe' in command_lower or 'what do you see' in command_lower:
            return {
                "intent": "describe_scene",
                "steps": [
                    {
                        "action": "perceive",
                        "description": "Analyze the current scene",
                        "parameters": {},
                        "expected_outcome": "Scene analysis complete"
                    },
                    {
                        "action": "describe",
                        "description": "Generate scene description",
                        "parameters": {},
                        "expected_outcome": "Scene described to user"
                    }
                ],
                "confidence": 0.95,
                "summary": "Describing the current scene"
            }

        else:
            # Unknown command
            return {
                "intent": "unknown",
                "steps": [
                    {
                        "action": "request_clarification",
                        "description": "Ask for clarification",
                        "parameters": {"question": f"I'm not sure how to handle: {command}"},
                        "expected_outcome": "User provides clarification"
                    }
                ],
                "confidence": 0.2,
                "summary": f"Unable to understand command: {command}"
            }

    def execute_plan(self, plan: Dict):
        """Execute the generated plan."""
        intent = plan.get('intent', 'unknown')
        steps = plan.get('steps', [])

        self.get_logger().info(f'Executing plan for intent: {intent}')

        for step in steps:
            action = step.get('action', '')
            description = step.get('description', '')
            parameters = step.get('parameters', {})

            self.get_logger().info(f'Executing step: {description}')

            # Execute the action based on type
            if action == 'navigate':
                self.execute_navigation_action(parameters)
            elif action == 'grasp':
                self.execute_grasp_action(parameters)
            elif action == 'perceive':
                self.execute_perception_action(parameters)
            elif action == 'describe':
                self.execute_description_action(parameters)
            elif action == 'request_clarification':
                self.execute_clarification_action(parameters)

    def execute_navigation_action(self, params: Dict):
        """Execute navigation action."""
        target = params.get('target', 'destination')
        speed = params.get('speed', 0.5)

        # Create and publish motion command
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = 0.0  # For simplicity, straight movement

        self.motion_pub.publish(cmd)

        self.get_logger().info(f'Navigating toward {target} at speed {speed}')

    def execute_grasp_action(self, params: Dict):
        """Execute grasping action."""
        obj = params.get('object', 'item')
        force = params.get('force', 10)

        self.get_logger().info(f'Attempting to grasp {obj} with force {force}')

    def execute_perception_action(self, params: Dict):
        """Execute perception action."""
        self.get_logger().info('Performing perception analysis')

    def execute_description_action(self, params: Dict):
        """Execute description action."""
        description = "The robot sees a red ball, blue box, and chair in the room."
        response_msg = String()
        response_msg.data = description
        self.response_pub.publish(response_msg)

    def execute_clarification_action(self, params: Dict):
        """Execute clarification action."""
        question = params.get('question', 'Could you please clarify?')
        response_msg = String()
        response_msg.data = question
        self.response_pub.publish(response_msg)

def main(args=None):
    rclpy.init(args=args)
    llm_planner = BasicLLMPlanner()

    try:
        rclpy.spin(llm_planner)
    except KeyboardInterrupt:
        llm_planner.get_logger().info('Shutting down Basic LLM Planner')
    finally:
        llm_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Prompt Engineering for Robotics

### Effective Prompting Strategies

```python
# advanced_prompting.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
import json
import re
from typing import Dict, List, Any

class AdvancedPromptingRobot(Node):
    def __init__(self):
        super().__init__('advanced_prompting_robot')

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Publishers
        self.plan_pub = self.create_publisher(String, 'execution_plan', 10)
        self.response_pub = self.create_publisher(String, 'prompted_response', 10)

        # Context storage
        self.scene_context = {}
        self.command_history = []

        # Prompt templates
        self.prompt_templates = {
            'navigation': self.navigation_prompt_template,
            'manipulation': self.manipulation_prompt_template,
            'inspection': self.inspection_prompt_template,
            'interaction': self.social_interaction_prompt_template
        }

        self.get_logger().info('Advanced Prompting Robot initialized')

    def command_callback(self, msg):
        """Process command with advanced prompting."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Determine command type and select appropriate template
        command_type = self.classify_command(command)
        prompt_template = self.prompt_templates.get(command_type, self.generic_prompt_template)

        # Create context
        context = self.build_context(command)

        # Generate prompt
        prompt = prompt_template(command, context)

        # Process with LLM (simulated here)
        plan = self.process_with_advanced_prompting(prompt, command_type)

        if plan:
            # Publish plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.plan_pub.publish(plan_msg)

            # Publish response
            response_msg = String()
            response_msg.data = plan.get('summary', 'Plan generated successfully')
            self.response_pub.publish(response_msg)

        # Update command history
        self.command_history.append({
            'command': command,
            'timestamp': self.get_clock().now().to_msg(),
            'plan': plan
        })

    def classify_command(self, command: str) -> str:
        """Classify command type for appropriate prompting."""
        command_lower = command.lower()

        navigation_keywords = ['go', 'move', 'navigate', 'walk', 'drive', 'approach', 'toward']
        manipulation_keywords = ['grasp', 'pick', 'place', 'move', 'manipulate', 'lift', 'drop']
        inspection_keywords = ['look', 'see', 'find', 'locate', 'inspect', 'examine', 'describe']
        interaction_keywords = ['hello', 'hi', 'help', 'assist', 'talk', 'speak', 'answer']

        if any(keyword in command_lower for keyword in navigation_keywords):
            return 'navigation'
        elif any(keyword in command_lower for keyword in manipulation_keywords):
            return 'manipulation'
        elif any(keyword in command_lower for keyword in inspection_keywords):
            return 'inspection'
        elif any(keyword in command_lower for keyword in interaction_keywords):
            return 'interaction'
        else:
            return 'generic'

    def build_context(self, command: str) -> Dict[str, Any]:
        """Build context for prompting."""
        context = {
            'robot_capabilities': {
                'navigation': ['move_forward', 'turn_left', 'turn_right', 'stop'],
                'manipulation': ['grasp', 'release', 'move_arm', 'position_gripper'],
                'perception': ['object_detection', 'distance_measurement', 'scene_analysis']
            },
            'environment': {
                'objects': self.scene_context.get('objects', []),
                'obstacles': self.scene_context.get('obstacles', []),
                'navigation_goals': self.scene_context.get('navigation_goals', [])
            },
            'constraints': {
                'safety': 'avoid collisions and ensure safe operation',
                'efficiency': 'complete tasks efficiently',
                'accuracy': 'perform actions accurately'
            }
        }

        return context

    def navigation_prompt_template(self, command: str, context: Dict) -> str:
        """Template for navigation commands."""
        return f"""
You are a navigation planning assistant for a mobile robot. Given the current environment and the user's navigation command,
generate a detailed navigation plan.

Environment context:
- Available objects: {context['environment']['objects']}
- Obstacles: {context['environment']['obstacles']}
- Robot capabilities: {context['robot_capabilities']['navigation']}

User command: {command}

Generate a JSON plan with:
- "intent": navigation intent
- "waypoints": list of coordinates to visit
- "obstacle_avoidance": strategy for avoiding obstacles
- "safety_checks": points where safety verification is needed
- "success_criteria": how to verify task completion

Example response:
{{
    "intent": "navigate_to_destination",
    "waypoints": [[1.0, 0.0], [1.0, 2.0], [3.0, 2.0]],
    "obstacle_avoidance": "use reactive obstacle avoidance with 0.5m clearance",
    "safety_checks": ["at each waypoint", "before final approach"],
    "success_criteria": "robot reaches within 0.1m of destination"
}}

Response (JSON only):
"""

    def manipulation_prompt_template(self, command: str, context: Dict) -> str:
        """Template for manipulation commands."""
        return f"""
You are a manipulation planning assistant for a robotic arm. Given the current scene and the user's manipulation command,
generate a detailed manipulation plan.

Environment context:
- Detected objects: {context['environment']['objects']}
- Robot capabilities: {context['robot_capabilities']['manipulation']}

User command: {command}

Generate a JSON plan with:
- "intent": manipulation intent
- "grasping_strategy": approach for grasping the object
- "motion_sequence": sequence of arm movements
- "safety_considerations": safety checks during manipulation
- "verification_steps": how to verify successful manipulation

Example response:
{{
    "intent": "grasp_red_cup",
    "grasping_strategy": "top-down grasp with 2cm clearance from surface",
    "motion_sequence": [
        {{"action": "approach", "target": [0.5, 0.2, 0.3]}},
        {{"action": "descend", "target": [0.5, 0.2, 0.1]}},
        {{"action": "grasp", "force": 10}},
        {{"action": "lift", "height": 0.3}}
    ],
    "safety_considerations": ["avoid collisions", "monitor grasp force"],
    "verification_steps": ["check grasp success", "verify object lift"]
}}

Response (JSON only):
"""

    def inspection_prompt_template(self, command: str, context: Dict) -> str:
        """Template for inspection commands."""
        return f"""
You are an inspection planning assistant for a robot. Given the current scene and the user's inspection command,
generate a detailed inspection plan.

Environment context:
- Detected objects: {context['environment']['objects']}
- Robot capabilities: {context['robot_capabilities']['perception']}

User command: {command}

Generate a JSON plan with:
- "intent": inspection intent
- "viewpoints": positions/orientations for optimal viewing
- "inspection_sequence": order of inspection steps
- "analysis_requirements": what to analyze in each view
- "reporting_format": how to present findings

Example response:
{{
    "intent": "inspect_blue_box",
    "viewpoints": [
        {{"position": [0.0, 1.0, 0.5], "orientation": [0, 0, 0]}},
        {{"position": [1.0, 0.0, 0.5], "orientation": [0, 0, 90]}},
        {{"position": [0.0, -1.0, 0.5], "orientation": [0, 0, 180]}}
    ],
    "inspection_sequence": ["front_view", "side_view", "top_view"],
    "analysis_requirements": ["object shape", "surface condition", "surroundings"],
    "reporting_format": "verbal_description_with_key_features"
}}

Response (JSON only):
"""

    def social_interaction_prompt_template(self, command: str, context: Dict) -> str:
        """Template for social interaction commands."""
        return f"""
You are a social interaction assistant for a robot. Given the user's social command,
generate an appropriate response and possible actions.

User command: {command}

Generate a JSON plan with:
- "intent": social interaction intent
- "response": appropriate verbal response
- "gesture": suggested non-verbal action
- "tone": suggested tone of voice
- "follow_up": possible follow-up interactions

Example response:
{{
    "intent": "greeting_response",
    "response": "Hello! How can I assist you today?",
    "gesture": "slight nod or wave if equipped",
    "tone": "friendly and welcoming",
    "follow_up": "ask how robot can help"
}}

Response (JSON only):
"""

    def generic_prompt_template(self, command: str, context: Dict) -> str:
        """Generic template for unrecognized command types."""
        return f"""
You are a helpful robot assistant. Given the user's command, generate an appropriate plan.

User command: {command}

Environment context:
- Robot capabilities: {list(context['robot_capabilities'].keys())}
- Current objects: {context['environment']['objects']}

Generate a JSON plan with:
- "intent": the main goal
- "steps": sequence of actions to accomplish the goal
- "resources_needed": what robot capabilities are required
- "success_criteria": how to verify completion
- "potential_issues": possible challenges and solutions

Response (JSON only):
"""

    def process_with_advanced_prompting(self, prompt: str, command_type: str) -> Dict:
        """Process command with advanced prompting (simulated)."""
        # In a real implementation, this would call the LLM with the prompt
        # For simulation, return appropriate response based on command type

        command = prompt.split('\n')[-1].split(': ')[-1]  # Extract command from prompt

        if command_type == 'navigation':
            return {
                "intent": "navigate_to_destination",
                "waypoints": [[1.0, 0.0], [1.0, 2.0], [3.0, 2.0]],
                "obstacle_avoidance": "use reactive obstacle avoidance with 0.5m clearance",
                "safety_checks": ["at each waypoint", "before final approach"],
                "success_criteria": "robot reaches within 0.1m of destination",
                "summary": "Navigation plan generated with obstacle avoidance"
            }
        elif command_type == 'manipulation':
            return {
                "intent": "grasp_object",
                "grasping_strategy": "top-down grasp with 2cm clearance",
                "motion_sequence": [
                    {"action": "approach", "target": [0.5, 0.2, 0.3]},
                    {"action": "descend", "target": [0.5, 0.2, 0.1]},
                    {"action": "grasp", "force": 10},
                    {"action": "lift", "height": 0.3}
                ],
                "safety_considerations": ["avoid collisions", "monitor grasp force"],
                "verification_steps": ["check grasp success", "verify object lift"],
                "summary": "Manipulation plan generated with safety considerations"
            }
        elif command_type == 'inspection':
            return {
                "intent": "inspect_object",
                "viewpoints": [
                    {"position": [0.0, 1.0, 0.5], "orientation": [0, 0, 0]},
                    {"position": [1.0, 0.0, 0.5], "orientation": [0, 0, 90]},
                    {"position": [0.0, -1.0, 0.5], "orientation": [0, 0, 180]}
                ],
                "inspection_sequence": ["front_view", "side_view", "top_view"],
                "analysis_requirements": ["object shape", "surface condition", "surroundings"],
                "reporting_format": "verbal_description_with_key_features",
                "summary": "Inspection plan with multiple viewpoints"
            }
        elif command_type == 'interaction':
            return {
                "intent": "social_response",
                "response": "Hello! How can I assist you today?",
                "gesture": "slight nod or wave if equipped",
                "tone": "friendly and welcoming",
                "follow_up": "ask how robot can help",
                "summary": "Social interaction initiated with friendly response"
            }
        else:
            return {
                "intent": "unknown",
                "steps": [{"action": "request_clarification", "details": f"Unable to understand: {command}"}],
                "resources_needed": [],
                "success_criteria": "receive clarification",
                "potential_issues": ["command not understood", "need more information"],
                "summary": f"Unable to process command: {command}"
            }

    def detections_callback(self, msg):
        """Update scene context with object detections."""
        objects = []
        for detection in msg.detections:
            if detection.results:
                obj = {
                    'class': detection.results[0].hypothesis.class_id,
                    'confidence': detection.results[0].hypothesis.score,
                    'position': {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y,
                        'size_x': detection.bbox.size_x,
                        'size_y': detection.bbox.size_y
                    }
                }
                if obj['confidence'] > 0.5:
                    objects.append(obj)

        self.scene_context['objects'] = objects

    def scan_callback(self, msg):
        """Update scene context with obstacle information."""
        # Process LiDAR scan to identify obstacles
        obstacles = []
        for i, range_val in enumerate(msg.ranges):
            if msg.range_min < range_val < msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                if range_val < 1.0:  # Consider anything within 1m as obstacle
                    obstacles.append({'x': x, 'y': y, 'distance': range_val})

        self.scene_context['obstacles'] = obstacles

def main(args=None):
    rclpy.init(args=args)
    advanced_robot = AdvancedPromptingRobot()

    try:
        rclpy.spin(advanced_robot)
    except KeyboardInterrupt:
        advanced_robot.get_logger().info('Shutting down Advanced Prompting Robot')
    finally:
        advanced_robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Context-Aware Planning

### Maintaining and Using Context

```python
# context_aware_planning.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import BatteryState
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque

@dataclass
class RobotState:
    """Represents the current state of the robot."""
    position: Point
    battery_level: float
    timestamp: float
    task_queue: List[str]
    last_command: str
    execution_status: str

@dataclass
class EnvironmentalContext:
    """Represents the environmental context."""
    objects: List[Dict]
    obstacles: List[Dict]
    lighting_conditions: str
    acoustic_conditions: str
    timestamp: float

@dataclass
class TaskHistoryItem:
    """Represents a historical task execution."""
    command: str
    plan: Dict
    success: bool
    execution_time: float
    timestamp: float

class ContextAwarePlanner(Node):
    def __init__(self):
        super().__init__('context_aware_planner')

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.battery_sub = self.create_subscription(
            BatteryState, 'battery_state', self.battery_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )

        # Publishers
        self.plan_pub = self.create_publisher(String, 'context_aware_plan', 10)
        self.response_pub = self.create_publisher(String, 'context_response', 10)
        self.context_pub = self.create_publisher(String, 'current_context', 10)

        # Context storage
        self.robot_state = RobotState(
            position=Point(x=0.0, y=0.0, z=0.0),
            battery_level=1.0,
            timestamp=time.time(),
            task_queue=[],
            last_command="",
            execution_status="idle"
        )
        self.environmental_context = EnvironmentalContext(
            objects=[],
            obstacles=[],
            lighting_conditions="normal",
            acoustic_conditions="quiet",
            timestamp=time.time()
        )
        self.task_history = deque(maxlen=50)  # Keep last 50 tasks

        # Context windows
        self.short_term_context = deque(maxlen=10)
        self.long_term_context = deque(maxlen=100)

        # Planning parameters
        self.battery_threshold = 0.2  # 20% battery threshold
        self.max_plan_length = 20     # Maximum steps in a plan
        self.context_update_interval = 1.0  # seconds

        # Last context update time
        self.last_context_update = time.time()

        self.get_logger().info('Context-Aware Planner initialized')

    def command_callback(self, msg):
        """Process command with full context awareness."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Update robot state
        self.robot_state.last_command = command
        self.robot_state.timestamp = time.time()

        # Check battery level before planning
        if self.robot_state.battery_level < self.battery_threshold:
            response_msg = String()
            response_msg.data = "Battery level is low. Please recharge before executing complex tasks."
            self.response_pub.publish(response_msg)

            if "urgent" not in command.lower():
                return  # Don't execute non-urgent tasks with low battery

        # Generate plan with full context
        plan = self.generate_context_aware_plan(command)

        if plan:
            # Add to short-term context
            self.short_term_context.append({
                'command': command,
                'plan': plan,
                'timestamp': time.time()
            })

            # Publish plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.plan_pub.publish(plan_msg)

            # Publish response
            response_msg = String()
            response_msg.data = plan.get('summary', 'Plan generated successfully')
            self.response_pub.publish(response_msg)

            # Update execution status
            self.robot_state.execution_status = "planning"
        else:
            response_msg = String()
            response_msg.data = f"Could not generate plan for: {command}"
            self.response_pub.publish(response_msg)

    def generate_context_aware_plan(self, command: str) -> Optional[Dict]:
        """Generate plan considering all available context."""
        try:
            # Build comprehensive context
            full_context = self.build_comprehensive_context(command)

            # Check if we have sufficient context
            if not self.has_sufficient_context(full_context):
                return self.handle_insufficient_context(command)

            # Generate plan based on context
            plan = self.create_contextual_plan(command, full_context)

            # Validate plan against constraints
            if self.validate_plan_against_context(plan, full_context):
                # Add to history
                history_item = TaskHistoryItem(
                    command=command,
                    plan=plan,
                    success=True,
                    execution_time=0.0,  # Will be updated during execution
                    timestamp=time.time()
                )
                self.task_history.append(history_item)

                return plan
            else:
                return self.handle_invalid_plan(command, full_context)

        except Exception as e:
            self.get_logger().error(f'Context-aware planning error: {e}')
            return None

    def build_comprehensive_context(self, command: str) -> Dict[str, Any]:
        """Build comprehensive context for planning."""
        context = {
            'robot_state': asdict(self.robot_state),
            'environmental_context': asdict(self.environmental_context),
            'task_history': list(self.task_history),
            'current_time': datetime.now().isoformat(),
            'battery_level': self.robot_state.battery_level,
            'available_objects': self.environmental_context.objects,
            'obstacles': self.environmental_context.obstacles,
            'recent_interactions': list(self.short_term_context),
            'long_term_patterns': self.identify_long_term_patterns()
        }

        return context

    def has_sufficient_context(self, context: Dict) -> bool:
        """Check if we have sufficient context for planning."""
        # Check if we have recent environmental data
        env_age = time.time() - context['environmental_context']['timestamp']
        if env_age > 30:  # More than 30 seconds old
            self.get_logger().warn('Environmental context is outdated')
            return False

        # Check if we have recent robot state
        state_age = time.time() - context['robot_state']['timestamp']
        if state_age > 10:  # More than 10 seconds old
            self.get_logger().warn('Robot state is outdated')
            return False

        return True

    def identify_long_term_patterns(self) -> List[Dict]:
        """Identify patterns from long-term task history."""
        patterns = []

        if len(self.task_history) < 3:
            return patterns

        # Identify frequent task combinations
        task_sequences = {}
        recent_tasks = [item.command for item in list(self.task_history)[-10:]]

        for i in range(len(recent_tasks) - 1):
            seq = (recent_tasks[i], recent_tasks[i+1])
            task_sequences[seq] = task_sequences.get(seq, 0) + 1

        # Add frequent sequences as patterns
        for (task1, task2), count in task_sequences.items():
            if count >= 2:  # Occurred at least twice
                patterns.append({
                    'type': 'task_sequence',
                    'tasks': [task1, task2],
                    'frequency': count
                })

        return patterns

    def create_contextual_plan(self, command: str, context: Dict) -> Dict:
        """Create plan based on comprehensive context."""
        # Determine plan type based on command and context
        plan_type = self.determine_plan_type(command, context)

        if plan_type == 'navigation':
            return self.create_navigation_plan(command, context)
        elif plan_type == 'manipulation':
            return self.create_manipulation_plan(command, context)
        elif plan_type == 'inspection':
            return self.create_inspection_plan(command, context)
        elif plan_type == 'social':
            return self.create_social_plan(command, context)
        else:
            return self.create_generic_plan(command, context)

    def determine_plan_type(self, command: str, context: Dict) -> str:
        """Determine the appropriate plan type."""
        command_lower = command.lower()

        # Check for specific plan types
        if any(word in command_lower for word in ['go', 'move', 'navigate', 'walk', 'drive']):
            return 'navigation'
        elif any(word in command_lower for word in ['grasp', 'pick', 'place', 'lift', 'drop']):
            return 'manipulation'
        elif any(word in command_lower for word in ['look', 'see', 'find', 'locate', 'inspect']):
            return 'inspection'
        elif any(word in command_lower for word in ['hello', 'hi', 'help', 'talk', 'speak']):
            return 'social'
        else:
            # Determine based on available objects and robot state
            if context['available_objects']:
                return 'manipulation'  # Default to manipulation if objects available
            else:
                return 'navigation'   # Default to navigation otherwise

    def create_navigation_plan(self, command: str, context: Dict) -> Dict:
        """Create navigation-specific plan."""
        # Identify target location
        target_location = self.extract_target_location(command, context)

        plan = {
            "intent": "navigate_to_location",
            "target_location": target_location,
            "steps": [],
            "context_awareness": {
                "battery_level": context['battery_level'],
                "obstacles": len(context['obstacles']),
                "lighting": context['environmental_context']['lighting_conditions']
            },
            "safety_considerations": self.get_safety_considerations(context),
            "energy_optimization": self.should_optimize_for_energy(context),
            "success_criteria": f"reach within 0.2m of {target_location}",
            "summary": f"Navigating to {target_location}"
        }

        # Add steps based on context
        steps = self.generate_navigation_steps(target_location, context)
        plan["steps"] = steps

        return plan

    def create_manipulation_plan(self, command: str, context: Dict) -> Dict:
        """Create manipulation-specific plan."""
        # Identify target object
        target_object = self.extract_target_object(command, context)

        plan = {
            "intent": "manipulate_object",
            "target_object": target_object,
            "steps": [],
            "context_awareness": {
                "battery_level": context['battery_level'],
                "available_objects": [obj['class'] for obj in context['available_objects']],
                "robot_capabilities": ["grasping", "lifting", "placing"]
            },
            "safety_considerations": self.get_safety_considerations(context),
            "precision_requirements": self.get_precision_requirements(command),
            "success_criteria": f"successfully manipulate {target_object}",
            "summary": f"Manipulating {target_object}"
        }

        # Add steps based on context
        steps = self.generate_manipulation_steps(target_object, context)
        plan["steps"] = steps

        return plan

    def extract_target_location(self, command: str, context: Dict) -> str:
        """Extract target location from command."""
        # In a real system, this would use NLP to extract locations
        # For simulation, return a default location or the last visited place
        if 'kitchen' in command.lower():
            return 'kitchen'
        elif 'living room' in command.lower():
            return 'living_room'
        elif 'bedroom' in command.lower():
            return 'bedroom'
        else:
            # Return the location of the most interesting object
            if context['available_objects']:
                obj = context['available_objects'][0]
                return f"location_of_{obj['class']}"
            else:
                return "random_location"

    def extract_target_object(self, command: str, context: Dict) -> str:
        """Extract target object from command."""
        # In a real system, this would use NLP to extract objects
        # For simulation, match objects in the scene
        command_lower = command.lower()

        for obj in context['available_objects']:
            if obj['class'] in command_lower:
                return obj['class']

        # If no direct match, return the most prominent object
        if context['available_objects']:
            return context['available_objects'][0]['class']

        return "unknown_object"

    def generate_navigation_steps(self, target_location: str, context: Dict) -> List[Dict]:
        """Generate navigation steps based on context."""
        steps = []

        # Add battery check step if needed
        if context['battery_level'] < 0.3:
            steps.append({
                "action": "check_battery",
                "description": "Verify sufficient battery for navigation",
                "parameters": {"minimum_required": 0.1},
                "critical": True
            })

        # Add obstacle avoidance preparation
        if context['obstacles']:
            steps.append({
                "action": "plan_route",
                "description": "Calculate route avoiding known obstacles",
                "parameters": {"obstacle_buffer": 0.5},
                "critical": True
            })

        # Add navigation steps
        steps.extend([
            {
                "action": "localize",
                "description": "Confirm current position",
                "parameters": {},
                "critical": True
            },
            {
                "action": "navigate",
                "description": f"Move toward {target_location}",
                "parameters": {"target": target_location, "speed": 0.5},
                "critical": True
            },
            {
                "action": "verify_arrival",
                "description": "Confirm arrival at destination",
                "parameters": {"tolerance": 0.2},
                "critical": True
            }
        ])

        return steps

    def generate_manipulation_steps(self, target_object: str, context: Dict) -> List[Dict]:
        """Generate manipulation steps based on context."""
        steps = []

        # Add approach step
        steps.append({
            "action": "approach_object",
            "description": f"Move close to {target_object}",
            "parameters": {"object": target_object, "distance": 0.3},
            "critical": True
        })

        # Add grasping step
        steps.append({
            "action": "grasp_object",
            "description": f"Grasp the {target_object}",
            "parameters": {"object": target_object, "force": 10},
            "critical": True
        })

        # Add verification step
        steps.append({
            "action": "verify_grasp",
            "description": "Confirm successful grasp",
            "parameters": {"timeout": 5.0},
            "critical": True
        })

        return steps

    def get_safety_considerations(self, context: Dict) -> List[str]:
        """Get safety considerations based on context."""
        considerations = []

        if context['battery_level'] < 0.2:
            considerations.append("operate in energy-efficient mode")
        if context['environmental_context']['acoustic_conditions'] == "noisy":
            considerations.append("use visual feedback as primary verification")
        if context['obstacles']:
            considerations.append("maintain 0.5m clearance from obstacles")

        return considerations

    def should_optimize_for_energy(self, context: Dict) -> bool:
        """Determine if plan should be energy-optimized."""
        return context['battery_level'] < 0.3

    def get_precision_requirements(self, command: str) -> str:
        """Get precision requirements from command."""
        if 'carefully' in command.lower() or 'gently' in command.lower():
            return 'high'
        elif 'quickly' in command.lower() or 'fast' in command.lower():
            return 'low'
        else:
            return 'medium'

    def validate_plan_against_context(self, plan: Dict, context: Dict) -> bool:
        """Validate plan against current context."""
        # Check if plan is too long given battery level
        if len(plan.get('steps', [])) > self.max_plan_length:
            self.get_logger().warn('Plan exceeds maximum length')
            return False

        # Check if battery is sufficient for plan type
        if context['battery_level'] < 0.1 and plan['intent'] in ['navigation', 'manipulation']:
            self.get_logger().warn('Insufficient battery for complex plan')
            return False

        # Check if required objects are available
        if plan['intent'] == 'manipulation':
            target_obj = plan.get('target_object', '')
            available_classes = [obj['class'] for obj in context['available_objects']]
            if target_obj and target_obj not in available_classes:
                self.get_logger().warn(f'Target object {target_obj} not available')
                return False

        return True

    def handle_insufficient_context(self, command: str) -> Dict:
        """Handle case where context is insufficient."""
        return {
            "intent": "request_more_information",
            "steps": [{
                "action": "request_context",
                "description": "Request additional context information",
                "parameters": {"needed_info": ["object_locations", "obstacle_map", "current_pose"]},
                "critical": True
            }],
            "success_criteria": "receive requested context information",
            "summary": "Need more context to execute command safely"
        }

    def handle_invalid_plan(self, command: str, context: Dict) -> Dict:
        """Handle case where plan validation fails."""
        return {
            "intent": "abort_command",
            "steps": [{
                "action": "report_issue",
                "description": "Report why command cannot be executed",
                "parameters": {"issue": "plan validation failed"},
                "critical": True
            }],
            "success_criteria": "inform user of limitation",
            "summary": "Cannot execute command due to contextual constraints"
        }

    def odom_callback(self, msg):
        """Update robot position from odometry."""
        self.robot_state.position = msg.pose.pose.position
        self.robot_state.timestamp = time.time()

    def battery_callback(self, msg):
        """Update battery level."""
        self.robot_state.battery_level = msg.percentage
        self.robot_state.timestamp = time.time()

    def detections_callback(self, msg):
        """Update environmental context with detections."""
        objects = []
        for detection in msg.detections:
            if detection.results and detection.results[0].hypothesis.score > 0.5:
                obj = {
                    'class': detection.results[0].hypothesis.class_id,
                    'confidence': detection.results[0].hypothesis.score,
                    'position': {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y
                    }
                }
                objects.append(obj)

        self.environmental_context.objects = objects
        self.environmental_context.timestamp = time.time()

    def publish_current_context(self):
        """Publish current context for monitoring."""
        context_msg = String()
        context_dict = {
            'robot_state': asdict(self.robot_state),
            'environmental_context': asdict(self.environmental_context),
            'context_timestamp': time.time()
        }
        context_msg.data = json.dumps(context_dict)
        self.context_pub.publish(context_msg)

def main(args=None):
    rclpy.init(args=args)
    context_planner = ContextAwarePlanner()

    # Add timer to periodically publish context
    context_timer = context_planner.create_timer(5.0, context_planner.publish_current_context)

    try:
        rclpy.spin(context_planner)
    except KeyboardInterrupt:
        context_planner.get_logger().info('Shutting down Context-Aware Planner')
    finally:
        context_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Optimized LLM Integration

### Performance Optimization Techniques

```python
# optimized_llm_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
import torch
import threading
import queue
import time
from typing import Optional, Dict, Any
import gc

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers.generation import GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

class OptimizedLLMIntegration(Node):
    def __init__(self):
        super().__init__('optimized_llm_integration')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.plan_pub = self.create_publisher(String, 'optimized_plan', 10)
        self.performance_pub = self.create_publisher(Float32, 'llm_inference_time', 10)
        self.status_pub = self.create_publisher(Bool, 'llm_ready', 10)

        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Optimization settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.max_new_tokens = 256
        self.temperature = 0.1
        self.top_p = 0.9

        # Processing queues
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # Threading
        self.processing_thread = None
        self.running = False

        # Performance tracking
        self.inference_times = []
        self.request_count = 0

        # Load optimized model
        if TRANSFORMERS_AVAILABLE:
            self.load_optimized_model()
        else:
            self.get_logger().warn('Transformers not available for optimization')

        # Start processing thread
        self.start_processing_thread()

        # Publish status
        self.publish_status(True)

        self.get_logger().info(f'Optimized LLM Integration initialized on {self.device}')

    def load_optimized_model(self):
        """Load LLM with optimizations for robotics applications."""
        try:
            self.get_logger().info('Loading optimized LLM model...')

            # Use a smaller, more efficient model for robotics
            model_name = "microsoft/DialoGPT-small"  # Lightweight model

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.compute_dtype,
                device_map="auto",  # Automatically distribute across available devices
                load_in_8bit=True if torch.cuda.is_available() else False,  # 8-bit quantization if GPU available
                trust_remote_code=False
            )

            # Move model to device
            self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            # Enable attention optimization if available
            if hasattr(self.model.config, 'gradient_checkpointing'):
                self.model.gradient_checkpointing_enable()

            self.get_logger().info('Optimized LLM model loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load optimized model: {e}')
            self.model = None

    def start_processing_thread(self):
        """Start the processing thread."""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self.process_requests, daemon=True)
        self.processing_thread.start()

        self.get_logger().info('LLM processing thread started')

    def stop_processing_thread(self):
        """Stop the processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        self.get_logger().info('LLM processing thread stopped')

    def command_callback(self, msg):
        """Add command to processing queue."""
        command = msg.data
        request = {
            'command': command,
            'timestamp': time.time()
        }

        try:
            self.request_queue.put(request, timeout=0.1)
        except queue.Full:
            self.get_logger().warn('Request queue is full, dropping command')

    def process_requests(self):
        """Process requests in the queue."""
        while self.running:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=0.1)

                # Process the request
                start_time = time.time()
                plan = self.process_command_optimized(request['command'])
                processing_time = time.time() - start_time

                # Track performance
                self.inference_times.append(processing_time)
                if len(self.inference_times) > 100:  # Keep last 100 measurements
                    self.inference_times.pop(0)

                # Publish performance metric
                perf_msg = Float32()
                perf_msg.data = float(processing_time)
                self.performance_pub.publish(perf_msg)

                if plan:
                    # Publish plan
                    plan_msg = String()
                    plan_msg.data = plan
                    self.plan_pub.publish(plan_msg)

                    self.get_logger().info(f'Processed command in {processing_time:.3f}s')

                self.request_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Processing error: {e}')

    def process_command_optimized(self, command: str) -> Optional[str]:
        """Process command with optimized LLM inference."""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            # Simulate processing
            return self.simulate_optimized_processing(command)

        try:
            # Create prompt
            prompt = self.create_optimized_prompt(command)

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Generate response with optimizations
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Reduce repetitive outputs
                    num_return_sequences=1
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the plan portion (remove the prompt part)
            if prompt in response:
                plan_text = response[len(prompt):].strip()
            else:
                plan_text = response.strip()

            return plan_text

        except Exception as e:
            self.get_logger().error(f'Optimized processing error: {e}')
            return None

    def create_optimized_prompt(self, command: str) -> str:
        """Create optimized prompt for efficient processing."""
        # Use a concise, direct prompt structure
        return f"""Command: {command}
Plan: """

    def simulate_optimized_processing(self, command: str) -> str:
        """Simulate optimized processing when LLM is not available."""
        import random
        import json

        # Simulate a simple plan based on command
        if 'go' in command.lower() or 'move' in command.lower():
            plan = {
                "intent": "navigation",
                "steps": [
                    {"action": "navigate", "params": {"target": "destination", "speed": 0.5}},
                    {"action": "verify", "params": {"condition": "at_destination"}}
                ],
                "estimated_time": random.uniform(5, 30)
            }
        elif 'grasp' in command.lower() or 'pick' in command.lower():
            plan = {
                "intent": "manipulation",
                "steps": [
                    {"action": "approach", "params": {"target": "object", "distance": 0.3}},
                    {"action": "grasp", "params": {"force": 10}},
                    {"action": "verify", "params": {"condition": "grasp_success"}}
                ],
                "estimated_time": random.uniform(10, 45)
            }
        else:
            plan = {
                "intent": "unknown",
                "steps": [{"action": "request_clarification", "params": {"question": f"Could you clarify: {command}"}}],
                "estimated_time": 0
            }

        return json.dumps(plan)

    def get_average_inference_time(self) -> float:
        """Get average inference time."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def publish_status(self, ready: bool):
        """Publish LLM readiness status."""
        status_msg = Bool()
        status_msg.data = ready
        self.status_pub.publish(status_msg)

    def cleanup(self):
        """Clean up resources."""
        self.stop_processing_thread()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def main(args=None):
    rclpy.init(args=args)
    optimized_llm = OptimizedLLMIntegration()

    try:
        rclpy.spin(optimized_llm)
    except KeyboardInterrupt:
        optimized_llm.get_logger().info('Shutting down Optimized LLM Integration')
    finally:
        optimized_llm.cleanup()
        optimized_llm.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## LLM-ROS Integration

### Complete Integration with ROS 2

```python
# llm_ros_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Header
import json
import time
from typing import Dict, List, Optional
import threading

class LLMROSIntegration(Node):
    def __init__(self):
        super().__init__('llm_ros_integration')

        # Publishers
        self.plan_pub = self.create_publisher(String, 'llm_execution_plan', 10)
        self.response_pub = self.create_publisher(String, 'llm_response', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'llm_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )
        self.perception_sub = self.create_subscription(
            String, 'perception_data', self.perception_callback, 10
        )

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # LLM components (simulated for this example)
        self.llm_initialized = True
        self.current_plan = None
        self.plan_execution_active = False

        # Context storage
        self.environment_context = {}
        self.robot_capabilities = {
            'navigation': True,
            'manipulation': False,  # Assuming no manipulator for this example
            'perception': True
        }

        # Threading for plan execution
        self.execution_thread = None
        self.execution_lock = threading.Lock()

        self.get_logger().info('LLM-ROS Integration initialized')

    def command_callback(self, msg):
        """Process natural language command and generate ROS-compatible plan."""
        command = msg.data
        self.get_logger().info(f'Received natural language command: {command}')

        # Update status
        status_msg = String()
        status_msg.data = f"Processing command: {command}"
        self.status_pub.publish(status_msg)

        # Generate plan using LLM (simulated)
        plan = self.generate_ros_plan(command)

        if plan:
            # Store plan and publish
            self.current_plan = plan

            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.plan_pub.publish(plan_msg)

            # Execute plan
            self.execute_ros_plan(plan)

            # Generate response
            response = self.generate_response_for_plan(plan)
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)
        else:
            # Handle command that couldn't be processed
            response_msg = String()
            response_msg.data = f"Sorry, I couldn't understand or execute the command: {command}"
            self.response_pub.publish(response_msg)

    def generate_ros_plan(self, command: str) -> Optional[Dict]:
        """Generate ROS-compatible execution plan from natural language."""
        # In a real implementation, this would use an LLM to generate the plan
        # For this example, we'll simulate plan generation

        command_lower = command.lower()

        if 'go to' in command_lower or 'navigate to' in command_lower:
            # Extract target location (simplified extraction)
            target = self.extract_target_location(command_lower)

            plan = {
                "intent": "navigation",
                "actions": [
                    {
                        "type": "navigate",
                        "target_pose": self.get_target_pose(target),
                        "description": f"Navigate to {target}"
                    }
                ],
                "estimated_duration": 30,  # seconds
                "success_criteria": "reach_within_tolerance"
            }

        elif 'move forward' in command_lower:
            plan = {
                "intent": "motion",
                "actions": [
                    {
                        "type": "move_straight",
                        "distance": 1.0,  # meters
                        "speed": 0.5,     # m/s
                        "description": "Move forward 1 meter"
                    }
                ],
                "estimated_duration": 2,
                "success_criteria": "travel_distance_achieved"
            }

        elif 'turn left' in command_lower:
            plan = {
                "intent": "motion",
                "actions": [
                    {
                        "type": "rotate",
                        "angle": 90,    # degrees
                        "speed": 0.5,   # rad/s
                        "description": "Turn left 90 degrees"
                    }
                ],
                "estimated_duration": 3,
                "success_criteria": "rotation_achieved"
            }

        elif 'turn right' in command_lower:
            plan = {
                "intent": "motion",
                "actions": [
                    {
                        "type": "rotate",
                        "angle": -90,   # negative for right turn
                        "speed": 0.5,
                        "description": "Turn right 90 degrees"
                    }
                ],
                "estimated_duration": 3,
                "success_criteria": "rotation_achieved"
            }

        elif 'stop' in command_lower:
            plan = {
                "intent": "motion",
                "actions": [
                    {
                        "type": "stop",
                        "description": "Stop all movement"
                    }
                ],
                "estimated_duration": 1,
                "success_criteria": "zero_velocity"
            }

        else:
            # Unknown command
            plan = {
                "intent": "unknown",
                "actions": [
                    {
                        "type": "request_clarification",
                        "question": f"I'm not sure how to execute: {command}",
                        "description": "Request clarification from user"
                    }
                ],
                "estimated_duration": 0,
                "success_criteria": "receive_clarification"
            }

        return plan

    def extract_target_location(self, command: str) -> str:
        """Extract target location from command (simplified)."""
        # This is a simplified extraction - in reality, you'd use NLP
        if 'kitchen' in command:
            return 'kitchen'
        elif 'living room' in command:
            return 'living_room'
        elif 'bedroom' in command:
            return 'bedroom'
        elif 'charger' in command:
            return 'charging_station'
        else:
            return 'default_location'

    def get_target_pose(self, location: str) -> Dict:
        """Get predefined pose for a location."""
        # In a real system, these would come from a map or localization system
        poses = {
            'kitchen': {"x": 2.0, "y": 1.0, "theta": 0.0},
            'living_room': {"x": 0.0, "y": 0.0, "theta": 0.0},
            'bedroom': {"x": -1.0, "y": 2.0, "theta": 1.57},  # 90 degrees
            'charging_station': {"x": 3.0, "y": 0.0, "theta": 3.14}  # 180 degrees
        }

        return poses.get(location, {"x": 0.0, "y": 0.0, "theta": 0.0})

    def execute_ros_plan(self, plan: Dict):
        """Execute the generated ROS plan."""
        with self.execution_lock:
            if self.plan_execution_active:
                self.get_logger().warn('Plan execution already active, skipping')
                return

            self.plan_execution_active = True

        try:
            intent = plan.get('intent', 'unknown')
            actions = plan.get('actions', [])

            self.get_logger().info(f'Executing plan with intent: {intent}')

            for action in actions:
                action_type = action.get('type', 'unknown')
                description = action.get('description', 'No description')

                self.get_logger().info(f'Executing action: {description}')

                if action_type == 'navigate':
                    self.execute_navigation_action(action)
                elif action_type == 'move_straight':
                    self.execute_move_straight_action(action)
                elif action_type == 'rotate':
                    self.execute_rotate_action(action)
                elif action_type == 'stop':
                    self.execute_stop_action()
                elif action_type == 'request_clarification':
                    self.execute_request_clarification_action(action)
                else:
                    self.get_logger().warn(f'Unknown action type: {action_type}')

        except Exception as e:
            self.get_logger().error(f'Plan execution error: {e}')
        finally:
            with self.execution_lock:
                self.plan_execution_active = False

    def execute_navigation_action(self, action: Dict):
        """Execute navigation action using ROS 2 navigation."""
        target_pose_data = action.get('target_pose', {})

        # Create NavigateToPose goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(target_pose_data.get('x', 0.0))
        goal_msg.pose.pose.position.y = float(target_pose_data.get('y', 0.0))
        goal_msg.pose.pose.position.z = 0.0

        # Convert angle to quaternion (assuming z-axis rotation)
        import math
        angle = target_pose_data.get('theta', 0.0)
        goal_msg.pose.pose.orientation.z = math.sin(angle / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(angle / 2.0)

        # Send navigation goal
        if self.nav_client.wait_for_server(timeout_sec=1.0):
            future = self.nav_client.send_goal_async(goal_msg)
            # In a real implementation, you'd wait for the result
            self.get_logger().info(f'Navigation goal sent to: ({target_pose_data.get("x")}, {target_pose_data.get("y")})')
        else:
            self.get_logger().error('Navigation action client not available')

    def execute_move_straight_action(self, action: Dict):
        """Execute straight-line movement action."""
        distance = action.get('distance', 1.0)
        speed = action.get('speed', 0.5)

        # Calculate movement time
        move_time = distance / speed

        # Create velocity command
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

        # In a real implementation, you'd monitor odometry to verify distance traveled
        time.sleep(move_time)

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        self.get_logger().info(f'Moved straight {distance}m at {speed}m/s')

    def execute_rotate_action(self, action: Dict):
        """Execute rotation action."""
        angle_deg = action.get('angle', 90)
        speed = action.get('speed', 0.5)

        # Convert angle to radians
        angle_rad = math.radians(angle_deg)

        # Calculate rotation time
        rotation_time = abs(angle_rad) / speed

        # Create velocity command
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = speed if angle_deg > 0 else -speed

        self.cmd_vel_pub.publish(cmd)

        # In a real implementation, you'd monitor IMU/odometry to verify rotation
        time.sleep(rotation_time)

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        self.get_logger().info(f'Rotated {angle_deg} degrees at {speed}rad/s')

    def execute_stop_action(self):
        """Execute stop action."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info('Stop command executed')

    def execute_request_clarification_action(self, action: Dict):
        """Execute clarification request."""
        question = action.get('question', 'Could you please clarify?')

        response_msg = String()
        response_msg.data = question
        self.response_pub.publish(response_msg)

        self.get_logger().info(f'Requested clarification: {question}')

    def generate_response_for_plan(self, plan: Dict) -> str:
        """Generate natural language response for executed plan."""
        intent = plan.get('intent', 'unknown')
        actions = plan.get('actions', [])

        if intent == 'navigation':
            target = self.extract_target_from_plan(plan)
            return f"I'm navigating to the {target}. This should take about {plan.get('estimated_duration', 30)} seconds."
        elif intent == 'motion':
            if actions:
                description = actions[0].get('description', 'an action')
                return f"I'm executing: {description}."
        elif intent == 'unknown':
            return "I'm not sure how to execute that command. Could you please rephrase it?"

        return "I've executed the requested action."

    def extract_target_from_plan(self, plan: Dict) -> str:
        """Extract target location from navigation plan."""
        actions = plan.get('actions', [])
        for action in actions:
            if action.get('type') == 'navigate':
                target_pose = action.get('target_pose', {})
                # This would be enhanced to map coordinates to named locations
                return "destination"
        return "location"

    def perception_callback(self, msg):
        """Handle perception data to update context."""
        try:
            data = json.loads(msg.data)
            self.environment_context.update(data)
            self.get_logger().debug(f'Updated environment context: {list(data.keys())}')
        except json.JSONDecodeError:
            self.get_logger().warn('Could not decode perception data')

    def cleanup(self):
        """Clean up resources."""
        with self.execution_lock:
            if self.plan_execution_active:
                # Stop any ongoing execution
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)

def main(args=None):
    rclpy.init(args=args)
    llm_ros = LLMROSIntegration()

    try:
        rclpy.spin(llm_ros)
    except KeyboardInterrupt:
        llm_ros.get_logger().info('Shutting down LLM-ROS Integration')
    finally:
        llm_ros.cleanup()
        llm_ros.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive LLM cognitive planning for robotics:

- **LLM Integration Fundamentals**: Basic integration patterns and model selection
- **Advanced Prompt Engineering**: Effective prompting strategies for robotics
- **Context-Aware Planning**: Maintaining and using environmental and task context
- **Performance Optimization**: Techniques to optimize LLM inference for robotics
- **ROS 2 Integration**: Complete integration with ROS 2 messaging and actions

LLMs provide powerful cognitive capabilities that enable robots to understand complex natural language commands, reason about their environment, and generate detailed action plans for execution.

## Exercises

1. Integrate an LLM with your robotic system for cognitive planning
2. Implement advanced prompting strategies for different robot capabilities
3. Create a context-aware planning system that maintains environmental state
4. Optimize LLM inference for real-time robotic applications
5. Test your system with various natural language commands

## Quiz

1. What is the main advantage of using LLMs for robotic planning?
   a) Faster movement
   b) Natural language understanding and reasoning
   c) Better sensors
   d) Lower cost

2. Which of these is important for context-aware planning?
   a) Battery level monitoring
   b) Environmental state tracking
   c) Task history maintenance
   d) All of the above

3. What does "prompt engineering" refer to in LLM robotics?
   a) Hardware setup
   b) Designing effective input prompts for LLMs
   c) Motor calibration
   d) Sensor fusion

## Mini-Project: LLM-Powered Robot Assistant

Create a complete LLM-powered robot assistant with:
1. Natural language command interpretation
2. Context-aware planning considering environment and robot state
3. Integration with ROS 2 navigation and manipulation capabilities
4. Performance optimization for real-time operation
5. Testing with various complex natural language commands
6. Evaluation of planning accuracy and execution success