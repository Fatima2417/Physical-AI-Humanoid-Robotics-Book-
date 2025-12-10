---
sidebar_position: 12
---

# VLA Applications in Humanoid Robotics

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the unique challenges and opportunities in humanoid robotics applications
- Implement VLA systems for humanoid-specific tasks and behaviors
- Design control systems for humanoid robot locomotion and manipulation
- Integrate perception systems optimized for humanoid applications
- Apply machine learning techniques to humanoid robot behaviors
- Validate and test humanoid robot systems in real-world scenarios

## Introduction to Humanoid Robotics Applications

Humanoid robots represent one of the most challenging and exciting frontiers in robotics. Their human-like form factor enables natural interaction with human environments and makes them ideal candidates for VLA (Vision-Language-Action) systems. Unlike specialized robots, humanoid robots must handle a wide variety of tasks while maintaining balance and dexterity similar to humans.

### Humanoid Robot Architecture

```
Humanoid Robot Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision        │───→│   Language      │───→│   Action        │
│   (Head Cameras,│    │   (Speech Rec., │    │   (Locomotion,  │
│   Perception)   │    │   Command       │    │   Manipulation, │
│                 │    │   Understanding)│    │   Balance Ctrl) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Humanoid      │
                         │   Control       │
                         │   Framework     │
                         │   (Balance,     │
                         │   Coordination, │
                         │   Safety)       │
                         └─────────────────┘
```

### Key Humanoid Capabilities

1. **Bipedal Locomotion**: Walking, running, climbing stairs
2. **Dexterous Manipulation**: Fine motor control with anthropomorphic hands
3. **Social Interaction**: Natural human-robot interaction through gestures and speech
4. **Environmental Adaptation**: Operating in human-designed spaces
5. **Multi-modal Perception**: Processing visual, auditory, and tactile information

## Humanoid Locomotion Control

### Bipedal Walking Patterns

Bipedal locomotion in humanoid robots requires sophisticated control algorithms to maintain balance while moving:

```python
# humanoid_locomotion.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import String, Bool, Float32
from builtin_interfaces.msg import Duration
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import time

class HumanoidLocomotionController(Node):
    def __init__(self):
        super().__init__('humanoid_locomotion_controller')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.balance_status_pub = self.create_publisher(Bool, 'balance_status', 10)
        self.step_status_pub = self.create_publisher(String, 'step_status', 10)
        self.com_trajectory_pub = self.create_publisher(Pose, 'com_trajectory', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.velocity_cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.velocity_cmd_callback, 10)
        self.walk_cmd_sub = self.create_subscription(String, 'walk_command', self.walk_cmd_callback, 10)

        # Robot configuration
        self.left_leg_joints = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
                               'left_knee', 'left_ankle_pitch', 'left_ankle_roll']
        self.right_leg_joints = ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
                                'right_knee', 'right_ankle_pitch', 'right_ankle_roll']

        # Current joint positions
        self.current_joint_positions = {}
        self.current_joint_velocities = {}

        # IMU data
        self.imu_orientation = None
        self.imu_angular_velocity = None
        self.imu_linear_acceleration = None

        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_height = 0.05  # meters
        self.step_duration = 1.0  # seconds
        self.step_frequency = 0.5  # Hz (steps per second)
        self.stride_width = 0.2  # distance between feet (meters)

        # Balance control parameters
        self.com_reference = np.array([0.0, 0.0, 0.85])  # Desired CoM position
        self.com_gain = np.array([100.0, 100.0, 50.0])   # CoM control gains
        self.angular_gain = np.array([50.0, 50.0, 10.0])  # Angular control gains

        # Walking state
        self.walk_phase = 0.0  # 0 to 2π
        self.is_walking = False
        self.walk_direction = np.array([1.0, 0.0, 0.0])  # Forward direction
        self.walk_speed = 0.0  # Current walk speed

        # Balance state
        self.com_position = np.array([0.0, 0.0, 0.85])
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])

        # Walking pattern generation
        self.left_foot_trajectory = []
        self.right_foot_trajectory = []

        # Control timer
        self.control_timer = self.create_timer(0.02, self.locomotion_control_callback)  # 50 Hz

        # Timing
        self.last_control_time = time.time()

        self.get_logger().info('Humanoid Locomotion Controller initialized')

    def imu_callback(self, msg):
        """Update IMU data for balance control."""
        # Extract orientation
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        self.imu_orientation = R.from_quat(orientation)

        # Extract angular velocity
        self.imu_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Extract linear acceleration
        self.imu_linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def joint_state_callback(self, msg):
        """Update joint state information."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

        # Update CoM estimate (simplified - in reality use forward kinematics)
        self.update_com_estimate()

    def velocity_cmd_callback(self, msg):
        """Process velocity commands for walking."""
        linear_speed = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        angular_speed = abs(msg.angular.z)

        # Determine walking direction and speed
        if linear_speed > 0.01:  # Moving forward/backward
            self.walk_direction = np.array([msg.linear.x, msg.linear.y, 0.0])
            self.walk_direction = self.walk_direction / np.linalg.norm(self.walk_direction)
            self.walk_speed = linear_speed
        elif angular_speed > 0.01:  # Turning
            self.walk_speed = 0.0
            # Turning handled separately
        else:  # Stopped
            self.walk_speed = 0.0
            self.is_walking = False

        # Start walking if we have a command
        if self.walk_speed > 0.01:
            self.is_walking = True
        else:
            self.is_walking = False

    def walk_cmd_callback(self, msg):
        """Process high-level walk commands."""
        command = msg.data.lower()

        if command == 'start':
            self.is_walking = True
        elif command == 'stop':
            self.is_walking = False
        elif command == 'faster':
            self.step_frequency = min(self.step_frequency + 0.1, 1.0)  # Max 1 Hz
        elif command == 'slower':
            self.step_frequency = max(self.step_frequency - 0.1, 0.2)  # Min 0.2 Hz
        elif command == 'turn_left':
            self.turn_robot(-0.5)  # Turn left at 0.5 rad/s
        elif command == 'turn_right':
            self.turn_robot(0.5)   # Turn right at 0.5 rad/s

    def locomotion_control_callback(self):
        """Main locomotion control loop."""
        current_time = time.time()
        dt = current_time - self.last_control_time
        self.last_control_time = current_time

        if dt <= 0:
            return

        # Update walking phase if walking
        if self.is_walking:
            self.walk_phase += 2 * math.pi * self.step_frequency * dt
            self.walk_phase %= (2 * math.pi)

        # Generate walking pattern
        if self.is_walking:
            left_foot_pos, right_foot_pos = self.generate_walking_pattern(self.walk_phase)

            # Generate joint commands for walking
            joint_commands = self.calculate_leg_joints_for_walking(
                left_foot_pos, right_foot_pos, self.walk_phase
            )

            # Apply balance corrections
            balance_corrected_commands = self.apply_balance_corrections(joint_commands)

            # Publish joint commands
            self.publish_joint_commands(balance_corrected_commands)

            # Publish step status
            step_msg = String()
            step_msg.data = f"Left foot: ({left_foot_pos[0]:.2f}, {left_foot_pos[1]:.2f}, {left_foot_pos[2]:.2f}), " \
                           f"Right foot: ({right_foot_pos[0]:.2f}, {right_foot_pos[1]:.2f}, {right_foot_pos[2]:.2f}), " \
                           f"Phase: {self.walk_phase:.2f}"
            self.step_status_pub.publish(step_msg)

        # Update balance control
        self.balance_control()

        # Publish CoM trajectory
        com_msg = Pose()
        com_msg.position.x = float(self.com_position[0])
        com_msg.position.y = float(self.com_position[1])
        com_msg.position.z = float(self.com_position[2])
        self.com_trajectory_pub.publish(com_msg)

    def generate_walking_pattern(self, phase):
        """Generate walking pattern for both feet."""
        # Simplified walking pattern generation
        # In reality, this would use more sophisticated gait generation algorithms

        # Calculate foot positions based on walking phase
        # Left foot follows a pattern that's 180 degrees out of phase with right foot

        left_phase = phase
        right_phase = (phase + math.pi) % (2 * math.pi)

        # Calculate foot positions (simplified sinusoidal pattern)
        left_x = self.step_length * math.sin(left_phase) * 0.5
        left_y = self.stride_width / 2 * math.cos(left_phase)
        left_z = max(0, self.step_height * math.sin(left_phase)) if left_phase < math.pi else 0

        right_x = self.step_length * math.sin(right_phase) * 0.5
        right_y = -self.stride_width / 2 * math.cos(right_phase)
        right_z = max(0, self.step_height * math.sin(right_phase)) if right_phase < math.pi else 0

        # Apply walking direction
        left_pos = np.array([left_x, left_y, left_z])
        right_pos = np.array([right_x, right_y, right_z])

        # Transform to world coordinates based on walking direction
        if np.linalg.norm(self.walk_direction) > 0:
            direction_unit = self.walk_direction / np.linalg.norm(self.walk_direction)
            left_pos = self.transform_to_direction(left_pos, direction_unit)
            right_pos = self.transform_to_direction(right_pos, direction_unit)

        # Add current position offset
        left_pos[0] += self.current_joint_positions.get('base_x', 0.0)
        left_pos[1] += self.current_joint_positions.get('base_y', 0.0)
        right_pos[0] += self.current_joint_positions.get('base_x', 0.0)
        right_pos[1] += self.current_joint_positions.get('base_y', 0.0)

        return left_pos, right_pos

    def transform_to_direction(self, position, direction_unit):
        """Transform position to walking direction."""
        # Simple transformation - in reality would use more complex rotation
        transformed = position.copy()
        transformed[0] = position[0] * direction_unit[0] - position[1] * direction_unit[1]
        transformed[1] = position[0] * direction_unit[1] + position[1] * direction_unit[0]
        return transformed

    def calculate_leg_joints_for_walking(self, left_foot_pos, right_foot_pos, phase):
        """Calculate leg joint angles for desired foot positions."""
        # This would use inverse kinematics in a real implementation
        # For this example, we'll use a simplified approach

        # Calculate joint angles based on desired foot positions
        # This is a highly simplified version - real IK would be much more complex

        left_joints = self.calculate_inverse_kinematics_2d(
            left_foot_pos, leg_side='left'
        )
        right_joints = self.calculate_inverse_kinematics_2d(
            right_foot_pos, leg_side='right'
        )

        # Create joint state message
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.left_leg_joints + self.right_leg_joints
        joint_state.position = left_joints + right_joints

        return joint_state

    def calculate_inverse_kinematics_2d(self, target_pos, leg_side):
        """Calculate 2D inverse kinematics for leg."""
        # Simplified 2D IK for a 3-DOF leg (hip, knee, ankle pitch)
        # hip -> knee -> ankle -> foot

        # Leg lengths (simplified)
        thigh_length = 0.4  # meters
        shin_length = 0.4   # meters

        # Calculate target position relative to hip
        hip_x = 0.0  # Hip is at origin in leg coordinate system
        hip_y = -0.8  # Hip height (negative because it's below ground level in robot frame)

        rel_x = target_pos[0] - hip_x
        rel_y = target_pos[1] - hip_y

        # Calculate distance from hip to target
        distance_sq = rel_x**2 + rel_y**2
        distance = math.sqrt(distance_sq)

        # Check if target is reachable
        if distance > thigh_length + shin_length:
            # Target is too far, extend leg fully
            angle_to_target = math.atan2(rel_y, rel_x)
            hip_angle = angle_to_target
            knee_angle = 0.0
            ankle_angle = -hip_angle
        elif distance < abs(thigh_length - shin_length):
            # Target is too close, this shouldn't happen in normal walking
            hip_angle = math.atan2(rel_y, rel_x)
            knee_angle = math.pi if thigh_length > shin_length else -math.pi
            ankle_angle = -hip_angle - knee_angle
        else:
            # Calculate joint angles using law of cosines
            cos_knee_angle = (thigh_length**2 + shin_length**2 - distance_sq) / (2 * thigh_length * shin_length)
            cos_knee_angle = max(-1, min(1, cos_knee_angle))  # Clamp to valid range
            knee_angle = math.pi - math.acos(cos_knee_angle)

            cos_hip_angle = (thigh_length**2 + distance_sq - shin_length**2) / (2 * thigh_length * distance)
            cos_hip_angle = max(-1, min(1, cos_hip_angle))  # Clamp to valid range
            hip_angle = math.atan2(rel_y, rel_x) - math.acos(cos_hip_angle)

            # Ankle angle to maintain foot orientation
            ankle_angle = -hip_angle - knee_angle

        # Return joint angles for this leg
        # This is a simplified 3-DOF model - real humanoid legs have more DOFs
        if leg_side == 'left':
            return [0.0, hip_angle, knee_angle, ankle_angle, 0.0, 0.0]  # yaw, roll, pitch, knee, pitch, roll
        else:  # right
            return [0.0, hip_angle, knee_angle, ankle_angle, 0.0, 0.0]

    def apply_balance_corrections(self, joint_commands):
        """Apply balance corrections to joint commands."""
        if self.imu_orientation is None:
            return joint_commands

        # Get current orientation
        current_rpy = self.imu_orientation.as_euler('xyz')

        # Calculate balance error
        roll_error = current_rpy[0]  # Deviation from upright
        pitch_error = current_rpy[1]  # Deviation from upright

        # Calculate CoM error
        com_error = self.com_reference[:2] - self.com_position[:2]

        # Apply balance corrections
        balance_corrections = np.zeros(len(joint_commands.position))

        # Roll correction (affects hip and ankle rolls)
        balance_corrections[1] += roll_error * self.angular_gain[0]  # Left hip roll
        balance_corrections[5] += -roll_error * self.angular_gain[0]  # Left ankle roll
        balance_corrections[7] += roll_error * self.angular_gain[0]  # Right hip roll
        balance_corrections[11] += -roll_error * self.angular_gain[0]  # Right ankle roll

        # Pitch correction (affects hip pitch)
        balance_corrections[2] += pitch_error * self.angular_gain[1]  # Left hip pitch
        balance_corrections[8] += pitch_error * self.angular_gain[1]  # Right hip pitch

        # CoM correction (affects hip and ankle positions)
        balance_corrections[2] += com_error[0] * self.com_gain[0]  # Left hip pitch for x correction
        balance_corrections[8] += com_error[0] * self.com_gain[0]  # Right hip pitch for x correction
        balance_corrections[1] += com_error[1] * self.com_gain[1]  # Left hip roll for y correction
        balance_corrections[7] += com_error[1] * self.com_gain[1]  # Right hip roll for y correction

        # Apply corrections to joint commands
        corrected_commands = JointState()
        corrected_commands.header.stamp = joint_commands.header.stamp
        corrected_commands.name = joint_commands.name
        corrected_commands.position = [
            pos + corr for pos, corr in zip(joint_commands.position, balance_corrections)
        ]

        return corrected_commands

    def balance_control(self):
        """Update CoM estimate and balance control."""
        # In a real implementation, this would use full forward kinematics
        # For this example, we'll estimate CoM based on simplified model

        # Estimate CoM position based on joint angles (simplified)
        # This would use full kinematic chain in reality
        left_hip_pos = np.array([0.0, 0.1, -0.8])  # Simplified left hip position
        right_hip_pos = np.array([0.0, -0.1, -0.8])  # Simplified right hip position

        # Average hip position as rough CoM estimate
        estimated_com = (left_hip_pos + right_hip_pos) / 2
        estimated_com[2] = 0.85  # Set reasonable CoM height

        # Update CoM velocity and acceleration
        previous_com = self.com_position.copy()
        self.com_position = estimated_com
        self.com_velocity = (self.com_position - previous_com) / 0.02  # Assuming 50Hz control
        self.com_acceleration = (self.com_velocity - self.previous_com_velocity) / 0.02

        # Check balance status
        is_balanced = self.check_balance_status()
        balance_status_msg = Bool()
        balance_status_msg.data = is_balanced
        self.balance_status_pub.publish(balance_status_msg)

        # Store previous values
        self.previous_com_velocity = self.com_velocity.copy()

    def check_balance_status(self) -> bool:
        """Check if robot is in balance."""
        # Check CoM position relative to support polygon
        # This is simplified - real implementation would check against actual foot positions
        com_xy_error = np.linalg.norm(self.com_position[:2] - self.com_reference[:2])
        return com_xy_error < 0.1  # Within 10cm of reference

    def update_com_estimate(self):
        """Update CoM estimate based on current joint configuration."""
        # This would use full kinematic chain and mass distribution
        # For this example, return a simplified estimate
        pass

    def turn_robot(self, angular_velocity):
        """Turn the robot at specified angular velocity."""
        # This would implement turning gait pattern
        # For now, just log the command
        self.get_logger().info(f'Turning robot at {angular_velocity} rad/s')

    def set_walking_parameters(self, step_length, step_height, step_duration):
        """Set walking parameters."""
        self.step_length = max(0.1, min(0.5, step_length))  # Constrain to reasonable range
        self.step_height = max(0.02, min(0.1, step_height))
        self.step_duration = max(0.5, min(2.0, step_duration))
        self.step_frequency = 1.0 / self.step_duration

        self.get_logger().info(
            f'Walking parameters updated - Step: {self.step_length}m, '
            f'Height: {self.step_height}m, Duration: {self.step_duration}s'
        )

    def publish_joint_commands(self, joint_state):
        """Publish joint commands to robot."""
        # In a real implementation, this would send commands to robot actuators
        # For simulation, just log the commands
        self.joint_cmd_pub.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)
    locomotion_node = HumanoidLocomotionController()

    try:
        rclpy.spin(locomotion_node)
    except KeyboardInterrupt:
        locomotion_node.get_logger().info('Shutting down Humanoid Locomotion Controller')
    finally:
        locomotion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Humanoid Manipulation Systems

### Dexterous Hand Control and Manipulation

```python
# humanoid_manipulation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from vision_msgs.msg import Detection2DArray
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional

class HumanoidManipulationController(Node):
    def __init__(self):
        super().__init__('humanoid_manipulation_controller')

        # Publishers
        self.left_arm_cmd_pub = self.create_publisher(JointTrajectory, 'left_arm_controller/command', 10)
        self.right_arm_cmd_pub = self.create_publisher(JointTrajectory, 'right_arm_controller/command', 10)
        self.left_hand_cmd_pub = self.create_publisher(JointTrajectory, 'left_hand_controller/command', 10)
        self.right_hand_cmd_pub = self.create_publisher(JointTrajectory, 'right_hand_controller/command', 10)
        self.manipulation_status_pub = self.create_publisher(String, 'manipulation_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )
        self.manipulation_cmd_sub = self.create_subscription(
            String, 'manipulation_command', self.manipulation_cmd_callback, 10
        )

        # TF listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot configuration
        self.left_arm_joints = [
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_forearm_yaw', 'left_wrist_pitch', 'left_wrist_yaw'
        ]
        self.right_arm_joints = [
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw'
        ]
        self.left_hand_joints = [
            'left_thumb_joint', 'left_index_joint', 'left_middle_joint',
            'left_ring_joint', 'left_pinky_joint', 'left_palm_joint'
        ]
        self.right_hand_joints = [
            'right_thumb_joint', 'right_index_joint', 'right_middle_joint',
            'right_ring_joint', 'right_pinky_joint', 'right_palm_joint'
        ]

        # Current joint states
        self.current_joint_positions = {}
        self.current_joint_velocities = {}

        # End effector positions
        self.left_ee_position = np.array([0.3, 0.2, 0.8])  # Initial position
        self.right_ee_position = np.array([0.3, -0.2, 0.8])  # Initial position

        # Manipulation state
        self.left_hand_open = True
        self.right_hand_open = True
        self.left_hand_grasping = False
        self.right_hand_grasping = False

        # Grasping parameters
        self.grasp_force = 50.0  # Newtons
        self.release_force = 5.0  # Newtons
        self.grasp_tolerance = 0.01  # meters

        # Detected objects
        self.detected_objects = {}

        # Manipulation tasks queue
        self.manipulation_tasks = []

        # Control parameters
        self.manipulation_speed = 0.1  # rad/s for joints
        self.max_manipulation_attempts = 3

        # Performance tracking
        self.manipulation_success_count = 0
        self.manipulation_failure_count = 0

        self.get_logger().info('Humanoid Manipulation Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state information."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

        # Update end effector positions (simplified - in reality use forward kinematics)
        self.update_end_effector_positions()

    def detections_callback(self, msg):
        """Process object detections for manipulation."""
        self.detected_objects = {}

        for detection in msg.detections:
            if detection.results:
                class_name = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score

                if confidence > 0.6:  # Confidence threshold
                    # Convert bounding box center to 3D position (simplified)
                    # In a real system, this would use depth information
                    position_3d = self.estimate_3d_position_from_detection(detection)

                    self.detected_objects[class_name] = {
                        'position': position_3d,
                        'confidence': confidence,
                        'detection': detection
                    }

    def manipulation_cmd_callback(self, msg):
        """Process manipulation commands."""
        command = msg.data
        self.get_logger().info(f'Received manipulation command: {command}')

        # Parse manipulation command
        parsed_command = self.parse_manipulation_command(command)

        if parsed_command:
            self.execute_manipulation_task(parsed_command)

    def parse_manipulation_command(self, command: str) -> Optional[Dict]:
        """Parse manipulation command string."""
        command_lower = command.lower()

        if 'grasp' in command_lower or 'pick up' in command_lower or 'grab' in command_lower:
            # Extract object to grasp
            object_name = self.extract_object_name(command_lower)
            arm_choice = self.select_arm_for_grasping(object_name)

            return {
                'action': 'grasp',
                'object': object_name,
                'arm': arm_choice,
                'command': command
            }

        elif 'release' in command_lower or 'drop' in command_lower or 'let go' in command_lower:
            # Extract object to release
            object_name = self.extract_object_name(command_lower)
            arm_choice = self.select_arm_for_release(object_name)

            return {
                'action': 'release',
                'object': object_name,
                'arm': arm_choice,
                'command': command
            }

        elif 'move' in command_lower or 'go to' in command_lower:
            # Extract target position
            target_position = self.extract_target_position(command_lower)

            return {
                'action': 'move_to',
                'target': target_position,
                'command': command
            }

        elif 'wave' in command_lower:
            return {
                'action': 'wave',
                'arm': 'right',  # Default to right arm
                'command': command
            }

        elif 'point' in command_lower:
            # Extract target to point at
            target = self.extract_target_object(command_lower)

            return {
                'action': 'point',
                'target': target,
                'arm': 'right',  # Default to right arm
                'command': command
            }

        else:
            self.get_logger().warn(f'Unknown manipulation command: {command}')
            return None

    def extract_object_name(self, command: str) -> str:
        """Extract object name from command."""
        # Simple keyword matching - in reality use NLP
        object_keywords = ['ball', 'cup', 'bottle', 'box', 'chair', 'table', 'person', 'robot']

        for keyword in object_keywords:
            if keyword in command:
                return keyword

        return 'unknown_object'

    def extract_target_position(self, command: str) -> Optional[List[float]]:
        """Extract target position from command."""
        # This would use NLP to extract coordinates
        # For simulation, return some default positions
        if 'high' in command:
            return [0.5, 0.0, 1.2]  # High position
        elif 'low' in command:
            return [0.3, 0.0, 0.5]  # Low position
        elif 'left' in command:
            return [0.3, 0.3, 0.8]  # Left position
        elif 'right' in command:
            return [0.3, -0.3, 0.8]  # Right position
        else:
            return [0.4, 0.0, 0.8]  # Default position

    def select_arm_for_grasping(self, object_name: str) -> str:
        """Select which arm to use for grasping."""
        # In a real system, this would consider object location and arm accessibility
        # For simulation, alternate between arms or use proximity
        if object_name in self.detected_objects:
            obj_pos = self.detected_objects[object_name]['position']

            # Choose arm based on object position
            if obj_pos[1] > 0:  # Object is on the left side
                return 'left'
            else:  # Object is on the right side
                return 'right'
        else:
            # Default to right arm
            return 'right'

    def select_arm_for_release(self, object_name: str) -> str:
        """Select which arm to use for releasing."""
        # In a real system, this would check which hand is currently grasping
        # For simulation, return right arm
        return 'right'

    def execute_manipulation_task(self, command_dict: Dict):
        """Execute the parsed manipulation task."""
        action = command_dict['action']

        if action == 'grasp':
            self.execute_grasp_task(command_dict)
        elif action == 'release':
            self.execute_release_task(command_dict)
        elif action == 'move_to':
            self.execute_move_to_task(command_dict)
        elif action == 'wave':
            self.execute_wave_task(command_dict)
        elif action == 'point':
            self.execute_point_task(command_dict)
        else:
            self.get_logger().warn(f'Unknown manipulation action: {action}')

    def execute_grasp_task(self, command_dict: Dict):
        """Execute grasping task."""
        object_name = command_dict['object']
        arm = command_dict['arm']

        self.get_logger().info(f'Executing grasp task for {object_name} using {arm} arm')

        # Check if object is detected
        if object_name not in self.detected_objects:
            self.get_logger().warn(f'Object {object_name} not detected, cannot grasp')
            status_msg = String()
            status_msg.data = f'Object {object_name} not detected for grasping'
            self.manipulation_status_pub.publish(status_msg)
            return

        target_pos = self.detected_objects[object_name]['position']

        # Move arm to object position
        success = self.move_arm_to_position(arm, target_pos, approach_distance=0.1)

        if success:
            # Grasp the object
            grasp_success = self.grasp_object(arm, object_name)

            if grasp_success:
                self.manipulation_success_count += 1
                status_msg = String()
                status_msg.data = f'Successfully grasped {object_name} with {arm} hand'
                self.manipulation_status_pub.publish(status_msg)
                self.get_logger().info(f'Successfully grasped {object_name}')
            else:
                self.manipulation_failure_count += 1
                status_msg = String()
                status_msg.data = f'Failed to grasp {object_name} with {arm} hand'
                self.manipulation_status_pub.publish(status_msg)
                self.get_logger().error(f'Failed to grasp {object_name}')
        else:
            self.manipulation_failure_count += 1
            status_msg = String()
            status_msg.data = f'Failed to reach {object_name} with {arm} arm'
            self.manipulation_status_pub.publish(status_msg)
            self.get_logger().error(f'Failed to reach {object_name}')

    def execute_release_task(self, command_dict: Dict):
        """Execute release task."""
        object_name = command_dict['object']
        arm = command_dict['arm']

        self.get_logger().info(f'Executing release task for {object_name} from {arm} hand')

        # Release the object
        release_success = self.release_object(arm, object_name)

        if release_success:
            self.manipulation_success_count += 1
            status_msg = String()
            status_msg.data = f'Successfully released {object_name} from {arm} hand'
            self.manipulation_status_pub.publish(status_msg)
            self.get_logger().info(f'Successfully released {object_name}')
        else:
            self.manipulation_failure_count += 1
            status_msg = String()
            status_msg.data = f'Failed to release {object_name} from {arm} hand'
            self.manipulation_status_pub.publish(status_msg)
            self.get_logger().error(f'Failed to release {object_name}')

    def execute_move_to_task(self, command_dict: Dict):
        """Execute move to position task."""
        target_pos = command_dict['target']

        self.get_logger().info(f'Moving to position: {target_pos}')

        # Move both arms to the target position (or choose one)
        # For this example, move right arm
        success = self.move_arm_to_position('right', target_pos)

        if success:
            status_msg = String()
            status_msg.data = f'Moved right arm to position {target_pos}'
            self.manipulation_status_pub.publish(status_msg)
            self.get_logger().info(f'Successfully moved to position: {target_pos}')
        else:
            status_msg = String()
            status_msg.data = f'Failed to move to position {target_pos}'
            self.manipulation_status_pub.publish(status_msg)
            self.get_logger().error(f'Failed to move to position: {target_pos}')

    def execute_wave_task(self, command_dict: Dict):
        """Execute waving gesture."""
        arm = command_dict['arm']

        self.get_logger().info(f'Executing wave gesture with {arm} arm')

        # Create waving motion trajectory
        wave_trajectory = self.create_wave_trajectory(arm)

        if wave_trajectory:
            self.publish_arm_trajectory(arm, wave_trajectory)

            status_msg = String()
            status_msg.data = f'Waving gesture executed with {arm} arm'
            self.manipulation_status_pub.publish(status_msg)
        else:
            status_msg = String()
            status_msg.data = f'Failed to create wave trajectory for {arm} arm'
            self.manipulation_status_pub.publish(status_msg)

    def execute_point_task(self, command_dict: Dict):
        """Execute pointing gesture."""
        target = command_dict['target']
        arm = command_dict['arm']

        self.get_logger().info(f'Pointing at {target} with {arm} arm')

        # Find target position if it's a known object
        target_pos = None
        if target in self.detected_objects:
            target_pos = self.detected_objects[target]['position']
        else:
            # Use default position for unknown targets
            target_pos = [1.0, 0.0, 1.0]

        if target_pos:
            # Create pointing trajectory
            point_trajectory = self.create_point_trajectory(arm, target_pos)

            if point_trajectory:
                self.publish_arm_trajectory(arm, point_trajectory)

                status_msg = String()
                status_msg.data = f'Pointing at {target} with {arm} arm'
                self.manipulation_status_pub.publish(status_msg)
            else:
                status_msg = String()
                status_msg.data = f'Failed to create point trajectory for {target} with {arm} arm'
                self.manipulation_status_pub.publish(status_msg)
        else:
            status_msg = String()
            status_msg.data = f'Unknown target {target} for pointing'
            self.manipulation_status_pub.publish(status_msg)

    def move_arm_to_position(self, arm: str, target_pos: List[float], approach_distance: float = 0.0) -> bool:
        """Move arm to target position."""
        try:
            # In a real implementation, this would use inverse kinematics
            # For simulation, create a simple trajectory

            # Get current arm position (simplified)
            current_pos = self.left_ee_position if arm == 'left' else self.right_ee_position

            # Calculate trajectory points
            trajectory = JointTrajectory()
            trajectory.header.stamp = self.get_clock().now().to_msg()
            trajectory.joint_names = self.left_arm_joints if arm == 'left' else self.right_arm_joints

            # Create trajectory points
            num_points = 20  # 20 intermediate points
            for i in range(num_points + 1):
                ratio = i / num_points
                intermediate_pos = current_pos + ratio * (np.array(target_pos) - current_pos)

                point = JointTrajectoryPoint()

                # This is simplified - in reality, use IK to convert Cartesian position to joint angles
                # For simulation, create mock joint angles
                if arm == 'left':
                    joint_angles = self.mock_ik_left_arm(intermediate_pos)
                else:  # right
                    joint_angles = self.mock_ik_right_arm(intermediate_pos)

                point.positions = joint_angles
                point.time_from_start.sec = 0
                point.time_from_start.nanosec = int((i * 2.0 / num_points) * 1e9)  # 2 seconds total

                trajectory.points.append(point)

            # Publish trajectory
            publisher = self.left_arm_cmd_pub if arm == 'left' else self.right_arm_cmd_pub
            publisher.publish(trajectory)

            return True

        except Exception as e:
            self.get_logger().error(f'Error moving {arm} arm to position: {e}')
            return False

    def grasp_object(self, arm: str, object_name: str) -> bool:
        """Execute grasping action."""
        try:
            # Close hand to grasp object
            hand_trajectory = JointTrajectory()
            hand_trajectory.header.stamp = self.get_clock().now().to_msg()
            hand_trajectory.joint_names = self.left_hand_joints if arm == 'left' else self.right_hand_joints

            point = JointTrajectoryPoint()

            # Set joint positions for grasping (simplified)
            if arm == 'left':
                grasp_positions = [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0]  # Closed hand
            else:  # right
                grasp_positions = [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0]  # Closed hand

            point.positions = grasp_positions
            point.time_from_start.sec = 1  # 1 second to complete grasp
            point.time_from_start.nanosec = 0

            hand_trajectory.points = [point]

            # Publish hand command
            hand_publisher = self.left_hand_cmd_pub if arm == 'left' else self.right_hand_cmd_pub
            hand_publisher.publish(hand_trajectory)

            # Update state
            if arm == 'left':
                self.left_hand_grasping = True
                self.left_hand_open = False
            else:
                self.right_hand_grasping = True
                self.right_hand_open = False

            return True

        except Exception as e:
            self.get_logger().error(f'Error grasping object with {arm} hand: {e}')
            return False

    def release_object(self, arm: str, object_name: str) -> bool:
        """Execute release action."""
        try:
            # Open hand to release object
            hand_trajectory = JointTrajectory()
            hand_trajectory.header.stamp = self.get_clock().now().to_msg()
            hand_trajectory.joint_names = self.left_hand_joints if arm == 'left' else self.right_hand_joints

            point = JointTrajectoryPoint()

            # Set joint positions for open hand
            if arm == 'left':
                open_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Open hand
            else:  # right
                open_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Open hand

            point.positions = open_positions
            point.time_from_start.sec = 1  # 1 second to complete release
            point.time_from_start.nanosec = 0

            hand_trajectory.points = [point]

            # Publish hand command
            hand_publisher = self.left_hand_cmd_pub if arm == 'left' else self.right_hand_cmd_pub
            hand_publisher.publish(hand_trajectory)

            # Update state
            if arm == 'left':
                self.left_hand_grasping = False
                self.left_hand_open = True
            else:
                self.right_hand_grasping = False
                self.right_hand_open = True

            return True

        except Exception as e:
            self.get_logger().error(f'Error releasing object with {arm} hand: {e}')
            return False

    def create_wave_trajectory(self, arm: str) -> Optional[JointTrajectory]:
        """Create waving motion trajectory."""
        try:
            trajectory = JointTrajectory()
            trajectory.header.stamp = self.get_clock().now().to_msg()
            trajectory.joint_names = self.left_arm_joints if arm == 'left' else self.right_arm_joints

            # Define wave motion points
            wave_points = [
                [0.2, 0.1, 0.0, -0.5, 0.0, 0.1, 0.0],  # Starting position
                [0.2, 0.1, 0.2, -0.5, 0.1, 0.1, 0.1],  # Wave up
                [0.2, 0.1, 0.0, -0.5, -0.1, 0.1, -0.1], # Wave down
                [0.2, 0.1, 0.0, -0.5, 0.0, 0.1, 0.0]   # Return to start
            ]

            # Convert to trajectory points
            for i, positions in enumerate(wave_points):
                point = JointTrajectoryPoint()
                point.positions = positions
                point.time_from_start.sec = 0
                point.time_from_start.nanosec = int((i + 1) * 0.5 * 1e9)  # 0.5 seconds per point

                trajectory.points.append(point)

            return trajectory

        except Exception as e:
            self.get_logger().error(f'Error creating wave trajectory: {e}')
            return None

    def create_point_trajectory(self, arm: str, target_pos: List[float]) -> Optional[JointTrajectory]:
        """Create pointing trajectory."""
        try:
            trajectory = JointTrajectory()
            trajectory.header.stamp = self.get_clock().now().to_msg()
            trajectory.joint_names = self.left_arm_joints if arm == 'left' else self.right_arm_joints

            # Calculate pointing position based on target
            current_pos = self.left_ee_position if arm == 'left' else self.right_ee_position

            # For pointing, we want to orient the hand toward the target
            direction = np.array(target_pos) - current_pos
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction_unit = direction / distance
                # Extend arm toward target
                point_pos = current_pos + direction_unit * min(distance, 0.5)  # Limit extension to 0.5m
            else:
                point_pos = current_pos

            # Create trajectory to point position
            point = JointTrajectoryPoint()

            # This would use real IK in practice
            if arm == 'left':
                joint_angles = self.mock_ik_left_arm(point_pos)
            else:
                joint_angles = self.mock_ik_right_arm(point_pos)

            point.positions = joint_angles
            point.time_from_start.sec = 2  # 2 seconds to point
            point.time_from_start.nanosec = 0

            trajectory.points = [point]
            return trajectory

        except Exception as e:
            self.get_logger().error(f'Error creating point trajectory: {e}')
            return None

    def mock_ik_left_arm(self, target_pos: np.ndarray) -> List[float]:
        """Mock inverse kinematics for left arm (simplified)."""
        # This is a highly simplified mock - real IK would be much more complex
        # In a real implementation, use proper IK solvers like KDL, MoveIt!, or analytical solutions
        return [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0]  # Default positions

    def mock_ik_right_arm(self, target_pos: np.ndarray) -> List[float]:
        """Mock inverse kinematics for right arm (simplified)."""
        # This is a highly simplified mock - real IK would be much more complex
        return [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0]  # Default positions

    def estimate_3d_position_from_detection(self, detection) -> np.ndarray:
        """Estimate 3D position from 2D detection."""
        # This would use depth information in a real system
        # For simulation, return a mock 3D position
        return np.array([1.0, 0.0, 0.8])  # Default position

    def update_end_effector_positions(self):
        """Update end effector positions based on joint states."""
        # This would use forward kinematics in a real implementation
        # For simulation, keep default positions
        pass

    def publish_arm_trajectory(self, arm: str, trajectory: JointTrajectory):
        """Publish arm trajectory."""
        publisher = self.left_arm_cmd_pub if arm == 'left' else self.right_arm_cmd_pub
        publisher.publish(trajectory)

    def get_manipulation_performance(self) -> Dict[str, float]:
        """Get manipulation performance metrics."""
        total_attempts = self.manipulation_success_count + self.manipulation_failure_count
        success_rate = (self.manipulation_success_count / total_attempts * 100) if total_attempts > 0 else 0.0

        return {
            'success_rate': success_rate,
            'successful_manipulations': self.manipulation_success_count,
            'failed_manipulations': self.manipulation_failure_count,
            'total_attempts': total_attempts
        }

    def reset_manipulation_stats(self):
        """Reset manipulation statistics."""
        self.manipulation_success_count = 0
        self.manipulation_failure_count = 0
        self.get_logger().info('Manipulation statistics reset')

def main(args=None):
    rclpy.init(args=args)
    manipulation_node = HumanoidManipulationController()

    try:
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        manipulation_node.get_logger().info('Shutting down Humanoid Manipulation Controller')
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Humanoid Social Interaction

### Social Behavior and Communication

```python
# humanoid_social_interaction.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image, JointState
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Detection2DArray
import numpy as np
import math
import json
from typing import Dict, List, Optional
import time
import threading

class HumanoidSocialInteraction(Node):
    def __init__(self):
        super().__init__('humanoid_social_interaction')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'robot_speech', 10)
        self.gesture_cmd_pub = self.create_publisher(JointState, 'gesture_commands', 10)
        self.social_status_pub = self.create_publisher(String, 'social_status', 10)
        self.face_expression_pub = self.create_publisher(String, 'face_expression', 10)
        self.attention_target_pub = self.create_publisher(Pose, 'attention_target', 10)

        # Subscribers
        self.human_detection_sub = self.create_subscription(
            Detection2DArray, 'human_detections', self.human_detection_callback, 10
        )
        self.speech_recognition_sub = self.create_subscription(
            String, 'speech_recognition', self.speech_recognition_callback, 10
        )
        self.social_command_sub = self.create_subscription(
            String, 'social_command', self.social_command_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )

        # Social behavior state
        self.humans_detected = []
        self.closest_human = None
        self.attention_mode = 'passive'  # 'passive', 'engaged', 'focused'
        self.current_interaction = None
        self.interaction_history = []

        # Greeting and social behaviors
        self.enable_greetings = True
        self.greeting_distance = 2.0  # meters
        self.greeting_timeout = 30.0  # seconds between greetings

        # Emotional expressions
        self.current_emotion = 'neutral'
        self.emotional_responses = {
            'happy': ['smile', 'nod', 'wave'],
            'sad': ['frown', 'shake_head'],
            'surprised': ['widen_eyes', 'raise_eyebrows'],
            'confused': ['tilt_head', 'scrunch_eyes']
        }

        # Attention tracking
        self.last_greeting_time = {}
        self.interaction_timers = {}

        # Social behavior parameters
        self.social_distance = 1.0  # meters for comfortable interaction
        self.max_attention_duration = 60.0  # seconds before shifting attention

        # Gesture execution
        self.gesture_queue = []
        self.gesture_execution_active = False

        # Social monitoring timer
        self.social_monitor_timer = self.create_timer(0.5, self.social_monitor_callback)  # 2 Hz

        # Current robot state
        self.current_joint_positions = {}
        self.current_human_pose = None

        self.get_logger().info('Humanoid Social Interaction Controller initialized')

    def human_detection_callback(self, msg):
        """Process human detections for social interaction."""
        detected_humans = []

        for detection in msg.detections:
            if detection.results:
                # Check if it's a human detection
                class_name = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score

                if class_name.lower() in ['person', 'human'] and confidence > 0.7:
                    # Calculate approximate 3D position (simplified)
                    # In a real system, this would use depth information
                    distance_estimate = self.estimate_distance_from_detection(detection)
                    bearing = self.calculate_bearing_from_detection(detection)

                    human_info = {
                        'id': len(detected_humans),
                        'position': {
                            'x': distance_estimate * math.cos(bearing),
                            'y': distance_estimate * math.sin(bearing),
                            'distance': distance_estimate
                        },
                        'confidence': confidence,
                        'last_seen': time.time()
                    }

                    detected_humans.append(human_info)

        # Update humans list
        self.humans_detected = detected_humans

        # Find closest human
        if detected_humans:
            self.closest_human = min(detected_humans, key=lambda h: h['position']['distance'])
        else:
            self.closest_human = None

        # Handle greetings for newly detected humans
        self.handle_new_human_detections(detected_humans)

    def speech_recognition_callback(self, msg):
        """Process recognized speech for social responses."""
        recognized_text = msg.data.lower()

        # Check for social cues in speech
        if any(greeting_word in recognized_text for greeting_word in ['hello', 'hi', 'hey']):
            self.respond_to_greeting()
        elif any(acknowledgment in recognized_text for acknowledgment in ['thank you', 'thanks', 'ok', 'okay']):
            self.respond_to_acknowledgment()
        elif any(question_word in recognized_text for question_word in ['how are you', 'what are you doing', 'who are you']):
            self.respond_to_question(recognized_text)
        elif any(command in recognized_text for command in ['wave', 'nod', 'smile', 'frown']):
            self.execute_social_command_from_speech(recognized_text)

    def social_command_callback(self, msg):
        """Process direct social commands."""
        command = msg.data.lower()

        if command == 'greet':
            self.execute_greeting_behavior()
        elif command == 'wave':
            self.execute_gesture('wave')
        elif command == 'nod':
            self.execute_gesture('nod')
        elif command == 'smile':
            self.execute_face_expression('smile')
        elif command == 'frown':
            self.execute_face_expression('frown')
        elif command == 'look_at_closest':
            self.focus_attention_on_closest_human()
        elif command.startswith('follow:'):
            target = command.split(':', 1)[1]
            self.start_follow_behavior(target)
        elif command.startswith('follow_stop'):
            self.stop_follow_behavior()

    def social_monitor_callback(self):
        """Monitor social situation and adjust behavior."""
        current_time = time.time()

        # Update attention based on human positions
        self.update_attention()

        # Check for timeout interactions
        for human_id, start_time in list(self.interaction_timers.items()):
            if current_time - start_time > self.max_attention_duration:
                self.shift_attention_away(human_id)

        # Update social status
        self.publish_social_status()

        # Execute queued gestures
        self.execute_queued_gestures()

    def handle_new_human_detections(self, detected_humans):
        """Handle newly detected humans."""
        current_time = time.time()

        for human in detected_humans:
            human_id = human['id']
            distance = human['position']['distance']

            # Check if this is a new human (not seen recently)
            if human_id not in self.last_greeting_time:
                self.last_greeting_time[human_id] = 0

            time_since_last_greeting = current_time - self.last_greeting_time[human_id]

            # Greet if close enough and enough time has passed
            if (distance <= self.greeting_distance and
                time_since_last_greeting >= self.greeting_timeout and
                self.enable_greetings):

                self.execute_greeting_behavior_to_human(human_id)
                self.last_greeting_time[human_id] = current_time

    def execute_greeting_behavior(self):
        """Execute general greeting behavior."""
        if self.closest_human:
            self.execute_greeting_behavior_to_human(self.closest_human['id'])
        else:
            # No humans detected, maybe look around
            self.look_around_curiously()

    def execute_greeting_behavior_to_human(self, human_id):
        """Execute greeting behavior directed at specific human."""
        self.get_logger().info(f'Greeting human {human_id}')

        # Play greeting audio
        greeting_msg = String()
        greeting_msg.data = "Hello! Nice to meet you!"
        self.speech_pub.publish(greeting_msg)

        # Execute greeting gesture
        self.execute_gesture('wave')

        # Update interaction history
        self.interaction_history.append({
            'type': 'greeting',
            'target': human_id,
            'timestamp': time.time()
        })

    def respond_to_greeting(self):
        """Respond to human greeting."""
        self.get_logger().info('Responding to human greeting')

        response_msg = String()
        response_msg.data = "Hello! How can I help you today?"
        self.speech_pub.publish(response_msg)

        # Acknowledge with gesture
        self.execute_gesture('nod')

        # Update interaction history
        self.interaction_history.append({
            'type': 'greeting_response',
            'timestamp': time.time()
        })

    def respond_to_acknowledgment(self):
        """Respond to human acknowledgment."""
        self.get_logger().info('Responding to human acknowledgment')

        response_msg = String()
        response_msg.data = "You're welcome!"
        self.speech_pub.publish(response_msg)

        # Maybe smile
        self.execute_face_expression('smile')

        # Update interaction history
        self.interaction_history.append({
            'type': 'acknowledgment_response',
            'timestamp': time.time()
        })

    def respond_to_question(self, question: str):
        """Respond to human questions."""
        self.get_logger().info(f'Responding to question: {question}')

        # Simple responses based on question content
        if 'how are you' in question:
            response = "I'm doing well, thank you for asking! I'm here to help with robotics tasks."
        elif 'what are you doing' in question:
            response = "I'm assisting with robotics research and development. I can help with navigation, manipulation, and perception tasks."
        elif 'who are you' in question:
            response = "I'm a humanoid robot designed for research and assistance. I can understand natural language and perform various robotic tasks."
        else:
            response = "That's an interesting question. I'm designed to help with various robotics tasks and interact naturally with humans."

        response_msg = String()
        response_msg.data = response
        self.speech_pub.publish(response_msg)

        # Update interaction history
        self.interaction_history.append({
            'type': 'question_response',
            'question': question,
            'timestamp': time.time()
        })

    def execute_gesture(self, gesture_type: str):
        """Execute a social gesture."""
        self.get_logger().info(f'Executing gesture: {gesture_type}')

        # Create joint trajectory for gesture
        gesture_trajectory = self.create_gesture_trajectory(gesture_type)

        if gesture_trajectory:
            self.gesture_cmd_pub.publish(gesture_trajectory)

        # Add to gesture queue for execution
        self.gesture_queue.append({
            'type': gesture_type,
            'timestamp': time.time()
        })

    def execute_face_expression(self, expression: str):
        """Execute facial expression."""
        self.get_logger().info(f'Executing face expression: {expression}')

        expression_msg = String()
        expression_msg.data = expression
        self.face_expression_pub.publish(expression_msg)

    def create_gesture_trajectory(self, gesture_type: str) -> Optional[JointState]:
        """Create trajectory for social gesture."""
        try:
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()

            if gesture_type == 'wave':
                # Wave with right arm
                joint_state.name = ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                                  'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw']
                joint_state.position = [0.2, 0.1, 0.0, -0.5, 0.1, 0.1, 0.0]  # Starting position

                # This would be animated over time in a real implementation
                # For simulation, just return the starting position

            elif gesture_type == 'nod':
                # Nod head (simplified)
                joint_state.name = ['neck_pitch']
                joint_state.position = [-0.2]  # Look down slightly

            elif gesture_type == 'shake_head':
                # Shake head (simplified)
                joint_state.name = ['neck_yaw']
                joint_state.position = [0.2]  # Turn head to the side

            elif gesture_type == 'point':
                # Point gesture
                joint_state.name = ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                                  'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw']
                joint_state.position = [0.3, 0.0, 0.0, -0.8, 0.0, 0.0, 0.0]  # Pointing position

            else:
                return None

            return joint_state

        except Exception as e:
            self.get_logger().error(f'Error creating gesture trajectory: {e}')
            return None

    def execute_queued_gestures(self):
        """Execute gestures in the queue."""
        if not self.gesture_queue:
            return

        # Execute one gesture at a time to avoid conflicts
        gesture = self.gesture_queue.pop(0)
        self.get_logger().info(f'Executing queued gesture: {gesture["type"]}')

        # In a real implementation, this would animate the gesture over time
        # For simulation, just log the execution

    def focus_attention_on_closest_human(self):
        """Focus robot attention on the closest human."""
        if self.closest_human:
            # Calculate direction to human
            human_pos = self.closest_human['position']
            direction_to_human = math.atan2(human_pos['y'], human_pos['x'])

            # Turn head toward human (simplified)
            neck_trajectory = JointState()
            neck_trajectory.header.stamp = self.get_clock().now().to_msg()
            neck_trajectory.name = ['neck_yaw', 'neck_pitch']
            neck_trajectory.position = [direction_to_human, 0.0]  # Look toward human

            self.gesture_cmd_pub.publish(neck_trajectory)

            # Update attention mode
            self.attention_mode = 'focused'
            self.current_interaction = self.closest_human['id']

            # Start interaction timer
            self.interaction_timers[self.closest_human['id']] = time.time()

            self.get_logger().info(f'Focusing attention on human at {human_pos}')

            # Publish attention target
            target_pose = Pose()
            target_pose.position.x = float(human_pos['x'])
            target_pose.position.y = float(human_pos['y'])
            target_pose.position.z = 1.0  # Eye level
            self.attention_target_pub.publish(target_pose)

    def look_around_curiously(self):
        """Look around to detect humans."""
        # Simple head movement to look around
        curiosity_trajectory = JointState()
        curiosity_trajectory.header.stamp = self.get_clock().now().to_msg()
        curiosity_trajectory.name = ['neck_yaw', 'neck_pitch']
        curiosity_trajectory.position = [0.5, 0.1]  # Look to the right and slightly up

        self.gesture_cmd_pub.publish(curiosity_trajectory)

        self.attention_mode = 'passive'
        self.get_logger().info('Looking around curiously')

    def update_attention(self):
        """Update attention based on human positions and interactions."""
        if self.closest_human and self.closest_human['position']['distance'] < self.social_distance:
            if self.attention_mode == 'passive':
                self.attention_mode = 'engaged'
                self.get_logger().info('Engaging with nearby human')
        elif self.attention_mode != 'passive':
            self.attention_mode = 'passive'
            self.get_logger().info('Returning to passive attention mode')

    def start_follow_behavior(self, target: str):
        """Start following behavior."""
        self.get_logger().info(f'Starting to follow: {target}')

        # In a real implementation, this would start a navigation behavior
        # to follow the specified target (human, object, etc.)

        # For simulation, just log the intention
        self.get_logger().info(f'Following behavior started for {target}')

    def stop_follow_behavior(self):
        """Stop following behavior."""
        self.get_logger().info('Stopping follow behavior')

        # Stop any movement
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        self.get_logger().info('Follow behavior stopped')

    def estimate_distance_from_detection(self, detection) -> float:
        """Estimate distance from detection bounding box."""
        # In a real system, this would use depth camera data
        # For simulation, return a mock distance based on bounding box size
        # Larger bounding box = closer object
        bbox_size = detection.bbox.size_x * detection.bbox.size_y
        if bbox_size > 10000:  # Large bounding box
            return 0.5  # Close
        elif bbox_size > 5000:  # Medium bounding box
            return 1.0  # Medium distance
        else:
            return 2.0  # Far

    def calculate_bearing_from_detection(self, detection) -> float:
        """Calculate bearing (angle) from detection position."""
        # In a real system, this would use camera calibration
        # For simulation, return a mock bearing based on image position
        center_x = detection.bbox.center.x
        image_width = 640  # Assuming 640x480 image
        normalized_x = (center_x - image_width / 2) / (image_width / 2)  # -1 to 1
        return math.atan2(normalized_x, 1.0)  # Simple angle calculation

    def publish_social_status(self):
        """Publish current social interaction status."""
        status_info = {
            'attention_mode': self.attention_mode,
            'humans_detected': len(self.humans_detected),
            'closest_human_distance': self.closest_human['position']['distance'] if self.closest_human else float('inf'),
            'current_interaction': self.current_interaction,
            'gestures_executed': len(self.gesture_queue)
        }

        status_msg = String()
        status_msg.data = json.dumps(status_info)
        self.social_status_pub.publish(status_msg)

    def joint_state_callback(self, msg):
        """Update current joint positions."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def set_social_distance(self, distance: float):
        """Set social interaction distance."""
        self.social_distance = max(0.5, distance)  # Minimum 0.5m
        self.get_logger().info(f'Social distance set to: {self.social_distance}m')

    def set_greeting_parameters(self, enable: bool, distance: float, timeout: float):
        """Set greeting behavior parameters."""
        self.enable_greetings = enable
        self.greeting_distance = distance
        self.greeting_timeout = timeout

        self.get_logger().info(
            f'Greeting parameters - Enabled: {enable}, Distance: {distance}m, Timeout: {timeout}s'
        )

    def get_social_interaction_metrics(self) -> Dict[str, Any]:
        """Get social interaction performance metrics."""
        return {
            'humans_detected': len(self.humans_detected),
            'current_attention_mode': self.attention_mode,
            'total_interactions': len(self.interaction_history),
            'greetings_given': len([i for i in self.interaction_history if i['type'] == 'greeting']),
            'responses_given': len([i for i in self.interaction_history if 'response' in i['type']])
        }

def main(args=None):
    rclpy.init(args=args)
    social_node = HumanoidSocialInteraction()

    try:
        rclpy.spin(social_node)
    except KeyboardInterrupt:
        social_node.get_logger().info('Shutting down Humanoid Social Interaction Controller')
    finally:
        social_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deployment Validation and Testing

### Comprehensive Testing Framework

```python
# deployment_validation.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
import unittest
import time
import threading
from typing import Dict, Any, Optional
import numpy as np

class HumanoidDeploymentValidator(Node):
    def __init__(self):
        super().__init__('humanoid_deployment_validator')

        # Publishers for test results
        self.test_status_pub = self.create_publisher(String, 'test_status', 10)
        self.test_results_pub = self.create_publisher(String, 'test_results', 10)
        self.performance_metrics_pub = self.create_publisher(String, 'performance_metrics', 10)

        # Subscribers for system monitoring
        self.system_status_sub = self.create_subscription(
            String, 'system_status', self.system_status_callback, 10
        )
        self.performance_sub = self.create_subscription(
            Float32, 'performance_metric', self.performance_callback, 10
        )

        # Test configuration
        self.tests_running = False
        self.test_results = {}
        self.test_progress = 0.0
        self.current_test = None

        # Performance thresholds
        self.performance_thresholds = {
            'min_fps': 10.0,
            'max_latency': 0.1,  # seconds
            'max_cpu_usage': 80.0,  # percent
            'min_battery_life': 30.0,  # minutes
            'success_rate': 0.95  # 95% success rate
        }

        # Test suites
        self.test_suites = {
            'locomotion': self.locomotion_test_suite,
            'manipulation': self.manipulation_test_suite,
            'perception': self.perception_test_suite,
            'social_interaction': self.social_interaction_test_suite,
            'system_integration': self.system_integration_test_suite
        }

        # Performance tracking
        self.frame_rates = {'vision': [], 'control': [], 'communication': []}
        self.latencies = {'sensor_to_actuator': [], 'processing': [], 'network': []}
        self.cpu_usage_history = []
        self.memory_usage_history = []

        # Validation timer
        self.validation_timer = self.create_timer(1.0, self.validation_callback)

        self.get_logger().info('Humanoid Deployment Validator initialized')

    def system_status_callback(self, msg):
        """Monitor system status."""
        try:
            status_data = json.loads(msg.data)
            self.current_system_status = status_data
        except json.JSONDecodeError:
            self.get_logger().warn('Could not parse system status message')

    def performance_callback(self, msg):
        """Monitor performance metrics."""
        # This would receive various performance metrics from the system
        # For now, we'll just store the values
        pass

    def validation_callback(self):
        """Main validation callback."""
        if self.tests_running:
            # Continue running tests
            self.run_next_test()
        else:
            # Monitor system performance
            self.monitor_system_performance()

    def start_validation_suite(self, suite_name: str):
        """Start a specific validation suite."""
        if suite_name in self.test_suites:
            self.get_logger().info(f'Starting validation suite: {suite_name}')
            self.current_test_suite = suite_name
            self.tests_running = True
            self.test_results = {}
            self.test_progress = 0.0

            # Run the test suite
            test_suite = self.test_suites[suite_name]
            results = test_suite()

            # Publish results
            results_msg = String()
            results_msg.data = json.dumps(results)
            self.test_results_pub.publish(results_msg)

            self.tests_running = False
            self.current_test_suite = None

        else:
            self.get_logger().warn(f'Unknown test suite: {suite_name}')

    def locomotion_test_suite(self) -> Dict[str, Any]:
        """Test locomotion capabilities."""
        results = {
            'suite': 'locomotion',
            'tests': [],
            'summary': {}
        }

        # Test 1: Basic walking
        test1_result = self.test_basic_walking()
        results['tests'].append({
            'name': 'basic_walking',
            'passed': test1_result['success'],
            'details': test1_result['details'],
            'metrics': test1_result['metrics']
        })

        # Test 2: Turning capability
        test2_result = self.test_turning_capability()
        results['tests'].append({
            'name': 'turning_capability',
            'passed': test2_result['success'],
            'details': test2_result['details'],
            'metrics': test2_result['metrics']
        })

        # Test 3: Obstacle avoidance
        test3_result = self.test_obstacle_avoidance()
        results['tests'].append({
            'name': 'obstacle_avoidance',
            'passed': test3_result['success'],
            'details': test3_result['details'],
            'metrics': test3_result['metrics']
        })

        # Test 4: Balance maintenance
        test4_result = self.test_balance_maintenance()
        results['tests'].append({
            'name': 'balance_maintenance',
            'passed': test4_result['success'],
            'details': test4_result['details'],
            'metrics': test4_result['metrics']
        })

        # Calculate suite summary
        passed_tests = sum(1 for test in results['tests'] if test['passed'])
        total_tests = len(results['tests'])
        results['summary'] = {
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }

        return results

    def manipulation_test_suite(self) -> Dict[str, Any]:
        """Test manipulation capabilities."""
        results = {
            'suite': 'manipulation',
            'tests': [],
            'summary': {}
        }

        # Test 1: Object grasping
        test1_result = self.test_object_grasping()
        results['tests'].append({
            'name': 'object_grasping',
            'passed': test1_result['success'],
            'details': test1_result['details'],
            'metrics': test1_result['metrics']
        })

        # Test 2: Object placement
        test2_result = self.test_object_placement()
        results['tests'].append({
            'name': 'object_placement',
            'passed': test2_result['success'],
            'details': test2_result['details'],
            'metrics': test2_result['metrics']
        })

        # Test 3: Arm trajectory execution
        test3_result = self.test_arm_trajectory_execution()
        results['tests'].append({
            'name': 'arm_trajectory_execution',
            'passed': test3_result['success'],
            'details': test3_result['details'],
            'metrics': test3_result['metrics']
        })

        # Test 4: Precision manipulation
        test4_result = self.test_precision_manipulation()
        results['tests'].append({
            'name': 'precision_manipulation',
            'passed': test4_result['success'],
            'details': test4_result['details'],
            'metrics': test4_result['metrics']
        })

        # Calculate suite summary
        passed_tests = sum(1 for test in results['tests'] if test['passed'])
        total_tests = len(results['tests'])
        results['summary'] = {
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }

        return results

    def perception_test_suite(self) -> Dict[str, Any]:
        """Test perception capabilities."""
        results = {
            'suite': 'perception',
            'tests': [],
            'summary': {}
        }

        # Test 1: Object detection accuracy
        test1_result = self.test_object_detection_accuracy()
        results['tests'].append({
            'name': 'object_detection_accuracy',
            'passed': test1_result['success'],
            'details': test1_result['details'],
            'metrics': test1_result['metrics']
        })

        # Test 2: Human detection
        test2_result = self.test_human_detection()
        results['tests'].append({
            'name': 'human_detection',
            'passed': test2_result['success'],
            'details': test2_result['details'],
            'metrics': test2_result['metrics']
        })

        # Test 3: Depth estimation accuracy
        test3_result = self.test_depth_estimation_accuracy()
        results['tests'].append({
            'name': 'depth_estimation_accuracy',
            'passed': test3_result['success'],
            'details': test3_result['details'],
            'metrics': test3_result['metrics']
        })

        # Test 4: Scene understanding
        test4_result = self.test_scene_understanding()
        results['tests'].append({
            'name': 'scene_understanding',
            'passed': test4_result['success'],
            'details': test4_result['details'],
            'metrics': test4_result['metrics']
        })

        # Calculate suite summary
        passed_tests = sum(1 for test in results['tests'] if test['passed'])
        total_tests = len(results['tests'])
        results['summary'] = {
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }

        return results

    def social_interaction_test_suite(self) -> Dict[str, Any]:
        """Test social interaction capabilities."""
        results = {
            'suite': 'social_interaction',
            'tests': [],
            'summary': {}
        }

        # Test 1: Greeting response
        test1_result = self.test_greeting_response()
        results['tests'].append({
            'name': 'greeting_response',
            'passed': test1_result['success'],
            'details': test1_result['details'],
            'metrics': test1_result['metrics']
        })

        # Test 2: Speech recognition accuracy
        test2_result = self.test_speech_recognition_accuracy()
        results['tests'].append({
            'name': 'speech_recognition_accuracy',
            'passed': test2_result['success'],
            'details': test2_result['details'],
            'metrics': test2_result['metrics']
        })

        # Test 3: Attention tracking
        test3_result = self.test_attention_tracking()
        results['tests'].append({
            'name': 'attention_tracking',
            'passed': test3_result['success'],
            'details': test3_result['details'],
            'metrics': test3_result['metrics']
        })

        # Test 4: Gesture execution
        test4_result = self.test_gesture_execution()
        results['tests'].append({
            'name': 'gesture_execution',
            'passed': test4_result['success'],
            'details': test4_result['details'],
            'metrics': test4_result['metrics']
        })

        # Calculate suite summary
        passed_tests = sum(1 for test in results['tests'] if test['passed'])
        total_tests = len(results['tests'])
        results['summary'] = {
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }

        return results

    def system_integration_test_suite(self) -> Dict[str, Any]:
        """Test system integration capabilities."""
        results = {
            'suite': 'system_integration',
            'tests': [],
            'summary': {}
        }

        # Test 1: Multi-module coordination
        test1_result = self.test_multi_module_coordination()
        results['tests'].append({
            'name': 'multi_module_coordination',
            'passed': test1_result['success'],
            'details': test1_result['details'],
            'metrics': test1_result['metrics']
        })

        # Test 2: Real-time performance
        test2_result = self.test_real_time_performance()
        results['tests'].append({
            'name': 'real_time_performance',
            'passed': test2_result['success'],
            'details': test2_result['details'],
            'metrics': test2_result['metrics']
        })

        # Test 3: Communication reliability
        test3_result = self.test_communication_reliability()
        results['tests'].append({
            'name': 'communication_reliability',
            'passed': test3_result['success'],
            'details': test3_result['details'],
            'metrics': test3_result['metrics']
        })

        # Test 4: Safety system integration
        test4_result = self.test_safety_system_integration()
        results['tests'].append({
            'name': 'safety_system_integration',
            'passed': test4_result['success'],
            'details': test4_result['details'],
            'metrics': test4_result['metrics']
        })

        # Calculate suite summary
        passed_tests = sum(1 for test in results['tests'] if test['passed'])
        total_tests = len(results['tests'])
        results['summary'] = {
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }

        return results

    def test_basic_walking(self) -> Dict[str, Any]:
        """Test basic walking capability."""
        # Simulate walking test
        # In a real implementation, this would command the robot to walk and verify success
        success = np.random.random() > 0.1  # 90% success rate in simulation

        return {
            'success': success,
            'details': 'Robot successfully walked 5 meters forward' if success else 'Robot failed to walk properly',
            'metrics': {
                'distance_traveled': 5.0 if success else np.random.uniform(0, 3),
                'walking_speed': 0.5 if success else 0.1,
                'balance_maintained': success,
                'execution_time': 10.0 if success else 5.0
            }
        }

    def test_turning_capability(self) -> Dict[str, Any]:
        """Test turning capability."""
        # Simulate turning test
        success = np.random.random() > 0.15  # 85% success rate

        return {
            'success': success,
            'details': 'Robot successfully turned 90 degrees' if success else 'Robot failed to turn properly',
            'metrics': {
                'turn_angle': 90.0 if success else np.random.uniform(45, 89),
                'turn_accuracy': 0.95 if success else 0.7,
                'balance_during_turn': success,
                'execution_time': 3.0 if success else 2.0
            }
        }

    def test_obstacle_avoidance(self) -> Dict[str, Any]:
        """Test obstacle avoidance capability."""
        # Simulate obstacle avoidance test
        success = np.random.random() > 0.2  # 80% success rate

        return {
            'success': success,
            'details': 'Robot successfully avoided obstacles' if success else 'Robot collided with obstacle',
            'metrics': {
                'obstacles_detected': 3 if success else np.random.randint(1, 3),
                'avoidance_success_rate': 1.0 if success else 0.6,
                'path_efficiency': 0.85 if success else 0.4,
                'execution_time': 15.0 if success else 8.0
            }
        }

    def test_balance_maintenance(self) -> Dict[str, Any]:
        """Test balance maintenance capability."""
        # Simulate balance test
        success = np.random.random() > 0.05  # 95% success rate

        return {
            'success': success,
            'details': 'Robot maintained balance throughout operation' if success else 'Robot lost balance',
            'metrics': {
                'balance_stability': 0.98 if success else 0.6,
                'com_deviation': 0.05 if success else 0.2,
                'recovery_attempts': 0 if success else 1,
                'test_duration': 60.0
            }
        }

    def test_object_grasping(self) -> Dict[str, Any]:
        """Test object grasping capability."""
        # Simulate grasping test
        success = np.random.random() > 0.25  # 75% success rate

        return {
            'success': success,
            'details': 'Robot successfully grasped target object' if success else 'Robot failed to grasp object',
            'metrics': {
                'grasp_success_rate': 0.75 if success else 0.4,
                'grasp_force_applied': 10.0 if success else 5.0,
                'object_preservation': success,
                'execution_time': 8.0 if success else 4.0
            }
        }

    def test_object_placement(self) -> Dict[str, Any]:
        """Test object placement capability."""
        # Simulate placement test
        success = np.random.random() > 0.3  # 70% success rate

        return {
            'success': success,
            'details': 'Robot successfully placed object at target location' if success else 'Robot failed to place object',
            'metrics': {
                'placement_accuracy': 0.9 if success else 0.6,
                'target_reached': success,
                'object_orientation': 0.95 if success else 0.7,
                'execution_time': 12.0 if success else 6.0
            }
        }

    def test_arm_trajectory_execution(self) -> Dict[str, Any]:
        """Test arm trajectory execution."""
        # Simulate trajectory test
        success = np.random.random() > 0.1  # 90% success rate

        return {
            'success': success,
            'details': 'Robot successfully executed arm trajectory' if success else 'Robot failed to execute trajectory',
            'metrics': {
                'trajectory_accuracy': 0.92 if success else 0.5,
                'execution_time': 5.0 if success else 3.0,
                'smoothness_score': 0.88 if success else 0.4,
                'joint_limit_compliance': success
            }
        }

    def test_precision_manipulation(self) -> Dict[str, Any]:
        """Test precision manipulation capability."""
        # Simulate precision test
        success = np.random.random() > 0.4  # 60% success rate (more challenging)

        return {
            'success': success,
            'details': 'Robot successfully performed precision manipulation' if success else 'Robot failed precision task',
            'metrics': {
                'precision_error': 0.005 if success else 0.02,
                'task_completion_rate': 0.6 if success else 0.3,
                'force_control_accuracy': 0.9 if success else 0.6,
                'execution_time': 20.0 if success else 10.0
            }
        }

    def test_object_detection_accuracy(self) -> Dict[str, Any]:
        """Test object detection accuracy."""
        # Simulate detection test
        success = np.random.random() > 0.15  # 85% success rate

        return {
            'success': success,
            'details': 'Object detection performed within acceptable accuracy' if success else 'Object detection accuracy below threshold',
            'metrics': {
                'detection_rate': 0.9 if success else 0.6,
                'false_positive_rate': 0.05 if success else 0.15,
                'false_negative_rate': 0.02 if success else 0.1,
                'processing_time': 0.03 if success else 0.05
            }
        }

    def test_human_detection(self) -> Dict[str, Any]:
        """Test human detection capability."""
        # Simulate human detection test
        success = np.random.random() > 0.1  # 90% success rate

        return {
            'success': success,
            'details': 'Human detection performed successfully' if success else 'Human detection failed',
            'metrics': {
                'human_detection_rate': 0.95 if success else 0.7,
                'identification_accuracy': 0.88 if success else 0.5,
                'detection_range': 3.0 if success else 2.0,
                'processing_time': 0.04 if success else 0.06
            }
        }

    def test_depth_estimation_accuracy(self) -> Dict[str, Any]:
        """Test depth estimation accuracy."""
        # Simulate depth estimation test
        success = np.random.random() > 0.2  # 80% success rate

        return {
            'success': success,
            'details': 'Depth estimation performed within acceptable accuracy' if success else 'Depth estimation accuracy below threshold',
            'metrics': {
                'depth_accuracy': 0.02 if success else 0.05,  # meters
                'measurement_range': 5.0 if success else 3.0,  # meters
                'confidence_score': 0.85 if success else 0.6,
                'processing_time': 0.02 if success else 0.04
            }
        }

    def test_scene_understanding(self) -> Dict[str, Any]:
        """Test scene understanding capability."""
        # Simulate scene understanding test
        success = np.random.random() > 0.25  # 75% success rate

        return {
            'success': success,
            'details': 'Scene understanding performed successfully' if success else 'Scene understanding failed',
            'metrics': {
                'object_classification_accuracy': 0.8 if success else 0.5,
                'spatial_relationship_identification': 0.85 if success else 0.4,
                'context_awareness_score': 0.75 if success else 0.3,
                'processing_time': 0.1 if success else 0.15
            }
        }

    def test_greeting_response(self) -> Dict[str, Any]:
        """Test greeting response capability."""
        # Simulate greeting test
        success = np.random.random() > 0.1  # 90% success rate

        return {
            'success': success,
            'details': 'Robot responded appropriately to greeting' if success else 'Robot failed to respond to greeting',
            'metrics': {
                'response_time': 1.5 if success else 3.0,
                'appropriateness_score': 0.9 if success else 0.5,
                'social_engagement': 0.85 if success else 0.4,
                'execution_time': 2.0 if success else 4.0
            }
        }

    def test_speech_recognition_accuracy(self) -> Dict[str, Any]:
        """Test speech recognition accuracy."""
        # Simulate speech recognition test
        success = np.random.random() > 0.15  # 85% success rate

        return {
            'success': success,
            'details': 'Speech recognition performed within acceptable accuracy' if success else 'Speech recognition accuracy below threshold',
            'metrics': {
                'word_accuracy_rate': 0.88 if success else 0.6,
                'command_understanding_rate': 0.9 if success else 0.5,
                'noise_robustness': 0.8 if success else 0.4,
                'processing_time': 0.2 if success else 0.4
            }
        }

    def test_attention_tracking(self) -> Dict[str, Any]:
        """Test attention tracking capability."""
        # Simulate attention tracking test
        success = np.random.random() > 0.2  # 80% success rate

        return {
            'success': success,
            'details': 'Attention tracking performed successfully' if success else 'Attention tracking failed',
            'metrics': {
                'tracking_accuracy': 0.92 if success else 0.6,
                'response_time_to_attention_shift': 0.3 if success else 0.8,
                'sustained_attention_duration': 30.0 if success else 15.0,
                'execution_time': 60.0
            }
        }

    def test_gesture_execution(self) -> Dict[str, Any]:
        """Test gesture execution capability."""
        # Simulate gesture execution test
        success = np.random.random() > 0.1  # 90% success rate

        return {
            'success': success,
            'details': 'Gesture executed successfully' if success else 'Gesture execution failed',
            'metrics': {
                'gesture_accuracy': 0.88 if success else 0.5,
                'timing_accuracy': 0.9 if success else 0.6,
                'smoothness_score': 0.85 if success else 0.4,
                'execution_time': 3.0 if success else 2.0
            }
        }

    def test_multi_module_coordination(self) -> Dict[str, Any]:
        """Test multi-module coordination."""
        # Simulate coordination test
        success = np.random.random() > 0.15  # 85% success rate

        return {
            'success': success,
            'details': 'Modules coordinated successfully' if success else 'Module coordination failed',
            'metrics': {
                'coordination_success_rate': 0.85 if success else 0.6,
                'inter_module_communication_latency': 0.01 if success else 0.05,
                'system_synchronization': success,
                'execution_time': 30.0 if success else 15.0
            }
        }

    def test_real_time_performance(self) -> Dict[str, Any]:
        """Test real-time performance."""
        # Simulate performance test
        success = np.random.random() > 0.1  # 90% success rate

        return {
            'success': success,
            'details': 'System maintained real-time performance' if success else 'System failed real-time requirements',
            'metrics': {
                'average_frame_rate': 25.0 if success else 15.0,
                'worst_case_latency': 0.05 if success else 0.1,
                'deadline_compliance_rate': 0.98 if success else 0.8,
                'test_duration': 120.0
            }
        }

    def test_communication_reliability(self) -> Dict[str, Any]:
        """Test communication reliability."""
        # Simulate communication test
        success = np.random.random() > 0.05  # 95% success rate

        return {
            'success': success,
            'details': 'Communication maintained reliability' if success else 'Communication reliability issues detected',
            'metrics': {
                'message_delivery_rate': 0.99 if success else 0.85,
                'average_latency': 0.005 if success else 0.02,
                'packet_loss_rate': 0.001 if success else 0.05,
                'test_duration': 60.0
            }
        }

    def test_safety_system_integration(self) -> Dict[str, Any]:
        """Test safety system integration."""
        # Simulate safety test
        success = np.random.random() > 0.02  # 98% success rate

        return {
            'success': success,
            'details': 'Safety systems integrated and functioning' if success else 'Safety system issues detected',
            'metrics': {
                'emergency_stop_response_time': 0.05 if success else 0.2,
                'collision_detection_accuracy': 0.98 if success else 0.7,
                'safe_operation_rate': 1.0 if success else 0.9,
                'test_duration': 180.0
            }
        }

    def monitor_system_performance(self):
        """Monitor overall system performance."""
        # This would continuously monitor system performance metrics
        # For simulation, we'll just publish some mock performance data

        performance_data = {
            'cpu_usage': np.random.uniform(20, 60),
            'memory_usage': np.random.uniform(40, 70),
            'gpu_usage': np.random.uniform(10, 80) if torch.cuda.is_available() else 0,
            'temperature': np.random.uniform(35, 65),
            'battery_level': np.random.uniform(0.3, 1.0),
            'network_latency': np.random.uniform(5, 50),  # ms
            'current_fps': np.random.uniform(15, 30)
        }

        performance_msg = String()
        performance_msg.data = json.dumps(performance_data)
        self.performance_metrics_pub.publish(performance_msg)

        # Store history for trend analysis
        self.cpu_usage_history.append(performance_data['cpu_usage'])
        self.memory_usage_history.append(performance_data['memory_usage'])

        if len(self.cpu_usage_history) > 100:
            self.cpu_usage_history.pop(0)
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history.pop(0)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get overall validation summary."""
        return {
            'total_tests_run': len(self.test_results),
            'tests_passed': sum(1 for result in self.test_results.values() if result.get('success', False)),
            'tests_failed': sum(1 for result in self.test_results.values() if not result.get('success', True)),
            'overall_success_rate': len([r for r in self.test_results.values() if r.get('success', False)]) / len(self.test_results) if self.test_results else 0,
            'last_validation_time': time.time(),
            'system_performance': {
                'avg_cpu_usage': np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0,
                'avg_memory_usage': np.mean(self.memory_usage_history) if self.memory_usage_history else 0,
                'current_temperature': np.random.uniform(35, 65),  # Mock value
                'battery_level': np.random.uniform(0.3, 1.0)  # Mock value
            }
        }

    def reset_validation_state(self):
        """Reset validation state."""
        self.test_results.clear()
        self.test_progress = 0.0
        self.current_test = None
        self.tests_running = False
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()

        self.get_logger().info('Validation state reset')

def main(args=None):
    rclpy.init(args=args)
    validator = HumanoidDeploymentValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down Humanoid Deployment Validator')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered real-world deployment considerations for humanoid robots with VLA systems:

- **Hardware Platform Considerations**: Optimizing for resource-constrained systems
- **Model Optimization**: Quantization, ONNX conversion, and performance optimization
- **Unity Robotics Integration**: Setting up Unity for robotics applications
- **Social Interaction Systems**: Implementing natural human-robot interaction
- **Deployment Validation**: Comprehensive testing and validation frameworks

Deploying VLA systems to real humanoid robots requires careful consideration of hardware constraints, safety requirements, and real-world operational conditions.

## Exercises

1. Set up a Unity project with robotics packages
2. Implement model optimization techniques for your VLA system
3. Create a deployment validation framework for your robot
4. Test your system with various real-world scenarios
5. Optimize your system for resource-constrained platforms

## Quiz

1. What is the main advantage of Unity over traditional simulators for robotics?
   a) Lower cost
   b) Photorealistic graphics and advanced sensor simulation
   c) Simpler interface
   d) Better documentation

2. What does "quantization" mean in model optimization?
   a) Adding more data
   b) Reducing precision to decrease model size and increase speed
   c) Increasing model complexity
   d) Adding more layers

3. What is the recommended approach for new robotics projects?
   a) Always use Gazebo Classic
   b) Always use New Gazebo
   c) Evaluate requirements and choose appropriately
   d) Use both simultaneously

## Mini-Project: Complete Deployment System

Create a complete deployment system with:
1. Optimized models for embedded platforms
2. Unity simulation environment for testing
3. Comprehensive validation framework
4. Performance monitoring and optimization
5. Real-world testing procedures
6. Safety and reliability mechanisms
7. Documentation of deployment procedures