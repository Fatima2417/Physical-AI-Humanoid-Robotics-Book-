---
sidebar_position: 8
---

# Humanoid Control Systems

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the unique challenges of controlling humanoid robots
- Implement balance and locomotion control systems for bipedal robots
- Design manipulation control for humanoid arms and hands
- Integrate perception with humanoid control for autonomous behavior
- Implement whole-body control frameworks for coordinated motion
- Validate and optimize humanoid control performance

## Introduction to Humanoid Control

Humanoid robots present unique control challenges due to their complex morphology, multiple degrees of freedom, and the need to maintain balance while performing tasks. Unlike wheeled robots, humanoid robots must actively control their balance and coordinate multiple limbs simultaneously.

### Humanoid Control Architecture

```
Humanoid Control Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   High-Level    │───→│   Mid-Level     │───→│   Low-Level     │
│   Tasks         │    │   Motion        │    │   Joint         │
│   (Walk, Grasp, │    │   Planning &    │    │   Control       │
│   Speak)        │    │   Balance       │    │   (PD Control,  │
│                 │    │   (Trajectories,│    │   Torque Ctrl)  │
└─────────────────┘    │   IK, Whole-    │    └─────────────────┘
         │               │   Body Ctrl)    │           │
         └───────────────┼─────────────────┼───────────┘
                         │                 │
                 ┌─────────────────┐       │
                 │   Perception    │       │
                 │   Integration   │       │
                 │   (Vision,      │       │
                 │   IMU, Force)   │       │
                 └─────────────────┘       │
                         │                 │
                         └─────────────────┘
```

### Key Control Challenges

1. **Balance Control**: Maintaining center of mass within support polygon
2. **Locomotion**: Coordinated leg movement for walking, running, jumping
3. **Manipulation**: Coordinated arm and hand movement for grasping
4. **Whole-Body Coordination**: Coordinating multiple limbs simultaneously
5. **Disturbance Rejection**: Handling external forces and perturbations
6. **Dynamic Stability**: Maintaining stability during motion

## Balance Control Systems

### Center of Mass Control

Maintaining balance in humanoid robots requires precise control of the center of mass (CoM) position relative to the support polygon formed by the feet.

```python
# balance_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Point, Vector3, Wrench
from std_msgs.msg import Float64, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import time

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Publishers
        self.com_position_pub = self.create_publisher(Point, 'balance/com_position', 10)
        self.com_velocity_pub = self.create_publisher(Vector3, 'balance/com_velocity', 10)
        self.com_acceleration_pub = self.create_publisher(Vector3, 'balance/com_acceleration', 10)
        self.balance_status_pub = self.create_publisher(Bool, 'balance/status', 10)
        self.compensation_cmd_pub = self.create_publisher(JointTrajectory, 'balance/compensation_commands', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.force_torque_sub = self.create_subscription(Wrench, 'ft_sensors', self.force_torque_callback, 10)

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.imu_data = None
        self.force_torque_data = None

        # Balance control parameters
        self.com_position = np.array([0.0, 0.0, 0.0])
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])
        self.support_polygon = []  # Points defining support polygon
        self.com_reference = np.array([0.0, 0.0, 0.85])  # Reference CoM position

        # Control gains
        self.com_kp = np.array([100.0, 100.0, 100.0])  # Proportional gains
        self.com_kd = np.array([20.0, 20.0, 20.0])     # Derivative gains
        self.com_ki = np.array([10.0, 10.0, 10.0])     # Integral gains

        # Integral terms for CoM control
        self.com_error_integral = np.array([0.0, 0.0, 0.0])

        # Balance thresholds
        self.balance_margin = 0.05  # meters margin for balance
        self.max_com_velocity = 0.5  # m/s
        self.max_com_acceleration = 2.0  # m/s²

        # Timing
        self.last_update_time = None

        # Control timer
        self.control_timer = self.create_timer(0.01, self.balance_control_callback)  # 100 Hz

        self.get_logger().info('Balance Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data for orientation and angular velocity."""
        self.imu_data = msg

        # Extract orientation
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Convert to rotation matrix
        rotation = R.from_quat(orientation).as_matrix()

        # Extract angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def joint_state_callback(self, msg):
        """Update joint state information."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

        # Update CoM position based on current joint configuration
        self.update_com_position()

    def force_torque_callback(self, msg):
        """Process force/torque sensor data."""
        self.force_torque_data = msg

    def update_com_position(self):
        """Update center of mass position based on kinematics."""
        # This is a simplified CoM calculation
        # In a real implementation, you'd use full kinematic chain and mass distribution
        # For simulation, we'll calculate a simplified CoM

        # Get positions of key body parts (simplified)
        # In reality, you'd use forward kinematics and mass distribution
        torso_mass = 10.0  # kg
        head_mass = 2.0    # kg
        leg_mass = 5.0     # kg each
        arm_mass = 3.0     # kg each

        # Simplified CoM calculation based on joint positions
        # This would be much more complex in reality with proper kinematic chains
        total_mass = torso_mass + head_mass + 2*leg_mass + 2*arm_mass

        # Calculate CoM position (simplified - in real system use full kinematics)
        com_x = 0.0  # Calculate based on actual joint positions
        com_y = 0.0  # Calculate based on actual joint positions
        com_z = 0.85  # Typical CoM height for humanoid

        # Apply any offset based on joint configuration
        # This is where you'd use forward kinematics in a real system
        if 'left_hip_pitch' in self.joint_positions:
            hip_offset = self.joint_positions['left_hip_pitch'] * 0.02  # Simplified effect
            com_x += hip_offset

        self.com_position = np.array([com_x, com_y, com_z])

    def balance_control_callback(self):
        """Main balance control loop."""
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_update_time is None:
            self.last_update_time = current_time
            return

        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if dt <= 0:
            return

        # Update CoM velocity and acceleration
        previous_com_velocity = self.com_velocity.copy()
        self.com_velocity = (self.com_position - self.previous_com_position) / dt if hasattr(self, 'previous_com_position') else np.array([0.0, 0.0, 0.0])
        self.com_acceleration = (self.com_velocity - previous_com_velocity) / dt if hasattr(self, 'previous_com_velocity') else np.array([0.0, 0.0, 0.0])

        # Store previous values for next iteration
        self.previous_com_position = self.com_position.copy()
        self.previous_com_velocity = self.com_velocity.copy()

        # Calculate CoM error
        com_error = self.com_reference - self.com_position

        # Update integral term
        self.com_error_integral += com_error * dt

        # Apply integral windup protection
        max_integral = 10.0
        self.com_error_integral = np.clip(self.com_error_integral, -max_integral, max_integral)

        # Calculate CoM control effort using PID
        com_derivative = self.com_velocity  # We're controlling velocity, so derivative of position is velocity
        com_control_effort = (
            self.com_kp * com_error +
            self.com_kd * (-com_derivative) +  # Negative because we want to reduce velocity
            self.com_ki * self.com_error_integral
        )

        # Limit control effort
        max_control = np.array([50.0, 50.0, 100.0])  # Different limits for x,y,z
        com_control_effort = np.clip(com_control_effort, -max_control, max_control)

        # Check if robot is balanced
        is_balanced = self.is_robot_balanced(com_error)

        # Generate compensation commands
        compensation_commands = self.generate_compensation_commands(com_control_effort)

        # Publish balance information
        com_msg = Point()
        com_msg.x = float(self.com_position[0])
        com_msg.y = float(self.com_position[1])
        com_msg.z = float(self.com_position[2])
        self.com_position_pub.publish(com_msg)

        vel_msg = Vector3()
        vel_msg.x = float(self.com_velocity[0])
        vel_msg.y = float(self.com_velocity[1])
        vel_msg.z = float(self.com_velocity[2])
        self.com_velocity_pub.publish(vel_msg)

        acc_msg = Vector3()
        acc_msg.x = float(self.com_acceleration[0])
        acc_msg.y = float(self.com_acceleration[1])
        acc_msg.z = float(self.com_acceleration[2])
        self.com_acceleration_pub.publish(acc_msg)

        status_msg = Bool()
        status_msg.data = is_balanced
        self.balance_status_pub.publish(status_msg)

        if compensation_commands:
            self.compensation_cmd_pub.publish(compensation_commands)

        # Log balance status periodically
        if int(current_time) % 5 == 0:  # Every 5 seconds
            self.get_logger().info(
                f'Balance Status - CoM: ({self.com_position[0]:.3f}, {self.com_position[1]:.3f}, {self.com_position[2]:.3f}), '
                f'Error: ({com_error[0]:.3f}, {com_error[1]:.3f}, {com_error[2]:.3f}), '
                f'Balanced: {is_balanced}'
            )

    def is_robot_balanced(self, com_error):
        """Check if robot is within balance margins."""
        # Check if CoM is within support polygon with margin
        # This is a simplified check - in reality you'd check against actual support polygon
        com_xy_error = math.sqrt(com_error[0]**2 + com_error[1]**2)
        return com_xy_error < self.balance_margin and abs(com_error[2]) < 0.1

    def generate_compensation_commands(self, control_effort):
        """Generate joint commands to compensate for balance errors."""
        # This would generate actual joint trajectory commands in a real system
        # For simulation, we'll create mock commands that would compensate for CoM error

        # In a real implementation, you'd use:
        # 1. Inverse kinematics to find joint positions for desired CoM
        # 2. Whole-body control to coordinate multiple joints
        # 3. Optimization to find best joint configuration

        # For this example, create a simple trajectory that would compensate
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = [
            'left_hip_pitch', 'left_knee', 'left_ankle_pitch',
            'right_hip_pitch', 'right_knee', 'right_ankle_pitch',
            'torso_yaw'
        ]

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000  # 50ms

        # Calculate compensation based on CoM error
        # This is highly simplified - real implementation would be much more complex
        compensation_values = []

        # Hip adjustments based on CoM x/y error
        hip_compensation_x = -control_effort[0] * 0.001  # Scale factor
        hip_compensation_y = -control_effort[1] * 0.001  # Scale factor

        compensation_values.extend([
            hip_compensation_x,  # left_hip_pitch
            0.0,                 # left_knee (no direct compensation)
            -hip_compensation_x * 0.5,  # left_ankle_pitch
            hip_compensation_x,  # right_hip_pitch
            0.0,                 # right_knee (no direct compensation)
            -hip_compensation_x * 0.5,  # right_ankle_pitch
            -control_effort[1] * 0.0005  # torso_yaw for lateral balance
        ])

        point.positions = compensation_values
        point.velocities = [0.0] * len(compensation_values)  # Start with zero velocity
        point.accelerations = [0.0] * len(compensation_values)  # Start with zero acceleration

        trajectory.points = [point]
        return trajectory

    def set_balance_reference(self, x, y, z):
        """Set reference CoM position."""
        self.com_reference = np.array([x, y, z])
        self.get_logger().info(f'Balance reference set to: ({x}, {y}, {z})')

    def adjust_balance_gains(self, kp, ki, kd):
        """Adjust balance control gains."""
        self.com_kp = np.array(kp)
        self.com_ki = np.array(ki)
        self.com_kd = np.array(kd)
        self.get_logger().info(f'Balance gains adjusted - Kp: {kp}, Ki: {ki}, Kd: {kd}')

def main(args=None):
    rclpy.init(args=args)
    balance_node = BalanceController()

    try:
        rclpy.spin(balance_node)
    except KeyboardInterrupt:
        balance_node.get_logger().info('Shutting down Balance Controller')
    finally:
        balance_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Locomotion Control

### Walking Pattern Generation

```python
# walking_pattern_generator.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64
import numpy as np
import math
from scipy import signal
import time

class WalkingPatternGenerator(Node):
    def __init__(self):
        super().__init__('walking_pattern_generator')

        # Publishers
        self.walk_trajectory_pub = self.create_publisher(JointTrajectory, 'walking_trajectory', 10)
        self.footstep_plan_pub = self.create_publisher(Pose, 'footstep_plan', 10)
        self.walk_status_pub = self.create_publisher(String, 'walk_status', 10)

        # Subscribers
        self.walk_command_sub = self.create_subscription(
            Twist, 'cmd_walk', self.walk_command_callback, 10
        )
        self.balance_state_sub = self.create_subscription(
            Bool, 'balance/status', self.balance_status_callback, 10
        )

        # Walking parameters
        self.step_length = 0.3      # meters
        self.step_width = 0.2       # meters (distance between feet)
        self.step_height = 0.05     # meters (foot lift height)
        self.step_duration = 1.0    # seconds per step
        self.walk_phase = 0.0       # Current phase of walking cycle (0 to 2π)
        self.is_walking = False
        self.is_balanced = True

        # Gait parameters
        self.stride_frequency = 0.5  # Hz (steps per second)
        self.duty_factor = 0.6       # Fraction of stride in contact with ground
        self.phase_offset = np.pi    # Phase difference between legs

        # Robot kinematic parameters (simplified)
        self.hip_height = 0.8        # Height of hip from ground
        self.leg_length = 0.7        # Length of leg

        # Current walking command
        self.desired_velocity = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.walk_command_timestamp = None

        # Walking pattern generation
        self.pattern_generation_timer = self.create_timer(0.02, self.generate_walking_pattern)  # 50 Hz

        # Performance tracking
        self.step_count = 0
        self.last_step_time = None

        self.get_logger().info('Walking Pattern Generator initialized')

    def walk_command_callback(self, msg):
        """Process walking commands."""
        self.desired_velocity = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.angular.z
        ])
        self.walk_command_timestamp = self.get_clock().now()

        # Start walking if not already walking and command is non-zero
        if not self.is_walking and np.linalg.norm(self.desired_velocity[:2]) > 0.01:
            self.is_walking = True
            self.get_logger().info('Starting walking pattern generation')

        # Stop walking if command is zero
        if self.is_walking and np.linalg.norm(self.desired_velocity) < 0.01:
            self.is_walking = False
            self.get_logger().info('Stopping walking pattern generation')

    def balance_status_callback(self, msg):
        """Update balance status."""
        self.is_balanced = msg.data

    def generate_walking_pattern(self):
        """Generate walking pattern based on desired velocity."""
        if not self.is_walking or not self.is_balanced:
            # If not walking or not balanced, send neutral stance
            if self.is_walking and not self.is_balanced:
                self.get_logger().warn('Robot not balanced, stopping walk')
                self.is_walking = False
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Calculate walking phase based on desired velocity and time
        if self.last_step_time is None:
            self.last_step_time = current_time

        dt = current_time - self.last_step_time
        self.last_step_time = current_time

        # Adjust step frequency based on desired speed
        speed = np.linalg.norm(self.desired_velocity[:2])
        if speed > 0.1:  # Only adjust frequency if moving
            # Increase frequency with speed (simplified)
            adjusted_frequency = self.stride_frequency * (1 + speed * 0.5)
        else:
            adjusted_frequency = self.stride_frequency

        # Update walking phase
        self.walk_phase += 2 * math.pi * adjusted_frequency * dt

        # Generate trajectory for current phase
        trajectory = self.calculate_walking_trajectory(self.walk_phase)

        if trajectory:
            self.walk_trajectory_pub.publish(trajectory)

        # Log walking status periodically
        self.step_count += 1
        if self.step_count % 100 == 0:  # Every 2 seconds at 50Hz
            self.get_logger().info(
                f'Walking - Phase: {self.walk_phase:.2f}, Speed: {speed:.2f}m/s, '
                f'Freq: {adjusted_frequency:.2f}Hz'
            )

    def calculate_walking_trajectory(self, phase):
        """Calculate joint trajectory for walking at given phase."""
        # This implements a simplified walking pattern generator
        # In reality, you'd use more sophisticated gait generation algorithms

        # Calculate left and right leg phases
        left_leg_phase = phase
        right_leg_phase = (phase + self.phase_offset) % (2 * math.pi)

        # Calculate foot positions using simple sinusoidal patterns
        left_foot_x = self.calculate_foot_position(left_leg_phase, 'left')
        right_foot_x = self.calculate_foot_position(right_leg_phase, 'right')

        # Convert foot positions to joint angles using inverse kinematics
        left_hip_angle, left_knee_angle, left_ankle_angle = self.inverse_kinematics_2d(
            left_foot_x, -self.hip_height, 'left'
        )
        right_hip_angle, right_knee_angle, right_ankle_angle = self.inverse_kinematics_2d(
            right_foot_x, -self.hip_height, 'right'
        )

        # Calculate arm swing to counteract leg motion (simplified)
        arm_swing_amplitude = 0.1 * (self.desired_velocity[0] / 0.5)  # Scale with forward speed
        left_arm_angle = arm_swing_amplitude * math.sin(phase)
        right_arm_angle = arm_swing_amplitude * math.sin(phase + math.pi)

        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = [
            'left_hip_pitch', 'left_knee', 'left_ankle_pitch',
            'right_hip_pitch', 'right_knee', 'right_ankle_pitch',
            'left_shoulder_pitch', 'right_shoulder_pitch'
        ]

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 20000000  # Next cycle at 50Hz

        # Set joint positions
        point.positions = [
            left_hip_angle, left_knee_angle, left_ankle_angle,
            right_hip_angle, right_knee_angle, right_ankle_angle,
            left_arm_angle, right_arm_angle
        ]

        # Calculate velocities (approximate derivative)
        if hasattr(self, 'previous_positions'):
            dt = 0.02  # 50Hz
            velocities = []
            for i in range(len(point.positions)):
                vel = (point.positions[i] - self.previous_positions[i]) / dt
                velocities.append(vel)
            point.velocities = velocities
        else:
            point.velocities = [0.0] * len(point.positions)

        self.previous_positions = point.positions.copy()

        trajectory.points = [point]
        return trajectory

    def calculate_foot_position(self, phase, leg_side):
        """Calculate foot position for given leg and phase."""
        # Simple walking pattern: sinusoidal movement
        # This is highly simplified - real walking patterns are more complex

        # Determine support and swing phases based on duty factor
        support_phase_start = 0
        support_phase_end = self.duty_factor * 2 * math.pi
        swing_phase_start = support_phase_end
        swing_phase_end = 2 * math.pi

        if support_phase_start <= phase % (2 * math.pi) <= support_phase_end:
            # Foot is in support phase (on ground)
            # Move foot backward relative to body as CoM moves forward
            foot_progress = (phase % (2 * math.pi)) / support_phase_end
            step_progress = foot_progress * self.step_length

            # Apply desired velocity effect
            velocity_effect = self.desired_velocity[0] * (foot_progress / self.duty_factor)

            if leg_side == 'left':
                foot_x = -step_progress + velocity_effect
            else:  # right
                foot_x = -step_progress + velocity_effect

        else:
            # Foot is in swing phase (lifting and moving forward)
            swing_phase_local = (phase % (2 * math.pi)) - swing_phase_start
            swing_progress = swing_phase_local / (swing_phase_end - swing_phase_start)

            # Swing foot forward
            if leg_side == 'left':
                foot_x = -(1 - self.duty_factor) * self.step_length * (1 - swing_progress)
            else:  # right
                foot_x = -(1 - self.duty_factor) * self.step_length * (1 - swing_progress)

            # Add vertical lifting motion during swing
            if 0.2 < swing_progress < 0.8:  # Lift during middle of swing
                vertical_lift = self.step_height * math.sin(math.pi * (swing_progress - 0.2) / 0.6)
            else:
                vertical_lift = 0.0

        return foot_x

    def inverse_kinematics_2d(self, foot_x, foot_z, leg_side):
        """Simple 2D inverse kinematics for leg."""
        # Simplified 2D inverse kinematics for a 3-DOF leg (hip, knee, ankle pitch)
        # hip -> knee -> ankle -> foot

        # Leg lengths
        thigh_length = self.leg_length * 0.5
        shin_length = self.leg_length * 0.5

        # Calculate distances
        distance_sq = foot_x**2 + foot_z**2
        distance = math.sqrt(distance_sq)

        # Check if position is reachable
        if distance > thigh_length + shin_length:
            # Position is too far, extend leg fully
            angle_to_foot = math.atan2(foot_z, foot_x)
            hip_angle = angle_to_foot
            knee_angle = 0.0
            ankle_angle = -hip_angle
        elif distance < abs(thigh_length - shin_length):
            # Position is too close, this shouldn't happen in normal walking
            hip_angle = math.atan2(foot_z, foot_x)
            knee_angle = math.pi if thigh_length > shin_length else -math.pi
            ankle_angle = -hip_angle - knee_angle
        else:
            # Calculate joint angles using law of cosines
            cos_knee_angle = (thigh_length**2 + shin_length**2 - distance_sq) / (2 * thigh_length * shin_length)
            cos_knee_angle = max(-1, min(1, cos_knee_angle))  # Clamp to valid range
            knee_angle = math.pi - math.acos(cos_knee_angle)

            cos_hip_angle = (thigh_length**2 + distance_sq - shin_length**2) / (2 * thigh_length * distance)
            cos_hip_angle = max(-1, min(1, cos_hip_angle))  # Clamp to valid range
            hip_angle = math.atan2(foot_z, foot_x) - math.acos(cos_hip_angle)

            # Ankle angle to maintain foot orientation
            ankle_angle = -hip_angle - knee_angle

        return hip_angle, knee_angle, ankle_angle

    def set_walking_parameters(self, step_length, step_width, step_height, step_duration):
        """Set walking pattern parameters."""
        self.step_length = max(0.1, min(0.5, step_length))  # Constrain to reasonable range
        self.step_width = max(0.1, min(0.4, step_width))
        self.step_height = max(0.02, min(0.1, step_height))
        self.step_duration = max(0.5, min(2.0, step_duration))

        self.stride_frequency = 1.0 / self.step_duration
        self.get_logger().info(
            f'Walking parameters updated - Step: {self.step_length}m, '
            f'Width: {self.step_width}m, Height: {self.step_height}m, '
            f'Duration: {self.step_duration}s, Frequency: {self.stride_frequency}Hz'
        )

    def stop_walking(self):
        """Stop walking pattern generation."""
        self.is_walking = False
        self.desired_velocity = np.array([0.0, 0.0, 0.0])
        self.get_logger().info('Walking stopped')

def main(args=None):
    rclpy.init(args=args)
    walker = WalkingPatternGenerator()

    try:
        rclpy.spin(walker)
    except KeyboardInterrupt:
        walker.get_logger().info('Shutting down Walking Pattern Generator')
    finally:
        walker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Manipulation Control

### Arm and Hand Control Systems

```python
# manipulation_control.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import time

class ManipulationController(Node):
    def __init__(self):
        super().__init__('manipulation_controller')

        # Publishers
        self.arm_trajectory_pub = self.create_publisher(JointTrajectory, 'arm_controller/command', 10)
        self.hand_trajectory_pub = self.create_publisher(JointTrajectory, 'hand_controller/command', 10)
        self.end_effector_pose_pub = self.create_publisher(Pose, 'end_effector_pose', 10)
        self.manipulation_status_pub = self.create_publisher(String, 'manipulation_status', 10)
        self.grasp_visualization_pub = self.create_publisher(MarkerArray, 'grasp_visualization', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.manipulation_command_sub = self.create_subscription(
            Pose, 'manipulation_command', self.manipulation_command_callback, 10
        )
        self.grasp_command_sub = self.create_subscription(
            String, 'grasp_command', self.grasp_command_callback, 10
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.joint_positions = {}
        self.left_arm_joints = ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
                               'left_elbow_pitch', 'left_forearm_yaw', 'left_wrist_pitch', 'left_wrist_yaw']
        self.right_arm_joints = ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                                'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw']
        self.left_hand_joints = ['left_hand_finger_1', 'left_hand_finger_2', 'left_hand_finger_3']
        self.right_hand_joints = ['right_hand_finger_1', 'right_hand_finger_2', 'right_hand_finger_3']

        # Manipulation state
        self.left_end_effector_pose = Pose()
        self.right_end_effector_pose = Pose()
        self.manipulation_target = None
        self.active_arm = 'right'  # Default to right arm

        # Inverse kinematics parameters
        self.ik_iterations = 100
        self.ik_tolerance = 0.001
        self.jacobian_step_size = 0.01

        # Gripper control parameters
        self.gripper_open_position = 0.05  # Fully open
        self.gripper_closed_position = 0.0  # Fully closed
        self.gripper_approach_position = 0.02  # Just enough to grip

        # Control parameters
        self.max_joint_velocity = 1.0  # rad/s
        self.max_joint_acceleration = 2.0  # rad/s²

        # Kinematic chain information (simplified)
        self.arm_chain_lengths = {
            'upper_arm': 0.3,   # shoulder to elbow
            'forearm': 0.25,    # elbow to wrist
            'hand': 0.15       # wrist to fingertips
        }

        # Timer for end-effector pose calculation
        self.pose_calc_timer = self.create_timer(0.05, self.calculate_end_effector_poses)  # 20 Hz

        self.get_logger().info('Manipulation Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state information."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def manipulation_command_callback(self, msg):
        """Process manipulation command."""
        self.manipulation_target = msg
        self.get_logger().info(f'Received manipulation target: ({msg.position.x}, {msg.position.y}, {msg.position.z})')

        # Determine which arm to use based on target position
        # For now, use the default arm
        trajectory = self.generate_arm_trajectory(msg, self.active_arm)

        if trajectory:
            if self.active_arm == 'left':
                self.arm_trajectory_pub.publish(trajectory)
            else:
                self.arm_trajectory_pub.publish(trajectory)

            status_msg = String()
            status_msg.data = f'Generating trajectory for {self.active_arm} arm to reach target'
            self.manipulation_status_pub.publish(status_msg)

    def grasp_command_callback(self, msg):
        """Process grasp command."""
        command = msg.data.lower()
        self.get_logger().info(f'Received grasp command: {command}')

        if command in ['grasp', 'grab', 'close']:
            trajectory = self.generate_grasp_trajectory('close')
        elif command in ['release', 'open']:
            trajectory = self.generate_grasp_trajectory('open')
        elif command in ['approach', 'prepare']:
            trajectory = self.generate_grasp_trajectory('approach')
        else:
            self.get_logger().warn(f'Unknown grasp command: {command}')
            return

        if trajectory:
            self.hand_trajectory_pub.publish(trajectory)

            status_msg = String()
            status_msg.data = f'Executing grasp command: {command}'
            self.manipulation_status_pub.publish(status_msg)

    def generate_arm_trajectory(self, target_pose, arm_side):
        """Generate trajectory to move arm to target pose."""
        # Get current joint positions for the specified arm
        if arm_side == 'left':
            current_joints = [self.joint_positions.get(name, 0.0) for name in self.left_arm_joints]
            joint_names = self.left_arm_joints
        else:  # right
            current_joints = [self.joint_positions.get(name, 0.0) for name in self.right_arm_joints]
            joint_names = self.right_arm_joints

        # Perform inverse kinematics to find target joint positions
        target_joints = self.inverse_kinematics(target_pose, arm_side, current_joints)

        if target_joints is None:
            self.get_logger().error(f'Could not find IK solution for {arm_side} arm')
            return None

        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start.sec = 2  # 2 seconds to reach target
        point.time_from_start.nanosec = 0

        # Calculate velocities and accelerations using cubic interpolation
        current_positions = np.array(current_joints)
        target_positions = np.array(target_joints)

        # Generate smooth trajectory using cubic polynomial
        duration = 2.0  # seconds
        num_points = 20  # 10 points per second
        dt = duration / num_points

        trajectory.points = []
        for i in range(num_points + 1):
            t = i * dt / duration  # Normalized time (0 to 1)

            # Cubic interpolation: smooth start and end
            # h00 = 2*t³ - 3*t² + 1 (start position coefficient)
            # h10 = t³ - 2*t² + t (start velocity coefficient)
            # h01 = -2*t³ + 3*t² (end position coefficient)
            # h11 = t³ - t² (end velocity coefficient)
            # For smooth motion with zero start/end velocities: p(t) = h00*p0 + h01*p1

            interp_factor = 3*t**2 - 2*t**3  # Smooth interpolation factor
            positions = current_positions + interp_factor * (target_positions - current_positions)

            point = JointTrajectoryPoint()
            point.positions = positions.tolist()
            point.time_from_start.sec = int(i * dt)
            point.time_from_start.nanosec = int((i * dt - int(i * dt)) * 1e9)

            if i > 0:
                # Calculate velocities (finite difference)
                prev_positions = trajectory.points[-1].positions
                velocities = [(curr - prev) / dt for curr, prev in zip(positions, prev_positions)]
                point.velocities = velocities

            trajectory.points.append(point)

        return trajectory

    def inverse_kinematics(self, target_pose, arm_side, current_joints):
        """Perform inverse kinematics to find joint angles for target pose."""
        # This is a simplified analytical IK solver
        # For a 7-DOF arm, numerical methods like Jacobian transpose/pseudo-inverse are typically used

        # Extract target position
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        target_z = target_pose.position.z

        # Get current end-effector position (simplified - in reality use forward kinematics)
        current_ee_pos = self.calculate_end_effector_position(current_joints, arm_side)

        # Initialize joint angles
        joint_angles = np.array(current_joints)

        # Iterative IK using Jacobian transpose method
        for iteration in range(self.ik_iterations):
            # Calculate current end-effector position
            current_pos = self.forward_kinematics(joint_angles, arm_side)

            # Calculate error
            error = np.array([target_x, target_y, target_z]) - current_pos[:3]

            # Check if we're close enough
            if np.linalg.norm(error) < self.ik_tolerance:
                break

            # Calculate Jacobian
            jacobian = self.calculate_jacobian(joint_angles, arm_side)

            # Update joint angles using Jacobian transpose
            delta_thetas = self.jacobian_step_size * jacobian.T @ error
            joint_angles += delta_thetas

        # Check if solution is valid (within joint limits)
        if self.is_joint_limits_valid(joint_angles):
            return joint_angles.tolist()
        else:
            self.get_logger().warn('IK solution violates joint limits')
            return None

    def forward_kinematics(self, joint_angles, arm_side):
        """Calculate end-effector position from joint angles (simplified)."""
        # Simplified forward kinematics for a 7-DOF arm
        # In reality, you'd use the DH parameters or transformation matrices

        # This is a highly simplified version - real FK would use proper transformation matrices
        q = joint_angles  # Joint angles

        # Calculate position based on joint angles (simplified model)
        # This assumes a simple kinematic chain
        upper_arm_len = self.arm_chain_lengths['upper_arm']
        forearm_len = self.arm_chain_lengths['forearm']
        hand_len = self.arm_chain_lengths['hand']

        # Simplified calculation - in reality use proper transformation matrices
        # Shoulder to elbow
        elbow_x = upper_arm_len * math.cos(q[0]) * math.cos(q[1])
        elbow_y = upper_arm_len * math.sin(q[0]) * math.cos(q[1])
        elbow_z = upper_arm_len * math.sin(q[1])

        # Elbow to wrist
        wrist_x = elbow_x + forearm_len * math.cos(q[0] + q[3]) * math.cos(q[1])
        wrist_y = elbow_y + forearm_len * math.sin(q[0] + q[3]) * math.cos(q[1])
        wrist_z = elbow_z + forearm_len * math.sin(q[1])

        # Wrist to end-effector
        ee_x = wrist_x + hand_len * math.cos(q[0] + q[3] + q[5]) * math.cos(q[1])
        ee_y = wrist_y + hand_len * math.sin(q[0] + q[3] + q[5]) * math.cos(q[1])
        ee_z = wrist_z + hand_len * math.sin(q[1])

        return np.array([ee_x, ee_y, ee_z, 0, 0, 0])  # x, y, z, roll, pitch, yaw

    def calculate_jacobian(self, joint_angles, arm_side):
        """Calculate Jacobian matrix for the arm."""
        # Simplified Jacobian calculation
        # In reality, you'd derive this analytically or numerically for your specific robot

        # This is a placeholder - real implementation would be complex
        # and specific to your robot's kinematic structure
        jacobian = np.zeros((6, len(joint_angles)))  # 6 DOF (pos + ori), N joints

        # For a real implementation, you'd calculate partial derivatives
        # of end-effector position/orientation w.r.t. each joint angle
        # This requires knowledge of the robot's kinematic chain

        # Return a dummy Jacobian for this example
        for i in range(min(6, len(joint_angles))):
            jacobian[i, i] = 1.0  # Diagonal elements

        return jacobian

    def is_joint_limits_valid(self, joint_angles):
        """Check if joint angles are within limits."""
        # Define joint limits (simplified - would be robot-specific)
        min_limits = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
        max_limits = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        for angle, min_lim, max_lim in zip(joint_angles, min_limits, max_limits):
            if angle < min_lim or angle > max_lim:
                return False
        return True

    def calculate_end_effector_position(self, joint_angles, arm_side):
        """Calculate end-effector position from joint angles."""
        # This would use forward kinematics in a real implementation
        # For this example, return a placeholder
        return np.array([0.0, 0.0, 0.0])

    def calculate_end_effector_poses(self):
        """Calculate and publish end-effector poses."""
        # Calculate end-effector poses based on current joint positions
        # This would use forward kinematics in a real implementation

        # For this example, we'll create mock poses
        current_time = self.get_clock().now()

        # Left arm end-effector (mock)
        left_ee = Pose()
        left_ee.position.x = 0.3  # Mock position
        left_ee.position.y = 0.2
        left_ee.position.z = 0.8
        left_ee.orientation.w = 1.0  # No rotation

        # Right arm end-effector (mock)
        right_ee = Pose()
        right_ee.position.x = 0.3  # Mock position
        right_ee.position.y = -0.2
        right_ee.position.z = 0.8
        right_ee.orientation.w = 1.0  # No rotation

        # Publish poses
        left_ee.header.stamp = current_time.to_msg()
        left_ee.header.frame_id = 'left_end_effector'
        # self.left_end_effector_pose_pub.publish(left_ee)  # Uncomment when publisher is added

        right_ee.header.stamp = current_time.to_msg()
        right_ee.header.frame_id = 'right_end_effector'
        self.right_end_effector_pose_pub.publish(right_ee)

    def generate_grasp_trajectory(self, grasp_type):
        """Generate trajectory for gripper control."""
        if self.active_arm == 'left':
            joint_names = self.left_hand_joints
            if grasp_type == 'close':
                positions = [self.gripper_closed_position] * len(self.left_hand_joints)
            elif grasp_type == 'open':
                positions = [self.gripper_open_position] * len(self.left_hand_joints)
            else:  # approach
                positions = [self.gripper_approach_position] * len(self.left_hand_joints)
        else:  # right
            joint_names = self.right_hand_joints
            if grasp_type == 'close':
                positions = [self.gripper_closed_position] * len(self.right_hand_joints)
            elif grasp_type == 'open':
                positions = [self.gripper_open_position] * len(self.right_hand_joints)
            else:  # approach
                positions = [self.gripper_approach_position] * len(self.right_hand_joints)

        # Create trajectory
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 1  # 1 second to complete grasp
        point.time_from_start.nanosec = 0

        trajectory.points = [point]
        return trajectory

    def set_active_arm(self, arm_side):
        """Set which arm to use for manipulation."""
        if arm_side in ['left', 'right']:
            self.active_arm = arm_side
            self.get_logger().info(f'Active arm set to: {arm_side}')
        else:
            self.get_logger().warn(f'Invalid arm side: {arm_side}. Use "left" or "right"')

    def move_to_home_position(self):
        """Move arm to home position."""
        home_positions = {
            'left': [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0],  # Left arm home
            'right': [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]  # Right arm home
        }

        if self.active_arm in home_positions:
            trajectory = JointTrajectory()
            trajectory.header.stamp = self.get_clock().now().to_msg()
            trajectory.joint_names = (self.left_arm_joints if self.active_arm == 'left'
                                    else self.right_arm_joints)

            point = JointTrajectoryPoint()
            point.positions = home_positions[self.active_arm]
            point.time_from_start.sec = 3  # 3 seconds to go home
            point.time_from_start.nanosec = 0

            trajectory.points = [point]

            self.arm_trajectory_pub.publish(trajectory)
            self.get_logger().info(f'Moving {self.active_arm} arm to home position')

def main(args=None):
    rclpy.init(args=args)
    manipulator = ManipulationController()

    try:
        rclpy.spin(manipulator)
    except KeyboardInterrupt:
        manipulator.get_logger().info('Shutting down Manipulation Controller')
    finally:
        manipulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Whole-Body Control Framework

### Coordinated Motion Control

```python
# whole_body_control.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import time

class WholeBodyController(Node):
    def __init__(self):
        super().__init__('whole_body_controller')

        # Publishers
        self.body_trajectory_pub = self.create_publisher(JointTrajectory, 'whole_body_command', 10)
        self.upper_body_pub = self.create_publisher(JointTrajectory, 'upper_body_command', 10)
        self.lower_body_pub = self.create_publisher(JointTrajectory, 'lower_body_command', 10)
        self.whole_body_status_pub = self.create_publisher(String, 'whole_body_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.body_command_sub = self.create_subscription(
            String, 'body_command', self.body_command_callback, 10
        )
        self.pose_command_sub = self.create_subscription(
            Pose, 'body_pose_command', self.pose_command_callback, 10
        )

        # Robot state
        self.joint_positions = OrderedDict()
        self.joint_velocities = OrderedDict()
        self.joint_efforts = OrderedDict()

        # Robot configuration
        self.torso_joints = ['torso_yaw', 'torso_pitch', 'torso_roll']
        self.head_joints = ['neck_yaw', 'neck_pitch', 'neck_roll']
        self.left_arm_joints = ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
                               'left_elbow_pitch', 'left_forearm_yaw', 'left_wrist_pitch', 'left_wrist_yaw']
        self.right_arm_joints = ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                                'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw']
        self.left_leg_joints = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
                               'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll']
        self.right_leg_joints = ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
                                'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll']

        # All joints in order
        self.all_joints = (self.torso_joints + self.head_joints +
                          self.left_arm_joints + self.right_arm_joints +
                          self.left_leg_joints + self.right_leg_joints)

        # Task priorities and weights
        self.task_weights = {
            'balance': 1.0,
            'manipulation': 0.8,
            'locomotion': 0.7,
            'posture': 0.5
        }

        # Control parameters
        self.control_frequency = 100  # Hz
        self.max_joint_velocity = 2.0
        self.max_joint_acceleration = 5.0

        # Task-specific targets
        self.balance_target = np.array([0.0, 0.0, 0.85])  # Desired CoM position
        self.manipulation_targets = {}  # {arm: target_pose}
        self.locomotion_target = np.array([0.0, 0.0, 0.0])  # Desired movement

        # Whole-body control parameters
        self.enable_balance_constraint = True
        self.enable_manipulation_constraint = True
        self.enable_posture_constraint = True

        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.whole_body_control_callback)

        self.get_logger().info('Whole Body Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state information."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def body_command_callback(self, msg):
        """Process whole body commands."""
        command = msg.data.lower()
        self.get_logger().info(f'Received whole body command: {command}')

        if command.startswith('move_to:'):
            # Parse target: move_to:x,y,z
            try:
                coords = command.split(':')[1].split(',')
                target = [float(c.strip()) for c in coords]
                if len(target) == 3:
                    trajectory = self.generate_whole_body_trajectory_to_point(target)
                    if trajectory:
                        self.body_trajectory_pub.publish(trajectory)
            except Exception as e:
                self.get_logger().error(f'Could not parse move_to command: {e}')

        elif command.startswith('posture:'):
            # Move to predefined posture: posture:home, posture:sit, etc.
            posture = command.split(':')[1].strip()
            trajectory = self.generate_posture_trajectory(posture)
            if trajectory:
                self.body_trajectory_pub.publish(trajectory)

        elif command == 'dance':
            trajectory = self.generate_dance_trajectory()
            if trajectory:
                self.body_trajectory_pub.publish(trajectory)

        elif command == 'wave':
            trajectory = self.generate_wave_trajectory()
            if trajectory:
                self.body_trajectory_pub.publish(trajectory)

        # Update status
        status_msg = String()
        status_msg.data = f'Executing command: {command}'
        self.whole_body_status_pub.publish(status_msg)

    def pose_command_callback(self, msg):
        """Process pose commands for specific body parts."""
        # This would handle more complex pose-based commands
        # For now, we'll use it to set manipulation targets
        self.manipulation_targets['right'] = msg
        self.get_logger().info('Received pose command for manipulation')

    def whole_body_control_callback(self):
        """Main whole-body control loop."""
        if not self.joint_positions:
            return

        # Get current joint configuration
        current_configuration = np.array([
            self.joint_positions.get(joint, 0.0) for joint in self.all_joints
        ])

        # Calculate desired configuration based on all active tasks
        desired_configuration = self.calculate_whole_body_solution(current_configuration)

        if desired_configuration is not None:
            # Create trajectory message
            trajectory = JointTrajectory()
            trajectory.header.stamp = self.get_clock().now().to_msg()
            trajectory.joint_names = self.all_joints

            point = JointTrajectoryPoint()
            point.positions = desired_configuration.tolist()
            point.time_from_start.nanosec = int(1e9 / self.control_frequency)  # Next control cycle

            # Calculate velocities using difference from current
            current_pos_array = np.array(current_configuration)
            velocities = (desired_configuration - current_pos_array) * self.control_frequency
            point.velocities = velocities.tolist()

            trajectory.points = [point]
            self.body_trajectory_pub.publish(trajectory)

    def calculate_whole_body_solution(self, current_configuration):
        """Calculate whole-body joint configuration using task-priority framework."""
        # This implements a simplified whole-body controller
        # In reality, you'd use more sophisticated methods like:
        # - Task-priority inverse kinematics
        # - Hierarchical quadratic programming
        # - Operational space control

        # Initialize solution as current configuration
        solution = current_configuration.copy()

        # Apply balance constraint (highest priority)
        if self.enable_balance_constraint:
            balance_correction = self.calculate_balance_correction(solution)
            solution += balance_correction * self.task_weights['balance']

        # Apply manipulation constraint (high priority)
        if self.enable_manipulation_constraint and self.manipulation_targets:
            for arm, target in self.manipulation_targets.items():
                manipulation_correction = self.calculate_manipulation_correction(solution, arm, target)
                if manipulation_correction is not None:
                    solution += manipulation_correction * self.task_weights['manipulation']

        # Apply locomotion constraint (medium priority)
        if np.any(self.locomotion_target != 0):
            locomotion_correction = self.calculate_locomotion_correction(solution)
            solution += locomotion_correction * self.task_weights['locomotion']

        # Apply posture constraint (lowest priority)
        if self.enable_posture_constraint:
            posture_correction = self.calculate_posture_correction(solution)
            solution += posture_correction * self.task_weights['posture']

        # Apply joint limits
        solution = self.apply_joint_limits(solution)

        return solution

    def calculate_balance_correction(self, current_configuration):
        """Calculate joint adjustments for balance maintenance."""
        # This would use more sophisticated balance control
        # For this example, return a simple correction based on CoM error

        # Calculate current CoM (simplified)
        current_com = self.estimate_com_position(current_configuration)

        # Calculate error from desired CoM
        com_error = self.balance_target - current_com

        # Map CoM error to joint corrections (simplified mapping)
        correction = np.zeros(len(self.all_joints))

        # Legs: adjust for CoM position
        leg_start_idx = len(self.torso_joints + self.head_joints + self.left_arm_joints + self.right_arm_joints)
        correction[leg_start_idx:leg_start_idx+len(self.left_leg_joints)] = com_error[0] * 0.1  # Hip adjustment
        correction[leg_start_idx+len(self.left_leg_joints):leg_start_idx+len(self.left_leg_joints)+len(self.right_leg_joints)] = com_error[0] * 0.1  # Hip adjustment

        # Torso: adjust for CoM height
        torso_start_idx = 0
        correction[torso_start_idx:torso_start_idx+len(self.torso_joints)] = com_error[2] * 0.05  # Torso adjustment

        return correction

    def estimate_com_position(self, configuration):
        """Estimate center of mass position from joint configuration (simplified)."""
        # This is a very simplified CoM estimation
        # In reality, you'd use the full kinematic chain and mass distribution

        # For this example, return a mock CoM based on leg positions
        left_ankle_idx = self.all_joints.index('left_ankle_pitch') if 'left_ankle_pitch' in self.all_joints else 0
        right_ankle_idx = self.all_joints.index('right_ankle_pitch') if 'right_ankle_pitch' in self.all_joints else 0

        # Estimate CoM position based on ankle positions and torso
        torso_idx = self.all_joints.index('torso_pitch') if 'torso_pitch' in self.all_joints else 0

        # Simplified calculation
        com_x = configuration[torso_idx] * 0.1  # Torso pitch affects CoM x
        com_y = (configuration[left_ankle_idx] + configuration[right_ankle_idx]) * 0.05  # Average ankle position
        com_z = 0.85 + configuration[torso_idx] * 0.02  # Base height + torso effect

        return np.array([com_x, com_y, com_z])

    def calculate_manipulation_correction(self, current_configuration, arm, target):
        """Calculate joint adjustments for manipulation task."""
        if arm not in ['left', 'right'] or target is None:
            return None

        # Get current end-effector position for the specified arm
        current_ee_pos = self.calculate_end_effector_position(current_configuration, arm)

        # Calculate desired position difference
        target_pos = np.array([target.position.x, target.position.y, target.position.z])
        pos_error = target_pos - current_ee_pos

        # Map position error to joint corrections (simplified)
        correction = np.zeros(len(self.all_joints))

        # Determine which joints to adjust based on arm
        if arm == 'left':
            arm_start_idx = len(self.torso_joints + self.head_joints)
            arm_end_idx = arm_start_idx + len(self.left_arm_joints)
        else:  # right
            arm_start_idx = len(self.torso_joints + self.head_joints + self.left_arm_joints)
            arm_end_idx = arm_start_idx + len(self.right_arm_joints)

        # Apply proportional correction to arm joints
        for i in range(arm_start_idx, arm_end_idx):
            # Map position error to joint space (simplified)
            joint_correction = pos_error[0] * 0.01 + pos_error[1] * 0.01 + pos_error[2] * 0.01
            correction[i] = joint_correction

        return correction

    def calculate_locomotion_correction(self, current_configuration):
        """Calculate joint adjustments for locomotion."""
        # This would adjust joints for walking pattern
        # For this example, return corrections that would support forward motion

        correction = np.zeros(len(self.all_joints))

        # Adjust legs for walking
        left_leg_start = len(self.torso_joints + self.head_joints + self.left_arm_joints + self.right_arm_joints)
        right_leg_start = left_leg_start + len(self.left_leg_joints)

        # Simple walking pattern adjustment
        current_time = time.time()
        left_phase = math.sin(current_time * 2)  # 2 Hz walking
        right_phase = math.sin(current_time * 2 + math.pi)  # Opposite phase

        # Apply walking pattern to legs
        correction[left_leg_start:left_leg_start+len(self.left_leg_joints)] = [
            0, 0, left_phase * 0.1,  # Hip adjustments
            left_phase * 0.2,         # Knee adjustment
            0, 0                      # Ankle adjustments
        ]

        correction[right_leg_start:right_leg_start+len(self.right_leg_joints)] = [
            0, 0, right_phase * 0.1,  # Hip adjustments
            right_phase * 0.2,         # Knee adjustment
            0, 0                      # Ankle adjustments
        ]

        # Adjust arms to counteract leg motion
        correction[len(self.torso_joints):len(self.torso_joints)+len(self.left_arm_joints)] = [
            0, 0, 0, -right_phase * 0.1, 0, 0, 0  # Left arm
        ]
        correction[len(self.torso_joints)+len(self.left_arm_joints):len(self.torso_joints)+len(self.left_arm_joints)+len(self.right_arm_joints)] = [
            0, 0, 0, -left_phase * 0.1, 0, 0, 0  # Right arm
        ]

        return correction

    def calculate_posture_correction(self, current_configuration):
        """Calculate joint adjustments for desired posture."""
        # Return correction toward neutral posture
        correction = np.zeros(len(self.all_joints))

        for i, joint in enumerate(self.all_joints):
            # Move toward neutral position (0 for most joints)
            current_pos = current_configuration[i]
            neutral_pos = 0.0  # Simplified neutral position

            # Calculate correction to move toward neutral
            correction[i] = (neutral_pos - current_pos) * 0.01  # Small proportional correction

        return correction

    def apply_joint_limits(self, configuration):
        """Apply joint limits to configuration."""
        # Define joint limits (simplified - would be robot-specific)
        min_limits = [-3.14] * len(self.all_joints)  # -π for all joints
        max_limits = [3.14] * len(self.all_joints)   # π for all joints

        # Apply limits
        limited_configuration = np.clip(configuration, min_limits, max_limits)

        return limited_configuration

    def generate_whole_body_trajectory_to_point(self, target_point):
        """Generate trajectory to move to a specific 3D point."""
        # This would use whole-body IK to find joint configuration
        # For this example, return a simple trajectory

        current_pos = self.estimate_com_position(list(self.joint_positions.values()))
        target = np.array(target_point)

        # Calculate difference
        diff = target - current_pos

        # Create trajectory to move CoM toward target
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.all_joints

        # Generate smooth trajectory over 3 seconds
        duration = 3.0
        num_points = int(duration * self.control_frequency)

        for i in range(num_points + 1):
            t = i / num_points  # Normalized time (0 to 1)
            interp_factor = 3*t**2 - 2*t**3  # Smooth interpolation

            # Calculate intermediate target
            intermediate_target = current_pos + interp_factor * diff

            # Find joint configuration for intermediate target (simplified)
            joint_positions = self.find_configuration_for_target(intermediate_target)

            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start.sec = int(i * duration / num_points)
            point.time_from_start.nanosec = int((i * duration / num_points - int(i * duration / num_points)) * 1e9)

            trajectory.points.append(point)

        return trajectory

    def find_configuration_for_target(self, target_com):
        """Find joint configuration for desired CoM position (simplified)."""
        # This is a simplified approach - real implementation would use whole-body IK
        current_positions = [self.joint_positions.get(joint, 0.0) for joint in self.all_joints]

        # Adjust leg joints to move CoM
        leg_joints_start = len(self.torso_joints + self.head_joints + self.left_arm_joints + self.right_arm_joints)

        # Simple adjustment based on CoM error
        com_error = target_com - self.estimate_com_position(current_positions)

        new_positions = current_positions.copy()
        for i in range(leg_joints_start, len(self.all_joints)):
            # Adjust leg joints proportionally to CoM error
            new_positions[i] += com_error[0] * 0.1 + com_error[1] * 0.1

        return new_positions

    def generate_posture_trajectory(self, posture_name):
        """Generate trajectory for predefined postures."""
        if posture_name == 'home':
            target_positions = self.get_home_posture()
        elif posture_name == 'sit':
            target_positions = self.get_sitting_posture()
        elif posture_name == 'stand':
            target_positions = self.get_standing_posture()
        else:
            self.get_logger().warn(f'Unknown posture: {posture_name}')
            return None

        # Create trajectory to move to posture
        current_positions = [self.joint_positions.get(joint, 0.0) for joint in self.all_joints]

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.all_joints

        # Smooth interpolation over 2 seconds
        duration = 2.0
        num_points = int(duration * self.control_frequency)

        for i in range(num_points + 1):
            t = i / num_points
            interp_factor = 3*t**2 - 2*t**3  # Smooth interpolation

            intermediate_positions = []
            for curr, target in zip(current_positions, target_positions):
                pos = curr + interp_factor * (target - curr)
                intermediate_positions.append(pos)

            point = JointTrajectoryPoint()
            point.positions = intermediate_positions
            point.time_from_start.sec = int(i * duration / num_points)
            point.time_from_start.nanosec = int((i * duration / num_points - int(i * duration / num_points)) * 1e9)

            trajectory.points.append(point)

        return trajectory

    def get_home_posture(self):
        """Get home posture joint positions."""
        positions = []
        for joint in self.all_joints:
            if 'hip' in joint or 'knee' in joint or 'ankle' in joint:
                positions.append(0.0)  # Straight legs in home position
            elif 'shoulder' in joint or 'elbow' in joint or 'wrist' in joint:
                positions.append(0.0)  # Arms at sides in home position
            elif 'neck' in joint:
                positions.append(0.0)  # Looking straight ahead
            elif 'torso' in joint:
                positions.append(0.0)  # Upright torso
            else:
                positions.append(0.0)  # Default neutral position
        return positions

    def get_standing_posture(self):
        """Get standing posture joint positions."""
        positions = []
        for joint in self.all_joints:
            if 'hip' in joint:
                positions.append(0.0)  # Neutral hip
            elif 'knee' in joint:
                positions.append(0.0)  # Straight knees
            elif 'ankle' in joint:
                positions.append(0.0)  # Neutral ankles
            elif 'shoulder' in joint:
                positions.append(0.0)  # Arms at sides
            elif 'elbow' in joint:
                positions.append(-1.57)  # Slightly bent elbows
            elif 'neck' in joint:
                positions.append(0.0)  # Looking forward
            else:
                positions.append(0.0)  # Default
        return positions

    def get_sitting_posture(self):
        """Get sitting posture joint positions."""
        positions = []
        for joint in self.all_joints:
            if 'hip' in joint:
                positions.append(1.57)  # Bent hips (~90 degrees)
            elif 'knee' in joint:
                positions.append(1.57)  # Bent knees (~90 degrees)
            elif 'ankle' in joint:
                positions.append(0.0)  # Neutral ankles
            elif 'shoulder' in joint:
                positions.append(0.2)  # Slightly raised arms for balance
            elif 'elbow' in joint:
                positions.append(-0.785)  # Bent elbows (~45 degrees)
            elif 'neck' in joint:
                positions.append(0.1)  # Slightly tilted forward
            else:
                positions.append(0.0)  # Default
        return positions

    def generate_wave_trajectory(self):
        """Generate waving motion trajectory."""
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.all_joints

        # Define wave motion over 5 seconds
        duration = 5.0
        num_points = int(duration * self.control_frequency)

        current_positions = [self.joint_positions.get(joint, 0.0) for joint in self.all_joints]

        for i in range(num_points + 1):
            t = i / num_points * 2 * math.pi  # Full wave cycle

            new_positions = current_positions.copy()

            # Wave motion for right arm
            right_arm_start = len(self.torso_joints + self.head_joints + self.left_arm_joints)
            right_arm_end = right_arm_start + len(self.right_arm_joints)

            # Shoulder movement
            new_positions[right_arm_start] = math.sin(t) * 0.5      # Pitch
            new_positions[right_arm_start + 1] = math.cos(t) * 0.3  # Roll
            new_positions[right_arm_start + 3] = math.sin(t * 2) * 0.8  # Elbow

            # Add some torso movement for natural look
            torso_start = 0
            new_positions[torso_start] = math.sin(t * 0.5) * 0.1  # Subtle torso sway

            point = JointTrajectoryPoint()
            point.positions = new_positions
            point.time_from_start.sec = int(i * duration / num_points)
            point.time_from_start.nanosec = int((i * duration / num_points - int(i * duration / num_points)) * 1e9)

            trajectory.points.append(point)

        return trajectory

def main(args=None):
    rclpy.init(args=args)
    wb_controller = WholeBodyController()

    try:
        rclpy.spin(wb_controller)
    except KeyboardInterrupt:
        wb_controller.get_logger().info('Shutting down Whole Body Controller')
    finally:
        wb_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Perception Integration for Humanoid Control

### Sensor Fusion for Humanoid Robots

```python
# perception_integration_humanoid.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import Pose, Twist, Vector3
from std_msgs.msg import Float32, Bool, String
from vision_msgs.msg import Detection2DArray
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
from collections import deque
import threading

class HumanoidPerceptionIntegration(Node):
    def __init__(self):
        super().__init__('humanoid_perception_integration')

        # Publishers
        self.perception_status_pub = self.create_publisher(String, 'perception_status', 10)
        self.fused_pose_pub = self.create_publisher(Pose, 'fused_pose', 10)
        self.safety_alert_pub = self.create_publisher(Bool, 'safety_alert', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.camera_detections_sub = self.create_subscription(Detection2DArray, 'camera/detections', self.detections_callback, 10)
        self.camera_image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # TF listener for sensor fusion
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Sensor data storage
        self.imu_data = None
        self.joint_states = None
        self.scan_data = None
        self.detection_data = None
        self.image_data = None

        # State estimation
        self.robot_pose = Pose()
        self.robot_velocity = Twist()
        self.balance_state = np.array([0.0, 0.0, 0.0])  # Roll, Pitch, Yaw
        self.com_position = np.array([0.0, 0.0, 0.0])

        # Sensor fusion parameters
        self.imu_weight = 0.8
        self.vision_weight = 0.1
        self.odometry_weight = 0.1

        # Safety parameters
        self.balance_threshold = 0.3  # radians
        self.obstacle_distance_threshold = 0.5  # meters
        self.enable_safety_monitoring = True

        # Data buffering for fusion
        self.imu_buffer = deque(maxlen=10)
        self.pose_buffer = deque(maxlen=20)

        # Threading for sensor fusion
        self.fusion_thread = threading.Thread(target=self.sensor_fusion_loop, daemon=True)
        self.fusion_lock = threading.Lock()
        self.new_data_available = threading.Event()
        self.fusion_thread.start()

        # Performance tracking
        self.fusion_times = []
        self.frame_count = 0

        self.get_logger().info('Humanoid Perception Integration initialized')

    def imu_callback(self, msg):
        """Process IMU data."""
        self.imu_data = msg

        # Extract orientation
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Convert to Euler angles (roll, pitch, yaw)
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion(orientation)
        self.balance_state = np.array(euler)

        # Add to buffer for fusion
        with self.fusion_lock:
            self.imu_buffer.append({
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'orientation': orientation,
                'angular_velocity': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
                'linear_acceleration': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            })

        self.new_data_available.set()

    def joint_state_callback(self, msg):
        """Process joint state data."""
        self.joint_states = msg

        # Update CoM based on joint configuration (simplified)
        self.update_com_from_joints()

    def scan_callback(self, msg):
        """Process laser scan data."""
        self.scan_data = msg

        # Check for obstacles
        if self.enable_safety_monitoring:
            self.check_obstacle_safety(msg)

    def detections_callback(self, msg):
        """Process camera detections."""
        self.detection_data = msg

        # Process detections for humanoid-specific tasks
        self.process_humanoid_detections(msg)

    def image_callback(self, msg):
        """Process camera image."""
        self.image_data = msg
        # Image processing would happen here

    def sensor_fusion_loop(self):
        """Main sensor fusion loop running in background thread."""
        while rclpy.ok():
            try:
                self.new_data_available.wait(timeout=0.1)
                self.new_data_available.clear()

                start_time = time.time()

                with self.fusion_lock:
                    # Perform sensor fusion
                    fused_pose = self.perform_sensor_fusion()

                    if fused_pose:
                        # Publish fused pose
                        self.fused_pose_pub.publish(fused_pose)

                        # Check balance
                        balance_ok = self.check_balance_safety()

                        # Publish safety status
                        safety_msg = Bool()
                        safety_msg.data = balance_ok
                        self.safety_alert_pub.publish(safety_msg)

                # Track performance
                fusion_time = time.time() - start_time
                self.fusion_times.append(fusion_time)

                if len(self.fusion_times) > 100:
                    self.fusion_times.pop(0)

                # Log performance periodically
                self.frame_count += 1
                if self.frame_count % 50 == 0:  # Every 50 fusion cycles
                    avg_time = sum(self.fusion_times) / len(self.fusion_times)
                    self.get_logger().info(
                        f'Sensor fusion - Avg: {avg_time*1000:.1f}ms, '
                        f'Balance: ({self.balance_state[0]:.3f}, {self.balance_state[1]:.3f})'
                    )

            except Exception as e:
                self.get_logger().error(f'Sensor fusion error: {e}')
                time.sleep(0.01)

    def perform_sensor_fusion(self) -> Pose:
        """Perform sensor fusion to estimate robot pose."""
        if not self.imu_data:
            return None

        # For this example, we'll create a fused pose based on IMU and other sensors
        # In a real system, you'd use a Kalman filter or similar

        fused_pose = Pose()

        # Use IMU for orientation (most reliable for balance)
        fused_pose.orientation = self.imu_data.orientation

        # Use joint states and kinematics for position (simplified)
        if self.joint_states:
            # Estimate position based on odometry from joint movements
            # This is a simplified approach - real implementation would use forward kinematics
            pass

        # Add timestamp
        fused_pose.header.stamp = self.get_clock().now().to_msg()
        fused_pose.header.frame_id = 'odom'

        return fused_pose

    def update_com_from_joints(self):
        """Update CoM estimate based on joint configuration."""
        if not self.joint_states:
            return

        # This would use forward kinematics and mass distribution
        # For this example, return a simplified estimate
        self.com_position = np.array([0.0, 0.0, 0.85])  # Fixed height for demonstration

        # In a real system, you'd:
        # 1. Calculate each link's contribution to CoM based on joint angles
        # 2. Weight by link masses
        # 3. Sum to get total CoM position

    def check_balance_safety(self) -> bool:
        """Check if robot is within balance safety limits."""
        # Check roll and pitch angles
        roll, pitch, yaw = self.balance_state

        if abs(roll) > self.balance_threshold or abs(pitch) > self.balance_threshold:
            self.get_logger().warn(f'Balance limit exceeded: Roll={roll:.3f}, Pitch={pitch:.3f}')
            return False

        return True

    def check_obstacle_safety(self, scan_msg):
        """Check for safety using laser scan data."""
        if len(scan_msg.ranges) == 0:
            return

        # Check for obstacles in front of robot
        front_start = len(scan_msg.ranges) // 2 - len(scan_msg.ranges) // 12  # -15 degrees
        front_end = len(scan_msg.ranges) // 2 + len(scan_msg.ranges) // 12    # +15 degrees

        front_ranges = scan_msg.ranges[front_start:front_end]
        valid_ranges = [r for r in front_ranges if scan_msg.range_min < r < scan_msg.range_max]

        if valid_ranges:
            min_distance = min(valid_ranges)
            if min_distance < self.obstacle_distance_threshold:
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, safety alert!')

                # Publish safety alert
                safety_msg = Bool()
                safety_msg.data = False  # Unsafe
                self.safety_alert_pub.publish(safety_msg)

    def process_humanoid_detections(self, detections_msg):
        """Process detections for humanoid-specific tasks."""
        if not detections_msg.detections:
            return

        # Look for humans, objects of interest, etc.
        humans_detected = 0
        objects_of_interest = []

        for detection in detections_msg.detections:
            if detection.results:
                class_name = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score

                if class_name == 'person' and confidence > 0.7:
                    humans_detected += 1
                elif confidence > 0.8:  # High confidence object detection
                    objects_of_interest.append({
                        'class': class_name,
                        'confidence': confidence,
                        'position': {
                            'x': detection.bbox.center.x,
                            'y': detection.bbox.center.y
                        }
                    })

        # Update perception status
        status_msg = String()
        status_msg.data = f'Humans: {humans_detected}, Objects: {len(objects_of_interest)}'
        self.perception_status_pub.publish(status_msg)

        # In a real system, you'd use this information for:
        # - Social navigation around humans
        # - Object manipulation planning
        # - Attention and gaze control
        # - Safety considerations

    def get_robot_state(self) -> Dict:
        """Get current robot state including pose, balance, and safety."""
        return {
            'pose': self.robot_pose,
            'balance_state': self.balance_state.tolist(),
            'com_position': self.com_position.tolist(),
            'balance_safe': self.check_balance_safety(),
            'timestamp': time.time()
        }

def main(args=None):
    rclpy.init(args=args)
    perception_node = HumanoidPerceptionIntegration()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down Humanoid Perception Integration')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive humanoid control systems:

- **Balance Control**: Center of mass control and stability maintenance
- **Locomotion**: Walking pattern generation and gait control
- **Manipulation**: Arm and hand control for object interaction
- **Whole-Body Control**: Coordinated motion of all robot parts
- **Perception Integration**: Sensor fusion for humanoid-specific tasks

Humanoid robots require sophisticated control systems that coordinate multiple subsystems while maintaining balance and achieving complex manipulation tasks.

## Exercises

1. Implement balance control for a simulated humanoid robot
2. Create walking pattern generation for bipedal locomotion
3. Develop manipulation control for humanoid arms
4. Integrate whole-body control with perception systems
5. Test your humanoid control system with various tasks

## Quiz

1. What is the main challenge in humanoid robot control compared to wheeled robots?
   a) More expensive
   b) Need to maintain balance while moving
   c) Harder to program
   d) Slower movement

2. What does CoM stand for in robotics?
   a) Center of Mass
   b) Center of Motion
   c) Control of Movement
   d) Center of Mechanics

3. Which component is essential for whole-body control?
   a) Single joint controller
   b) Coordination of multiple subsystems
   c) Simple sensor
   d) Basic actuator

## Mini-Project: Complete Humanoid Controller

Create a complete humanoid robot controller with:
1. Balance maintenance system with CoM control
2. Walking pattern generation for bipedal locomotion
3. Arm manipulation control for object interaction
4. Whole-body coordination system
5. Perception integration for environment awareness
6. Safety monitoring and disturbance rejection
7. Testing with various tasks and scenarios