---
sidebar_position: 5
---

# Control Pipelines

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement robust control systems for robotic applications
- Implement various control strategies including PID, model predictive control, and learning-based control
- Integrate perception and control systems for closed-loop robotic behavior
- Develop control pipelines that handle uncertainty and disturbances
- Validate and optimize control system performance in simulation and real-world scenarios

## Introduction to Robotic Control Systems

Robotic control systems are responsible for translating high-level goals into low-level actuator commands that drive the robot's behavior. In the Isaac ecosystem, control pipelines are designed to work seamlessly with perception systems to create intelligent, adaptive robots.

### Control System Architecture

```
Robotic Control Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   High-Level    │───→│   Mid-Level     │───→│   Low-Level     │
│   Planning      │    │   Control       │    │   Actuation     │
│   (Navigation,  │    │   (Trajectory    │    │   (Motor        │
│   Manipulation) │    │   Following,    │    │   Commands)     │
│                 │    │   Servoing)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Perception    │
                         │   Feedback      │
                         │   (State       │
                         │   Estimation)   │
                         └─────────────────┘
```

### Key Control Concepts

1. **Feedforward Control**: Proactive control based on desired trajectories
2. **Feedback Control**: Corrective control based on state estimation
3. **Adaptive Control**: Control that adjusts parameters based on system changes
4. **Robust Control**: Control that handles uncertainties and disturbances

## PID Control Systems

Proportional-Integral-Derivative (PID) control is fundamental to many robotic systems due to its simplicity and effectiveness.

### PID Controller Implementation

```python
# pid_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
import math

class PIDController(Node):
    def __init__(self):
        super().__init__('pid_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.target_sub = self.create_subscription(
            Vector3, 'target_velocity', self.target_callback, 10
        )

        # PID parameters for linear velocity
        self.kp_linear = 1.0
        self.ki_linear = 0.1
        self.kd_linear = 0.05

        # PID parameters for angular velocity
        self.kp_angular = 2.0
        self.ki_angular = 0.2
        self.kd_angular = 0.05

        # PID state variables
        self.linear_error_sum = 0.0
        self.linear_error_prev = 0.0
        self.angular_error_sum = 0.0
        self.angular_error_prev = 0.0

        # Current and target states
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0

        # Control timer
        self.control_timer = self.create_timer(0.02, self.control_callback)  # 50 Hz

        # Time tracking
        self.last_time = self.get_clock().now()

        self.get_logger().info('PID Controller initialized')

    def odom_callback(self, msg):
        """Update current velocity from odometry."""
        self.current_linear_vel = math.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2 +
            msg.twist.twist.linear.z**2
        )
        self.current_angular_vel = msg.twist.twist.angular.z

    def target_callback(self, msg):
        """Update target velocity."""
        self.target_linear_vel = msg.x
        self.target_angular_vel = msg.z

    def control_callback(self):
        """Main PID control loop."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt <= 0:
            return

        # Calculate errors
        linear_error = self.target_linear_vel - self.current_linear_vel
        angular_error = self.target_angular_vel - self.current_angular_vel

        # Update error integrals
        self.linear_error_sum += linear_error * dt
        self.angular_error_sum += angular_error * dt

        # Calculate derivatives
        linear_error_deriv = (linear_error - self.linear_error_prev) / dt
        angular_error_deriv = (angular_error - self.angular_error_prev) / dt

        # Store current errors for next iteration
        self.linear_error_prev = linear_error
        self.angular_error_prev = angular_error

        # Calculate PID outputs
        linear_output = (
            self.kp_linear * linear_error +
            self.ki_linear * self.linear_error_sum +
            self.kd_linear * linear_error_deriv
        )

        angular_output = (
            self.kp_angular * angular_error +
            self.ki_angular * self.angular_error_sum +
            self.kd_angular * angular_error_deriv
        )

        # Apply output limits
        linear_output = max(min(linear_output, 1.0), -1.0)
        angular_output = max(min(angular_output, 1.0), -1.0)

        # Create and publish velocity command
        cmd = Twist()
        cmd.linear.x = float(linear_output)
        cmd.angular.z = float(angular_output)

        self.cmd_vel_pub.publish(cmd)

        # Log control performance periodically
        if int(current_time.nanoseconds / 1e9) % 5 == 0:  # Every 5 seconds
            self.get_logger().info(
                f'PID Control - Target: ({self.target_linear_vel:.2f}, {self.target_angular_vel:.2f}), '
                f'Current: ({self.current_linear_vel:.2f}, {self.current_angular_vel:.2f}), '
                f'Output: ({linear_output:.2f}, {angular_output:.2f})'
            )

    def tune_pid_parameters(self, kp, ki, kd, control_type='linear'):
        """Dynamically tune PID parameters."""
        if control_type == 'linear':
            self.kp_linear = kp
            self.ki_linear = ki
            self.kd_linear = kd
        elif control_type == 'angular':
            self.kp_angular = kp
            self.ki_angular = ki
            self.kd_angular = kd

        self.get_logger().info(f'PID parameters tuned for {control_type}: Kp={kp}, Ki={ki}, Kd={kd}')

def main(args=None):
    rclpy.init(args=args)
    pid_controller = PIDController()

    try:
        rclpy.spin(pid_controller)
    except KeyboardInterrupt:
        pid_controller.get_logger().info('Shutting down PID Controller')
    finally:
        pid_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Trajectory Following Control

### Pure Pursuit Controller

The pure pursuit algorithm is commonly used for path following in mobile robots.

```python
# pure_pursuit_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10
        )
        self.lookahead_pub = self.create_publisher(Marker, 'lookahead_point', 10)

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Path following parameters
        self.lookahead_distance = 1.0  # meters
        self.linear_velocity = 0.5     # m/s
        self.max_angular_velocity = 1.0 # rad/s
        self.path_tolerance = 0.2      # meters

        # Path storage
        self.path = []
        self.path_index = 0

        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_callback)  # 20 Hz

        self.get_logger().info('Pure Pursuit Controller initialized')

    def odom_callback(self, msg):
        """Update robot position from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def path_callback(self, msg):
        """Receive and store path."""
        self.path = []
        for pose_stamped in msg.poses:
            point = [
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ]
            self.path.append(point)

        self.path_index = 0  # Reset to beginning of path
        self.get_logger().info(f'New path received with {len(self.path)} points')

    def control_callback(self):
        """Main control loop for path following."""
        if not self.path:
            # Stop robot if no path
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return

        # Find lookahead point
        lookahead_point = self.find_lookahead_point()
        if lookahead_point is None:
            # Stop robot if no lookahead point found
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return

        # Calculate control commands
        linear_vel, angular_vel = self.calculate_control(lookahead_point)

        # Create and publish velocity command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel

        self.cmd_vel_pub.publish(cmd)

        # Publish visualization marker for lookahead point
        self.publish_lookahead_marker(lookahead_point)

        # Check if path is completed
        if self.is_path_completed():
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info('Path following completed')

    def find_lookahead_point(self):
        """Find the point on the path that is lookahead_distance away."""
        if len(self.path) < 2:
            return None

        # Start from current path index
        for i in range(self.path_index, len(self.path)):
            path_point = self.path[i]

            # Calculate distance to this path point
            dx = path_point[0] - self.current_x
            dy = path_point[1] - self.current_y
            distance = math.sqrt(dx*dx + dy*dy)

            # If we find a point at or beyond lookahead distance, return it
            if distance >= self.lookahead_distance:
                self.path_index = i
                return path_point

        # If we reach the end of the path, return the last point
        if self.path:
            return self.path[-1]

        return None

    def calculate_control(self, lookahead_point):
        """Calculate linear and angular velocities to reach lookahead point."""
        # Vector from robot to lookahead point
        dx = lookahead_point[0] - self.current_x
        dy = lookahead_point[1] - self.current_y

        # Calculate angle to lookahead point
        angle_to_point = math.atan2(dy, dx)
        angle_error = angle_to_point - self.current_yaw

        # Normalize angle to [-π, π]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Calculate angular velocity (proportional to angle error)
        angular_vel = max(min(angle_error * 1.5, self.max_angular_velocity), -self.max_angular_velocity)

        # Calculate linear velocity based on angular velocity (reduce speed when turning sharply)
        linear_vel = self.linear_velocity * max(0.3, 1.0 - abs(angular_vel) / self.max_angular_velocity)

        return linear_vel, angular_vel

    def is_path_completed(self):
        """Check if the robot has reached the end of the path."""
        if not self.path:
            return True

        # Check distance to last point in path
        last_point = self.path[-1]
        dx = last_point[0] - self.current_x
        dy = last_point[1] - self.current_y
        distance = math.sqrt(dx*dx + dy*dy)

        return distance < self.path_tolerance

    def publish_lookahead_marker(self, point):
        """Publish visualization marker for lookahead point."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.lookahead_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    controller = PurePursuitController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Pure Pursuit Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Model Predictive Control (MPC)

Model Predictive Control is an advanced control technique that optimizes control actions over a prediction horizon.

```python
# mpc_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import math
from scipy.optimize import minimize
import time

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10
        )
        self.mpc_path_pub = self.create_publisher(Path, 'mpc_predicted_path', 10)

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_v = 0.0

        # MPC parameters
        self.horizon = 10          # Prediction horizon steps
        self.dt = 0.1              # Time step (seconds)
        self.max_linear_vel = 1.0  # m/s
        self.max_angular_vel = 1.0 # rad/s
        self.max_accel = 2.0       # m/s²
        self.max_alpha = 2.0       # rad/s²

        # Path following
        self.path = []
        self.path_index = 0
        self.lookahead_distance = 1.0

        # MPC state
        self.mpc_controls = []  # Store predicted control sequence
        self.mpc_states = []    # Store predicted states

        # Control timer
        self.control_timer = self.create_timer(0.05, self.mpc_control_callback)  # 20 Hz

        self.get_logger().info('MPC Controller initialized')

    def odom_callback(self, msg):
        """Update robot state from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract orientation
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Extract linear velocity magnitude
        self.current_v = math.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2
        )

    def path_callback(self, msg):
        """Receive and store path."""
        self.path = []
        for pose_stamped in msg.poses:
            point = [
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ]
            self.path.append(point)

        self.path_index = 0
        self.get_logger().info(f'MPC: New path with {len(self.path)} points')

    def mpc_control_callback(self):
        """Main MPC control loop."""
        if not self.path:
            # Stop robot if no path
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return

        # Solve MPC optimization problem
        optimal_controls = self.solve_mpc()

        if optimal_controls is not None and len(optimal_controls) > 0:
            # Apply first control in sequence
            linear_vel, angular_vel = optimal_controls[0]

            # Create and publish velocity command
            cmd = Twist()
            cmd.linear.x = float(linear_vel)
            cmd.angular.z = float(angular_vel)

            self.cmd_vel_pub.publish(cmd)

            # Store predicted states and controls for visualization
            self.mpc_controls = optimal_controls
            self.predict_states(optimal_controls)

            # Publish predicted path for visualization
            self.publish_predicted_path()
        else:
            # Stop robot if MPC fails
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)

    def solve_mpc(self):
        """Solve the MPC optimization problem."""
        # Initial state
        x0 = np.array([self.current_x, self.current_y, self.current_yaw, self.current_v])

        # Define cost function
        def cost_function(controls_flat):
            # Reshape flat controls array to 2D (linear_vel, angular_vel)
            controls = controls_flat.reshape((self.horizon, 2))

            # Predict states using the controls
            states = self.predict_trajectory(x0, controls)

            # Calculate cost
            total_cost = 0.0

            # Path following cost
            for i, state in enumerate(states):
                path_cost = self.calculate_path_cost(state, i)
                total_cost += path_cost

            # Control effort cost (minimize control changes)
            for i in range(len(controls)):
                total_cost += 0.1 * (controls[i, 0]**2 + controls[i, 1]**2)  # Minimize control effort
                if i > 0:
                    total_cost += 0.05 * ((controls[i, 0] - controls[i-1, 0])**2 +
                                         (controls[i, 1] - controls[i-1, 1])**2)  # Minimize control changes

            # Terminal cost
            if len(states) > 0:
                terminal_cost = self.calculate_path_cost(states[-1], self.horizon)
                total_cost += 2.0 * terminal_cost

            return total_cost

        # Define constraints
        def control_constraints(controls_flat):
            controls = controls_flat.reshape((self.horizon, 2))
            constraints = []

            for i in range(self.horizon):
                # Velocity limits
                constraints.append(self.max_linear_vel - abs(controls[i, 0]))
                constraints.append(self.max_angular_vel - abs(controls[i, 1]))

                # Acceleration limits (if not the first control)
                if i > 0:
                    constraints.append(self.max_accel - abs(controls[i, 0] - controls[i-1, 0]) / self.dt)
                    constraints.append(self.max_alpha - abs(controls[i, 1] - controls[i-1, 1]) / self.dt)

            return np.array(constraints)

        # Initial guess for controls (current velocity)
        initial_controls = np.zeros(2 * self.horizon)
        for i in range(self.horizon):
            initial_controls[2*i] = self.current_v  # linear velocity
            initial_controls[2*i + 1] = 0.0         # angular velocity

        # Optimization bounds
        bounds = []
        for i in range(self.horizon):
            bounds.extend([(-self.max_linear_vel, self.max_linear_vel),    # linear velocity bounds
                          (-self.max_angular_vel, self.max_angular_vel)])  # angular velocity bounds

        # Solve optimization problem
        try:
            result = minimize(
                cost_function,
                initial_controls,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': control_constraints},
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            if result.success:
                # Reshape result to control sequence
                optimal_controls = result.x.reshape((self.horizon, 2))
                return optimal_controls
            else:
                self.get_logger().warn(f'MPC optimization failed: {result.message}')
                return None
        except Exception as e:
            self.get_logger().error(f'MPC optimization error: {e}')
            return None

    def predict_trajectory(self, initial_state, controls):
        """Predict robot trajectory given initial state and control sequence."""
        states = [initial_state]

        current_state = initial_state.copy()

        for control in controls:
            linear_vel, angular_vel = control

            # Simple kinematic model for differential drive
            # x_dot = v * cos(theta)
            # y_dot = v * sin(theta)
            # theta_dot = omega
            # v_dot = linear_acc (assumed to track linear_vel instantaneously for simplicity)

            new_x = current_state[0] + linear_vel * math.cos(current_state[2]) * self.dt
            new_y = current_state[1] + linear_vel * math.sin(current_state[2]) * self.dt
            new_theta = current_state[2] + angular_vel * self.dt
            new_v = linear_vel  # Simplified velocity model

            current_state = np.array([new_x, new_y, new_theta, new_v])
            states.append(current_state)

        return states

    def calculate_path_cost(self, state, time_step):
        """Calculate cost of deviating from path."""
        if not self.path:
            return 0.0

        # Find closest point on path to current state
        min_dist = float('inf')
        closest_point = None

        for path_point in self.path:
            dist = math.sqrt((state[0] - path_point[0])**2 + (state[1] - path_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = path_point

        if closest_point is None:
            return 0.0

        # Path following cost increases with distance from path
        path_cost = 5.0 * min_dist**2  # Quadratic cost for path deviation

        # Add orientation cost (penalize when robot is not aligned with path direction)
        if len(self.path) > 1 and self.path_index < len(self.path) - 1:
            # Calculate desired orientation based on path direction
            next_idx = min(self.path_index + 1, len(self.path) - 1)
            path_dx = self.path[next_idx][0] - self.path[self.path_index][0]
            path_dy = self.path[next_idx][1] - self.path[self.path_index][1]
            desired_yaw = math.atan2(path_dy, path_dx)

            yaw_error = state[2] - desired_yaw
            # Normalize angle
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            orientation_cost = 2.0 * abs(yaw_error)
            path_cost += orientation_cost

        return path_cost

    def predict_states(self, controls):
        """Predict states for visualization."""
        x0 = np.array([self.current_x, self.current_y, self.current_yaw, self.current_v])
        self.mpc_states = self.predict_trajectory(x0, controls)

    def publish_predicted_path(self):
        """Publish predicted path for visualization."""
        if not self.mpc_states:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for i, state in enumerate(self.mpc_states):
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = path_msg.header.stamp
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = float(state[0])
            pose_stamped.pose.position.y = float(state[1])
            pose_stamped.pose.position.z = 0.05 * i  # Slightly raise for visualization

            # Convert yaw to quaternion
            cy = math.cos(state[2] * 0.5)
            sy = math.sin(state[2] * 0.5)
            pose_stamped.pose.orientation.z = sy
            pose_stamped.pose.orientation.w = cy

            path_msg.poses.append(pose_stamped)

        self.mpc_path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    mpc_controller = MPCController()

    try:
        rclpy.spin(mpc_controller)
    except KeyboardInterrupt:
        mpc_controller.get_logger().info('Shutting down MPC Controller')
    finally:
        mpc_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Adaptive Control Systems

Adaptive control systems adjust their parameters based on changing conditions or uncertainties.

```python
# adaptive_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import numpy as np
import math

class AdaptiveController(Node):
    def __init__(self):
        super().__init__('adaptive_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.target_sub = self.create_subscription(
            Vector3, 'target_velocity', self.target_callback, 10
        )

        # Adaptive PID parameters
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05

        # Adaptive parameter bounds
        self.kp_min, self.kp_max = 0.5, 3.0
        self.ki_min, self.ki_max = 0.01, 0.5
        self.kd_min, self.kd_max = 0.01, 0.2

        # PID state variables
        self.error_sum = 0.0
        self.error_prev = 0.0
        self.derivative_filtered = 0.0  # For noise filtering

        # Current and target states
        self.current_vel = 0.0
        self.target_vel = 0.0

        # Adaptive control parameters
        self.gamma = 0.01  # Learning rate
        self.sigma = 0.1   # Sigma modification term (for stability)
        self.adaptation_enabled = True

        # Environmental adaptation variables
        self.obstacle_distance = float('inf')
        self.surface_condition = 1.0  # 1.0 = normal, <1.0 = slippery, >1.0 = high traction

        # Control timer
        self.control_timer = self.create_timer(0.02, self.adaptive_control_callback)  # 50 Hz

        # Time tracking
        self.last_time = self.get_clock().now()

        self.get_logger().info('Adaptive Controller initialized')

    def odom_callback(self, msg):
        """Update current velocity from odometry."""
        self.current_vel = math.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2 +
            msg.twist.twist.linear.z**2
        )

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection and surface estimation."""
        # Find minimum distance in front of robot (simplified)
        if len(msg.ranges) > 0:
            front_start = len(msg.ranges) // 2 - len(msg.ranges) // 12  # -15 degrees
            front_end = len(msg.ranges) // 2 + len(msg.ranges) // 12    # +15 degrees

            front_ranges = msg.ranges[front_start:front_end]
            valid_ranges = [r for r in front_ranges if msg.range_min < r < msg.range_max]

            if valid_ranges:
                self.obstacle_distance = min(valid_ranges)
            else:
                self.obstacle_distance = float('inf')

        # Estimate surface condition from scan (simplified)
        # In reality, this would use IMU, wheel encoders, or other sensors
        self.estimate_surface_condition(msg)

    def estimate_surface_condition(self, scan_msg):
        """Estimate surface condition based on scan data."""
        # This is a simplified estimation
        # In practice, you'd use IMU data, wheel slip detection, etc.
        if self.obstacle_distance < 1.0:
            # In cluttered environments, assume more careful driving is needed
            self.surface_condition = 0.8
        else:
            self.surface_condition = 1.0

    def target_callback(self, msg):
        """Update target velocity."""
        self.target_vel = math.sqrt(msg.x**2 + msg.y**2 + msg.z**2)

    def adaptive_control_callback(self):
        """Main adaptive control loop."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt <= 0:
            return

        # Calculate error
        error = self.target_vel - self.current_vel

        # Update error integral with anti-windup
        self.error_sum = self.error_sum + error * dt
        # Anti-windup: limit integral term
        max_integral = 5.0
        self.error_sum = max(min(self.error_sum, max_integral), -max_integral)

        # Calculate error derivative with filtering to reduce noise
        if dt > 0:
            error_deriv = (error - self.error_prev) / dt
            # First-order filter for derivative
            alpha = 0.2  # Filter coefficient
            self.derivative_filtered = alpha * error_deriv + (1 - alpha) * self.derivative_filtered
        else:
            self.derivative_filtered = 0.0

        # Store current error for next iteration
        self.error_prev = error

        # Calculate base PID output
        pid_output = (
            self.kp * error +
            self.ki * self.error_sum +
            self.kd * self.derivative_filtered
        )

        # Adaptive parameter adjustment based on system conditions
        if self.adaptation_enabled:
            self.update_adaptive_parameters(error, dt)

        # Apply surface condition scaling
        scaled_output = pid_output * self.surface_condition

        # Apply output limits
        max_output = 2.0
        final_output = max(min(scaled_output, max_output), -max_output)

        # Create and publish velocity command
        cmd = Twist()
        cmd.linear.x = float(final_output)
        # Add some angular control based on obstacle avoidance
        cmd.angular.z = self.calculate_obstacle_avoidance()

        self.cmd_vel_pub.publish(cmd)

        # Log adaptive parameters periodically
        if int(current_time.nanoseconds / 1e9) % 3 == 0:  # Every 3 seconds
            self.get_logger().info(
                f'Adaptive Control - Kp: {self.kp:.3f}, Ki: {self.ki:.3f}, Kd: {self.kd:.3f}, '
                f'Surface: {self.surface_condition:.2f}, Obstacle Dist: {self.obstacle_distance:.2f}m'
            )

    def update_adaptive_parameters(self, error, dt):
        """Update PID parameters based on system performance."""
        # Sensitivity functions (partial derivatives of control output w.r.t. parameters)
        phi_kp = error
        phi_ki = self.error_sum * dt
        phi_kd = self.derivative_filtered * dt

        # Parameter update using gradient descent with sigma modification
        # This helps maintain stability while adapting
        self.kp = self.kp + self.gamma * error * phi_kp / (1 + self.sigma * abs(phi_kp))
        self.ki = self.ki + self.gamma * error * phi_ki / (1 + self.sigma * abs(phi_ki))
        self.kd = self.kd + self.gamma * error * phi_kd / (1 + self.sigma * abs(phi_kd))

        # Apply parameter bounds
        self.kp = max(min(self.kp, self.kp_max), self.kp_min)
        self.ki = max(min(self.ki, self.ki_max), self.ki_min)
        self.kd = max(min(self.kd, self.kd_max), self.kd_min)

        # Adapt parameters based on environmental conditions
        if self.obstacle_distance < 2.0:  # Close obstacles
            # Reduce gains for more careful control
            self.kp *= 0.8
            self.ki *= 0.8
            self.kd *= 1.2  # Increase derivative gain for more damping
        elif self.surface_condition < 0.9:  # Slippery surface
            # Reduce gains to avoid wheel slip
            self.kp *= 0.7
            self.ki *= 0.6
            self.kd *= 0.8

    def calculate_obstacle_avoidance(self):
        """Calculate angular velocity for obstacle avoidance."""
        if self.obstacle_distance < 1.0:
            # If obstacle is close, turn away
            # This is a simplified approach - in reality you'd use more sophisticated methods
            if self.obstacle_distance < 0.5:
                return 0.5  # Sharp turn
            else:
                return 0.2  # Gentle turn
        return 0.0  # No obstacle, go straight

    def enable_adaptation(self, enable=True):
        """Enable or disable parameter adaptation."""
        self.adaptation_enabled = enable
        status = "enabled" if enable else "disabled"
        self.get_logger().info(f'Adaptive parameter tuning {status}')

def main(args=None):
    rclpy.init(args=args)
    adaptive_controller = AdaptiveController()

    try:
        rclpy.spin(adaptive_controller)
    except KeyboardInterrupt:
        adaptive_controller.get_logger().info('Shutting down Adaptive Controller')
    finally:
        adaptive_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Learning-Based Control

### Reinforcement Learning for Control

```python
# rl_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import numpy as np
import math
import random

class RLController(Node):
    def __init__(self):
        super().__init__('rl_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # RL parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01

        # State and action spaces
        self.state_bins = [5, 5, 3]  # [distance_to_goal, obstacle_distance, angle_to_goal]
        self.action_space = 6  # Discrete actions: [fast_forward, slow_forward, turn_left, turn_right, backward, stop]

        # Q-table initialization
        self.q_table = np.zeros((*self.state_bins, self.action_space))

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacle_distances = np.array([])

        # Goal (for simulation purposes)
        self.goal_x = 5.0
        self.goal_y = 5.0

        # RL state tracking
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0.0

        # Control timer
        self.control_timer = self.create_timer(0.1, self.rl_control_callback)  # 10 Hz

        self.get_logger().info('RL Controller initialized')

    def odom_callback(self, msg):
        """Update robot position from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract orientation
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        """Process laser scan data."""
        if len(msg.ranges) > 0:
            self.obstacle_distances = np.array(msg.ranges)
            # Replace invalid ranges with max range
            self.obstacle_distances[self.obstacle_distances < msg.range_min] = msg.range_max
            self.obstacle_distances[self.obstacle_distances > msg.range_max] = msg.range_max

    def rl_control_callback(self):
        """Main RL control loop."""
        # Get current discrete state
        current_state = self.get_discrete_state()

        # Choose action using epsilon-greedy policy
        action = self.choose_action(current_state)

        # Execute action
        cmd_vel = self.action_to_velocity(action)
        self.cmd_vel_pub.publish(cmd_vel)

        # Calculate reward
        reward = self.calculate_reward()

        # Update Q-table if we have a previous state-action pair
        if self.previous_state is not None and self.previous_action is not None:
            self.update_q_table(self.previous_state, self.previous_action,
                              self.previous_reward, current_state)

        # Update state tracking
        self.previous_state = current_state
        self.previous_action = action
        self.previous_reward = reward

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay

    def get_discrete_state(self):
        """Convert continuous state to discrete state."""
        # Calculate distance to goal
        dist_to_goal = math.sqrt((self.goal_x - self.current_x)**2 + (self.goal_y - self.current_y)**2)

        # Calculate angle to goal
        angle_to_goal = math.atan2(self.goal_y - self.current_y, self.goal_x - self.current_x)
        angle_diff = angle_to_goal - self.current_yaw
        # Normalize angle to [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Get minimum obstacle distance
        if len(self.obstacle_distances) > 0:
            min_obstacle_dist = np.min(self.obstacle_distances)
        else:
            min_obstacle_dist = 10.0  # Default to far distance

        # Discretize state variables
        dist_idx = min(int(dist_to_goal / 2.0), self.state_bins[0] - 1)  # Max 10m range
        obs_idx = min(int(min_obstacle_dist / 1.0), self.state_bins[1] - 1)  # Max 5m range
        angle_idx = 0 if angle_diff < -math.pi/3 else (1 if angle_diff < math.pi/3 else 2)

        return (dist_idx, obs_idx, angle_idx)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: choose best known action
            return int(np.argmax(self.q_table[state]))

    def action_to_velocity(self, action):
        """Convert discrete action to velocity command."""
        cmd = Twist()

        if action == 0:  # Fast forward
            cmd.linear.x = 0.8
            cmd.angular.z = 0.0
        elif action == 1:  # Slow forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif action == 2:  # Turn left
            cmd.linear.x = 0.2
            cmd.angular.z = 0.5
        elif action == 3:  # Turn right
            cmd.linear.x = 0.2
            cmd.angular.z = -0.5
        elif action == 4:  # Backward
            cmd.linear.x = -0.2
            cmd.angular.z = 0.0
        else:  # Stop (action == 5)
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def calculate_reward(self):
        """Calculate reward based on current state."""
        # Calculate distance to goal
        dist_to_goal = math.sqrt((self.goal_x - self.current_x)**2 + (self.goal_y - self.current_y)**2)

        # Calculate minimum obstacle distance
        min_obstacle_dist = np.min(self.obstacle_distances) if len(self.obstacle_distances) > 0 else 10.0

        # Reward components
        goal_reward = -dist_to_goal  # Negative distance (closer is better)

        # Obstacle penalty (large penalty for very close obstacles)
        obstacle_penalty = 0
        if min_obstacle_dist < 0.5:
            obstacle_penalty = -100  # Large penalty for collision risk
        elif min_obstacle_dist < 1.0:
            obstacle_penalty = -10 / min_obstacle_dist  # Inverse penalty

        # Small time penalty to encourage efficiency
        time_penalty = -0.1

        # Combined reward
        reward = goal_reward * 0.1 + obstacle_penalty + time_penalty

        # Bonus for reaching goal (within 0.5m)
        if dist_to_goal < 0.5:
            reward += 100  # Large positive reward for reaching goal

        return reward

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm."""
        current_q = self.q_table[state][action]

        # Calculate maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state])

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward +
                     self.discount_factor * max_next_q - current_q)

        # Update Q-value
        self.q_table[state][action] = new_q

    def save_policy(self, filename):
        """Save the learned Q-table to file."""
        np.save(filename, self.q_table)
        self.get_logger().info(f'Policy saved to {filename}')

    def load_policy(self, filename):
        """Load a Q-table from file."""
        try:
            self.q_table = np.load(filename)
            self.get_logger().info(f'Policy loaded from {filename}')
        except FileNotFoundError:
            self.get_logger().warn(f'Policy file {filename} not found, starting with zeros')

def main(args=None):
    rclpy.init(args=args)
    rl_controller = RLController()

    try:
        rclpy.spin(rl_controller)
    except KeyboardInterrupt:
        rl_controller.get_logger().info('Shutting down RL Controller')
        rl_controller.save_policy('learned_policy.npy')
    finally:
        rl_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Control System Integration

### Integrating Perception and Control

```python
# perception_control_integration.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray, Detection3D
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import numpy as np
import math
import cv2

class PerceptionControlIntegration(Node):
    def __init__(self):
        super().__init__('perception_control_integration')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detections', self.detections_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10
        )

        # Status publishers
        self.safety_status_pub = self.create_publisher(Bool, 'safety_status', 10)
        self.control_effort_pub = self.create_publisher(Float32, 'control_effort', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_v = 0.0

        # Perception data
        self.detections = []
        self.scan_data = None

        # Navigation path
        self.path = []
        self.path_index = 0

        # Control parameters
        self.linear_velocity = 0.5
        self.angular_velocity_limit = 1.0
        self.safety_distance = 0.8
        self.person_following_distance = 2.0

        # Control mode
        self.control_mode = 'navigation'  # 'navigation', 'person_following', 'obstacle_avoidance'
        self.followed_person_id = None

        # Control timer
        self.control_timer = self.create_timer(0.05, self.integrated_control_callback)  # 20 Hz

        self.get_logger().info('Perception-Control Integration initialized')

    def odom_callback(self, msg):
        """Update robot state from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract orientation
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Extract linear velocity magnitude
        self.current_v = math.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2
        )

    def detections_callback(self, msg):
        """Process perception detections."""
        self.detections = msg.detections

        # Check if there are person detections for person following
        person_detections = [det for det in self.detections
                           if det.results and det.results[0].hypothesis.class_id == 'person']

        if person_detections and self.control_mode == 'person_following':
            # Select the person closest to the center of the image
            center_x = 320  # Assuming 640x480 image
            closest_person = min(person_detections,
                               key=lambda det: abs(det.bbox.center.x - center_x))

            # Switch to person following mode if we have a detection
            if self.followed_person_id is None:
                self.followed_person_id = closest_person.results[0].hypothesis.class_id
                self.get_logger().info(f'Starting to follow person: {self.followed_person_id}')

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection."""
        self.scan_data = msg

    def path_callback(self, msg):
        """Receive navigation path."""
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.path_index = 0
        self.control_mode = 'navigation'  # Switch to navigation mode when new path received
        self.get_logger().info(f'New navigation path with {len(self.path)} waypoints')

    def integrated_control_callback(self):
        """Main integrated control loop."""
        # Determine control mode based on perception input
        self.update_control_mode()

        # Execute appropriate control strategy
        if self.control_mode == 'person_following':
            cmd_vel = self.person_following_control()
        elif self.control_mode == 'obstacle_avoidance':
            cmd_vel = self.obstacle_avoidance_control()
        else:  # navigation
            cmd_vel = self.navigation_control()

        # Apply safety checks
        cmd_vel = self.safety_check(cmd_vel)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish safety status
        safety_status = Bool()
        safety_status.data = self.is_safe_to_move()
        self.safety_status_pub.publish(safety_status)

        # Publish control effort (magnitude of velocity command)
        control_effort = Float32()
        control_effort.data = math.sqrt(cmd_vel.linear.x**2 + cmd_vel.angular.z**2)
        self.control_effort_pub.publish(control_effort)

    def update_control_mode(self):
        """Update control mode based on perception input."""
        # Check for person detections to switch to person following
        if self.detections:
            person_detections = [det for det in self.detections
                               if det.results and det.results[0].hypothesis.class_id == 'person']

            if person_detections and self.control_mode != 'person_following':
                self.get_logger().info('Switching to person following mode')
                self.control_mode = 'person_following'

    def person_following_control(self):
        """Control strategy for following a person."""
        if not self.detections:
            # No detections, stop robot
            return Twist()

        # Find the person being followed
        person_detection = None
        for det in self.detections:
            if (det.results and
                det.results[0].hypothesis.class_id == 'person' and
                self.followed_person_id in det.results[0].hypothesis.class_id):
                person_detection = det
                break

        if not person_detection:
            # Person lost, stop robot
            self.get_logger().warn('Person lost, stopping robot')
            self.control_mode = 'navigation'  # Switch back to navigation
            self.followed_person_id = None
            return Twist()

        # Calculate person position relative to image center
        image_center_x = 320  # Assuming 640x480 image
        person_center_x = person_detection.bbox.center.x
        x_error = person_center_x - image_center_x

        # Calculate distance control based on bounding box size
        bbox_size = person_detection.bbox.size_x * person_detection.bbox.size_y
        # Assuming that larger bounding box means closer person
        desired_distance = 50000 / (bbox_size + 1)  # Simplified inverse relationship
        distance_error = self.person_following_distance - desired_distance

        # PID-like control
        linear_vel = max(min(0.5 - distance_error * 0.01, 0.8), -0.2)  # Forward/backward
        angular_vel = max(min(-x_error * 0.005, self.angular_velocity_limit), -self.angular_velocity_limit)  # Turn

        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel

        return cmd

    def navigation_control(self):
        """Control strategy for following a navigation path."""
        if not self.path or self.path_index >= len(self.path):
            # No path or path completed, stop robot
            return Twist()

        # Get current target point
        target_x, target_y = self.path[self.path_index]

        # Calculate distance and angle to target
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance_to_target = math.sqrt(dx*dx + dy*dy)

        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - self.current_yaw

        # Normalize angle to [-π, π]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Control outputs
        if distance_to_target < 0.2:  # Close to current waypoint
            # Move to next waypoint
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.get_logger().info('Navigation completed')
                return Twist()  # Stop when path completed

            # Get new target
            target_x, target_y = self.path[self.path_index]
            dx = target_x - self.current_x
            dy = target_y - self.current_y
            distance_to_target = math.sqrt(dx*dx + dy*dy)
            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - self.current_yaw

        # PID control for navigation
        linear_vel = max(min(distance_to_target * 0.5, self.linear_velocity), 0)
        angular_vel = max(min(angle_error * 2.0, self.angular_velocity_limit), -self.angular_velocity_limit)

        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel

        return cmd

    def obstacle_avoidance_control(self):
        """Control strategy for obstacle avoidance."""
        if self.scan_data is None:
            return Twist()

        # Find minimum distance in front of robot
        front_start = len(self.scan_data.ranges) // 2 - len(self.scan_data.ranges) // 12
        front_end = len(self.scan_data.ranges) // 2 + len(self.scan_data.ranges) // 12

        front_ranges = self.scan_data.ranges[front_start:front_end]
        valid_ranges = [r for r in front_ranges if self.scan_data.range_min < r < self.scan_data.range_max]

        if not valid_ranges:
            # No obstacles detected in front, continue forward
            cmd = Twist()
            cmd.linear.x = self.linear_velocity
            return cmd

        min_distance = min(valid_ranges)

        if min_distance < self.safety_distance:
            # Obstacle too close, turn away
            cmd = Twist()

            # Determine turn direction based on obstacle distribution
            left_ranges = front_ranges[:len(front_ranges)//2]
            right_ranges = front_ranges[len(front_ranges)//2:]

            left_clear = any(r > self.safety_distance for r in left_ranges if self.scan_data.range_min < r < self.scan_data.range_max)
            right_clear = any(r > self.safety_distance for r in right_ranges if self.scan_data.range_min < r < self.scan_data.range_max)

            if left_clear and not right_clear:
                cmd.angular.z = 0.5  # Turn left
            elif right_clear and not left_clear:
                cmd.angular.z = -0.5  # Turn right
            elif left_clear and right_clear:
                # Both sides clear, turn toward the side with more clearance
                left_avg = np.mean([r for r in left_ranges if self.scan_data.range_min < r < self.scan_data.range_max])
                right_avg = np.mean([r for r in right_ranges if self.scan_data.range_min < r < self.scan_data.range_max])
                cmd.angular.z = 0.5 if left_avg > right_avg else -0.5
            else:
                # Neither side clear, turn in a random direction
                cmd.angular.z = 0.5 if np.random.random() > 0.5 else -0.5

            return cmd
        else:
            # Safe to move forward
            cmd = Twist()
            cmd.linear.x = self.linear_velocity
            return cmd

    def safety_check(self, cmd_vel):
        """Apply safety checks to velocity commands."""
        if self.scan_data is None:
            return Twist()  # Stop if no sensor data

        # Emergency stop if very close to obstacle
        front_ranges = self.scan_data.ranges[len(self.scan_data.ranges)//2-10:len(self.scan_data.ranges)//2+10]
        very_close_ranges = [r for r in front_ranges if self.scan_data.range_min < r < 0.3]

        if very_close_ranges:
            # Emergency stop
            self.get_logger().warn('Very close obstacle detected, emergency stop!')
            return Twist()

        return cmd_vel

    def is_safe_to_move(self):
        """Check if it's safe to move."""
        if self.scan_data is None:
            return False

        front_ranges = self.scan_data.ranges[len(self.scan_data.ranges)//2-20:len(self.scan_data.ranges)//2+20]
        close_ranges = [r for r in front_ranges if self.scan_data.range_min < r < self.safety_distance]

        return len(close_ranges) == 0

def main(args=None):
    rclpy.init(args=args)
    integration_node = PerceptionControlIntegration()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        integration_node.get_logger().info('Shutting down Perception-Control Integration')
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Control System Validation

### Validating Control Performance

```python
# control_validation.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Vector3
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import math
import time
from collections import deque

class ControlValidation(Node):
    def __init__(self):
        super().__init__('control_validation')

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.cmd_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10
        )
        self.target_sub = self.create_subscription(
            Vector3, 'target_pose', self.target_callback, 10
        )

        # Publishers for validation metrics
        self.stability_pub = self.create_publisher(Float32, 'control_stability', 10)
        self.accuracy_pub = self.create_publisher(Float32, 'control_accuracy', 10)
        self.smoothness_pub = self.create_publisher(Float32, 'control_smoothness', 10)
        self.response_time_pub = self.create_publisher(Float32, 'control_response_time', 10)
        self.error_pub = self.create_publisher(Float32, 'tracking_error', 10)
        self.status_pub = self.create_publisher(Bool, 'control_status', 10)

        # Robot state tracking
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_v = 0.0
        self.current_omega = 0.0

        # Command tracking
        self.command_v = 0.0
        self.command_omega = 0.0

        # Path tracking
        self.path = []
        self.target_x = 0.0
        self.target_y = 0.0

        # Performance tracking
        self.position_errors = deque(maxlen=100)
        self.velocity_errors = deque(maxlen=100)
        self.control_commands = deque(maxlen=50)
        self.timestamps = deque(maxlen=100)

        # Performance thresholds
        self.stability_threshold = 0.1
        self.accuracy_threshold = 0.3
        self.smoothness_threshold = 0.5

        # Timing
        self.start_time = time.time()

        self.get_logger().info('Control Validation initialized')

    def odom_callback(self, msg):
        """Update robot state from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract orientation
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Extract velocities
        self.current_v = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.current_omega = msg.twist.twist.angular.z

        # Record timestamp and state
        current_time = time.time()
        self.timestamps.append(current_time)

        # Calculate errors
        if self.path or (self.target_x != 0 or self.target_y != 0):
            # Calculate distance to target/path
            if self.path:
                # For path following, find distance to closest path point
                min_dist = float('inf')
                for point in self.path:
                    dist = math.sqrt((self.current_x - point[0])**2 + (self.current_y - point[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                error = min_dist
            else:
                # For target following
                error = math.sqrt((self.current_x - self.target_x)**2 + (self.current_y - self.target_y)**2)

            self.position_errors.append(error)
            self.velocity_errors.append(abs(self.current_v - self.command_v))

    def cmd_callback(self, msg):
        """Track control commands."""
        self.command_v = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.command_omega = msg.angular.z

        # Store command for smoothness analysis
        self.control_commands.append((self.command_v, self.command_omega))

    def path_callback(self, msg):
        """Receive path for validation."""
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.get_logger().info(f'Validation: Path with {len(self.path)} points')

    def target_callback(self, msg):
        """Receive target position."""
        self.target_x = msg.x
        self.target_y = msg.y

    def calculate_stability(self):
        """Calculate control stability metric."""
        if len(self.position_errors) < 10:
            return 0.0

        # Calculate variance of position errors (lower variance = more stable)
        errors_array = np.array(list(self.position_errors))
        variance = np.var(errors_array)

        # Normalize to 0-1 scale (inverted, so lower variance = higher stability)
        stability = 1.0 / (1.0 + variance)
        return min(stability, 1.0)  # Cap at 1.0

    def calculate_accuracy(self):
        """Calculate control accuracy metric."""
        if not self.position_errors:
            return 0.0

        # Calculate average error (lower = more accurate)
        avg_error = np.mean(list(self.position_errors))

        # Convert to accuracy score (inverted)
        accuracy = 1.0 / (1.0 + avg_error)
        return min(accuracy, 1.0)

    def calculate_smoothness(self):
        """Calculate control smoothness metric."""
        if len(self.control_commands) < 2:
            return 1.0  # Perfectly smooth if only one command

        # Calculate changes in control commands
        v_changes = []
        omega_changes = []

        commands_list = list(self.control_commands)
        for i in range(1, len(commands_list)):
            v_change = abs(commands_list[i][0] - commands_list[i-1][0])
            omega_change = abs(commands_list[i][1] - commands_list[i-1][1])

            v_changes.append(v_change)
            omega_changes.append(omega_change)

        # Average change (lower = smoother)
        avg_v_change = np.mean(v_changes) if v_changes else 0
        avg_omega_change = np.mean(omega_changes) if omega_changes else 0

        # Combine into smoothness score
        total_change = avg_v_change + avg_omega_change
        smoothness = 1.0 / (1.0 + total_change * 10)  # Scale factor for normalization
        return min(smoothness, 1.0)

    def calculate_response_time(self):
        """Calculate control response time."""
        # This would typically measure how quickly the system responds to step inputs
        # For simulation, we'll return a normalized value based on error convergence
        if len(self.position_errors) < 20:
            return 0.0

        # Look at the rate of error reduction
        recent_errors = list(self.position_errors)[-20:]
        older_errors = list(self.position_errors)[-40:-20] if len(self.position_errors) >= 40 else list(self.position_errors)[:20]

        if not older_errors:
            return 0.0

        recent_avg = np.mean(recent_errors)
        older_avg = np.mean(older_errors)

        # If errors are decreasing, response is good
        if older_avg > recent_avg:
            improvement_rate = (older_avg - recent_avg) / older_avg
            return min(improvement_rate * 5, 1.0)  # Scale to 0-1
        else:
            return 0.0

    def calculate_tracking_error(self):
        """Calculate overall tracking error."""
        if not self.position_errors:
            return 0.0

        return float(np.mean(list(self.position_errors)))

    def publish_validation_metrics(self):
        """Publish all validation metrics."""
        # Stability
        stability = Float32()
        stability.data = self.calculate_stability()
        self.stability_pub.publish(stability)

        # Accuracy
        accuracy = Float32()
        accuracy.data = self.calculate_accuracy()
        self.accuracy_pub.publish(accuracy)

        # Smoothness
        smoothness = Float32()
        smoothness.data = self.calculate_smoothness()
        self.smoothness_pub.publish(smoothness)

        # Response time
        response_time = Float32()
        response_time.data = self.calculate_response_time()
        self.response_time_pub.publish(response_time)

        # Tracking error
        error = Float32()
        error.data = self.calculate_tracking_error()
        self.error_pub.publish(error)

        # Overall status
        status = Bool()
        status.data = (stability.data >= self.stability_threshold and
                      accuracy.data >= self.accuracy_threshold and
                      smoothness.data >= self.smoothness_threshold)
        self.status_pub.publish(status)

        # Log metrics periodically
        current_time = time.time()
        if int(current_time - self.start_time) % 5 == 0:  # Every 5 seconds
            self.get_logger().info(
                f'Control Validation - '
                f'Stability: {stability.data:.3f}, '
                f'Accuracy: {accuracy.data:.3f}, '
                f'Smoothness: {smoothness.data:.3f}, '
                f'Error: {error.data:.3f}, '
                f'Status: {"GOOD" if status.data else "POOR"}'
            )

    def validation_timer_callback(self):
        """Timer callback for validation."""
        self.publish_validation_metrics()

def main(args=None):
    rclpy.init(args=args)
    validation_node = ControlValidation()

    # Add a timer for validation metrics
    validation_node.validation_timer = validation_node.create_timer(1.0, validation_node.validation_timer_callback)

    try:
        rclpy.spin(validation_node)
    except KeyboardInterrupt:
        validation_node.get_logger().info('Shutting down Control Validation')
    finally:
        validation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive control systems for robotics:

- **PID Control**: Fundamental proportional-integral-derivative control systems
- **Trajectory Following**: Pure pursuit and path following algorithms
- **Model Predictive Control**: Advanced control with prediction and optimization
- **Adaptive Control**: Systems that adjust parameters based on conditions
- **Learning-Based Control**: Reinforcement learning approaches for control
- **Perception-Action Integration**: Combining perception and control systems
- **Control Validation**: Measuring and validating control system performance

These control systems form the backbone of autonomous robotic behavior, enabling robots to execute complex tasks with precision and adaptability.

## Exercises

1. Implement a PID controller for a mobile robot
2. Create a trajectory following system using pure pursuit
3. Develop an MPC controller for path following
4. Implement adaptive control parameters based on environmental conditions
5. Validate your control system with appropriate performance metrics

## Quiz

1. What does the "P" in PID control stand for?
   a) Proportional
   b) Predictive
   c) Performance
   d) Position

2. What is the main advantage of Model Predictive Control (MPC)?
   a) Simplicity
   b) Optimization over prediction horizon
   c) Lower computational cost
   d) No need for system model

3. What does "exploration vs exploitation" refer to in reinforcement learning?
   a) Hardware vs software solutions
   b) Balancing trying new actions vs using known good actions
   c) High-level vs low-level control
   d) Sensor vs actuator trade-offs

## Mini-Project: Integrated Control System

Create a complete control system with:
1. Multiple control strategies (PID, trajectory following, adaptive)
2. Integration with perception systems for closed-loop control
3. Learning components for improved performance over time
4. Comprehensive validation and performance monitoring
5. Smooth transitions between different control modes