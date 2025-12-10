---
sidebar_position: 4
---

# Robot Modeling and Simulation

## Learning Objectives

By the end of this chapter, you will be able to:
- Create detailed 3D robot models for simulation environments
- Implement realistic physics properties for robot models
- Simulate robot kinematics and dynamics accurately
- Integrate robot models with control systems in simulation
- Validate robot models for both Gazebo and Unity environments

## Introduction to Robot Modeling

Robot modeling is the process of creating digital representations of physical robots that accurately simulate their behavior in virtual environments. This involves creating both visual models for rendering and physical models for physics simulation.

### Key Components of Robot Models

```
Robot Model Components:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Visual Model  │    │   Physical      │    │   Kinematic     │
│   (Geometry,    │    │   Model         │    │   Model         │
│   Materials)    │    │   (Mass, Inertia│    │   (Joints,      │
│                 │    │   , Collisions) │    │   Transforms)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Control       │
                         │   Interfaces    │
                         │   (ROS Topics)  │
                         └─────────────────┘
```

### Model Fidelity Levels

Different applications require different levels of model fidelity:

- **Visual Fidelity**: Appearance and rendering quality
- **Physical Fidelity**: Accurate mass, inertia, and collision properties
- **Kinematic Fidelity**: Accurate joint relationships and motion
- **Dynamic Fidelity**: Accurate force and torque responses

## 3D Modeling for Robotics

### Model Requirements

Robot models for simulation should meet these requirements:

1. **Accurate Dimensions**: Match physical robot dimensions exactly
2. **Proper Scaling**: Use correct units (typically meters)
3. **Clean Geometry**: Avoid overlapping meshes or non-manifold geometry
4. **Appropriate Detail**: Balance visual quality with performance
5. **Proper Hierarchy**: Organize links in kinematic chain order

### Modeling Best Practices

```xml
<!-- Example: Proper URDF structure for a simple robot -->
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.08"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel_link"/>
    <origin xyz="0 0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="left_wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel_link"/>
    <origin xyz="0 -0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="right_wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>
</robot>
```

### Modeling Tools Comparison

| Tool | Best For | Pros | Cons |
|------|----------|------|------|
| Blender | Complex geometry, textures | Free, powerful, extensible | Steep learning curve |
| Fusion 360 | CAD modeling | Accurate dimensions, parametric | Proprietary, subscription required |
| SolidWorks | Engineering precision | Industry standard, accurate | Expensive, complex |
| Maya/3ds Max | Animation, visual effects | Professional tools, powerful | Expensive, overkill for robotics |

## Physics Properties and Simulation

### Mass and Inertia

Accurate mass and inertia properties are crucial for realistic simulation:

```python
# Calculate_inertial_properties.py
import numpy as np

def calculate_cylinder_inertia(mass, radius, length):
    """
    Calculate inertia tensor for a cylinder
    Ixx = Iyy = (1/12) * m * (3*r² + h²)
    Izz = (1/2) * m * r²
    """
    ixx = iyy = (1/12) * mass * (3 * radius**2 + length**2)
    izz = (1/2) * mass * radius**2

    return {
        'ixx': ixx,
        'iyy': iyy,
        'izz': izz,
        'ixy': 0,
        'ixz': 0,
        'iyz': 0
    }

def calculate_box_inertia(mass, width, depth, height):
    """
    Calculate inertia tensor for a box
    Ixx = (1/12) * m * (h² + d²)
    Iyy = (1/12) * m * (w² + h²)
    Izz = (1/12) * m * (w² + d²)
    """
    ixx = (1/12) * mass * (height**2 + depth**2)
    iyy = (1/12) * mass * (width**2 + height**2)
    izz = (1/12) * mass * (width**2 + depth**2)

    return {
        'ixx': ixx,
        'iyy': iyy,
        'izz': izz,
        'ixy': 0,
        'ixz': 0,
        'iyz': 0
    }

# Example usage
cylinder_props = calculate_cylinder_inertia(5.0, 0.15, 0.1)
box_props = calculate_box_inertia(2.0, 0.5, 0.3, 0.2)

print("Cylinder inertia:", cylinder_props)
print("Box inertia:", box_props)
```

### Collision Properties

Collision models should be simplified for performance while maintaining accuracy:

```xml
<!-- Simplified collision model -->
<link name="complex_part">
  <!-- Visual model - detailed -->
  <visual>
    <geometry>
      <mesh filename="complex_detailed_model.stl"/>
    </geometry>
  </visual>

  <!-- Collision model - simplified -->
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.2"/>
    </geometry>
  </collision>

  <!-- Or use multiple simple shapes -->
  <collision>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
</link>
```

### Material Properties

Material properties affect physics simulation:

```xml
<!-- Material properties in SDF format -->
<material name="rubber_wheel">
  <pbr>
    <metal>
      <albedo_map>materials/textures/rubber_albedo.png</albedo_map>
      <normal_map>materials/textures/rubber_normal.png</normal_map>
      <metalness_map>materials/textures/rubber_metalness.png</metalness_map>
      <roughness_map>materials/textures/rubber_roughness.png</roughness_map>
    </metal>
  </pbr>
</material>

<!-- Physics properties -->
<collision name="wheel_collision">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>  <!-- Coefficient of friction -->
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
      <threshold>100000</threshold>
    </bounce>
  </surface>
</collision>
```

## Kinematic Modeling

### Forward Kinematics

Forward kinematics calculates the position of end-effectors based on joint angles:

```python
# forward_kinematics.py
import numpy as np

class DifferentialDriveKinematics:
    def __init__(self, wheel_radius, wheel_base):
        self.wheel_radius = wheel_radius  # meters
        self.wheel_base = wheel_base      # meters

    def forward_kinematics(self, left_wheel_vel, right_wheel_vel, dt):
        """
        Calculate robot velocity from wheel velocities
        """
        # Linear and angular velocity
        linear_vel = self.wheel_radius * (left_wheel_vel + right_wheel_vel) / 2.0
        angular_vel = self.wheel_radius * (right_wheel_vel - left_wheel_vel) / self.wheel_base

        return linear_vel, angular_vel

    def position_update(self, x, y, theta, linear_vel, angular_vel, dt):
        """
        Update robot position based on velocities
        """
        # Integrate velocities
        dx = linear_vel * np.cos(theta) * dt
        dy = linear_vel * np.sin(theta) * dt
        dtheta = angular_vel * dt

        # Update position
        x += dx
        y += dy
        theta += dtheta

        return x, y, theta

# Example usage
kinematics = DifferentialDriveKinematics(wheel_radius=0.05, wheel_base=0.3)
linear_vel, angular_vel = kinematics.forward_kinematics(1.0, 1.2, 0.1)
```

### Inverse Kinematics

Inverse kinematics calculates required joint angles to achieve desired end-effector positions:

```python
# inverse_kinematics.py
import numpy as np

class TwoLinkArmIK:
    def __init__(self, link1_length, link2_length):
        self.l1 = link1_length
        self.l2 = link2_length

    def inverse_kinematics(self, x, y):
        """
        Calculate joint angles for 2-link planar arm
        """
        # Distance from origin to target
        r = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > (self.l1 + self.l2):
            return None  # Target out of reach

        if r < abs(self.l1 - self.l2):
            return None  # Target too close

        # Calculate second joint angle
        cos_theta2 = (self.l1**2 + self.l2**2 - r**2) / (2 * self.l1 * self.l2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

        # Calculate first joint angle
        k1 = self.l1 + self.l2 * np.cos(theta2)
        k2 = self.l2 * np.sin(theta2)

        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return theta1, theta2

# Example usage
ik_solver = TwoLinkArmIK(link1_length=0.5, link2_length=0.4)
angles = ik_solver.inverse_kinematics(0.6, 0.3)
```

## Dynamic Simulation

### Rigid Body Dynamics

Understanding rigid body dynamics is essential for realistic simulation:

```python
# rigid_body_dynamics.py
import numpy as np

class RigidBodyDynamics:
    def __init__(self, mass, inertia_tensor):
        self.mass = mass
        self.inertia_tensor = np.array(inertia_tensor)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # quaternion
        self.angular_velocity = np.zeros(3)

    def update(self, forces, torques, dt):
        """
        Update rigid body state based on applied forces and torques
        """
        # Linear motion: F = ma => a = F/m
        linear_acceleration = forces / self.mass
        self.velocity += linear_acceleration * dt
        self.position += self.velocity * dt

        # Angular motion: τ = Iα => α = I^(-1)τ
        angular_acceleration = np.linalg.solve(self.inertia_tensor, torques)
        self.angular_velocity += angular_acceleration * dt

        # Update orientation using quaternion integration
        self.integrate_orientation(dt)

        return self.position, self.velocity, self.orientation, self.angular_velocity

    def integrate_orientation(self, dt):
        """
        Integrate orientation using quaternion mathematics
        """
        # Convert angular velocity to quaternion derivative
        omega_quat = np.array([0, *self.angular_velocity])
        orientation_quat = self.orientation

        # Quaternion derivative: dq/dt = 0.5 * q * ω
        quat_derivative = 0.5 * self.quaternion_multiply(orientation_quat, omega_quat)

        # Integrate
        new_orientation = orientation_quat + quat_derivative * dt

        # Normalize quaternion
        self.orientation = new_orientation / np.linalg.norm(new_orientation)

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])
```

## Control System Integration

### Joint Control in Simulation

Integrating control systems with simulation models:

```python
# joint_controller.py
import numpy as np

class JointController:
    def __init__(self, joint_name, kp=10.0, ki=0.0, kd=0.1):
        self.joint_name = joint_name
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        self.error_sum = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def compute_control(self, desired_position, current_position, dt):
        """
        Compute control effort using PID control
        """
        if self.previous_time is None:
            self.previous_time = 0.0

        # Calculate error
        error = desired_position - current_position

        # Calculate error derivative
        if dt > 0:
            error_derivative = (error - self.previous_error) / dt
        else:
            error_derivative = 0.0

        # Update error sum
        self.error_sum += error * dt

        # Calculate control output
        p_term = self.kp * error
        i_term = self.ki * self.error_sum
        d_term = self.kd * error_derivative

        control_output = p_term + i_term + d_term

        # Update previous values
        self.previous_error = error
        self.previous_time += dt

        return control_output

class RobotController:
    def __init__(self):
        self.joint_controllers = {
            'left_wheel': JointController('left_wheel', kp=5.0, ki=0.1, kd=0.05),
            'right_wheel': JointController('right_wheel', kp=5.0, ki=0.1, kd=0.05)
        }

    def compute_wheel_commands(self, desired_linear_vel, desired_angular_vel,
                              current_left_vel, current_right_vel, dt):
        """
        Compute wheel velocity commands from desired robot velocities
        """
        # Convert robot velocities to wheel velocities
        wheel_radius = 0.05  # meters
        wheel_base = 0.3     # meters

        left_wheel_cmd = (desired_linear_vel - desired_angular_vel * wheel_base / 2) / wheel_radius
        right_wheel_cmd = (desired_linear_vel + desired_angular_vel * wheel_base / 2) / wheel_radius

        # Apply PID control
        left_effort = self.joint_controllers['left_wheel'].compute_control(
            left_wheel_cmd, current_left_vel, dt
        )

        right_effort = self.joint_controllers['right_wheel'].compute_control(
            right_wheel_cmd, current_right_vel, dt
        )

        return left_effort, right_effort
```

## Gazebo-Specific Modeling

### Gazebo Plugins for Robot Models

```xml
<!-- Gazebo-specific plugins in URDF -->
<robot name="gazebo_robot">
  <!-- Include Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.1</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Sensor plugins -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_optical_frame</frame_name>
        <topic_name>camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Gazebo World Integration

```xml
<!-- Example world file with robot -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="robot_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include robot model -->
    <include>
      <uri>model://simple_robot</uri>
    </include>

    <!-- Position robot -->
    <model name="simple_robot">
      <pose>0 0 0.1 0 0 0</pose>
    </model>

    <!-- Add obstacles -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Unity-Specific Modeling

### Unity Robot Components

```csharp
// UnityRobotModel.cs
using UnityEngine;
using Unity.Robotics.Core;
using Unity.Robotics.ROSTCPConnector;
using geometry_msgs;

public class UnityRobotModel : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float wheelRadius = 0.05f;
    public float wheelBase = 0.3f;
    public float maxLinearSpeed = 1.0f;
    public float maxAngularSpeed = 1.0f;

    [Header("Joint References")]
    public Transform leftWheel;
    public Transform rightWheel;
    public Transform robotBody;

    [Header("ROS Settings")]
    public string cmdVelTopic = "/cmd_vel";

    private float leftWheelVelocity = 0.0f;
    private float rightWheelVelocity = 0.0f;
    private float linearVelocity = 0.0f;
    private float angularVelocity = 0.0f;

    private ISubscriber<geometry_msgs.msg.Twist> cmdVelSubscriber;

    void Start()
    {
        // Subscribe to ROS commands
        var ros = ROS2UnityComponent.Instance;
        if (ros != null)
        {
            cmdVelSubscriber = ros.ROS2Socket.subscribe<geometry_msgs.msg.Twist>(
                cmdVelTopic, ProcessVelocityCommand
            );
        }
    }

    void ProcessVelocityCommand(geometry_msgs.msg.Twist cmd)
    {
        // Convert robot velocity to wheel velocities
        float linear = Mathf.Clamp(cmd.linear.x, -maxLinearSpeed, maxLinearSpeed);
        float angular = Mathf.Clamp(cmd.angular.z, -maxAngularSpeed, maxAngularSpeed);

        leftWheelVelocity = (linear - angular * wheelBase / 2) / wheelRadius;
        rightWheelVelocity = (linear + angular * wheelBase / 2) / wheelRadius;

        linearVelocity = linear;
        angularVelocity = angular;
    }

    void Update()
    {
        // Update wheel rotations based on velocities
        if (leftWheel != null)
        {
            leftWheel.Rotate(Vector3.right, leftWheelVelocity * Time.deltaTime * Mathf.Rad2Deg);
        }

        if (rightWheel != null)
        {
            rightWheel.Rotate(Vector3.right, rightWheelVelocity * Time.deltaTime * Mathf.Rad2Deg);
        }

        // Update robot position (simplified)
        if (robotBody != null)
        {
            robotBody.Translate(Vector3.forward * linearVelocity * Time.deltaTime);
            robotBody.Rotate(Vector3.up, angularVelocity * Time.deltaTime * Mathf.Rad2Deg);
        }
    }
}
```

### Unity Physics Configuration

```csharp
// UnityRobotPhysics.cs
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class UnityRobotPhysics : MonoBehaviour
{
    [Header("Physics Configuration")]
    public float robotMass = 10.0f;
    public float wheelFriction = 0.8f;
    public float wheelRadius = 0.05f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.mass = robotMass;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

        // Configure wheel colliders if using Unity's wheel system
        ConfigureWheelColliders();
    }

    void ConfigureWheelColliders()
    {
        // Configure each wheel collider
        WheelCollider[] wheelColliders = GetComponentsInChildren<WheelCollider>();
        foreach (WheelCollider wheel in wheelColliders)
        {
            wheel.radius = wheelRadius;
            wheel.mass = 1.0f;
            wheel.wheelDampingRate = 0.25f;

            // Configure friction
            WheelFrictionCurve forwardFriction = wheel.forwardFriction;
            forwardFriction.extremumSlip = 1.0f;
            forwardFriction.extremumValue = wheelFriction;
            forwardFriction.asymptoteSlip = 2.0f;
            forwardFriction.asymptoteValue = wheelFriction * 0.5f;
            wheel.forwardFriction = forwardFriction;
        }
    }

    void FixedUpdate()
    {
        // Apply physics-based control
        ApplyPhysicsControl();
    }

    void ApplyPhysicsControl()
    {
        // This would be where you apply forces based on your control system
        // For a differential drive robot, you'd apply forces to simulate wheel torques
    }
}
```

## Validation and Testing

### Model Validation Checklist

- [ ] **Visual accuracy**: Model matches physical dimensions
- [ ] **Physics properties**: Mass and inertia values are correct
- [ ] **Collision geometry**: Properly configured for simulation
- [ ] **Kinematic chain**: Joints are properly connected
- [ ] **Control interfaces**: ROS topics/services are properly configured
- [ ] **Sensor placement**: Sensors are correctly positioned
- [ ] **Performance**: Model runs efficiently in simulation

### Testing Scripts

```python
# model_validation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

class ModelValidator(Node):
    def __init__(self):
        super().__init__('model_validator')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )

        # Validation data
        self.joint_positions = {}
        self.odom_data = None
        self.test_results = {}

    def joint_state_callback(self, msg):
        """Process joint state messages."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def odom_callback(self, msg):
        """Process odometry messages."""
        self.odom_data = msg

    def validate_model(self):
        """Run comprehensive model validation."""
        self.get_logger().info('Starting model validation...')

        # Test 1: Joint movement
        self.test_joint_movement()

        # Test 2: Robot motion
        self.test_robot_motion()

        # Test 3: Sensor data
        self.test_sensor_data()

        self.print_validation_results()

    def test_joint_movement(self):
        """Test that joints respond to commands."""
        self.get_logger().info('Testing joint movement...')

        # Send command to move joints
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward
        cmd.angular.z = 0.0

        # Publish command for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2.0:
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Check if joints moved
        initial_positions = self.joint_positions.copy()

        # Stop robot
        cmd.linear.x = 0.0
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Verify joint positions changed
        moved = False
        for joint, pos in self.joint_positions.items():
            if joint in initial_positions:
                if abs(pos - initial_positions[joint]) > 0.01:
                    moved = True
                    break

        self.test_results['joint_movement'] = moved
        self.get_logger().info(f'Joint movement test: {"PASS" if moved else "FAIL"}')

    def test_robot_motion(self):
        """Test that robot moves as expected."""
        self.get_logger().info('Testing robot motion...')

        # Record initial position
        initial_odom = self.odom_data
        if initial_odom is None:
            self.test_results['robot_motion'] = False
            self.get_logger().info('Robot motion test: FAIL (no odometry data)')
            return

        initial_pos = [
            initial_odom.pose.pose.position.x,
            initial_odom.pose.pose.position.y,
            initial_odom.pose.pose.position.z
        ]

        # Move robot
        cmd = Twist()
        cmd.linear.x = 0.5
        start_time = time.time()
        while time.time() - start_time < 2.0:
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stop robot
        cmd.linear.x = 0.0
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Check if position changed
        if self.odom_data is not None:
            final_pos = [
                self.odom_data.pose.pose.position.x,
                self.odom_data.pose.pose.position.y,
                self.odom_data.pose.pose.position.z
            ]

            distance_moved = sum([(f - i)**2 for f, i in zip(final_pos, initial_pos)])**0.5
            motion_valid = distance_moved > 0.1  # Moved at least 10cm

            self.test_results['robot_motion'] = motion_valid
            self.get_logger().info(f'Robot motion test: {"PASS" if motion_valid else "FAIL"}')
        else:
            self.test_results['robot_motion'] = False
            self.get_logger().info('Robot motion test: FAIL (no final odometry)')

    def print_validation_results(self):
        """Print validation results."""
        self.get_logger().info('Model validation results:')
        for test, result in self.test_results.items():
            status = 'PASS' if result else 'FAIL'
            self.get_logger().info(f'  {test}: {status}')

def main(args=None):
    rclpy.init(args=args)
    validator = ModelValidator()

    # Run validation
    validator.validate_model()

    validator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive robot modeling and simulation techniques:

- **3D modeling**: Creating accurate visual and collision models
- **Physics properties**: Setting up mass, inertia, and material properties
- **Kinematic modeling**: Forward and inverse kinematics for robot motion
- **Dynamic simulation**: Rigid body dynamics for realistic behavior
- **Control integration**: Connecting models with control systems
- **Environment-specific setup**: Gazebo and Unity integration
- **Validation**: Testing and verifying model accuracy

Proper robot modeling is essential for effective simulation, ensuring that behaviors learned in simulation transfer well to real robots.

## Exercises

1. Create a URDF model of a simple differential drive robot
2. Implement forward kinematics for your robot model
3. Validate your model in both Gazebo and Unity
4. Add sensors to your robot model and test their outputs

## Quiz

1. What is the primary purpose of collision geometry in robot models?
   a) To make the robot look realistic
   b) To define how the robot interacts with the environment in physics simulation
   c) To store sensor data
   d) To control robot movement

2. What does the inertia tensor represent in a robot model?
   a) The robot's visual appearance
   b) How mass is distributed and affects rotational motion
   c) The robot's maximum speed
   d) The robot's sensor configuration

3. What is the difference between forward and inverse kinematics?
   a) Forward calculates joint angles from end-effector position, inverse does the opposite
   b) Forward is for mobile robots, inverse is for manipulators
   c) Inverse calculates joint angles from end-effector position, forward does the opposite
   d) There is no difference

## Mini-Project: Complete Robot Model

Create a complete robot model with:
1. Accurate 3D geometry with proper mass and inertia properties
2. Differential drive kinematics implementation
3. Integration with ROS 2 control systems
4. Sensor simulation (camera and LiDAR)
5. Validation tests to verify model accuracy
6. Documentation of the modeling process and validation results