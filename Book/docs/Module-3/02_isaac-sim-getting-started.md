---
sidebar_position: 2
---

# Isaac Sim Getting Started

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure Isaac Sim on your development system
- Create and manipulate objects in the Isaac Sim environment
- Set up realistic sensor configurations for robotic applications
- Implement basic robot control within Isaac Sim
- Integrate Isaac Sim with ROS 2 for robotics workflows

## Installing Isaac Sim

Isaac Sim can be installed through multiple methods depending on your development environment and requirements.

### Prerequisites

Before installing Isaac Sim, ensure you have:

- **NVIDIA GPU**: Compatible with CUDA (GTX 1060 or better recommended)
- **Graphics Drivers**: Latest NVIDIA drivers (470.63.01 or newer)
- **CUDA Toolkit**: Version 11.8 or 12.x
- **Operating System**: Ubuntu 20.04/22.04 or Windows 10/11
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB free space minimum

### Installation Methods

#### Method 1: Omniverse Launcher (Recommended)

1. **Download Omniverse Launcher**:
   - Visit https://developer.nvidia.com/omniverse-downloads
   - Download and install the Omniverse Launcher

2. **Install Isaac Sim**:
   - Open Omniverse Launcher
   - Go to "Assets" tab
   - Search for "Isaac Sim"
   - Click "Install" to download and install Isaac Sim

3. **Launch Isaac Sim**:
   - Go to "My Assets" tab
   - Find Isaac Sim and click "Launch"

#### Method 2: Docker Installation

```bash
# Pull the Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:latest

# Run Isaac Sim container with GPU support
docker run --gpus all -it --rm \
  --network=host \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume=$HOME/.Xauthority:/root/.Xauthority:rw \
  --volume=$PWD:/workspace \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --name isaac_sim \
  nvcr.io/nvidia/isaac-sim:latest
```

#### Method 3: Native Installation (Linux)

```bash
# Download Isaac Sim from NVIDIA Developer website
# Follow the installation instructions for your platform
# The installer will handle dependencies and setup
```

### Verification of Installation

After installation, verify Isaac Sim is working:

```python
# test_isaac_sim.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import is_prim_path_valid
import carb

def test_isaac_sim():
    """Test basic Isaac Sim functionality."""
    print("Testing Isaac Sim installation...")

    try:
        # Initialize Isaac Sim world
        world = World(stage_units_in_meters=1.0)
        print("✓ World initialized successfully")

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path:
            print(f"✓ Assets root path: {assets_root_path}")
        else:
            print("✗ Could not find assets root path")
            return False

        # Add a simple object to the stage
        object_path = "/World/Box"
        from omni.isaac.core.objects import DynamicCuboid
        world.scene.add(
            DynamicCuboid(
                prim_path=object_path,
                name="test_box",
                position=[0, 0, 1.0],
                size=0.1
            )
        )
        print("✓ Object added to stage successfully")

        # Reset the world
        world.reset()
        print("✓ World reset successfully")

        # Step the simulation
        world.step(render=True)
        print("✓ Simulation step completed successfully")

        # Clean up
        world.clear()
        print("✓ Isaac Sim test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Isaac Sim test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_isaac_sim()
    if success:
        print("\nIsaac Sim installation verified successfully!")
    else:
        print("\nIsaac Sim installation verification failed!")
```

## Basic Scene Setup

### Creating Your First Scene

Let's create a basic scene with a ground plane and a simple robot:

```python
# basic_scene.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.viewports import set_viewport_camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import OmniGlass, OmniPBR
import numpy as np

class BasicScene:
    def __init__(self):
        # Initialize the world with 1-meter stage units
        self.world = World(stage_units_in_meters=1.0)

        # Set up the stage
        self.setup_stage()

        # Add objects to the scene
        self.add_objects()

        # Setup camera view
        self.setup_camera()

    def setup_stage(self):
        """Set up the basic stage with ground plane."""
        # Add default ground plane
        self.world.scene.add_default_ground_plane()

        # Set basic physics parameters
        self.world.set_settings(
            enable_scene_query_support=True,
            enable_rendering_graph_pruning=True,
            enable_gyroscopic_forces=True,
            default_physics_dt=1.0/60.0,
            use_gpu_pipeline=True  # Enable GPU physics if available
        )

        print("Stage setup completed")

    def add_objects(self):
        """Add objects to the scene."""
        # Add a simple robot (using a basic cube as placeholder)
        robot_path = "/World/Robot"
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path=robot_path,
                name="simple_robot",
                position=[0.0, 0.0, 0.5],
                size=0.2,
                color=np.array([0.8, 0.2, 0.2])
            )
        )

        # Add some obstacles
        obstacle1 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Obstacle1",
                name="obstacle1",
                position=[1.0, 0.0, 0.2],
                size=0.3,
                color=np.array([0.2, 0.2, 0.8])
            )
        )

        obstacle2 = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Obstacle2",
                name="obstacle2",
                position=[-1.0, 0.5, 0.15],
                size=[0.4, 0.2, 0.3],
                color=np.array([0.2, 0.8, 0.2])
            )
        )

        # Add a glass object to demonstrate materials
        glass_material = OmniGlass(
            prim_path="/World/Looks/glass_material",
            dynamic_friction=0.4,
            glass_color=np.array([0.8, 0.9, 1.0])
        )

        glass_object = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/GlassObject",
                name="glass_object",
                position=[0.0, 1.0, 0.5],
                size=0.15,
                color=np.array([0.8, 0.9, 1.0]),
                mass=0.1
            )
        )

        print("Objects added to scene")

    def setup_camera(self):
        """Set up the viewport camera."""
        # Set the camera position to view the scene well
        set_viewport_camera(
            "/OmniverseKit_Persp",
            np.array([3.0, 3.0, 2.0]),
            np.array([0.0, 0.0, 0.5])
        )

        print("Camera setup completed")

    def run_simulation(self, steps=1000):
        """Run the simulation for a specified number of steps."""
        print(f"Running simulation for {steps} steps...")

        self.world.reset()

        for i in range(steps):
            if i % 100 == 0:
                print(f"Step {i}/{steps}")

            # Step the world
            self.world.step(render=True)

            # Add simple robot movement after some steps
            if i > 100:
                # Move the robot in a simple pattern
                current_pos = self.robot.get_world_pose()[0]
                new_x = current_pos[0] + 0.001 * np.sin(i * 0.01)
                new_y = current_pos[1] + 0.001 * np.cos(i * 0.01)
                self.robot.set_world_pose(position=np.array([new_x, new_y, 0.5]))

        print("Simulation completed")

    def cleanup(self):
        """Clean up the scene."""
        self.world.clear()
        print("Scene cleaned up")

def main():
    # Create and run the basic scene
    scene = BasicScene()

    try:
        scene.run_simulation(steps=500)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        scene.cleanup()

if __name__ == "__main__":
    main()
```

## Robot Setup in Isaac Sim

### Using Pre-built Robot Models

Isaac Sim comes with several pre-built robot models that you can use:

```python
# robot_setup.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

class RobotSetup:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.setup_scene()

    def setup_scene(self):
        """Set up the scene with a robot."""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if not assets_root_path:
            print("Could not find assets root path")
            return

        # Add a pre-built robot (using Carter robot as example)
        robot_path = "/World/Carter"
        try:
            self.robot = self.world.scene.add(
                Robot(
                    prim_path=robot_path,
                    name="carter_robot",
                    usd_path=f"{assets_root_path}/Isaac/Robots/Carter/carter_navigation.usd",
                    position=[0.0, 0.0, 0.1]
                )
            )
            print("Carter robot loaded successfully")
        except Exception as e:
            print(f"Could not load Carter robot: {e}")
            # Fallback to a simple cube robot
            from omni.isaac.core.objects import DynamicCuboid
            self.robot = self.world.scene.add(
                DynamicCuboid(
                    prim_path=robot_path,
                    name="simple_robot",
                    position=[0.0, 0.0, 0.2],
                    size=0.3
                )
            )
            print("Fallback simple robot created")

    def control_robot(self):
        """Example robot control."""
        if self.robot is None:
            return

        # For a differential drive robot, we can control wheel velocities
        # This is a simplified example - actual control would depend on the robot type
        try:
            # Reset the world first
            self.world.reset()

            for i in range(1000):
                # Step the simulation
                self.world.step(render=True)

                # Apply simple control after initial steps
                if i > 50:
                    # For a real robot with joints, you would use:
                    # self.robot.apply_articulation_actions(
                    #     ArticulationAction(joint_positions=[...],
                    #                        joint_velocities=[...])
                    # )

                    # For our simple cube, just move it
                    current_pos = self.robot.get_world_pose()[0]
                    new_x = current_pos[0] + 0.001
                    new_y = current_pos[1] + 0.0005 * np.sin(i * 0.02)
                    self.robot.set_world_pose(position=np.array([new_x, new_y, 0.2]))

                if i % 200 == 0:
                    print(f"Robot position: {self.robot.get_world_pose()[0]}")

        except Exception as e:
            print(f"Error controlling robot: {e}")

    def cleanup(self):
        """Clean up the scene."""
        self.world.clear()

def main():
    robot_setup = RobotSetup()

    try:
        robot_setup.control_robot()
    except KeyboardInterrupt:
        print("Robot control interrupted by user")
    finally:
        robot_setup.cleanup()

if __name__ == "__main__":
    main()
```

## Sensor Configuration

### Adding Sensors to Robots

Isaac Sim provides realistic sensor simulation. Here's how to add and configure sensors:

```python
# sensor_setup.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera, LidarRtx
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, Sdf, UsdGeom
import numpy as np

class SensorSetup:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.camera = None
        self.lidar = None
        self.setup_scene_with_sensors()

    def setup_scene_with_sensors(self):
        """Set up scene with robot and sensors."""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if not assets_root_path:
            print("Could not find assets root path")
            return

        # Add a simple robot platform
        robot_path = "/World/Robot"
        from omni.isaac.core.objects import DynamicCuboid
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path=robot_path,
                name="sensor_robot",
                position=[0.0, 0.0, 0.2],
                size=0.3,
                color=np.array([0.8, 0.2, 0.2])
            )
        )

        # Add a sensor mast on top of the robot
        sensor_mast_path = f"{robot_path}/SensorMast"
        sensor_mast = self.world.scene.add(
            DynamicCuboid(
                prim_path=sensor_mast_path,
                name="sensor_mast",
                position=[0.0, 0.0, 0.3],
                size=[0.05, 0.05, 0.2],
                color=np.array([0.5, 0.5, 0.5])
            )
        )

        # Add a camera sensor
        self.setup_camera_sensor(sensor_mast_path)

        # Add a LiDAR sensor
        self.setup_lidar_sensor(sensor_mast_path)

    def setup_camera_sensor(self, parent_path):
        """Set up a camera sensor on the robot."""
        try:
            # Create camera sensor
            camera_path = f"{parent_path}/Camera"

            # Add camera prim to stage first
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.DefinePrim(camera_path, "Camera")

            # Set camera properties
            camera_prim.GetAttribute("focalLength").Set(24.0)
            camera_prim.GetAttribute("horizontalAperture").Set(36.0)
            camera_prim.GetAttribute("verticalAperture").Set(20.25)

            # Create Isaac camera sensor
            from omni.isaac.core.utils.viewports import create_viewport_window
            from omni.kit.viewport.utility import get_active_viewport

            # The camera will be used for RGB and depth data
            print("Camera sensor configured")

        except Exception as e:
            print(f"Could not set up camera sensor: {e}")

    def setup_lidar_sensor(self, parent_path):
        """Set up a LiDAR sensor on the robot."""
        try:
            # Define LiDAR parameters
            lidar_config = {
                "rotation_frequency": 10,  # Hz
                "number_of_channels": 16,
                "points_per_channel": 1875,
                "horizontal_resolution": 3,  # degrees
                "vertical_resolution": 2,    # degrees
                "horizontal_min_angle": -np.pi,   # -180 degrees
                "horizontal_max_angle": np.pi,    # 180 degrees
                "range_min": 0.1,    # meters
                "range_max": 25.0,   # meters
                "return_mode": "closest"  # closest, strongest, all
            }

            lidar_path = f"{parent_path}/Lidar"

            # Create LiDAR sensor (using Isaac Sim's built-in LiDAR)
            # Note: This is a simplified example; actual implementation
            # would depend on the specific LiDAR model being simulated
            print("LiDAR sensor configured with parameters:")
            for key, value in lidar_config.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"Could not set up LiDAR sensor: {e}")

    def run_sensor_simulation(self, steps=500):
        """Run simulation with sensors."""
        print("Starting sensor simulation...")

        self.world.reset()

        for i in range(steps):
            if i % 100 == 0:
                print(f"Sensor simulation step: {i}/{steps}")

            # Step the world
            self.world.step(render=True)

            # Move the robot in a pattern to generate sensor data
            if self.robot and i > 50:
                current_pos = self.robot.get_world_pose()[0]
                new_x = current_pos[0] + 0.002 * np.cos(i * 0.01)
                new_y = current_pos[1] + 0.002 * np.sin(i * 0.01)
                self.robot.set_world_pose(position=np.array([new_x, new_y, 0.2]))

                # In a real implementation, you would:
                # - Read sensor data
                # - Process the data
                # - Use it for navigation/perception tasks

                if i % 50 == 0:
                    robot_pos = self.robot.get_world_pose()[0]
                    print(f"Robot at position: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}]")

    def cleanup(self):
        """Clean up the scene."""
        self.world.clear()
        print("Sensor setup cleaned up")

def main():
    sensor_setup = SensorSetup()

    try:
        sensor_setup.run_sensor_simulation()
    except KeyboardInterrupt:
        print("Sensor simulation interrupted by user")
    finally:
        sensor_setup.cleanup()

if __name__ == "__main__":
    main()
```

## ROS 2 Integration

### Connecting Isaac Sim to ROS 2

Isaac Sim can be integrated with ROS 2 through the Isaac ROS bridge, allowing you to control robots and process sensor data using standard ROS 2 tools:

```python
# isaac_ros_integration.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import get_prim_at_path
import carb
import numpy as np

# Import ROS 2 components
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import LaserScan, Image, CameraInfo
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    print("ROS 2 Python libraries not available. Install ROS 2 Humble to enable ROS integration.")
    ROS_AVAILABLE = False

class IsaacROSBridge(Node if ROS_AVAILABLE else object):
    def __init__(self):
        if ROS_AVAILABLE:
            super().__init__('isaac_ros_bridge')

            # Create ROS publishers and subscribers
            self.cmd_vel_sub = self.create_subscription(
                Twist, 'cmd_vel', self.cmd_vel_callback, 10
            )
            self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
            self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)
            self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
            self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

            # Timer for publishing sensor data
            self.pub_timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz

            # Robot control variables
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
        else:
            print("Initializing without ROS functionality")

        # Isaac Sim components
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.setup_isaac_world()

    def setup_isaac_world(self):
        """Set up Isaac Sim world."""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add a simple robot
        robot_path = "/World/ROSRobot"
        from omni.isaac.core.objects import DynamicCuboid
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path=robot_path,
                name="ros_robot",
                position=[0.0, 0.0, 0.2],
                size=0.3,
                color=np.array([0.2, 0.8, 0.2])
            )
        )

        print("Isaac Sim world with ROS bridge set up")

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS."""
        if ROS_AVAILABLE:
            self.linear_velocity = msg.linear.x
            self.angular_velocity = msg.angular.z
            self.get_logger().debug(f"Received cmd_vel: linear={self.linear_velocity}, angular={self.angular_velocity}")

    def publish_sensor_data(self):
        """Publish sensor data to ROS topics."""
        if not ROS_AVAILABLE or not self.robot:
            return

        # Get robot pose and publish as odometry
        position, orientation = self.robot.get_world_pose()
        linear_vel, angular_vel = self.robot.get_linear_velocity(), self.robot.get_angular_velocity()

        # Create and publish odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]

        # Convert orientation quaternion (simplified)
        odom_msg.pose.pose.orientation.x = orientation[0]
        odom_msg.pose.pose.orientation.y = orientation[1]
        odom_msg.pose.pose.orientation.z = orientation[2]
        odom_msg.pose.pose.orientation.w = orientation[3]

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Create and publish laser scan (simulated)
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 2 * np.pi / 360  # 360 points
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 30.0

        # Simulate some range data (in a real implementation, this would come from a LiDAR sensor)
        scan_msg.ranges = [25.0] * 360  # No obstacles detected

        # Add some simulated obstacles
        for i in range(45, 135):  # Front right
            scan_msg.ranges[i] = 2.0
        for i in range(225, 315):  # Back left
            scan_msg.ranges[i] = 1.5

        self.scan_pub.publish(scan_msg)

    def run_simulation(self, steps=1000):
        """Run the simulation loop."""
        print("Starting Isaac Sim with ROS bridge...")

        self.world.reset()

        for i in range(steps):
            if i % 100 == 0:
                print(f"ROS bridge simulation step: {i}/{steps}")

            # Step Isaac Sim
            self.world.step(render=True)

            # Apply robot control based on ROS commands
            if self.robot and i > 50:
                # Simple differential drive kinematics
                current_pos = self.robot.get_world_pose()[0]
                current_rot = self.robot.get_world_pose()[1]  # This is a simplified approach

                # Update position based on velocity commands
                # In a real implementation, this would use proper kinematics
                delta_x = self.linear_velocity * 0.01  # Scale factor
                delta_y = 0  # For simplicity, only move in x direction
                new_x = current_pos[0] + delta_x
                new_y = current_pos[1] + delta_y

                # Apply angular velocity effect
                # This is a simplified approach - proper rotation would require quaternion math
                self.robot.set_world_pose(position=np.array([new_x, new_y, 0.2]))

            # Process ROS callbacks
            if ROS_AVAILABLE:
                rclpy.spin_once(self, timeout_sec=0)

    def cleanup(self):
        """Clean up resources."""
        self.world.clear()
        if ROS_AVAILABLE:
            self.destroy_node()
        print("Isaac ROS bridge cleaned up")

def main():
    if ROS_AVAILABLE:
        rclpy.init()

    bridge = IsaacROSBridge()

    try:
        bridge.run_simulation()
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        bridge.cleanup()
        if ROS_AVAILABLE:
            rclpy.shutdown()

if __name__ == "__main__":
    main()
```

## Advanced Isaac Sim Features

### Domain Randomization for AI Training

Domain randomization is a key feature of Isaac Sim for training robust AI models:

```python
# domain_randomization.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import random

class DomainRandomization:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.objects = []
        self.materials = []
        self.setup_randomized_environment()

    def setup_randomized_environment(self):
        """Set up an environment with randomized properties."""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Create random materials
        self.create_random_materials()

        # Add randomized objects
        self.add_randomized_objects()

        print("Domain randomized environment created")

    def create_random_materials(self):
        """Create materials with randomized properties."""
        for i in range(10):
            # Random color
            color = np.array([
                random.uniform(0.1, 1.0),
                random.uniform(0.1, 1.0),
                random.uniform(0.1, 1.0)
            ])

            # Random roughness
            roughness = random.uniform(0.1, 0.9)

            # Random metallic
            metallic = random.uniform(0.0, 1.0)

            material_name = f"random_material_{i}"
            material_path = f"/World/Looks/{material_name}"

            try:
                material = OmniPBR(
                    prim_path=material_path,
                    color=color,
                    roughness=roughness,
                    metallic=metallic
                )
                self.materials.append(material)
            except Exception as e:
                print(f"Could not create material {material_name}: {e}")

    def add_randomized_objects(self):
        """Add objects with randomized properties."""
        for i in range(20):
            # Random position
            x = random.uniform(-3.0, 3.0)
            y = random.uniform(-3.0, 3.0)
            z = random.uniform(0.2, 2.0)

            # Random size
            size = random.uniform(0.1, 0.5)

            # Random color from our materials or basic color
            if self.materials:
                color_idx = random.randint(0, len(self.materials)-1)
                color = self.materials[color_idx].color
            else:
                color = np.array([
                    random.uniform(0.1, 1.0),
                    random.uniform(0.1, 1.0),
                    random.uniform(0.1, 1.0)
                ])

            # Random shape (simplified - using cubes with different properties)
            object_path = f"/World/Object_{i}"

            # Randomly choose between dynamic and fixed objects
            if random.choice([True, False]):
                obj = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=object_path,
                        name=f"dynamic_obj_{i}",
                        position=[x, y, z],
                        size=size,
                        color=color
                    )
                )
            else:
                obj = self.world.scene.add(
                    VisualCuboid(
                        prim_path=object_path,
                        name=f"visual_obj_{i}",
                        position=[x, y, z],
                        size=size,
                        color=color
                    )
                )

            self.objects.append(obj)

    def randomize_environment(self):
        """Randomize the environment for the next training episode."""
        print("Randomizing environment...")

        # Move objects to new random positions
        for i, obj in enumerate(self.objects):
            # Random new position
            x = random.uniform(-3.0, 3.0)
            y = random.uniform(-3.0, 3.0)
            z = random.uniform(0.2, 2.0)

            # Random new size
            new_size = random.uniform(0.1, 0.5)

            # For simplicity, just move the object
            # In a real implementation, you'd update all properties
            obj.set_world_pose(position=np.array([x, y, z]))

        # Randomize lighting
        self.randomize_lighting()

        # Randomize camera properties if applicable
        self.randomize_camera_properties()

    def randomize_lighting(self):
        """Randomize lighting conditions."""
        # In a real implementation, you would modify light sources
        # This is a placeholder for the concept
        print("Lighting randomized")

    def randomize_camera_properties(self):
        """Randomize camera properties for training."""
        # This would modify camera parameters like exposure, noise, etc.
        print("Camera properties randomized")

    def run_training_episode(self, steps=500):
        """Run a training episode with domain randomization."""
        print(f"Starting training episode with {len(self.objects)} objects...")

        self.world.reset()

        for i in range(steps):
            if i % 100 == 0:
                print(f"Training step: {i}/{steps}")

            # Step the simulation
            self.world.step(render=True)

            # At random intervals, randomize the environment
            if i > 0 and i % 200 == 0:  # Randomize every 200 steps
                self.randomize_environment()

    def cleanup(self):
        """Clean up the environment."""
        self.world.clear()
        print("Domain randomization environment cleaned up")

def main():
    dr = DomainRandomization()

    try:
        # Run multiple episodes with domain randomization
        for episode in range(3):
            print(f"\nStarting training episode {episode + 1}/3")
            dr.run_training_episode(steps=300)
            dr.randomize_environment()  # Randomize between episodes
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        dr.cleanup()

if __name__ == "__main__":
    main()
```

## Performance Optimization

### Optimizing Isaac Sim Performance

To get the best performance from Isaac Sim, consider these optimization techniques:

```python
# performance_optimization.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.settings import set_carb_setting
import carb
import numpy as np

class PerformanceOptimization:
    def __init__(self):
        # Set performance-related settings before creating world
        self.setup_performance_settings()

        # Initialize world with performance optimizations
        self.world = World(
            stage_units_in_meters=1.0,
            rendering_dt=1.0/60.0,  # Rendering rate
            physics_dt=1.0/60.0,    # Physics rate
            stage_enable_lights=False  # Disable lights for performance
        )

        self.setup_optimized_scene()

    def setup_performance_settings(self):
        """Configure performance-related settings."""
        # Set rendering quality settings
        set_carb_setting("/app/profiling/captureOnExit", False)
        set_carb_setting("/app/profiling/maxGpuGraphs", 100)
        set_carb_setting("/app/profiling/maxCpuGraphs", 100)

        # Physics settings for performance
        set_carb_setting("/physics/solverType", "TGS")  # Use TGS solver
        set_carb_setting("/physics/broadphaseType", "MBP")  # Multi-box pruning

        # Rendering settings
        set_carb_setting("/rtx/sceneDb/enableUpdate", True)
        set_carb_setting("/rtx/sceneDb/enableUpdateTask", True)

        print("Performance settings configured")

    def setup_optimized_scene(self):
        """Set up a scene optimized for performance."""
        # Add minimal ground plane
        self.world.scene.add_default_ground_plane()

        # Add a simple robot
        robot_path = "/World/PerformanceRobot"
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path=robot_path,
                name="perf_robot",
                position=[0.0, 0.0, 0.2],
                size=0.3,
                color=np.array([0.2, 0.6, 1.0])
            )
        )

        # Use simple shapes instead of complex meshes
        # Limit number of objects in scene
        # Use appropriate collision geometries

        print("Optimized scene created")

    def run_optimized_simulation(self, steps=1000):
        """Run simulation with performance optimizations."""
        print(f"Starting optimized simulation for {steps} steps...")

        # Reset world
        self.world.reset()

        # Performance monitoring
        import time
        start_time = time.time()

        for i in range(steps):
            step_start = time.time()

            # Step the simulation
            self.world.step(render=True)

            # Move robot in a simple pattern
            if i > 50:
                current_pos = self.robot.get_world_pose()[0]
                new_x = current_pos[0] + 0.001
                self.robot.set_world_pose(position=np.array([new_x, current_pos[1], 0.2]))

            # Performance logging
            if i % 200 == 0:
                step_time = time.time() - step_start
                avg_time = (time.time() - start_time) / (i + 1)
                print(f"Step {i}: {step_time:.4f}s per step, avg: {avg_time:.4f}s")

        total_time = time.time() - start_time
        avg_step_time = total_time / steps
        print(f"Simulation completed in {total_time:.2f}s ({avg_step_time:.4f}s per step, {1/avg_step_time:.1f} FPS)")

    def cleanup(self):
        """Clean up resources."""
        self.world.clear()
        print("Performance optimization test cleaned up")

def main():
    perf_test = PerformanceOptimization()

    try:
        perf_test.run_optimized_simulation()
    except KeyboardInterrupt:
        print("Performance test interrupted by user")
    finally:
        perf_test.cleanup()

if __name__ == "__main__":
    main()
```

## Summary

This chapter covered getting started with Isaac Sim:

- **Installation**: Multiple methods for setting up Isaac Sim
- **Basic Scene Setup**: Creating and configuring simulation environments
- **Robot Integration**: Adding and controlling robots in simulation
- **Sensor Configuration**: Setting up realistic sensors for robotics
- **ROS Integration**: Connecting Isaac Sim with ROS 2 workflows
- **Advanced Features**: Domain randomization for AI training
- **Performance Optimization**: Techniques for efficient simulation

Isaac Sim provides a powerful platform for robotics development with realistic physics, rendering, and sensor simulation capabilities.

## Exercises

1. Install Isaac Sim and run the basic scene example
2. Add a pre-built robot model to your simulation
3. Configure camera and LiDAR sensors on your robot
4. Implement basic robot control using keyboard input
5. Experiment with domain randomization techniques

## Quiz

1. What is the recommended minimum GPU for Isaac Sim?
   a) GTX 1050
   b) GTX 1060 or better
   c) RTX 2060
   d) Any GPU with OpenGL support

2. Which of these is NOT a method for installing Isaac Sim?
   a) Omniverse Launcher
   b) Docker
   c) Native installation
   d) PyPI package manager

3. What is domain randomization used for in Isaac Sim?
   a) Reducing simulation costs
   b) Training more robust AI models
   c) Improving graphics quality
   d) Increasing physics accuracy

## Mini-Project: Complete Robot Simulation

Create a complete robot simulation in Isaac Sim with:
1. A robot model with realistic physics properties
2. Camera and LiDAR sensors configured properly
3. ROS 2 integration for control and sensor data
4. Basic navigation or manipulation task
5. Performance optimization for real-time simulation
6. Documentation of your implementation process