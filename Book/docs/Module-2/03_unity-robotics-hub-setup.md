---
sidebar_position: 3
---

# Unity Robotics Hub Setup

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure Unity Hub and Unity Editor for robotics applications
- Set up the Unity Robotics Hub and ML-Agents toolkit
- Integrate Unity with ROS 2 using the ROS TCP Connector
- Create basic robotic environments in Unity
- Implement sensor simulation in Unity for robotics applications

## Introduction to Unity for Robotics

Unity has emerged as a powerful platform for robotics simulation, offering photorealistic rendering, advanced physics simulation, and flexible environment creation capabilities. The Unity Robotics Hub provides the necessary tools and packages to bridge Unity with ROS 2, enabling realistic sensor simulation and robot control.

### Why Unity for Robotics?

Unity offers several advantages for robotics simulation:

- **Photorealistic Graphics**: High-fidelity rendering for realistic sensor simulation
- **Flexible Environment Design**: Powerful tools for creating complex, varied environments
- **Physics Simulation**: Accurate physics with realistic material properties
- **Machine Learning Integration**: ML-Agents toolkit for training AI systems
- **Cross-Platform Deployment**: Deploy to various platforms including VR/AR
- **Asset Store**: Extensive library of 3D models and environments
- **Active Development**: Continuous updates and improvements

### Unity vs Traditional Robotics Simulators

```
Unity Advantages:
┌─────────────────┐    ┌─────────────────┐
│   Graphics      │    │   Environment   │
│   Quality       │    │   Flexibility   │
│   • Photoreal   │    │   • Visual      │
│   • Lighting    │    │   • Procedural  │
│   • Materials   │    │   • Scripted    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
              Robotics Applications
              (Sensor Simulation, Training)
```

## Installing Unity Hub and Unity Editor

### Prerequisites

Before installing Unity, ensure your system meets the requirements:

- **Operating System**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: 20GB+ free space for Unity installation
- **Graphics**: DirectX 10, OpenGL 3.3, or Vulkan compatible GPU

### Installing Unity Hub

Unity Hub is the recommended way to manage Unity installations:

1. **Download Unity Hub**:
   - Visit https://unity.com/download
   - Download Unity Hub for your operating system

2. **Install Unity Hub**:
   - Run the installer
   - Follow the installation wizard
   - Create or log in to your Unity ID

3. **Configure Unity Hub**:
   - Set your preferred installation directory
   - Configure proxy settings if needed for enterprise environments

### Installing Unity Editor

Unity 2021.3 LTS or newer is recommended for robotics applications:

1. **Open Unity Hub**
2. **Click "Installs" tab**
3. **Click "Add" to install a new Unity version**
4. **Select Unity 2021.3 LTS or newer**
5. **Select modules**:
   - Unity Editor
   - Android Build Support (if needed)
   - iOS Build Support (if needed)
   - Linux Build Support (if needed)
   - Windows Build Support (if needed)

6. **Install** the selected components

### Unity Editor Settings for Robotics

After installation, configure Unity Editor for robotics applications:

```csharp
// Editor configuration for robotics projects
// File -> Build Settings -> Player Settings

// Player Settings Configuration:
// - Other Settings:
//   * Scripting Backend: Mono or IL2CPP
//   * Api Compatibility Level: .NET Standard 2.1
//   * Target Architecture: x64 for robotics applications
// - XR Settings:
//   * VR Supported: Enable if using VR for teleoperation
//   * VR SDKs: OpenVR, Oculus, etc. (if needed)
```

## Setting Up Unity Robotics Hub

### Installing Unity Robotics Hub Packages

The Unity Robotics Hub includes several essential packages:

1. **ROS TCP Connector**: Communication between Unity and ROS 2
2. **Unity Perception**: Advanced sensor simulation and dataset generation
3. **ML-Agents**: Machine learning for intelligent agents
4. **XR Packages**: For VR/AR applications (optional)

### Installing ROS TCP Connector

The ROS TCP Connector enables communication between Unity and ROS 2:

1. **Open Unity Editor**
2. **Create or open a robotics project**
3. **Go to Window -> Package Manager**
4. **Click the "+" icon in the top-left corner**
5. **Select "Add package from git URL..."**
6. **Enter the URL**: `https://github.com/Unity-Technologies/ROS-TCP-Connector.git`
7. **Click "Add"**

### Installing Unity Perception

Unity Perception provides advanced sensor simulation:

1. **In Package Manager, click "+"**
2. **Select "Add package from git URL..."**
3. **Enter the URL**: `https://github.com/Unity-Technologies/Unity-Perception.git`
4. **Click "Add"**

### Installing ML-Agents

ML-Agents enables machine learning training:

1. **In Package Manager, click "+"**
2. **Select "Add package from git URL..."**
3. **Enter the URL**: `https://github.com/Unity-Technologies/ml-agents.git`
4. **Click "Add"**

## Project Setup for Robotics

### Creating a New Robotics Project

1. **Open Unity Hub**
2. **Click "New Project"**
3. **Select "3D (Built-in Render Pipeline)"** (for robotics applications)
4. **Name your project** (e.g., "RoboticsSimulation")
5. **Choose a location** to save your project
6. **Click "Create Project"**

### Project Structure for Robotics

A well-organized robotics project should follow this structure:

```
RoboticsSimulation/
├── Assets/
│   ├── Scripts/              # C# scripts for robot control and sensors
│   │   ├── RobotControl/
│   │   ├── Sensors/
│   │   └── Communication/
│   ├── Models/               # 3D models of robots and objects
│   ├── Materials/            # Material definitions
│   ├── Scenes/               # Unity scene files
│   │   ├── Main.unity
│   │   └── Training.unity
│   ├── Prefabs/              # Reusable robot and object prefabs
│   ├── Textures/             # Texture files
│   ├── Config/               # Configuration files
│   └── Plugins/              # External libraries and DLLs
├── Packages/                 # Package manifest and dependencies
└── ProjectSettings/          # Project configuration
```

### Basic Scene Setup

Create a basic scene for robotics simulation:

```csharp
// BasicRobotScene.cs - Example script for scene setup
using UnityEngine;

public class BasicRobotScene : MonoBehaviour
{
    [Header("Environment Settings")]
    public float gravity = -9.81f;
    public Color ambientLight = Color.gray;

    [Header("Robot Spawning")]
    public GameObject robotPrefab;
    public Transform spawnPoint;

    void Start()
    {
        // Configure physics
        Physics.gravity = new Vector3(0, gravity, 0);

        // Set ambient lighting
        RenderSettings.ambientLight = ambientLight;

        // Spawn robot if prefab exists
        if (robotPrefab != null && spawnPoint != null)
        {
            Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
        }
    }
}
```

## ROS TCP Connector Integration

### Setting Up ROS TCP Connector

The ROS TCP Connector provides communication between Unity and ROS 2:

```csharp
// ROSConnectorExample.cs
using ROS2;
using UnityEngine;

public class ROSConnectorExample : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIP = "127.0.0.1";  // ROS master IP
    public int rosPort = 10000;          // ROS communication port
    public float connectionTimeout = 10.0f;

    private ROS2UnityComponent ros2Unity;
    private ROS2Socket ros2Socket;

    void Start()
    {
        // Initialize ROS connection
        InitializeROSConnection();
    }

    void InitializeROSConnection()
    {
        ros2Unity = GetComponent<ROS2UnityComponent>();
        if (ros2Unity != null)
        {
            ros2Unity.ROS2ServerURL = rosIP;
            ros2Unity.ROS2ServerPort = rosPort;
            ros2Unity.Connect();

            // Wait for connection
            Invoke("CheckConnection", 1.0f);
        }
        else
        {
            Debug.LogError("ROS2UnityComponent not found on this GameObject!");
        }
    }

    void CheckConnection()
    {
        if (ros2Unity != null && ros2Unity.Ok())
        {
            Debug.Log("Successfully connected to ROS!");
        }
        else
        {
            Debug.LogWarning("Still connecting to ROS...");
            Invoke("CheckConnection", 1.0f);
        }
    }

    void OnDestroy()
    {
        if (ros2Unity != null && ros2Unity.Ok())
        {
            ros2Unity.Shutdown();
        }
    }
}
```

### Creating a ROS Publisher in Unity

```csharp
// UnityPublisher.cs
using ROS2;
using UnityEngine;
using std_msgs;

public class UnityPublisher : MonoBehaviour
{
    [Header("Publisher Settings")]
    public string topicName = "/unity_sensor_data";
    public float publishRate = 10.0f;  // Hz

    private ROS2Socket ros2Socket;
    private IPublisher<std_msgs.msg.Float32> publisher;
    private float publishTimer;

    void Start()
    {
        // Initialize publisher
        InitializePublisher();
    }

    void InitializePublisher()
    {
        ros2Socket = GetComponent<ROS2UnityComponent>().ROS2Socket;
        if (ros2Socket != null)
        {
            publisher = ros2Socket.advertise<std_msgs.msg.Float32>(topicName);
        }
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        if (publishTimer >= 1.0f / publishRate)
        {
            PublishData();
            publishTimer = 0.0f;
        }
    }

    void PublishData()
    {
        if (publisher != null)
        {
            var msg = new std_msgs.msg.Float32();
            msg.data = Random.Range(0.0f, 100.0f);  // Example sensor data
            publisher.Publish(msg);
        }
    }
}
```

### Creating a ROS Subscriber in Unity

```csharp
// UnitySubscriber.cs
using ROS2;
using UnityEngine;
using geometry_msgs;

public class UnitySubscriber : MonoBehaviour
{
    [Header("Subscriber Settings")]
    public string topicName = "/cmd_vel";

    private ROS2Socket ros2Socket;
    private ISubscriber<geometry_msgs.msg.Twist> subscriber;

    [Header("Robot Movement")]
    public float linearSpeed = 1.0f;
    public float angularSpeed = 1.0f;

    private float linearVelocity = 0.0f;
    private float angularVelocity = 0.0f;

    void Start()
    {
        // Initialize subscriber
        InitializeSubscriber();
    }

    void InitializeSubscriber()
    {
        ros2Socket = GetComponent<ROS2UnityComponent>().ROS2Socket;
        if (ros2Socket != null)
        {
            subscriber = ros2Socket.subscribe<geometry_msgs.msg.Twist>(
                topicName,
                ReceiveCommand
            );
        }
    }

    void ReceiveCommand(geometry_msgs.msg.Twist msg)
    {
        linearVelocity = msg.linear.x * linearSpeed;
        angularVelocity = msg.angular.z * angularSpeed;
    }

    void Update()
    {
        // Apply movement based on received commands
        transform.Translate(Vector3.forward * linearVelocity * Time.deltaTime);
        transform.Rotate(Vector3.up, angularVelocity * Time.deltaTime);
    }
}
```

## Unity Perception for Advanced Sensors

### Setting Up Camera Sensors

Unity Perception provides advanced camera simulation:

```csharp
// CameraSensor.cs
using UnityEngine;
using Unity.Robotics.Sensors;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine.Rendering;
using System.Collections;

public class CameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fieldOfView = 60.0f;

    private Camera unityCamera;
    private RenderTexture renderTexture;

    void Start()
    {
        SetupCamera();
    }

    void SetupCamera()
    {
        unityCamera = GetComponent<Camera>();
        if (unityCamera == null)
        {
            unityCamera = gameObject.AddComponent<Camera>();
        }

        unityCamera.fieldOfView = fieldOfView;
        unityCamera.allowMSAA = false;  // For consistent rendering

        // Create render texture for sensor data
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        unityCamera.targetTexture = renderTexture;
    }

    void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
        }
    }
}
```

### Setting Up LiDAR Sensors

Unity Perception includes realistic LiDAR simulation:

```csharp
// LidarSensor.cs
using UnityEngine;
using Unity.Robotics.Sensors;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public int numberOfRays = 360;
    public float minAngle = -180f;
    public float maxAngle = 180f;
    public float maxDistance = 30f;
    public float updateRate = 10f;

    [Header("Noise Settings")]
    public float noiseStdDev = 0.01f;

    private float updateTimer;
    private List<float> ranges;

    void Start()
    {
        ranges = new List<float>(new float[numberOfRays]);
    }

    void Update()
    {
        updateTimer += Time.deltaTime;
        if (updateTimer >= 1.0f / updateRate)
        {
            SimulateLidarScan();
            updateTimer = 0.0f;
        }
    }

    void SimulateLidarScan()
    {
        for (int i = 0; i < numberOfRays; i++)
        {
            float angle = minAngle + (maxAngle - minAngle) * i / (numberOfRays - 1);
            float rayAngle = Mathf.Deg2Rad * angle;

            Vector3 direction = new Vector3(
                Mathf.Cos(rayAngle),
                0,
                Mathf.Sin(rayAngle)
            );

            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction),
                              out hit, maxDistance))
            {
                float distance = hit.distance;
                // Add noise to simulate real sensor
                distance += Random.Range(-noiseStdDev, noiseStdDev);
                ranges[i] = Mathf.Clamp(distance, 0, maxDistance);
            }
            else
            {
                ranges[i] = maxDistance;  // No obstacle detected
            }
        }
    }

    public float[] GetRanges()
    {
        return ranges.ToArray();
    }
}
```

## Practical Example: Complete Robot Setup

Here's a complete example combining all the components:

```csharp
// CompleteRobotController.cs
using UnityEngine;
using ROS2;
using geometry_msgs;
using sensor_msgs;
using Unity.Robotics.ROSTCPConnector;

public class CompleteRobotController : MonoBehaviour
{
    [Header("ROS Settings")]
    public string cmdVelTopic = "/cmd_vel";
    public string laserScanTopic = "/scan";
    public string odomTopic = "/odom";

    [Header("Robot Settings")]
    public float maxLinearSpeed = 1.0f;
    public float maxAngularSpeed = 1.0f;
    public float wheelRadius = 0.1f;
    public float wheelBase = 0.5f;

    [Header("Sensor Settings")]
    public int laserRays = 360;
    public float laserMaxRange = 30.0f;

    private ROS2UnityComponent ros2Unity;
    private ROS2Socket ros2Socket;

    // Publishers and subscribers
    private IPublisher<sensor_msgs.msg.LaserScan> laserPublisher;
    private IPublisher<nav_msgs.msg.Odometry> odomPublisher;
    private ISubscriber<geometry_msgs.msg.Twist> cmdSub;

    // Robot state
    private float linearVelocity = 0.0f;
    private float angularVelocity = 0.0f;
    private Vector3 position = Vector3.zero;
    private Quaternion rotation = Quaternion.identity;

    // Sensor simulation
    private float[] laserRanges;

    void Start()
    {
        // Initialize ROS connection
        ros2Unity = GetComponent<ROS2UnityComponent>();
        if (ros2Unity != null)
        {
            ros2Unity.ROS2ServerURL = "127.0.0.1";
            ros2Unity.ROS2ServerPort = 10000;
            ros2Unity.Connect();

            Invoke("InitializeROSCommunication", 1.0f);
        }

        // Initialize sensor data
        laserRanges = new float[laserRays];
        for (int i = 0; i < laserRays; i++)
        {
            laserRanges[i] = laserMaxRange;
        }
    }

    void InitializeROSCommunication()
    {
        ros2Socket = ros2Unity.ROS2Socket;
        if (ros2Socket != null)
        {
            // Initialize publishers
            laserPublisher = ros2Socket.advertise<sensor_msgs.msg.LaserScan>(laserScanTopic);
            odomPublisher = ros2Socket.advertise<nav_msgs.msg.Odometry>(odomTopic);

            // Initialize subscriber
            cmdSub = ros2Socket.subscribe<geometry_msgs.msg.Twist>(
                cmdVelTopic,
                ProcessVelocityCommand
            );

            Debug.Log("ROS communication initialized successfully");
        }
    }

    void ProcessVelocityCommand(geometry_msgs.msg.Twist msg)
    {
        linearVelocity = Mathf.Clamp(msg.linear.x, -maxLinearSpeed, maxLinearSpeed);
        angularVelocity = Mathf.Clamp(msg.angular.z, -maxAngularSpeed, maxAngularSpeed);
    }

    void Update()
    {
        // Update robot position based on velocity commands
        UpdateRobotPosition();

        // Simulate sensors
        SimulateLidar();

        // Publish sensor data
        PublishSensorData();
    }

    void UpdateRobotPosition()
    {
        // Simple differential drive kinematics
        float deltaTime = Time.deltaTime;

        // Update rotation
        float deltaRotation = angularVelocity * deltaTime;
        rotation = Quaternion.Euler(0, rotation.eulerAngles.y + deltaRotation * Mathf.Rad2Deg, 0);

        // Update position
        Vector3 forward = rotation * Vector3.forward;
        position += forward * linearVelocity * deltaTime;

        // Apply new transform
        transform.position = position;
        transform.rotation = rotation;
    }

    void SimulateLidar()
    {
        // Cast rays for LiDAR simulation
        for (int i = 0; i < laserRays; i++)
        {
            float angle = Mathf.Lerp(-Mathf.PI, Mathf.PI, (float)i / (laserRays - 1));
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, laserMaxRange))
            {
                laserRanges[i] = hit.distance;
            }
            else
            {
                laserRanges[i] = laserMaxRange;
            }
        }
    }

    void PublishSensorData()
    {
        if (ros2Socket != null && ros2Socket.Ok())
        {
            // Publish laser scan
            if (laserPublisher != null)
            {
                var laserMsg = new sensor_msgs.msg.LaserScan();
                laserMsg.header.stamp = new builtin_interfaces.msg.Time();
                laserMsg.header.frame_id = "laser_frame";
                laserMsg.angle_min = -Mathf.PI;
                laserMsg.angle_max = Mathf.PI;
                laserMsg.angle_increment = (2 * Mathf.PI) / laserRays;
                laserMsg.time_increment = 0.0f;
                laserMsg.scan_time = 0.1f;  // 10Hz
                laserMsg.range_min = 0.1f;
                laserMsg.range_max = laserMaxRange;
                laserMsg.ranges = laserRanges;

                laserPublisher.Publish(laserMsg);
            }

            // Publish odometry
            if (odomPublisher != null)
            {
                var odomMsg = new nav_msgs.msg.Odometry();
                odomMsg.header.stamp = new builtin_interfaces.msg.Time();
                odomMsg.header.frame_id = "odom";
                odomMsg.child_frame_id = "base_link";

                // Position
                odomMsg.pose.pose.position.x = position.x;
                odomMsg.pose.pose.position.y = position.z;  // Unity Z -> ROS Y
                odomMsg.pose.pose.position.z = position.y;  // Unity Y -> ROS Z

                // Orientation (simplified)
                odomMsg.pose.pose.orientation.x = rotation.x;
                odomMsg.pose.pose.orientation.y = rotation.y;
                odomMsg.pose.pose.orientation.z = rotation.z;
                odomMsg.pose.pose.orientation.w = rotation.w;

                // Velocity (simplified)
                odomMsg.twist.twist.linear.x = linearVelocity;
                odomMsg.twist.twist.angular.z = angularVelocity;

                odomPublisher.Publish(odomMsg);
            }
        }
    }

    void OnDestroy()
    {
        if (ros2Unity != null && ros2Unity.Ok())
        {
            ros2Unity.Shutdown();
        }
    }
}
```

## Testing the Setup

### Basic Testing Script

```csharp
// UnityRobotTester.cs
using UnityEngine;

public class UnityRobotTester : MonoBehaviour
{
    [Header("Test Configuration")]
    public GameObject robotController;
    public float testDuration = 10.0f;
    public float testInterval = 1.0f;

    private float testTimer = 0.0f;
    private bool isTesting = false;

    void Start()
    {
        StartTest();
    }

    void StartTest()
    {
        isTesting = true;
        testTimer = 0.0f;
        Debug.Log("Starting Unity robot simulation test...");
    }

    void Update()
    {
        if (isTesting)
        {
            testTimer += Time.deltaTime;

            if (testTimer >= testInterval)
            {
                PerformTestCheck();
                testTimer = 0.0f;
            }

            if (testTimer >= testDuration)
            {
                EndTest();
            }
        }
    }

    void PerformTestCheck()
    {
        // Check if robot controller is functioning
        if (robotController != null)
        {
            Debug.Log($"Robot position: {robotController.transform.position}");
            Debug.Log($"Robot rotation: {robotController.transform.rotation.eulerAngles}");
        }
        else
        {
            Debug.LogError("Robot controller not found!");
        }
    }

    void EndTest()
    {
        isTesting = false;
        Debug.Log("Unity robot simulation test completed.");
    }
}
```

## Troubleshooting Common Issues

### Connection Issues

**Problem**: Cannot connect to ROS
**Solution**:
- Ensure ROS bridge is running: `ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1 -p ROS_TCP_PORT:=10000`
- Check firewall settings
- Verify IP and port configuration

### Performance Issues

**Problem**: Low frame rate in Unity
**Solutions**:
- Reduce rendering quality during simulation
- Simplify physics colliders
- Reduce sensor update rates
- Use occlusion culling

### Sensor Accuracy

**Problem**: Sensor data doesn't match expectations
**Solutions**:
- Verify sensor placement and orientation
- Check coordinate system conversions
- Validate physics properties of objects
- Adjust sensor noise parameters

## Summary

This chapter covered the complete setup of Unity for robotics applications:

- **Unity Hub and Editor**: Installation and configuration for robotics
- **ROS TCP Connector**: Integration with ROS 2 for communication
- **Unity Perception**: Advanced sensor simulation capabilities
- **Complete robot setup**: Integration of all components into a functional system

Unity provides powerful capabilities for robotics simulation, particularly for applications requiring photorealistic graphics, advanced sensor simulation, or machine learning training environments.

## Exercises

1. Install Unity Hub and the Robotics Hub packages
2. Create a simple robot that can receive velocity commands from ROS 2
3. Implement a basic LiDAR sensor simulation in Unity
4. Test communication between Unity and ROS 2 nodes

## Quiz

1. What is the primary purpose of the ROS TCP Connector in Unity?
   a) To provide physics simulation
   b) To enable communication between Unity and ROS 2
   c) To render graphics
   d) To simulate sensors

2. Which Unity package provides advanced sensor simulation?
   a) ML-Agents
   b) Unity Perception
   c) XR packages
   d) Physics package

3. What is the recommended Unity version for robotics applications?
   a) Unity 2020.1
   b) Unity 2021.3 LTS or newer
   c) Unity 2019.4 LTS
   d) Any Unity version

## Mini-Project: Unity Robot Simulation

Create a complete Unity robot simulation with:
1. A differential drive robot model
2. ROS 2 communication for control and sensor data
3. Camera and LiDAR sensor simulation
4. A simple navigation environment
5. Integration with ROS 2 navigation stack

Test your simulation by controlling the robot from ROS 2 nodes and verifying sensor data transmission.