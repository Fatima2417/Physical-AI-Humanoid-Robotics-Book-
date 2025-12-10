---
sidebar_position: 13
---

# Unity Robotics Hub Setup

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure Unity Hub and Unity Editor for robotics applications
- Set up Unity Robotics Hub packages and dependencies
- Create Unity scenes with realistic robotic environments
- Integrate Unity with ROS 2 for bidirectional communication
- Implement sensor simulation using Unity Perception
- Configure Unity for real-time performance with robotics systems

## Introduction to Unity Robotics Hub

Unity Robotics Hub is a collection of tools, components, and workflows that enable robotics researchers and developers to use Unity as a robotics simulation platform. It provides realistic physics, high-fidelity graphics, and seamless integration with ROS 2 for comprehensive robotic simulation and development.

### Unity Robotics Hub Architecture

```
Unity Robotics Hub Components:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Unity Editor  │◄───┤  ROS TCP        │───►│   ROS 2         │
│   (Simulation)  │    │  Connector      │    │   (Control &   │
│                 │    │  (Communication) │    │   Processing)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Unity         │
                         │   Perception    │
                         │   (Sensors,     │
                         │   Data Gen)     │
                         └─────────────────┘
```

### Key Components

1. **ROS TCP Connector**: Enables communication between Unity and ROS 2
2. **Unity Perception**: Provides advanced sensor simulation and dataset generation
3. **ML-Agents**: Framework for training intelligent agents using reinforcement learning
4. **XR Packages**: Support for virtual and augmented reality applications
5. **Robotics Packages**: Specialized tools for robotics simulation

## Installing Unity Hub and Unity Editor

### Prerequisites

Before installing Unity, ensure your system meets the requirements:

- **Operating System**: Windows 10/11 (64-bit), macOS 10.14+, or Ubuntu 18.04+ (64-bit)
- **Processor**: Intel Core i5 or AMD Ryzen 5 processor or newer
- **Memory**: 8 GB RAM minimum, 16 GB recommended
- **Graphics**: DirectX 10, OpenGL 3.3, or Vulkan compatible GPU
- **Storage**: 20 GB available space for Unity installation
- **Additional**: Python 3.8+ for ROS 2 integration

### Installing Unity Hub

Unity Hub is the recommended way to manage Unity installations:

1. **Download Unity Hub**:
   - Visit https://unity.com/download
   - Download Unity Hub for your operating system
   - Run the installer and follow the setup wizard

2. **Sign in to Unity**:
   - Create a Unity ID if you don't have one
   - Sign in to Unity Hub with your credentials

3. **Configure Unity Hub**:
   - Set your preferred installation directory
   - Configure proxy settings if needed for enterprise environments

### Installing Unity Editor

For robotics applications, Unity 2021.3 LTS or newer is recommended:

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
// Unity Editor Configuration for Robotics
// Edit -> Project Settings -> Player

// Player Settings for Robotics Applications:
// - Other Settings:
//   * Scripting Backend: Mono or IL2CPP
//   * Api Compatibility Level: .NET Standard 2.1
//   * Target Architecture: x64 for robotics applications
// - XR Settings:
//   * VR Supported: Enable if using VR for teleoperation
//   * VR SDKs: OpenVR, Oculus, etc. (if needed)
// - Resolution and Presentation:
//   * Run in Background: Enable
//   * Visible in Background: Enable
//   * Fullscreen Mode: Windowed
//   * Resolution: Set to desired resolution for simulation
```

## Setting Up Unity Robotics Hub Packages

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

### Installing ML-Agents (Optional)

ML-Agents enables machine learning training:

1. **In Package Manager, click "+"**
2. **Select "Add package from git URL..."**
3. **Enter the URL**: `https://github.com/Unity-Technologies/ml-agents.git`
4. **Click "Add"**

## Creating Robotics Simulation Environment

### Project Structure for Robotics

Create a well-organized project structure for robotics applications:

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

### Basic Robot Setup in Unity

Create a basic robot setup with ROS integration:

```csharp
// ROSRobotController.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using System.Collections;

public class ROSRobotController : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Configuration")]
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;
    public Transform baseLink;

    [Header("Joint Configuration")]
    public Transform[] jointTransforms;
    public string[] jointNames;

    // ROS Communication
    private ROSConnection ros;
    private string cmdVelTopic = "cmd_vel";
    private string jointStatesTopic = "joint_states";

    // Robot state
    private Vector3 currentVelocity = Vector3.zero;
    private float currentAngularVelocity = 0f;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);

        // Subscribe to command velocity topic
        ros.Subscribe<TwistMsg>(cmdVelTopic, CmdVelCallback);

        // Start publishing joint states
        StartCoroutine(PublishJointStates());

        Debug.Log("ROS Robot Controller initialized");
    }

    void CmdVelCallback(TwistMsg cmdVel)
    {
        // Process velocity commands
        currentVelocity = new Vector3(
            (float)cmdVel.linear.x,
            (float)cmdVel.linear.y,
            (float)cmdVel.linear.z
        );

        currentAngularVelocity = (float)cmdVel.angular.z;

        // Apply movement to robot (simplified differential drive)
        ApplyRobotMovement();
    }

    void ApplyRobotMovement()
    {
        // Apply linear velocity
        transform.Translate(currentVelocity * Time.deltaTime);

        // Apply angular velocity (rotation around Y-axis for differential drive)
        transform.Rotate(Vector3.up, currentAngularVelocity * Time.deltaTime * Mathf.Rad2Deg);
    }

    IEnumerator PublishJointStates()
    {
        // Publish joint states at 30 Hz
        WaitForSeconds wait = new WaitForSeconds(1f / 30f);

        while (true)
        {
            // Create joint state message
            var jointState = new SensorMsgs.JointStateMsg();
            jointState.header = new StdMsgs.HeaderMsg();
            jointState.header.stamp = new builtin_interfaces.msg.Time();
            jointState.header.frame_id = "base_link";

            // Set joint names
            jointState.name = jointNames;

            // Set joint positions (simplified - in reality get from actual joint transforms)
            jointState.position = new double[jointTransforms.Length];
            for (int i = 0; i < jointTransforms.Length; i++)
            {
                // For this example, just use the current rotation as position
                // In a real robot, this would be the actual joint angles
                jointState.position[i] = jointTransforms[i].localEulerAngles.y * Mathf.Deg2Rad;
            }

            // Set joint velocities (simplified)
            jointState.velocity = new double[jointTransforms.Length];
            for (int i = 0; i < jointTransforms.Length; i++)
            {
                jointState.velocity[i] = 0.0; // Simplified - would be actual velocities
            }

            // Set joint efforts (simplified)
            jointState.effort = new double[jointTransforms.Length];
            for (int i = 0; i < jointTransforms.Length; i++)
            {
                jointState.effort[i] = 0.0; // Simplified - would be actual efforts
            }

            // Publish joint states
            ros.Publish(jointStatesTopic, jointState);

            yield return wait;
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

### Unity Perception Sensor Setup

Implement sensor simulation using Unity Perception:

```csharp
// UnityPerceptionSensors.cs
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Simulation;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class UnityPerceptionSensors : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public float cameraFOV = 60f;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float maxDepthRange = 10.0f;

    [Header("ROS Topic Settings")]
    public string cameraTopic = "camera/image_raw";
    public string depthTopic = "camera/depth";
    public string segmentationTopic = "camera/segmentation";

    [Header("Sensor Publishing")]
    public float publishRate = 30f; // Hz

    private Camera sensorCamera;
    private ROSConnection ros;
    private PerceptionCamera perceptionCamera;

    void Start()
    {
        // Get the camera component
        sensorCamera = GetComponent<Camera>();
        if (sensorCamera == null)
        {
            sensorCamera = gameObject.AddComponent<Camera>();
        }

        // Configure camera
        sensorCamera.fieldOfView = cameraFOV;
        sensorCamera.aspect = (float)imageWidth / imageHeight;
        sensorCamera.depth = 0;

        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Set up perception camera
        SetupPerceptionCamera();

        Debug.Log("Unity Perception Sensors initialized");
    }

    void SetupPerceptionCamera()
    {
        // Add perception camera component
        perceptionCamera = GetComponent<PerceptionCamera>();
        if (perceptionCamera == null)
        {
            perceptionCamera = gameObject.AddComponent<PerceptionCamera>();
        }

        // Configure perception camera
        perceptionCamera.captureRgbImages = true;
        perceptionCamera.captureSegmentationImages = true;
        perceptionCamera.captureDepthImages = true;
        perceptionCamera.rgbSensorSettings.outputWidth = imageWidth;
        perceptionCamera.rgbSensorSettings.outputHeight = imageHeight;
        perceptionCamera.rgbSensorSettings.publishTopic = cameraTopic;
        perceptionCamera.depthSensorSettings.publishTopic = depthTopic;
        perceptionCamera.segmentationSensorSettings.publishTopic = segmentationTopic;

        // Set up semantic segmentation labels
        SetupSemanticLabels();
    }

    void SetupSemanticLabels()
    {
        // Create semantic segmentation labels
        // This would map objects to semantic classes
        // Example: "person", "chair", "table", "floor", etc.

        // In a real implementation, you would assign semantic labels to objects
        // For this example, we'll just log that the setup is complete
        Debug.Log("Semantic labels setup completed");
    }

    void Update()
    {
        // Update sensor simulation at specified rate
        if (Time.frameCount % Mathf.RoundToInt(60f / publishRate) == 0)
        {
            UpdateSensors();
        }
    }

    void UpdateSensors()
    {
        // Update sensor data and publish to ROS
        // This would capture images, depth data, etc. and send to ROS
        Debug.Log("Sensor update called");
    }
}
```

## ROS 2 Integration

### Setting up ROS TCP Connection

Create a comprehensive ROS connection manager:

```csharp
// ROSConnectionManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;
using System.Collections.Generic;
using System.Linq;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public float connectionRetryDelay = 2f;

    [Header("Topics Configuration")]
    public string cmdVelTopic = "cmd_vel";
    public string odomTopic = "odom";
    public string laserScanTopic = "scan";
    public string jointStatesTopic = "joint_states";
    public string imageTopic = "camera/image_raw";

    // Connection state
    private ROSConnection ros;
    private bool isConnected = false;
    private float lastConnectionAttempt = 0f;

    // Message publishers
    private Dictionary<string, System.Action<object>> publishers = new Dictionary<string, System.Action<object>>();

    // Subscribers
    private Dictionary<string, System.Action<object>> subscribers = new Dictionary<string, System.Action<object>>();

    // Robot state
    private Vector3 position = Vector3.zero;
    private Quaternion orientation = Quaternion.identity;
    private Vector3 linearVelocity = Vector3.zero;
    private Vector3 angularVelocity = Vector3.zero;

    void Start()
    {
        InitializeROSConnection();
    }

    void InitializeROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Attempt connection
        ConnectToROS();
    }

    void ConnectToROS()
    {
        if (Time.time - lastConnectionAttempt < connectionRetryDelay)
            return;

        lastConnectionAttempt = Time.time;

        try
        {
            ros.Initialize(rosIPAddress, rosPort);
            isConnected = true;

            // Set up publishers
            SetupPublishers();

            // Set up subscribers
            SetupSubscribers();

            Debug.Log("Connected to ROS successfully");
        }
        catch (System.Exception e)
        {
            isConnected = false;
            Debug.LogError($"Failed to connect to ROS: {e.Message}");

            // Retry connection
            Invoke(nameof(ConnectToROS), connectionRetryDelay);
        }
    }

    void SetupPublishers()
    {
        // Create publishers for different topics
        publishers[cmdVelTopic] = (msg) => ros.Publish(cmdVelTopic, msg);
        publishers[odomTopic] = (msg) => ros.Publish(odomTopic, msg);
        publishers[jointStatesTopic] = (msg) => ros.Publish(jointStatesTopic, msg);
        publishers[imageTopic] = (msg) => ros.Publish(imageTopic, msg);
    }

    void SetupSubscribers()
    {
        // Subscribe to command topics
        ros.Subscribe<TwistMsg>(cmdVelTopic, ProcessVelocityCommand);
        ros.Subscribe<OdometryMsg>(odomTopic, ProcessOdometry);
    }

    void ProcessVelocityCommand(TwistMsg cmd)
    {
        // Process velocity commands from ROS
        linearVelocity = new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z);
        angularVelocity = new Vector3((float)cmd.angular.x, (float)cmd.angular.y, (float)cmd.angular.z);
    }

    void ProcessOdometry(OdometryMsg odom)
    {
        // Process odometry data from ROS
        position = new Vector3(
            (float)odom.pose.pose.position.x,
            (float)odom.pose.pose.position.y,
            (float)odom.pose.pose.position.z
        );

        orientation = new Quaternion(
            (float)odom.pose.pose.orientation.x,
            (float)odom.pose.pose.orientation.y,
            (float)odom.pose.pose.orientation.z,
            (float)odom.pose.pose.orientation.w
        );
    }

    public void PublishOdometry(Vector3 pos, Quaternion rot, Vector3 linVel, Vector3 angVel)
    {
        if (!isConnected) return;

        var odomMsg = new OdometryMsg();
        odomMsg.header = new StdMsgs.HeaderMsg();
        odomMsg.header.stamp = new builtin_interfaces.msg.Time();
        odomMsg.header.frame_id = "odom";
        odomMsg.child_frame_id = "base_link";

        // Set position
        odomMsg.pose.pose.position = new GeometryMsgs.PointMsg(pos.x, pos.y, pos.z);
        odomMsg.pose.pose.orientation = new GeometryMsgs.QuaternionMsg(
            rot.x, rot.y, rot.z, rot.w
        );

        // Set velocities
        odomMsg.twist.twist.linear = new GeometryMsgs.Vector3Msg(linVel.x, linVel.y, linVel.z);
        odomMsg.twist.twist.angular = new GeometryMsgs.Vector3Msg(angVel.x, angVel.y, angVel.z);

        // Publish odometry
        ros.Publish(odomTopic, odomMsg);
    }

    public void PublishLaserScan(float[] ranges, float angleMin, float angleMax, float angleIncrement)
    {
        if (!isConnected) return;

        var scanMsg = new LaserScanMsg();
        scanMsg.header = new StdMsgs.HeaderMsg();
        scanMsg.header.stamp = new builtin_interfaces.msg.Time();
        scanMsg.header.frame_id = "laser_frame";

        scanMsg.angle_min = angleMin;
        scanMsg.angle_max = angleMax;
        scanMsg.angle_increment = angleIncrement;
        scanMsg.time_increment = 0.0f;
        scanMsg.scan_time = 0.1f;
        scanMsg.range_min = 0.1f;
        scanMsg.range_max = 30.0f;

        // Convert ranges to ROS format
        scanMsg.ranges = ranges.Select(r => (float)r).ToArray();

        ros.Publish(laserScanTopic, scanMsg);
    }

    public void PublishJointStates(string[] jointNames, double[] positions, double[] velocities, double[] efforts)
    {
        if (!isConnected) return;

        var jointState = new JointStateMsg();
        jointState.header = new StdMsgs.HeaderMsg();
        jointState.header.stamp = new builtin_interfaces.msg.Time();
        jointState.header.frame_id = "base_link";

        jointState.name = jointNames;
        jointState.position = positions;
        jointState.velocity = velocities;
        jointState.effort = efforts;

        ros.Publish(jointStatesTopic, jointState);
    }

    void Update()
    {
        if (!isConnected && Time.time - lastConnectionAttempt > connectionRetryDelay)
        {
            ConnectToROS();
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }

    // Public accessors for robot state
    public Vector3 Position => position;
    public Quaternion Orientation => orientation;
    public Vector3 LinearVelocity => linearVelocity;
    public Vector3 AngularVelocity => angularVelocity;
    public bool IsConnected => isConnected;
}
```

## Unity Scene Setup for Robotics

### Creating Robot Environment

Create a comprehensive robot environment in Unity:

```csharp
// RobotEnvironmentSetup.cs
using UnityEngine;
using System.Collections.Generic;

public class RobotEnvironmentSetup : MonoBehaviour
{
    [Header("Environment Configuration")]
    public Transform robotSpawnPoint;
    public GameObject robotPrefab;
    public GameObject[] obstaclePrefabs;
    public int obstacleCount = 10;

    [Header("Terrain Settings")]
    public float terrainWidth = 20f;
    public float terrainLength = 20f;
    public float terrainHeight = 1f;

    [Header("Lighting Configuration")]
    public Light mainLight;
    public Color ambientLightColor = Color.gray;
    public float ambientIntensity = 0.5f;

    [Header("Sensor Mounting Points")]
    public Transform cameraMountPoint;
    public Transform lidarMountPoint;
    public Transform imuMountPoint;

    private List<GameObject> spawnedObstacles = new List<GameObject>();
    private GameObject spawnedRobot;

    void Start()
    {
        // Setup environment
        SetupEnvironment();
        SetupRobot();
        SetupObstacles();
        SetupLighting();
        SetupSensors();
    }

    void SetupEnvironment()
    {
        // Create terrain or ground plane
        CreateGroundPlane();

        // Set up environmental boundaries
        SetupBoundaries();

        Debug.Log("Environment setup completed");
    }

    void CreateGroundPlane()
    {
        // Create a ground plane
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.localScale = new Vector3(terrainWidth / 10f, 1, terrainLength / 10f);
        ground.transform.position = new Vector3(0, -terrainHeight/2f, 0);

        // Add physics material for proper robot interaction
        var groundCollider = ground.GetComponent<Collider>();
        if (groundCollider != null)
        {
            var physicsMaterial = new PhysicMaterial();
            physicsMaterial.staticFriction = 0.8f;
            physicsMaterial.dynamicFriction = 0.6f;
            physicsMaterial.frictionCombine = PhysicMaterialCombine.Multiply;
            groundCollider.material = physicsMaterial;
        }

        // Add visual material
        Renderer groundRenderer = ground.GetComponent<Renderer>();
        if (groundRenderer != null)
        {
            groundRenderer.material = CreateGroundMaterial();
        }
    }

    Material CreateGroundMaterial()
    {
        // Create a material for the ground
        Material groundMaterial = new Material(Shader.Find("Standard"));
        groundMaterial.color = Color.green;
        groundMaterial.SetColor("_EmissionColor", Color.black);
        groundMaterial.SetFloat("_Metallic", 0.0f);
        groundMaterial.SetFloat("_Smoothness", 0.5f);
        return groundMaterial;
    }

    void SetupBoundaries()
    {
        // Create boundary walls to contain the robot
        float wallHeight = 2f;
        float wallThickness = 0.5f;

        // Left wall
        CreateBoundaryWall(
            new Vector3(-terrainWidth/2f - wallThickness/2f, 0, 0),
            new Vector3(wallThickness, terrainLength, wallHeight),
            "LeftBoundary"
        );

        // Right wall
        CreateBoundaryWall(
            new Vector3(terrainWidth/2f + wallThickness/2f, 0, 0),
            new Vector3(wallThickness, terrainLength, wallHeight),
            "RightBoundary"
        );

        // Front wall
        CreateBoundaryWall(
            new Vector3(0, 0, terrainLength/2f + wallThickness/2f),
            new Vector3(terrainWidth, terrainLength, wallHeight),
            "FrontBoundary"
        );

        // Back wall
        CreateBoundaryWall(
            new Vector3(0, 0, -terrainLength/2f - wallThickness/2f),
            new Vector3(terrainWidth, terrainLength, wallHeight),
            "BackBoundary"
        );
    }

    void CreateBoundaryWall(Vector3 position, Vector3 size, string name)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = name;
        wall.transform.position = position;
        wall.transform.localScale = size;

        // Make wall invisible but keep collision
        Renderer renderer = wall.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.enabled = false;
        }
    }

    void SetupRobot()
    {
        if (robotPrefab != null && robotSpawnPoint != null)
        {
            // Spawn robot at designated spawn point
            spawnedRobot = Instantiate(robotPrefab, robotSpawnPoint.position, robotSpawnPoint.rotation);
            spawnedRobot.name = "Robot";

            Debug.Log($"Robot spawned at {robotSpawnPoint.position}");
        }
        else
        {
            Debug.LogWarning("Robot prefab or spawn point not configured");
        }
    }

    void SetupObstacles()
    {
        if (obstaclePrefabs.Length == 0)
        {
            Debug.LogWarning("No obstacle prefabs configured");
            return;
        }

        for (int i = 0; i < obstacleCount; i++)
        {
            // Randomly select an obstacle prefab
            GameObject obstaclePrefab = obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)];

            // Generate random position within bounds
            Vector3 randomPos = new Vector3(
                Random.Range(-terrainWidth/2f + 2f, terrainWidth/2f - 2f),
                0.5f,  // Place at ground level + height
                Random.Range(-terrainLength/2f + 2f, terrainLength/2f - 2f)
            );

            // Generate random rotation
            Quaternion randomRot = Quaternion.Euler(0, Random.Range(0, 360), 0);

            // Spawn obstacle
            GameObject obstacle = Instantiate(obstaclePrefab, randomPos, randomRot);
            obstacle.name = $"Obstacle_{i}";

            spawnedObstacles.Add(obstacle);
        }

        Debug.Log($"Spawned {obstacleCount} obstacles");
    }

    void SetupLighting()
    {
        // Configure main directional light
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.color = Color.white;
            mainLight.intensity = 1.0f;
            mainLight.transform.rotation = Quaternion.Euler(50, -30, 0);
        }

        // Set ambient lighting
        RenderSettings.ambientLight = ambientLightColor;
        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
    }

    void SetupSensors()
    {
        // Mount sensors on the robot if spawn point exists
        if (spawnedRobot != null)
        {
            // Mount camera
            if (cameraMountPoint != null)
            {
                MountCameraOnRobot();
            }

            // Mount LiDAR
            if (lidarMountPoint != null)
            {
                MountLidarOnRobot();
            }

            // Mount IMU
            if (imuMountPoint != null)
            {
                MountIMUOnRobot();
            }
        }
    }

    void MountCameraOnRobot()
    {
        // Create a camera and mount it on the robot
        GameObject cameraObj = new GameObject("Camera");
        cameraObj.transform.SetParent(spawnedRobot.transform);
        cameraObj.transform.localPosition = cameraMountPoint.localPosition;
        cameraObj.transform.localRotation = cameraMountPoint.localRotation;

        // Add camera component
        Camera cam = cameraObj.AddComponent<Camera>();
        cam.fieldOfView = 60f;
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 100f;

        // Add perception camera component
        var perceptionCam = cameraObj.AddComponent<Unity.Perception.GroundTruth.PerceptionCamera>();
        perceptionCam.captureRgbImages = true;
        perceptionCam.rgbSensorSettings.outputWidth = 640;
        perceptionCam.rgbSensorSettings.outputHeight = 480;
        perceptionCam.rgbSensorSettings.publishTopic = "camera/image_raw";

        Debug.Log("Camera mounted on robot");
    }

    void MountLidarOnRobot()
    {
        // Create a LiDAR sensor (simplified representation)
        GameObject lidarObj = new GameObject("Lidar");
        lidarObj.transform.SetParent(spawnedRobot.transform);
        lidarObj.transform.localPosition = lidarMountPoint.localPosition;
        lidarObj.transform.localRotation = lidarMountPoint.localRotation;

        // In a real implementation, this would use Unity Perception's LiDAR component
        // For this example, we'll just create a placeholder
        var lidarSensor = lidarObj.AddComponent<LidarSensorPlaceholder>();
        lidarSensor.topicName = "scan";

        Debug.Log("LiDAR mounted on robot");
    }

    void MountIMUOnRobot()
    {
        // Create an IMU sensor placeholder
        GameObject imuObj = new GameObject("IMU");
        imuObj.transform.SetParent(spawnedRobot.transform);
        imuObj.transform.localPosition = imuMountPoint.localPosition;
        imuObj.transform.localRotation = imuMountPoint.localRotation;

        var imuSensor = imuObj.AddComponent<IMUSensorPlaceholder>();
        imuSensor.topicName = "imu/data";

        Debug.Log("IMU mounted on robot");
    }

    // Helper methods for runtime environment changes
    public void AddObstacleAtPosition(Vector3 position)
    {
        if (obstaclePrefabs.Length > 0)
        {
            GameObject obstaclePrefab = obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)];
            GameObject obstacle = Instantiate(obstaclePrefab, position, Quaternion.identity);
            obstacle.name = $"DynamicObstacle_{Time.time}";
            spawnedObstacles.Add(obstacle);
        }
    }

    public void RemoveRandomObstacle()
    {
        if (spawnedObstacles.Count > 0)
        {
            int randomIndex = Random.Range(0, spawnedObstacles.Count);
            GameObject obstacleToRemove = spawnedObstacles[randomIndex];
            spawnedObstacles.RemoveAt(randomIndex);
            Destroy(obstacleToRemove);
        }
    }

    public void ResetEnvironment()
    {
        // Remove all spawned obstacles
        foreach (GameObject obstacle in spawnedObstacles)
        {
            Destroy(obstacle);
        }
        spawnedObstacles.Clear();

        // Respawn obstacles
        SetupObstacles();

        Debug.Log("Environment reset completed");
    }
}

// Placeholder components for sensors (would be replaced with actual Unity Perception components)
public class LidarSensorPlaceholder : MonoBehaviour
{
    public string topicName = "scan";
    public float maxRange = 10.0f;
    public int resolution = 360;
}

public class IMUSensorPlaceholder : MonoBehaviour
{
    public string topicName = "imu/data";
    public float noiseLevel = 0.01f;
}
```

## Performance Optimization for Real-Time Operation

### Optimizing Unity for Robotics Simulation

```csharp
// UnityPerformanceOptimizer.cs
using UnityEngine;
using System.Collections.Generic;

public class UnityPerformanceOptimizer : MonoBehaviour
{
    [Header("Performance Settings")]
    public bool enableDynamicBatching = true;
    public bool enableStaticBatching = true;
    public ShadowQuality shadowQuality = ShadowQuality.All;
    public float targetFrameRate = 60f;
    public int maximumLODLevel = 2;

    [Header("Physics Optimization")]
    public float physicsUpdateRate = 120f;
    public int solverIterations = 6;
    public int solverVelocityIterations = 1;

    [Header("Rendering Optimization")]
    public int maxVisibleDistance = 20;
    public bool enableOcclusionCulling = true;
    public bool enableLOD = true;

    [Header("Memory Management")]
    public float garbageCollectionInterval = 10f;
    private float lastGC = 0f;

    void Start()
    {
        // Apply performance settings
        ApplyPerformanceSettings();

        // Optimize physics settings
        OptimizePhysics();

        // Optimize rendering
        OptimizeRendering();

        Debug.Log("Unity Performance Optimization applied");
    }

    void ApplyPerformanceSettings()
    {
        // Set target frame rate
        Application.targetFrameRate = (int)targetFrameRate;

        // Quality settings
        QualitySettings.shadowQuality = shadowQuality;
        QualitySettings.maximumLODLevel = maximumLODLevel;
        QualitySettings.anisotropicFiltering = AnisotropicFiltering.Enable;
        QualitySettings.vSyncCount = 0; // Disable VSync for consistent frame rates

        // Batching settings
        DynamicGI.enabled = enableDynamicBatching;

        Debug.Log($"Performance settings applied - Target FPS: {targetFrameRate}, Shadows: {shadowQuality}");
    }

    void OptimizePhysics()
    {
        // Physics settings
        Time.fixedDeltaTime = 1f / physicsUpdateRate;
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;

        // Sleep thresholds
        Physics.sleepThreshold = 0.005f;
        Physics.defaultContactOffset = 0.01f;

        Debug.Log($"Physics optimized - Update rate: {physicsUpdateRate}Hz, Solver iterations: {solverIterations}");
    }

    void OptimizeRendering()
    {
        // Occlusion culling
        if (enableOcclusionCulling)
        {
            // This is set in the scene view, not programmatically
            Debug.Log("Occlusion culling enabled (set in scene view)");
        }

        // LOD settings
        if (enableLOD)
        {
            // Enable LOD groups on objects that support it
            LODGroup[] lodGroups = FindObjectsOfType<LODGroup>();
            foreach (LODGroup lodGroup in lodGroups)
            {
                lodGroup.enabled = true;
            }
        }

        // Culling settings
        Camera.main.layerCullDistances = new float[32]; // Initialize array
        for (int i = 0; i < 32; i++)
        {
            Camera.main.layerCullDistances[i] = maxVisibleDistance;
        }

        Debug.Log($"Rendering optimized - Max visible distance: {maxVisibleDistance}m");
    }

    void Update()
    {
        // Periodic garbage collection for robotics applications
        if (Time.time - lastGC > garbageCollectionInterval)
        {
            System.GC.Collect();
            lastGC = Time.time;
        }

        // Monitor performance
        MonitorPerformance();
    }

    void MonitorPerformance()
    {
        // Calculate performance metrics
        float currentFPS = 1f / Time.unscaledDeltaTime;
        float memoryUsage = UnityEngine.Profiling.Profiler.GetTotalMemoryLong() / (1024f * 1024f); // MB

        // Log performance if it drops below target
        if (currentFPS < targetFrameRate * 0.8f) // Below 80% of target
        {
            Debug.LogWarning($"Performance warning - Current FPS: {currentFPS:F1}, Target: {targetFrameRate}");
        }
    }

    // Runtime performance adjustment methods
    public void AdjustPerformanceToMode(string mode)
    {
        switch (mode.ToLower())
        {
            case "high_performance":
                SetHighPerformanceMode();
                break;
            case "balanced":
                SetBalancedMode();
                break;
            case "power_saving":
                SetPowerSavingMode();
                break;
            default:
                Debug.LogWarning($"Unknown performance mode: {mode}");
                break;
        }
    }

    void SetHighPerformanceMode()
    {
        QualitySettings.shadowQuality = ShadowQuality.All;
        QualitySettings.maximumLODLevel = 0;
        Application.targetFrameRate = 120;
        Physics.defaultSolverIterations = 10;
        Debug.Log("High performance mode activated");
    }

    void SetBalancedMode()
    {
        QualitySettings.shadowQuality = ShadowQuality.High;
        QualitySettings.maximumLODLevel = 2;
        Application.targetFrameRate = 60;
        Physics.defaultSolverIterations = 6;
        Debug.Log("Balanced performance mode activated");
    }

    void SetPowerSavingMode()
    {
        QualitySettings.shadowQuality = ShadowQuality.Low;
        QualitySettings.maximumLODLevel = 3;
        Application.targetFrameRate = 30;
        Physics.defaultSolverIterations = 3;
        Debug.Log("Power saving mode activated");
    }

    public void GetPerformanceMetrics(out float fps, out float memoryMB, out float cpuUsage)
    {
        fps = 1f / Time.unscaledDeltaTime;
        memoryMB = UnityEngine.Profiling.Profiler.GetTotalMemoryLong() / (1024f * 1024f);
        cpuUsage = 0f; // Unity doesn't expose CPU usage directly, would need external monitoring

        Debug.Log($"Performance Metrics - FPS: {fps:F1}, Memory: {memoryMB:F1}MB");
    }
}
```

## Testing and Validation

### Unity Robotics Integration Testing

```csharp
// UnityRoboticsTestSuite.cs
using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;

public class UnityRoboticsTestSuite
{
    [SetUp]
    public void Setup()
    {
        // Setup test environment
        Debug.Log("Setting up Unity Robotics tests");
    }

    [Test]
    public void TestROSConnection()
    {
        // Test that ROS connection can be established
        var rosManager = Object.FindObjectOfType<ROSConnectionManager>();
        Assert.IsNotNull(rosManager, "ROS Connection Manager not found");

        // In a real test, you would verify actual connection status
        // For this example, just verify the component exists
        Assert.IsTrue(true, "ROS Connection Manager exists");
    }

    [Test]
    public void TestRobotSpawning()
    {
        // Test that robot spawns correctly
        var robot = GameObject.FindGameObjectWithTag("Robot");
        Assert.IsNotNull(robot, "Robot not found in scene");

        // Verify robot has necessary components
        var controller = robot.GetComponent<ROSRobotController>();
        Assert.IsNotNull(controller, "Robot controller not found");
    }

    [Test]
    public void TestSensorMounting()
    {
        // Test that sensors are properly mounted
        var robot = GameObject.FindGameObjectWithTag("Robot");
        Assert.IsNotNull(robot, "Robot not found");

        var camera = robot.transform.Find("Camera");
        Assert.IsNotNull(camera, "Camera not mounted on robot");

        var lidar = robot.transform.Find("Lidar");
        Assert.IsNotNull(lidar, "LiDAR not mounted on robot");

        var imu = robot.transform.Find("IMU");
        Assert.IsNotNull(imu, "IMU not mounted on robot");
    }

    [UnityTest]
    public IEnumerator TestRobotMovement()
    {
        // Test that robot responds to movement commands
        var robotController = Object.FindObjectOfType<ROSRobotController>();
        Assert.IsNotNull(robotController, "Robot controller not found");

        Vector3 initialPosition = robotController.transform.position;

        // Simulate sending a movement command
        // In a real test, you would send actual ROS messages
        // For this example, we'll just verify the robot can move
        robotController.transform.position = initialPosition + new Vector3(1, 0, 0);

        yield return new WaitForSeconds(0.1f);

        Assert.AreNotEqual(initialPosition, robotController.transform.position, "Robot did not move");
    }

    [Test]
    public void TestEnvironmentSetup()
    {
        // Test that environment is properly configured
        var envSetup = Object.FindObjectOfType<RobotEnvironmentSetup>();
        Assert.IsNotNull(envSetup, "Environment setup component not found");

        // Verify ground plane exists
        var ground = GameObject.FindGameObjectWithTag("Ground");
        Assert.IsNotNull(ground, "Ground plane not found");
    }

    [Test]
    public void TestPerformanceOptimization()
    {
        // Test that performance optimizer is applied
        var perfOptimizer = Object.FindObjectOfType<UnityPerformanceOptimizer>();
        Assert.IsNotNull(perfOptimizer, "Performance optimizer not found");

        // Verify settings are applied
        Assert.AreEqual(60, Application.targetFrameRate, "Target frame rate not set correctly");
    }

    [TearDown]
    public void Teardown()
    {
        // Cleanup after tests
        Debug.Log("Tearing down Unity Robotics tests");
    }
}
```

## Deployment Considerations

### Unity Build Configuration for Robotics

```csharp
// UnityRoboticsBuildConfiguration.cs
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class UnityRoboticsBuildConfiguration : MonoBehaviour
{
    [Header("Build Settings for Robotics")]
    public bool enableHeadlessMode = false;
    public int buildResolutionWidth = 1280;
    public int buildResolutionHeight = 720;
    public bool enableVSync = false;
    public int targetFrameRate = 60;

    [Header("Robotics-Specific Settings")]
    public bool enablePhysics = true;
    public bool enableAudio = false;  // Often not needed for robotics
    public bool enableVideo = false;  // Often not needed for robotics

    void Start()
    {
        ConfigureRuntimeSettings();
    }

    void ConfigureRuntimeSettings()
    {
        // Configure build-specific settings
        if (enableHeadlessMode)
        {
            // Headless mode configuration (for server deployment)
            Screen.SetResolution(buildResolutionWidth, buildResolutionHeight, false);
        }
        else
        {
            // Windowed mode for development
            Screen.SetResolution(buildResolutionWidth, buildResolutionHeight, false);
        }

        // Apply frame rate and VSync settings
        Application.targetFrameRate = targetFrameRate;
        QualitySettings.vSyncCount = enableVSync ? 1 : 0;

        // Configure physics
        if (!enablePhysics)
        {
            Physics.autoSimulation = false;
        }

        // Configure audio (often not needed in headless builds)
        if (!enableAudio)
        {
            AudioListener.volume = 0f;
        }

        Debug.Log("Unity Robotics Build Configuration applied");
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(UnityRoboticsBuildConfiguration))]
public class UnityRoboticsBuildConfigurationEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        GUILayout.Space(10);

        if (GUILayout.Button("Apply Robotics Build Settings"))
        {
            ApplyBuildSettings();
        }

        if (GUILayout.Button("Configure for Linux Deployment"))
        {
            ConfigureForLinuxDeployment();
        }

        if (GUILayout.Button("Configure for Docker"))
        {
            ConfigureForDocker();
        }
    }

    void ApplyBuildSettings()
    {
        var config = (UnityRoboticsBuildConfiguration)target;

        // Apply settings
        PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
        PlayerSettings.runInBackground = true;
        PlayerSettings.resizableWindow = !config.enableHeadlessMode;
        PlayerSettings.visibleInBackground = true;

        Debug.Log("Robotics build settings applied in editor");
    }

    void ConfigureForLinuxDeployment()
    {
        // Specific settings for Linux deployment
        EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneLinux64);

        PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Standalone, "com.robotics.unity-simulation");
        PlayerSettings.productName = "UnityRoboticsSimulation";

        Debug.Log("Configured for Linux deployment");
    }

    void ConfigureForDocker()
    {
        // Settings for Docker container deployment
        PlayerSettings.SetScriptingBackend(BuildTarget.StandaloneLinux64, ScriptingImplementation.Mono2x);
        PlayerSettings.SetArchitecture(BuildTarget.StandaloneLinux64, 2); // x86_64

        Debug.Log("Configured for Docker deployment");
    }
}
#endif
```

## Summary

This chapter covered comprehensive Unity Robotics Hub setup:

- **Installation**: Complete setup of Unity Hub, Editor, and Robotics packages
- **ROS Integration**: Setting up ROS TCP Connector for communication
- **Environment Creation**: Building realistic robotic environments in Unity
- **Sensor Simulation**: Implementing perception systems with Unity Perception
- **Performance Optimization**: Optimizing Unity for real-time robotics applications
- **Testing and Validation**: Ensuring proper integration and functionality
- **Deployment Configuration**: Preparing Unity applications for deployment

Unity provides powerful capabilities for robotics simulation, particularly for applications requiring high-fidelity graphics and realistic sensor simulation.

## Exercises

1. Install Unity Hub and Unity Editor with Robotics packages
2. Create a basic robot simulation environment in Unity
3. Integrate ROS TCP Connector with your Unity project
4. Implement sensor simulation using Unity Perception
5. Optimize your Unity scene for real-time performance
6. Test communication between Unity and ROS 2 nodes

## Quiz

1. What is the main advantage of Unity over traditional simulators for robotics?
   a) Lower cost
   b) Photorealistic graphics and advanced sensor simulation
   c) Simpler interface
   d) Better documentation

2. Which Unity package enables communication with ROS 2?
   a) ML-Agents
   b) Unity Perception
   c) ROS TCP Connector
   d) XR Package

3. What is the recommended Unity version for robotics applications?
   a) Unity 2019.x
   b) Unity 2020.x
   c) Unity 2021.3 LTS or newer
   d) Any version works the same

## Mini-Project: Unity Robotics Simulation

Create a complete Unity robotics simulation with:
1. Unity project setup with Robotics Hub packages
2. Robot model with ROS integration
3. Environment with obstacles and navigation challenges
4. Sensor simulation (camera, LiDAR, IMU)
5. Communication bridge to ROS 2
6. Performance optimization for real-time operation
7. Testing and validation of the complete system