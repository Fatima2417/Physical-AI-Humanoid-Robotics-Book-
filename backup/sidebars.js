// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro/01_introduction-to-physical-ai',
        'intro/02_fundamentals-of-embodied-intelligence'
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'module1-ros2/01_ros2-overview',
        'module1-ros2/02_ros2-nodes-topics-services',
        'module1-ros2/03_ros2-actions-and-parameters',
        'module1-ros2/04_ros2-launch-files-and-composition',
        'module1-ros2/05_ros2-practical-workflows'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity Simulation',
      items: [
        'module2-gazebo-unity/01_simulation-environments-overview',
        'module2-gazebo-unity/02_gazebo-classic-vs-fortress',
        'module2-gazebo-unity/03_unity-robotics-hub-setup',
        'module2-gazebo-unity/04_robot-modeling-and-simulation',
        'module2-gazebo-unity/05_sensor-integration-and-robotics'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac',
      items: [
        'module3-isaac/01_nvidia-isaac-overview',
        'module3-isaac/02_isaac-sim-getting-started',
        'module3-isaac/03_isaac-ros-bridge',
        'module3-isaac/04_perception-pipelines',
        'module3-isaac/05_control-pipelines'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'module4-vla/01_vision-language-action-overview',
        'module4-vla/02_whisper-integration',
        'module4-vla/03_llm-cognitive-planning',
        'module4-vla/04_ros2-action-execution'
      ],
    },
    {
      type: 'category',
      label: 'Capstone: Autonomous Humanoid',
      items: [
        'capstone/01_capstone-overview',
        'capstone/02_system-integration',
        'capstone/03_autonomous-behaviors',
        'capstone/04_final-project'
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/a_ros2-command-reference',
        'appendices/b_sensor-specifications',
        'appendices/c_urdf-tutorial',
        'appendices/d_troubleshooting-guide'
      ],
    }
  ],
};

module.exports = sidebars;