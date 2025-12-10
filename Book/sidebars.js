// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      collapsed: false, // Major sections visible by default
      collapsible: true,
      items: [
        // Subsections remain visible
        'Introduction/introduction-to-physical-ai',
        'Introduction/fundamentals-of-embodied-intelligence',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      collapsed: false, // Major sections visible by default
      collapsible: true,
      items: [
        // Subsections remain visible
        'Module-1/ros2-overview',
        'Module-1/ros2-nodes-topics-services',
        'Module-1/ros2-actions-and-parameters',
        'Module-1/ros2-launch-files-and-composition',
        'Module-1/ros2-practical-workflows',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity',
      collapsed: false, // Major sections visible by default
      collapsible: true,
      items: [
        // Subsections remain visible
        'Module-2/simulation-environments-overview',
        'Module-2/gazebo-classic-vs-fortress',
        'Module-2/unity-robotics-hub-setup',
        'Module-2/robot-modeling-and-simulation',
        'Module-2/sensor-integration-and-robotics',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac',
      collapsed: false, // Major sections visible by default
      collapsible: true,
      items: [
        // Subsections remain visible
        'Module-3/nvidia-isaac-overview',
        'Module-3/isaac-sim-getting-started',
        'Module-3/isaac-ros-bridge',
        'Module-3/perception-pipelines',
        'Module-3/control-pipelines',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      collapsed: false, // Major sections visible by default
      collapsible: true,
      items: [
        // Subsections remain visible
        'Module-4/vision-language-action-overview',
        'Module-4/vla-architectures',
        'Module-4/vla-training-methods',
        'Module-4/vla-evaluation-testing',
        'Module-4/vla-applications-humanoids',
        'Module-4/whisper-integration',
        'Module-4/llm-cognitive-planning',
        'Module-4/perception-integration',
        'Module-4/sensor-integration-ros2',
        'Module-4/ros2-action-execution',
        'Module-4/integration-patterns',
        'Module-4/humanoid-control-systems',
        'Module-4/real-world-deployment',
        'Module-4/vla-communication-protocols',
        'Module-4/vla-deployment-considerations',
        'Module-4/vla-summary-conclusion',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
