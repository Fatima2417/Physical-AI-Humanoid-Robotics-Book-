import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog', 'b2f'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive', '182'),
    exact: true
  },
  {
    path: '/blog/authors',
    component: ComponentCreator('/blog/authors', '0b7'),
    exact: true
  },
  {
    path: '/blog/authors/all-sebastien-lorber-articles',
    component: ComponentCreator('/blog/authors/all-sebastien-lorber-articles', '4a1'),
    exact: true
  },
  {
    path: '/blog/authors/yangshun',
    component: ComponentCreator('/blog/authors/yangshun', 'a68'),
    exact: true
  },
  {
    path: '/blog/first-blog-post',
    component: ComponentCreator('/blog/first-blog-post', '89a'),
    exact: true
  },
  {
    path: '/blog/long-blog-post',
    component: ComponentCreator('/blog/long-blog-post', '9ad'),
    exact: true
  },
  {
    path: '/blog/mdx-blog-post',
    component: ComponentCreator('/blog/mdx-blog-post', 'e9f'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags', '287'),
    exact: true
  },
  {
    path: '/blog/tags/docusaurus',
    component: ComponentCreator('/blog/tags/docusaurus', '704'),
    exact: true
  },
  {
    path: '/blog/tags/facebook',
    component: ComponentCreator('/blog/tags/facebook', '858'),
    exact: true
  },
  {
    path: '/blog/tags/hello',
    component: ComponentCreator('/blog/tags/hello', '299'),
    exact: true
  },
  {
    path: '/blog/tags/hola',
    component: ComponentCreator('/blog/tags/hola', '00d'),
    exact: true
  },
  {
    path: '/blog/welcome',
    component: ComponentCreator('/blog/welcome', 'd2b'),
    exact: true
  },
  {
    path: '/markdown-page',
    component: ComponentCreator('/markdown-page', '3d7'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'e0a'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'a12'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '448'),
            routes: [
              {
                path: '/docs/Introduction/fundamentals-of-embodied-intelligence',
                component: ComponentCreator('/docs/Introduction/fundamentals-of-embodied-intelligence', 'f6f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Introduction/introduction-to-physical-ai',
                component: ComponentCreator('/docs/Introduction/introduction-to-physical-ai', '2ee'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-1/ros2-actions-and-parameters',
                component: ComponentCreator('/docs/Module-1/ros2-actions-and-parameters', 'e14'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-1/ros2-launch-files-and-composition',
                component: ComponentCreator('/docs/Module-1/ros2-launch-files-and-composition', 'e00'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-1/ros2-nodes-topics-services',
                component: ComponentCreator('/docs/Module-1/ros2-nodes-topics-services', 'c5f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-1/ros2-overview',
                component: ComponentCreator('/docs/Module-1/ros2-overview', '7b3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-1/ros2-practical-workflows',
                component: ComponentCreator('/docs/Module-1/ros2-practical-workflows', '132'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-2/gazebo-classic-vs-fortress',
                component: ComponentCreator('/docs/Module-2/gazebo-classic-vs-fortress', '138'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-2/robot-modeling-and-simulation',
                component: ComponentCreator('/docs/Module-2/robot-modeling-and-simulation', 'fdd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-2/sensor-integration-and-robotics',
                component: ComponentCreator('/docs/Module-2/sensor-integration-and-robotics', '902'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-2/simulation-environments-overview',
                component: ComponentCreator('/docs/Module-2/simulation-environments-overview', '1fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-2/unity-robotics-hub-setup',
                component: ComponentCreator('/docs/Module-2/unity-robotics-hub-setup', 'f2b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-3/control-pipelines',
                component: ComponentCreator('/docs/Module-3/control-pipelines', 'e12'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-3/isaac-ros-bridge',
                component: ComponentCreator('/docs/Module-3/isaac-ros-bridge', '174'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-3/isaac-sim-getting-started',
                component: ComponentCreator('/docs/Module-3/isaac-sim-getting-started', '468'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-3/nvidia-isaac-overview',
                component: ComponentCreator('/docs/Module-3/nvidia-isaac-overview', '4d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-3/perception-pipelines',
                component: ComponentCreator('/docs/Module-3/perception-pipelines', '66e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/humanoid-control-systems',
                component: ComponentCreator('/docs/Module-4/humanoid-control-systems', '0ea'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/integration-patterns',
                component: ComponentCreator('/docs/Module-4/integration-patterns', 'b2f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/llm-cognitive-planning',
                component: ComponentCreator('/docs/Module-4/llm-cognitive-planning', 'bc1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/perception-integration',
                component: ComponentCreator('/docs/Module-4/perception-integration', '423'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/real-world-deployment',
                component: ComponentCreator('/docs/Module-4/real-world-deployment', '002'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/ros2-action-execution',
                component: ComponentCreator('/docs/Module-4/ros2-action-execution', 'cea'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/sensor-integration-ros2',
                component: ComponentCreator('/docs/Module-4/sensor-integration-ros2', 'd02'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/unity-robotics-hub-setup',
                component: ComponentCreator('/docs/Module-4/unity-robotics-hub-setup', 'd06'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vision-language-action-overview',
                component: ComponentCreator('/docs/Module-4/vision-language-action-overview', '3be'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-applications-humanoids',
                component: ComponentCreator('/docs/Module-4/vla-applications-humanoids', '607'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-architectures',
                component: ComponentCreator('/docs/Module-4/vla-architectures', '151'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-communication-protocols',
                component: ComponentCreator('/docs/Module-4/vla-communication-protocols', 'afa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-deployment-considerations',
                component: ComponentCreator('/docs/Module-4/vla-deployment-considerations', '0b5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-evaluation-testing',
                component: ComponentCreator('/docs/Module-4/vla-evaluation-testing', '7fb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-summary-conclusion',
                component: ComponentCreator('/docs/Module-4/vla-summary-conclusion', '083'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/vla-training-methods',
                component: ComponentCreator('/docs/Module-4/vla-training-methods', '4bb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/Module-4/whisper-integration',
                component: ComponentCreator('/docs/Module-4/whisper-integration', 'cde'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
