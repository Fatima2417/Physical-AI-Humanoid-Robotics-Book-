// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'A comprehensive guide to embodied intelligence and humanoid robotics',
  favicon: 'img/favicon.ico',

  // Main GitHub Pages URL (no subpath!)
  url: 'https://Fatima2417.github.io', 
  // Project subpath (must match your repo name, include trailing slash)
  baseUrl: '/',

  // GitHub pages deployment config
  organizationName: 'Fatima2417', // Your GitHub username
  projectName: 'Physical-AI-Humanoid-Robotics-Book-', // Your repo name
  deploymentBranch: 'gh-pages', // branch for GitHub Pages

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/Fatima2417/Physical-AI-Humanoid-Robotics-Book-/tree/main/',
        },
        blog: false, // Disable blog plugin
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            href: 'https://github.com/Fatima2417/Physical-AI-Humanoid-Robotics-Book-',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Textbook',
            items: [
              {
                label: 'Introduction',
                to: '/docs/Introduction/introduction-to-physical-ai',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/physical-ai',
              },
              {
                label: 'Robotics Stack Exchange',
                href: 'https://robotics.stackexchange.com/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Fatima2417/Physical-AI-Humanoid-Robotics-Book-',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['python', 'cpp', 'bash', 'json', 'yaml'],
      },
    }),
};

module.exports = config;
