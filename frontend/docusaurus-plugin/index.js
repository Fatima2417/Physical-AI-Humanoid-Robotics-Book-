const path = require('path');

module.exports = function (context, options) {
  const { siteConfig } = context;
  const config = options || {};

  return {
    name: 'docusaurus-plugin-rag-chatbot',

    getClientModules() {
      return [path.resolve(__dirname, './client-modules')];
    },

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@chatbot': path.resolve(__dirname, '../src'),
          },
        },
      };
    },

    // Add the chatbot to the head of the page
    injectHtmlTags() {
      return {
        postBodyTags: [
          `<div id="rag-chatbot-root"></div>`,
        ],
      };
    },

    // Add script to initialize the chatbot
    getThemePath() {
      return path.resolve(__dirname, './theme');
    },

    plugins: [
      [
        '@docusaurus/plugin-content-blog',
        {
          id: 'chatbot',
          path: path.resolve(__dirname, '../src'),
        },
      ],
    ],
  };
};

// Export a function that will render the chatbot
module.exports.renderChatbot = function (backendUrl = 'http://localhost:8000') {
  // This function would be used to render the chatbot in Docusaurus pages
  return `
    <script>
      // Initialize the chatbot when the page loads
      document.addEventListener('DOMContentLoaded', function() {
        // Create a container for the chatbot
        const chatbotContainer = document.createElement('div');
        chatbotContainer.id = 'rag-chatbot-container';
        document.body.appendChild(chatbotContainer);

        // In a real implementation, this would mount the React component
        console.log('RAG Chatbot initialized with backend URL:', '${backendUrl}');
      });
    </script>
  `;
};