// Client module to initialize the chatbot
import React from 'react';
import { createRoot } from 'react-dom/client';
import Chatbot from '@chatbot/components/Chatbot';

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('rag-chatbot-root');
  if (container) {
    const root = createRoot(container);

    // Get backend URL from site config or use default
    const backendUrl = window.chatbotConfig?.backendUrl || process.env.BACKEND_URL || 'http://localhost:8000';

    root.render(<Chatbot backendUrl={backendUrl} />);
  }
});