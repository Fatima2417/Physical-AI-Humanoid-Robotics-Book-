import React, { useState, useEffect, useRef } from 'react';
import apiService from '../services/api';
import '../styles/chatbot.css';

const Chatbot = ({ backendUrl = 'http://localhost:8000' }) => {
  // Update API service with the provided backend URL
  apiService.backendUrl = backendUrl;

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState('global'); // 'global' or 'selected'
  const [selectedText, setSelectedText] = useState('');
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  // Function to get selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      let data;
      if (mode === 'global') {
        // Global query to entire book corpus
        data = await apiService.queryGlobal(inputValue, true);
      } else {
        // Selected text query
        data = await apiService.querySelected(inputValue, selectedText, window.location.pathname);
      }

      // Add bot response to chat
      const botMessage = {
        id: Date.now() + 1,
        text: data.answer,
        sender: 'bot',
        citations: data.citations || [],
        timestamp: new Date(),
        confidence: data.confidence_score
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setError('Failed to get response from the server. Please try again.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="chatbot-header-left">
              <h3>Book Assistant</h3>
              <div className="mode-toggle">
                <button
                  className={mode === 'global' ? 'active' : ''}
                  onClick={() => setMode('global')}
                >
                  Global
                </button>
                <button
                  className={mode === 'selected' ? 'active' : ''}
                  onClick={() => setMode('selected')}
                  disabled={!selectedText}
                  title={selectedText ? 'Ask about selected text' : 'Select text on page first'}
                >
                  Selected Text
                </button>
              </div>
            </div>
            <div className="chatbot-header-right">
              <button onClick={clearChat} className="clear-btn" title="Clear chat">
                ✕
              </button>
              <button onClick={toggleChat} className="close-btn" title="Close chat">
                −
              </button>
            </div>
          </div>

          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your book assistant. Ask me anything about the Physical AI & Humanoid Robotics content.</p>
                {mode === 'selected' && selectedText && (
                  <p className="selected-text-preview">
                    <strong>Selected text:</strong> {selectedText.substring(0, 100)}...
                  </p>
                )}
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.sender}`}
                >
                  <div className="message-content">
                    <p>{message.text}</p>
                    {message.citations && message.citations.length > 0 && (
                      <div className="citations">
                        <h4>Sources:</h4>
                        <ul>
                          {message.citations.map((citation, idx) => (
                            <li key={idx}>
                              <a href={citation.section_path} target="_blank" rel="noopener noreferrer">
                                {citation.section_title}
                              </a>
                              {citation.page_number && <span> (p. {citation.page_number})</span>}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <p>Thinking...</p>
                </div>
              </div>
            )}
            {error && (
              <div className="message error">
                <div className="message-content">
                  <p>{error}</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-area">
            {mode === 'selected' && selectedText && (
              <div className="selected-text-indicator">
                Using selected text: "{selectedText.substring(0, 50)}..."
              </div>
            )}
            <div className="input-container">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  mode === 'global'
                    ? 'Ask about the book content...'
                    : selectedText
                      ? 'Ask about selected text...'
                      : 'Select text on the page first...'
                }
                disabled={mode === 'selected' && !selectedText}
                rows="1"
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading || (mode === 'selected' && !selectedText)}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      ) : (
        <button className="chatbot-toggle" onClick={toggleChat}>
          <span>🤖</span>
        </button>
      )}
    </div>
  );
};

export default Chatbot;