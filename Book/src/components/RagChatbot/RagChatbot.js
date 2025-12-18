import React, { useState, useRef, useEffect } from 'react';
import './RagChatbot.css';

const RagChatbot = ({ backendUrl = 'http://localhost:9000' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [mode, setMode] = useState('global'); // 'global' or 'selected_text'
  const [sessionId, setSessionId] = useState(null);

  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

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
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Check for simple greetings or non-content questions
      const lowerInput = inputValue.toLowerCase().trim();
      const greetingKeywords = ['hello', 'hi', 'hey', 'greetings', 'help', 'welcome'];
      const isGreeting = greetingKeywords.some(keyword => lowerInput.includes(keyword));

      let botMessage;

      if (isGreeting) {
        // Handle greetings with a friendly response
        botMessage = {
          id: Date.now() + 1,
          text: "Hello! I'm your Physical AI & Humanoid Robotics Assistant. I can help answer questions about the book content. Ask me anything about the topics covered in the book!",
          sender: 'bot',
          citations: [],
          confidence: 1.0,
          timestamp: new Date()
        };
      } else {
        // Make API call for actual questions
        let response;
        if (mode === 'selected_text' && selectedText) {
          // Use selected text mode
          response = await fetch(`${backendUrl}/api/v1/query-selected`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: inputValue,
              selected_text: selectedText,
              page_context: window.location.pathname,
              session_id: sessionId
            })
          });
        } else {
          // Use global mode
          response = await fetch(`${backendUrl}/api/v1/query`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: inputValue,
              session_id: sessionId,
              include_citations: true
            })
          });
        }

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update session ID if not set
        if (!sessionId && data.session_id) {
          setSessionId(data.session_id);
        }

        // Add bot response
        botMessage = {
          id: Date.now() + 1,
          text: data.answer,
          sender: 'bot',
          citations: data.citations || [],
          confidence: data.confidence_score,
          timestamp: new Date()
        };
      }

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'bot',
        isError: true,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const clearChat = () => {
    setMessages([]);
  };

  const switchMode = (newMode) => {
    setMode(newMode);
  };

  return (
    <>
      {/* Floating chat button */}
      {!isOpen && (
        <button
          className="rag-chatbot-float-button"
          onClick={toggleChat}
          title="Open RAG Chatbot"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H17.77L15.65 20.71C15.5309 20.9231 15.3401 21.0899 15.1099 21.1837C14.8797 21.2775 14.6225 21.2919 14.38 21.225C14.1375 21.1581 13.9231 21.0139 13.7758 20.8163C13.6285 20.6187 13.5571 20.3784 13.57 20.13L14.28 14.13C14.29 14.05 14.29 13.96 14.29 13.88C14.29 13.8 14.29 13.71 14.28 13.63L13.63 8.38C13.55 7.63 12.9 7 12 7H6C4.9 7 4 7.9 4 9V17C4 18.1 4.9 19 6 19H11.71C11.8 19 11.88 19 11.96 18.99C12.04 18.99 12.13 18.99 12.21 18.99L15.21 18.99C15.6 18.99 15.93 18.72 16.02 18.35L16.65 15.85C16.74 15.48 16.56 15.1 16.21 14.91L15.73 14.66L16.21 14.91C16.56 15.1 16.74 15.48 16.65 15.85L16.02 18.35C15.93 18.72 15.6 18.99 15.21 18.99H12.21C12.13 18.99 12.04 18.99 11.96 18.99C11.88 18.99 11.8 18.99 11.71 18.99H6C5.47 18.99 5 18.52 5 17.99V9C5 8.47 5.47 8 6 8H12C12.37 8 12.7 8.21 12.86 8.55L13.5 13.88C13.5 13.96 13.5 14.05 13.5 14.13L12.77 20.38L15.62 18.38L19 18.38C19.5304 18.38 20.0391 18.1693 20.4142 17.7942C20.7893 17.4191 21 16.9104 21 16.38V15ZM17 11C17 11.5523 16.5523 12 16 12C15.4477 12 15 11.5523 15 11C15 10.4477 15.4477 10 16 10C16.5523 10 17 10.4477 17 11ZM12 11C12 11.5523 11.5523 12 11 12C10.4477 12 10 11.5523 10 11C10 10.4477 10.4477 10 11 10C11.5523 10 12 10.4477 12 11ZM8 11C8 11.5523 7.55228 12 7 12C6.44772 12 6 11.5523 6 11C6 10.4477 6.44772 10 7 10C7.55228 10 8 10.4477 8 11Z" fill="currentColor"/>
          </svg>
        </button>
      )}

      {/* Chat modal */}
      {isOpen && (
        <div className="rag-chatbot-modal">
          <div className="rag-chatbot-container" ref={chatContainerRef}>
            {/* Chat header */}
            <div className="rag-chatbot-header">
              <div className="rag-chatbot-header-content">
                <h3>Physical AI & Humanoid Robotics Assistant</h3>
                <div className="rag-chatbot-controls">
                  <div className="rag-chatbot-mode-selector">
                    <button
                      className={`rag-chatbot-mode-btn ${mode === 'global' ? 'active' : ''}`}
                      onClick={() => switchMode('global')}
                      title="Global mode - search entire book"
                    >
                      Global
                    </button>
                    <button
                      className={`rag-chatbot-mode-btn ${mode === 'selected_text' ? 'active' : ''}`}
                      onClick={() => switchMode('selected_text')}
                      title="Selected text mode - answer based on highlighted text"
                      disabled={!selectedText}
                    >
                      Selected Text
                    </button>
                  </div>
                  <button
                    className="rag-chatbot-clear-btn"
                    onClick={clearChat}
                    title="Clear chat"
                  >
                    Clear
                  </button>
                  <button
                    className="rag-chatbot-close-btn"
                    onClick={toggleChat}
                    title="Close chat"
                  >
                    ×
                  </button>
                </div>
              </div>

              {mode === 'selected_text' && selectedText && (
                <div className="rag-chatbot-selected-text-preview">
                  <small>Selected: "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"</small>
                </div>
              )}
            </div>

            {/* Messages container */}
            <div className="rag-chatbot-messages">
              {messages.length === 0 ? (
                <div className="rag-chatbot-welcome">
                  <h4>Hello! I'm your Physical AI & Humanoid Robotics Assistant.</h4>
                  <p>Ask me anything about the book content!</p>
                  <div className="rag-chatbot-features">
                    <div className="rag-chatbot-feature">
                      <strong>Global Mode</strong>: Search entire book corpus
                    </div>
                    <div className="rag-chatbot-feature">
                      <strong>Selected Text Mode</strong>: Answer based on highlighted text
                    </div>
                  </div>
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`rag-chatbot-message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
                  >
                    <div className="rag-chatbot-message-content">
                      <div className="rag-chatbot-message-text">
                        {message.text}
                      </div>

                      {message.sender === 'bot' && message.citations && message.citations.length > 0 && (
                        <div className="rag-chatbot-citations">
                          <details className="rag-chatbot-citations-details">
                            <summary>Sources ({message.citations.length})</summary>
                            {message.citations.map((citation, index) => (
                              <div key={index} className="rag-chatbot-citation">
                                <div className="rag-chatbot-citation-title">
                                  {citation.section_title || 'Section'}
                                </div>
                                <div className="rag-chatbot-citation-snippet">
                                  {citation.text_snippet}
                                </div>
                              </div>
                            ))}
                          </details>
                        </div>
                      )}

                      {message.isError && (
                        <div className="rag-chatbot-error">
                          Error occurred. Please check your query or try again.
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="rag-chatbot-message bot-message">
                  <div className="rag-chatbot-message-content">
                    <div className="rag-chatbot-typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input area */}
            <form className="rag-chatbot-input-form" onSubmit={handleSubmit}>
              {mode === 'selected_text' && !selectedText && (
                <div className="rag-chatbot-warning">
                  Please select text on the page to use Selected Text mode.
                </div>
              )}
              <div className="rag-chatbot-input-container">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder={
                    mode === 'selected_text' && !selectedText
                      ? "Select text on the page first..."
                      : "Ask about the Physical AI & Humanoid Robotics book..."
                  }
                  disabled={isLoading || (mode === 'selected_text' && !selectedText)}
                  className="rag-chatbot-input"
                />
                <button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading || (mode === 'selected_text' && !selectedText)}
                  className="rag-chatbot-send-btn"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default RagChatbot;