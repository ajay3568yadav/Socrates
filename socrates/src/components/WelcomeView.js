import React, { useState, useRef, useEffect } from 'react';
import '../css/WelcomeView.css'; 

const WelcomeView = ({ onSendMessage, user, codePanelOpen = false }) => {
  const [inputValue, setInputValue] = useState('');
  const [activeTab, setActiveTab] = useState('All');
  const textareaRef = useRef(null);

  const tabs = ['All', 'Text', 'Image', 'Video', 'Music', 'Analytics'];
  
  const quickActions = [
    {
      icon: '📚',
      title: 'CUDA Basics',
      description: 'Learn the fundamentals of CUDA programming',
      question: 'What is CUDA and why is it useful?'
    },
    {
      icon: '⚡',
      title: 'Kernel Examples',
      description: 'Get started with your first CUDA kernel',
      question: 'Show me a simple CUDA kernel example'
    },
    {
      icon: '🧠',
      title: 'Memory Optimization',
      description: 'Master efficient memory usage patterns',
      question: 'How do I optimize memory access in CUDA?'
    },
    {
      icon: '🏆',
      title: 'Best Practices',
      description: 'Professional CUDA development tips',
      question: 'What are the best practices for CUDA programming?'
    }
  ];

  const handleSubmit = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleQuickAction = (question) => {
    onSendMessage(question);
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [inputValue]);

  const getUserName = () => {
    if (user?.user_metadata?.full_name) {
      return user.user_metadata.full_name;
    }
    if (user?.email) {
      return user.email.split('@')[0];
    }
    return 'there';
  };

  // Calculate width when code panel is open
  const contentWidth = codePanelOpen ? 'calc(100% - var(--code-panel-width, 50%))' : '100%';

  return (
    <div 
      className="welcome-container"
      style={{ 
        width: contentWidth,
        transition: 'width 0.3s ease',
        maxWidth: codePanelOpen ? 'none' : '100%'
      }}
    >
      <div className="welcome-icon">🚀</div>

      <h1 className="welcome-title">
        How can I help you today{user ? `, ${getUserName()}` : ''}?
      </h1>
      
      <p className="welcome-subtitle">
        Ask me anything about CUDA programming, GPU optimization, or parallel computing.
        I'm here to help you master GPU development!
      </p>

      <div className="quick-actions">
        {quickActions.map((action, index) => (
          <div 
            key={index}
            onClick={() => handleQuickAction(action.question)}
            className="quick-action"
          >
            <div className="quick-action-icon">{action.icon}</div>
            <div className="quick-action-title">{action.title}</div>
            <div className="quick-action-desc">{action.description}</div>
          </div>
        ))}
      </div>

      <div className="input-section">
        <div className="input-tabs">
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`input-tab ${activeTab === tab ? 'active' : ''}`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="input-container">
          <div className="input-icon">🚀</div>
          
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your prompt here..."
            className="input-field"
            rows={1}
          />
          
          <div className="input-actions">
            <button className="input-action">🎤</button>
            <button 
              onClick={handleSubmit}
              disabled={!inputValue.trim()}
              className="send-btn"
            >
              →
            </button>
          </div>
        </div>

        <p className="disclaimer">
          CUDA Tutor can make mistakes. Consider checking important information.
        </p>
      </div>
    </div>
  );
};

export default WelcomeView;