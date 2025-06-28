import React, { useState, useRef, useEffect } from 'react';
import '../css/ChatInput.css';

const ChatInput = ({ 
  onSendMessage, 
  isLoading, 
  splitPaneMode = false, 
  tutoringMode = false // NEW: Tutoring mode prop
}) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [message]);

  // Focus on mount and when tutoring mode changes
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [tutoringMode]);

  const getPlaceholder = () => {
    if (tutoringMode) {
      return "Type your answer or ask a question... (e.g., 'A' for complete beginner)";
    }
    return splitPaneMode 
      ? "Ask about the code or request changes..." 
      : "Ask me anything about CUDA programming...";
  };

  const getSendButtonColor = () => {
    if (tutoringMode) {
      return '#3b82f6';
    }
    return '#22c55e';
  };

  return (
    <div className={`chat-input-container ${splitPaneMode ? 'split-mode' : ''} ${tutoringMode ? 'tutoring-mode' : ''}`}>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={getPlaceholder()}
            className={`message-input ${tutoringMode ? 'tutoring' : ''}`}
            disabled={isLoading}
            rows={1}
          />
          <button
            type="submit"
            disabled={!message.trim() || isLoading}
            className={`send-button ${tutoringMode ? 'tutoring' : ''}`}
            style={{ 
              backgroundColor: !message.trim() || isLoading ? '#333' : getSendButtonColor() 
            }}
            title={tutoringMode ? "Send answer" : "Send message"}
          >
            {isLoading ? (
              <div className="loading-spinner">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <line x1="22" y1="2" x2="11" y2="13" stroke="currentColor" strokeWidth="2"/>
                <polygon points="22,2 15,22 11,13 2,9 22,2" fill="currentColor"/>
              </svg>
            )}
          </button>
        </div>
        
        {/* NEW: Tutoring mode hint */}
        {tutoringMode && (
          <div className="tutoring-hint">
            <span className="hint-icon">💡</span>
            <span className="hint-text">
              This is an interactive tutoring session. Answer questions to help me tailor the lesson to your level!
            </span>
          </div>
        )}
        
        {/* Existing disclaimer */}
        <div className={`input-disclaimer ${tutoringMode ? 'tutoring' : ''}`}>
          {tutoringMode 
            ? "🎓 Interactive tutoring session - ask questions anytime!"
            : "CUDA Tutor can make mistakes. Consider checking important information."
          }
        </div>
      </form>
    </div>
  );
};

export default ChatInput;