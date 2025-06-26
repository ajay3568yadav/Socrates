import React from 'react';
import '../css/TypingIndicator.css'; 
const TypingIndicator = () => {
  return (
    <div className="typing-indicator">
      <div className="message-avatar">🤖</div>
      <div className="typing-content">
        <div className="typing-text">
          <span>Thinking</span>
          <div className="typing-dots">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;