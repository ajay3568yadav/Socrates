import React, { useState, useRef, useEffect } from 'react';
import '../css/ChatInput.css';
const ChatInput = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
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

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [inputValue]);

  return (
    <div className="chat-input-area">
      <div className="chat-input-container">
        <div className="input-container">
          <div className="input-icon">🚀</div>
          
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="input-field"
            rows={1}
            disabled={isLoading}
          />
          
          <div className="input-actions">
            <button className="input-action">🎤</button>
            <button 
              onClick={handleSubmit}
              disabled={!inputValue.trim() || isLoading}
              className="send-btn"
            >
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;