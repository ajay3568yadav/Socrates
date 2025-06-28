import React, { useState, useRef, useEffect } from 'react';
import '../css/ChatInput.css';

const ChatInput = ({ onSendMessage, isLoading, splitPaneMode = false }) => {
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
    <div className={`chat-input-area ${splitPaneMode ? 'split-pane-mode' : ''}`}>
      <div className="chat-input-container">
        <div className="input-container">
          {/* Left side - Tools button */}
          <div className="input-left-actions">
            <button className="tools-btn" title="Add attachments or tools">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 5v14m-7-7h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <span className="tools-text">Tools</span>
          </div>
          
          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={splitPaneMode ? "Ask about the code..." : "Ask anything"}
            className="input-field"
            rows={1}
            disabled={isLoading}
          />
          
          {/* Right side - Voice and Send buttons */}
          <div className="input-right-actions">
            <button className="voice-btn" title="Voice input">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" stroke="currentColor" strokeWidth="2"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <line x1="8" y1="23" x2="16" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
            
            <button 
              onClick={handleSubmit}
              disabled={!inputValue.trim() || isLoading}
              className="send-btn"
              title="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>
        
        {/* Disclaimer text */}
        <div className="input-disclaimer">
          Socrates can make mistakes. Check important info.
        </div>
      </div>
    </div>
  );
};

export default ChatInput;