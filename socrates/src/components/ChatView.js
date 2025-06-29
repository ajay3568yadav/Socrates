import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import TypingIndicator from './TypingIndicator';
import ChatInput from './ChatInput';
import '../css/ChatView.css';

const ChatView = ({ 
  messages, 
  isLoading, 
  onSendMessage, 
  onOpenCodeEditor,
  splitPaneMode = false,
  tutoringMode = false,
  currentChatId
}) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const scrollToBottom = () => {
    // Multiple scroll methods for better compatibility
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
    
    // Fallback: scroll container to bottom
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom);
    }
  };

  useEffect(() => {
    // Scroll immediately when messages change
    scrollToBottom();
    
    // Also scroll after a short delay to handle dynamic content loading
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(timeoutId);
  }, [messages, isLoading]);

  // Force scroll when component mounts
  useEffect(() => {
    scrollToBottom();
  }, []);

  return (
    <div className={`chat-view ${splitPaneMode ? 'split-pane-mode' : ''} ${tutoringMode ? 'tutoring-mode' : ''}`}>
      {/* NEW: Tutoring Mode Header */}
      {tutoringMode && (
        <div className="tutoring-header">
          <div className="tutoring-header-content">
            <div className="tutoring-icon">🎓</div>
            <div className="tutoring-info">
              <div className="tutoring-title">Interactive Tutoring Session</div>
              <div className="tutoring-subtitle">Your AI tutor will guide you step by step</div>
            </div>
            <div className="tutoring-badge">
              <div className="tutoring-badge-dot"></div>
              <span>Live</span>
            </div>
          </div>
        </div>
      )}

      <div 
        className="messages-container" 
        ref={messagesContainerRef}
        onScroll={handleScroll}
      >
        <div className="messages-wrapper">
          {/* NEW: Tutoring mode welcome message */}
          {tutoringMode && messages.length === 0 && (
            <div className="tutoring-welcome">
              <div className="tutoring-welcome-icon">🎓</div>
              <div className="tutoring-welcome-content">
                <h3>Welcome to Tutoring Mode!</h3>
                <p>
                  Your AI tutor will guide you through interactive learning with questions, 
                  explanations, and hands-on exercises. Feel free to ask questions or 
                  request clarification at any time.
                </p>
                <div className="tutoring-tips">
                  <div className="tip-item">
                    <span className="tip-icon">💡</span>
                    <span>Ask for examples and detailed explanations</span>
                  </div>
                  <div className="tip-item">
                    <span className="tip-icon">❓</span>
                    <span>Answer questions to test your understanding</span>
                  </div>
                  <div className="tip-item">
                    <span className="tip-icon">🧪</span>
                    <span>Practice with code examples and exercises</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {messages.map((message) => (
          <Message 
            key={message.id} 
            message={message} 
            onSendMessage={onSendMessage}
            isLoading={isLoading}
            onOpenCodeEditor={onOpenCodeEditor}
            splitPaneMode={splitPaneMode}
            tutoringMode={tutoringMode}
            currentChatId={currentChatId} // ADD THIS LINE
          />
          ))}
          
          {isLoading && <TypingIndicator splitPaneMode={splitPaneMode} tutoringMode={tutoringMode} />}
          <div ref={messagesEndRef} style={{ height: '1px', marginTop: '20px' }} />
        </div>
        
        {/* Scroll to bottom button */}
        {showScrollButton && (
          <button 
            className="scroll-to-bottom-btn"
            onClick={scrollToBottom}
            title="Scroll to bottom"
          >
            ↓
          </button>
        )}
      </div>

      {/* Chat Input */}
      <div className="chat-input-wrapper">
        <ChatInput 
          onSendMessage={onSendMessage} 
          isLoading={isLoading}
          splitPaneMode={splitPaneMode}
          tutoringMode={tutoringMode} // NEW: Pass tutoring mode to input
        />
      </div>
    </div>
  );
};

export default ChatView;