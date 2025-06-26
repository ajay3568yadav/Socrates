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
  splitPaneMode = false
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
    <div className={`chat-view ${splitPaneMode ? 'split-pane-mode' : ''}`}>
      <div 
        className="messages-container" 
        ref={messagesContainerRef}
        onScroll={handleScroll}
      >
        <div className="messages-wrapper">
          {messages.map((message) => (
            <Message 
              key={message.id} 
              message={message} 
              onSendMessage={onSendMessage}
              isLoading={isLoading}
              onOpenCodeEditor={onOpenCodeEditor}
              splitPaneMode={splitPaneMode}
            />
          ))}
          
          {isLoading && <TypingIndicator />}
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
        />
      </div>
    </div>
  );
};

export default ChatView;