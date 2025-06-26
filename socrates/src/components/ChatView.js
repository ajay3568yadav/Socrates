import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import StreamingMessage from './StreamingMessage';
import TypingIndicator from './TypingIndicator';
import ChatInput from './ChatInput';

const ChatView = ({ messages, isLoading, onSendMessage }) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState(null);

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

  const handleStreamingComplete = () => {
    setStreamingMessageId(null);
  };

  useEffect(() => {
    // Scroll immediately when messages change
    scrollToBottom();
    
    // Set streaming message ID for the latest assistant message
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && !lastMessage.isUser && !isLoading) {
      setStreamingMessageId(lastMessage.id);
    }
    
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

  // Scroll during streaming
  useEffect(() => {
    if (streamingMessageId) {
      const interval = setInterval(scrollToBottom, 100);
      return () => clearInterval(interval);
    }
  }, [streamingMessageId]);

  return (
    <div className="chat-view">
      <div 
        className="messages-container" 
        ref={messagesContainerRef}
        onScroll={handleScroll}
      >
        <div className="messages-wrapper">
          {messages.map((message, index) => {
            // Check if this is the latest assistant message and should be streamed
            const shouldStream = !message.isUser && 
                               message.id === streamingMessageId && 
                               index === messages.length - 1;
                               
            return shouldStream ? (
              <StreamingMessage 
                key={message.id} 
                message={message} 
                onComplete={handleStreamingComplete}
              />
            ) : (
              <Message key={message.id} message={message} />
            );
          })}
          
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

      <ChatInput onSendMessage={onSendMessage} isLoading={isLoading} />
    </div>
  );
};

export default ChatView;