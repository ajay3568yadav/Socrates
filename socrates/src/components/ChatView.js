import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import TypingIndicator from './TypingIndicator';
import ChatInput from './ChatInput';
import CodePanel from './CodePanel';
import '../css/ChatView.css';
const ChatView = ({ messages, isLoading, onSendMessage }) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  
  // Code panel state
  const [codePanelOpen, setCodePanelOpen] = useState(false);
  const [currentCode, setCurrentCode] = useState('');
  const [currentLanguage, setCurrentLanguage] = useState('cuda');

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

  const handleOpenCodeEditor = (code, language = 'cuda') => {
    setCurrentCode(code);
    setCurrentLanguage(language);
    setCodePanelOpen(true);
  };

  const handleCloseCodePanel = () => {
    console.log('ChatView: Closing code panel'); // Debug log
    setCodePanelOpen(false);
    // Clean up any CSS variables
    document.documentElement.style.removeProperty('--code-panel-width');
  };

  const handleCodeReview = (reviewPrompt) => {
    if (onSendMessage) {
      onSendMessage(reviewPrompt);
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
    <div className={`chat-view ${codePanelOpen ? 'with-code-panel' : ''}`}>
      <div 
        className="messages-container" 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        style={{ 
          width: codePanelOpen ? 'calc(100% - var(--code-panel-width, 50%))' : '100%',
          transition: 'width 0.3s ease'
        }}
      >
        <div className="messages-wrapper">
          {messages.map((message) => (
            <Message 
              key={message.id} 
              message={message} 
              onSendMessage={onSendMessage}
              isLoading={isLoading}
              onOpenCodeEditor={handleOpenCodeEditor}
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

      <div 
        className="chat-input-container"
        style={{ 
          width: codePanelOpen ? 'calc(100% - var(--code-panel-width, 50%))' : '100%',
          transition: 'width 0.3s ease'
        }}
      >
        <ChatInput onSendMessage={onSendMessage} isLoading={isLoading} />
      </div>

      {/* Code Panel */}
      <CodePanel
        isOpen={codePanelOpen}
        onClose={handleCloseCodePanel}
        initialCode={currentCode}
        language={currentLanguage}
        onSendForReview={handleCodeReview}
        isLoading={isLoading}
        title={`${currentLanguage.toUpperCase()} Code Editor`}
      />
    </div>
  );
};

export default ChatView;