// Message.js
import React from 'react';

const Message = ({ message }) => {
  const isUser = message.role === 'user';

  const formatMessage = (content = '') => {
    // Convert code blocks
    content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    
    // Convert inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    return content;
  };

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div
          className="message-text"
          dangerouslySetInnerHTML={{
            __html: formatMessage(message.content)
          }}
        />
      </div>
    </div>
  );
};

export default Message;