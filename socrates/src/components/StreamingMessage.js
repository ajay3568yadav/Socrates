import React, { useState, useEffect, useRef } from 'react';

const StreamingMessage = ({ message, onComplete, onSendMessage, isLoading, onOpenCodeEditor }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const messageRef = useRef(null);
  const codeBlocksRef = useRef([]);

  useEffect(() => {
    if (!message.content || currentIndex >= message.content.length) {
      setIsTyping(false);
      if (onComplete) onComplete();
      return;
    }

    const typingSpeed = message.content.length > 1000 ? 10 : 30; // Faster for long messages
    
    const timer = setTimeout(() => {
      setDisplayedText(prev => prev + message.content[currentIndex]);
      setCurrentIndex(prev => prev + 1);
    }, typingSpeed);

    return () => clearTimeout(timer);
  }, [currentIndex, message.content, onComplete]);

  // Reset when message changes
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
    setIsTyping(true);
    codeBlocksRef.current = [];
  }, [message.id]);

  const extractCodeBlocks = (content) => {
    // Clear previous code blocks if starting fresh
    if (content === message.content) {
      codeBlocksRef.current = [];
    }
    
    // First, extract and replace code blocks with styled blocks
    let codeBlockIndex = 0;
    
    // Extract code blocks
    content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
      
      // Store code block data for event handlers (avoid duplicates)
      const existingBlock = codeBlocksRef.current.find(block => block.index === codeBlockIndex);
      if (!existingBlock) {
        codeBlocksRef.current.push({
          index: codeBlockIndex,
          language: lang || 'text', 
          code: code.trim()
        });
      }
      
      codeBlockIndex++;
      return placeholder;
    });
    
    // Convert inline code
    content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    // Restore code blocks with safe data attributes
    codeBlocksRef.current.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      
      const codeBlockHtml = `
        <div class="temp-code-block-container" style="margin: 12px 0; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; overflow: hidden;">
          <div class="temp-code-header" style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #161b22; border-bottom: 1px solid #30363d;">
            <span style="color: #1f6feb; font-size: 12px; font-weight: 500;">${block.language.toUpperCase()}</span>
            <div style="display: flex; gap: 8px;">
              <button class="copy-code-btn" data-code-index="${block.index}" 
                      style="background: none; border: none; color: #7d8590; cursor: pointer; font-size: 12px; padding: 4px;" 
                      title="Copy code">Copy</button>
              ${!isTyping ? `<button class="edit-code-btn" data-code-index="${block.index}" 
                      style="background: #238636; border: none; color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 500;" 
                      title="Edit in code panel">Edit</button>` : ''}
            </div>
          </div>
          <pre style="margin: 0; padding: 16px; overflow-x: auto; background: #0d1117; color: #f0f6fc; font-family: Monaco, monospace; font-size: 14px; line-height: 1.4;"><code>${block.code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
        </div>
      `;
      
      content = content.replace(placeholder, codeBlockHtml);
    });
    
    return content;
  };

  // Set up event listeners for code block buttons
  useEffect(() => {
    const messageElement = messageRef.current;
    if (!messageElement) return;

    const handleCopyClick = (event) => {
      const button = event.target.closest('.copy-code-btn');
      if (!button) return;
      
      const codeIndex = parseInt(button.dataset.codeIndex);
      const codeBlock = codeBlocksRef.current.find(block => block.index === codeIndex);
      
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.code).then(() => {
          const originalText = button.textContent;
          button.textContent = '✓';
          button.style.color = '#22c55e';
          setTimeout(() => {
            button.textContent = originalText;
            button.style.color = '#7d8590';
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy code:', err);
        });
      }
    };

    const handleEditClick = (event) => {
      const button = event.target.closest('.edit-code-btn');
      if (!button) return;
      
      const codeIndex = parseInt(button.dataset.codeIndex);
      const codeBlock = codeBlocksRef.current.find(block => block.index === codeIndex);
      
      if (codeBlock && onOpenCodeEditor) {
        onOpenCodeEditor(codeBlock.code, codeBlock.language);
      }
    };

    // Add event listeners
    messageElement.addEventListener('click', handleCopyClick);
    messageElement.addEventListener('click', handleEditClick);

    // Cleanup
    return () => {
      messageElement.removeEventListener('click', handleCopyClick);
      messageElement.removeEventListener('click', handleEditClick);
    };
  }, [displayedText, isTyping, onOpenCodeEditor]);

  const formatMessage = (content) => {
    return extractCodeBlocks(content);
  };

  return (
    <div ref={messageRef} className={`message ${message.isUser ? 'user' : 'assistant'}`}>
      <div className="message-avatar">
        {message.isUser ? '👤' : '🤖'}
      </div>
      
      <div className="message-content">
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatMessage(displayedText) }}
        />
        {isTyping && (
          <span className="typing-cursor">▋</span>
        )}
      </div>
    </div>
  );
};

export default StreamingMessage;