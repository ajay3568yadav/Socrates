import React, { useState, useEffect } from 'react';
import { highlightCudaCode } from '../utils/codeHighlighting';

const StreamingMessage = ({ message, onComplete }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!message.text || currentIndex >= message.text.length) {
      setIsTyping(false);
      if (onComplete) onComplete();
      return;
    }

    const typingSpeed = message.text.length > 1000 ? 10 : 30; // Faster for long messages
    
    const timer = setTimeout(() => {
      setDisplayedText(prev => prev + message.text[currentIndex]);
      setCurrentIndex(prev => prev + 1);
    }, typingSpeed);

    return () => clearTimeout(timer);
  }, [currentIndex, message.text, onComplete]);

  // Reset when message changes
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
    setIsTyping(true);
  }, [message.id]);

  const formatMessage = (content) => {
    // First, extract and replace code blocks with placeholders
    const codeBlocks = [];
    let codeBlockIndex = 0;
    
    // Extract code blocks
    content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
      codeBlocks.push({ language: lang || 'text', code: code.trim() });
      codeBlockIndex++;
      return placeholder;
    });
    
    // Convert inline code
    content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    // Restore code blocks with syntax highlighting
    codeBlocks.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      const highlightedCode = (block.language === 'cuda' || block.language === 'cpp' || block.language === 'c') 
        ? highlightCudaCode(block.code) 
        : block.code;
      
      const codeBlockHtml = `
        <div class="code-block-container">
          <div class="code-block-header">
            <span class="code-language">${block.language.toUpperCase()}</span>
            <button class="code-copy-btn" onclick="copyCodeToClipboard('${encodeURIComponent(block.code)}', this)" title="Copy code">📋</button>
          </div>
          <pre class="code-block"><code>${highlightedCode}</code></pre>
        </div>
      `;
      
      content = content.replace(placeholder, codeBlockHtml);
    });
    
    return content;
  };

  return (
    <div className={`message ${message.isUser ? 'user' : 'assistant'}`}>
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