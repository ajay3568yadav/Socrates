import React, { useEffect, useRef } from 'react';
import '../css/Message.css'; 

const Message = ({ message, onSendMessage, isLoading, onOpenCodeEditor, splitPaneMode = false }) => {
  const isUser = message.role === 'user';
  const messageRef = useRef(null);
  const codeBlocksRef = useRef([]);

  const formatTextContent = (content) => {
    if (!content) return '';
    
    // Convert inline code - more conservative approach
    let formatted = content.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');
    
    // Handle line breaks more carefully
    formatted = formatted.replace(/\n\n/g, '</p><p>');
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Wrap in paragraph tags if not already wrapped
    if (!formatted.startsWith('<p>') && !formatted.includes('<p>')) {
      formatted = '<p>' + formatted + '</p>';
    }
    
    return formatted;
  };

  const extractCodeBlocks = (content) => {
    // Clear previous code blocks
    codeBlocksRef.current = [];
    
    // First, extract and replace code blocks with styled blocks
    let codeBlockIndex = 0;
    
    // Extract code blocks with better regex
    content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
      
      // Store code block data for event handlers
      codeBlocksRef.current.push({
        index: codeBlockIndex,
        language: lang || 'text', 
        code: code.trim()
      });
      
      codeBlockIndex++;
      return placeholder;
    });
    
    // Convert inline code
    content = content.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');
    
    // Better paragraph and line break handling
    content = content.replace(/\n\n+/g, '</p><p>');
    content = content.replace(/\n/g, '<br>');
    
    // Wrap in paragraphs if needed
    if (!content.includes('<p>')) {
      content = '<p>' + content + '</p>';
    }
    
    // Clean up empty paragraphs
    content = content.replace(/<p><\/p>/g, '');
    content = content.replace(/<p>\s*<\/p>/g, '');
    
    // Restore code blocks with enhanced action buttons
    codeBlocksRef.current.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      
      const canCompile = ['c', 'cpp', 'cuda', 'python', 'javascript', 'typescript'].includes(block.language.toLowerCase());
      const compileButton = canCompile ? 
        `<button class="compile-code-btn" data-code-index="${block.index}" 
                title="Compile and run code">🔨 Compile</button>` : '';
      
      const codeBlockHtml = `
        <div class="temp-code-block-container" data-language="${block.language}">
          <div class="temp-code-header">
            <span>${block.language.toUpperCase()}</span>
            <div>
              <button class="copy-code-btn" data-code-index="${block.index}" 
                      title="Copy code">📋 Copy</button>
              <button class="edit-code-btn" data-code-index="${block.index}" 
                      title="Edit in code panel">📝 Edit</button>
              ${compileButton}
            </div>
          </div>
          <pre><code>${block.code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
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
          button.textContent = '✓ Copied';
          button.style.color = '#22c55e';
          setTimeout(() => {
            button.textContent = originalText;
            button.style.color = '';
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

    const handleCompileClick = (event) => {
      const button = event.target.closest('.compile-code-btn');
      if (!button) return;
      
      const codeIndex = parseInt(button.dataset.codeIndex);
      const codeBlock = codeBlocksRef.current.find(block => block.index === codeIndex);
      
      if (codeBlock && onOpenCodeEditor) {
        // Open in code editor
        onOpenCodeEditor(codeBlock.code, codeBlock.language);
        
        // Show feedback
        const originalText = button.textContent;
        const originalBg = button.style.background;
        button.textContent = '⏳ Opening...';
        button.style.background = '#059669';
        setTimeout(() => {
          button.textContent = originalText;
          button.style.background = originalBg;
        }, 2000);
        
        // Send a message to request test generation and compilation
        if (onSendMessage && !splitPaneMode) {
          setTimeout(() => {
            const compileRequest = `Please generate a test script and help me compile and run this ${codeBlock.language.toUpperCase()} code:\n\n\`\`\`${codeBlock.language}\n${codeBlock.code}\n\`\`\``;
            onSendMessage(compileRequest);
          }, 500);
        }
      }
    };

    // Add event listeners with delegation
    messageElement.addEventListener('click', handleCopyClick);
    messageElement.addEventListener('click', handleEditClick);
    messageElement.addEventListener('click', handleCompileClick);

    // Cleanup
    return () => {
      messageElement.removeEventListener('click', handleCopyClick);
      messageElement.removeEventListener('click', handleEditClick);
      messageElement.removeEventListener('click', handleCompileClick);
    };
  }, [message.content, onOpenCodeEditor, onSendMessage, splitPaneMode]);

  const renderContent = () => {
    if (isUser) {
      return (
        <div
          className="message-text"
          dangerouslySetInnerHTML={{
            __html: formatTextContent(message.content)
          }}
        />
      );
    }

    // For assistant messages with enhanced formatting
    const processedContent = extractCodeBlocks(message.content || '');
    
    return (
      <div
        className="message-text"
        dangerouslySetInnerHTML={{
          __html: processedContent
        }}
      />
    );
  };

  return (
    <div 
      ref={messageRef} 
      className={`message ${isUser ? 'user' : 'assistant'} ${splitPaneMode ? 'split-mode' : ''}`}
    >
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        {renderContent()}
        {message.isError && (
          <div className="message-error-indicator">
            <span className="error-icon">⚠️</span>
            <span className="error-text">This message contains an error</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default Message;