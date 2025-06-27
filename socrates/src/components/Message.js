import React, { useEffect, useRef } from 'react';
import '../css/Message.css'; 

const Message = ({ message, onSendMessage, isLoading, onOpenCodeEditor, splitPaneMode = false }) => {
  const isUser = message.role === 'user';
  const messageRef = useRef(null);
  const codeBlocksRef = useRef([]);

  const formatTextContent = (content) => {
    if (!content) return '';
    
    // Convert inline code
    let formatted = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Convert line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
  };

  const extractCodeBlocks = (content) => {
    // Clear previous code blocks
    codeBlocksRef.current = [];
    
    // First, extract and replace code blocks with styled blocks
    let codeBlockIndex = 0;
    
    // Extract code blocks
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
    content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    // Restore code blocks with enhanced action buttons
    codeBlocksRef.current.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      
      const canCompile = ['c', 'cpp', 'cuda', 'python'].includes(block.language.toLowerCase());
      const compileButton = canCompile ? 
        `<button class="compile-code-btn" data-code-index="${block.index}" 
                style="background: #f97316; border: none; color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 500; margin-left: 4px;" 
                title="Compile and run code">🔨 Compile</button>` : '';
      
      const codeBlockHtml = `
        <div class="temp-code-block-container" style="margin: 12px 0; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; overflow: hidden;">
          <div class="temp-code-header" style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #161b22; border-bottom: 1px solid #30363d;">
            <span style="color: #1f6feb; font-size: 12px; font-weight: 500;">${block.language.toUpperCase()}</span>
            <div style="display: flex; gap: 4px; align-items: center;">
              <button class="copy-code-btn" data-code-index="${block.index}" 
                      style="background: none; border: none; color: #7d8590; cursor: pointer; font-size: 12px; padding: 4px;" 
                      title="Copy code">📋</button>
              <button class="edit-code-btn" data-code-index="${block.index}" 
                      style="background: #238636; border: none; color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 500;" 
                      title="Edit in code panel">📝 Edit</button>
              ${compileButton}
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

    const handleCompileClick = (event) => {
      const button = event.target.closest('.compile-code-btn');
      if (!button) return;
      
      const codeIndex = parseInt(button.dataset.codeIndex);
      const codeBlock = codeBlocksRef.current.find(block => block.index === codeIndex);
      
      if (codeBlock && onOpenCodeEditor) {
        // Open in code editor and trigger compilation
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

    // Add event listeners
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

    // For assistant messages with code blocks
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
    <div ref={messageRef} className={`message ${isUser ? 'user' : 'assistant'} ${splitPaneMode ? 'split-mode' : ''}`}>
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