import React, { useEffect, useRef } from 'react';
import '../css/Message.css'; 
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const Message = ({ message, onSendMessage, isLoading, onOpenCodeEditor, splitPaneMode = false }) => {
  const isUser = message.role === 'user';
  const messageRef = useRef(null);
  const codeBlocksRef = useRef([]);

  // Parse content into blocks (text/code)
  const parseContentBlocks = (content) => {
    if (!content) return [];
    const blocks = [];
    let lastIndex = 0;
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    let match;
    let codeBlockIndex = 0;
    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        // Text before code block
        blocks.push({ type: 'text', content: content.slice(lastIndex, match.index) });
      }
      blocks.push({
        type: 'code',
        language: match[1] || 'text',
        code: match[2],
        index: codeBlockIndex++
      });
      lastIndex = codeBlockRegex.lastIndex;
    }
    if (lastIndex < content.length) {
      blocks.push({ type: 'text', content: content.slice(lastIndex) });
    }
    return blocks;
  };

  // Process markdown formatting for text content
  const processMarkdown = (text) => {
    if (!text) return '';
    
    let processed = text;
    
    // Process inline code first (highest priority to avoid conflicts)
    processed = processed.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Process bold text (**text** or __text__)
    processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // Process italic text (*text* or _text_)
    processed = processed.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    processed = processed.replace(/_([^_]+)_/g, '<em>$1</em>');
    
    // Process strikethrough (~~text~~)
    processed = processed.replace(/~~(.*?)~~/g, '<del>$1</del>');
    
    // Process headers (# ## ### etc.)
    processed = processed.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    processed = processed.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    processed = processed.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Process unordered lists (- or * at start of line)
    const lines = processed.split('\n');
    let inList = false;
    const processedLines = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const isListItem = /^[\s]*[-*]\s(.+)/.test(line);
      
      if (isListItem) {
        const content = line.replace(/^[\s]*[-*]\s/, '');
        if (!inList) {
          processedLines.push('<ul>');
          inList = true;
        }
        processedLines.push(`<li>${content}</li>`);
      } else {
        if (inList && line.trim() === '') {
          // Empty line in list - continue list
          processedLines.push('');
        } else if (inList) {
          // Non-list item - close list
          processedLines.push('</ul>');
          inList = false;
          processedLines.push(line);
        } else {
          processedLines.push(line);
        }
      }
    }
    
    // Close list if still open
    if (inList) {
      processedLines.push('</ul>');
    }
    
    processed = processedLines.join('\n');
    
    // Process ordered lists (1. 2. etc.)
    processed = processed.replace(/^\d+\.\s(.+)/gm, '<ol><li>$1</li></ol>');
    
    // Clean up multiple consecutive ol tags
    processed = processed.replace(/<\/ol>\s*<ol>/g, '');
    
    // Process links [text](url)
    processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Process line breaks
    processed = processed.replace(/\n/g, '<br>');
    
    return processed;
  };

  // Set up event listeners for code block buttons (copy/edit/compile)
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
          const originalText = button.innerHTML;
          button.innerHTML = '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 10L10 13L17 6" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>';
          setTimeout(() => {
            button.innerHTML = originalText;
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
        onOpenCodeEditor(codeBlock.code, codeBlock.language);
        const originalText = button.textContent;
        const originalBg = button.style.background;
        button.textContent = '⏳ Opening...';
        button.style.background = '#059669';
        setTimeout(() => {
          button.textContent = originalText;
          button.style.background = originalBg;
        }, 2000);
        if (onSendMessage && !splitPaneMode) {
          setTimeout(() => {
            const compileRequest = `Please generate a test script and help me compile and run this ${codeBlock.language.toUpperCase()} code:\n\n\`\`\`${codeBlock.language}\n${codeBlock.code}\n\`\`\`\n`;
            onSendMessage(compileRequest);
          }, 500);
        }
      }
    };
    
    messageElement.addEventListener('click', handleCopyClick);
    messageElement.addEventListener('click', handleEditClick);
    messageElement.addEventListener('click', handleCompileClick);
    
    return () => {
      messageElement.removeEventListener('click', handleCopyClick);
      messageElement.removeEventListener('click', handleEditClick);
      messageElement.removeEventListener('click', handleCompileClick);
    };
  }, [message.content, onOpenCodeEditor, onSendMessage, splitPaneMode]);

  // Render content with syntax highlighting for code blocks
  const renderContent = () => {
    // Parse and render blocks for both user and assistant
    const blocks = parseContentBlocks(message.content || '');
    // Store code blocks for event handlers
    codeBlocksRef.current = blocks.filter(b => b.type === 'code');
    
    return (
      <div className="message-text">
        {blocks.map((block, i) => {
          if (block.type === 'text') {
            // Process markdown for text blocks only
            const processedContent = processMarkdown(block.content);
            return (
              <span 
                key={i} 
                dangerouslySetInnerHTML={{ __html: processedContent }}
              />
            );
          } else if (block.type === 'code') {
            return (
              <div 
                className="temp-code-block-container" 
                key={i} 
                style={{ 
                  margin: '12px 0', 
                  borderRadius: 8, 
                  overflow: 'hidden', 
                  background: 'none', 
                  border: '1px solid #30363d' 
                }}
              >
                <div 
                  className="temp-code-header" 
                  style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center', 
                    padding: '6px 12px', 
                    background: '#1a1a1a', 
                    borderBottom: '1px solid #30363d', 
                    fontSize: 12 
                  }}
                >
                  <span style={{ color: '#76B900', fontWeight: 600, letterSpacing: 0.5 }}>
                    {block.language.toUpperCase()}
                  </span>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    <button 
                      className="copy-code-btn" 
                      data-code-index={block.index} 
                      style={{ 
                        background: 'none', 
                        border: 'none', 
                        color: '#7d8590', 
                        cursor: 'pointer', 
                        padding: 4, 
                        borderRadius: 4, 
                        transition: 'background 0.2s' 
                      }} 
                      title="Copy code"
                    >
                      <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="5" y="5" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                        <rect x="8" y="3" width="9" height="9" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                      </svg>
                    </button>
                    {onOpenCodeEditor && (
                      <button 
                        className="edit-code-btn" 
                        data-code-index={block.index} 
                        style={{ 
                          background: 'none', 
                          border: 'none', 
                          color: '#7d8590', 
                          cursor: 'pointer', 
                          padding: 4, 
                          borderRadius: 4, 
                          transition: 'background 0.2s' 
                        }} 
                        title="Edit in code panel"
                      >
                        <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M14.7 3.29a1 1 0 0 1 1.41 1.42l-8.5 8.5-2.12.71.71-2.12 8.5-8.5z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
                <SyntaxHighlighter
                  language={block.language}
                  style={vscDarkPlus}
                  customStyle={{ 
                    margin: 0, 
                    borderRadius: 0, 
                    fontSize: 14, 
                    background: '#1f1f1f', 
                    padding: 16 
                  }}
                  showLineNumbers={false}
                  wrapLongLines={true}
                >
                  {block.code}
                </SyntaxHighlighter>
              </div>
            );
          }
          return null;
        })}
      </div>
    );
  };

  return (
    <div ref={messageRef} className={`message ${isUser ? 'user' : 'assistant'} ${splitPaneMode ? 'split-mode' : ''}`}>
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