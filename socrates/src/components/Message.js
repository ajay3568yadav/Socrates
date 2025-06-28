import React, { useEffect, useRef, useState } from 'react';
import '../css/Message.css';
import '../css/SystemInfo.css'; 

const Message = ({ message, onSendMessage, isLoading, onOpenCodeEditor, splitPaneMode = false }) => {
  const isUser = message.role === 'user';
  const messageRef = useRef(null);
  const codeBlocksRef = useRef([]);
  const [showSystemModal, setShowSystemModal] = useState(false);

  // System Info Modal Component
  const SystemInfoModal = ({ performanceMetrics, onClose }) => {
    if (!performanceMetrics) return null;

    const { response_time_seconds, prompt_length, response_length, system_metrics, gpu_metrics } = performanceMetrics;

    return (
      <div className="system-modal-overlay" style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000
      }} onClick={onClose}>
        <div className="system-modal-content" style={{
          backgroundColor: '#0d1117',
          border: '1px solid #30363d',
          borderRadius: '12px',
          padding: '24px',
          maxWidth: '600px',
          maxHeight: '80vh',
          overflow: 'auto',
          color: '#f0f6fc',
          fontFamily: 'Monaco, monospace',
          fontSize: '14px',
          lineHeight: '1.6'
        }} onClick={(e) => e.stopPropagation()}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '20px',
            borderBottom: '1px solid #30363d',
            paddingBottom: '12px'
          }}>
            <h3 style={{
              margin: 0,
              color: '#1f6feb',
              fontSize: '18px',
              fontWeight: '600'
            }}>🖥️ Enhanced System Metrics</h3>
            <button
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                color: '#7d8590',
                cursor: 'pointer',
                fontSize: '20px',
                padding: '4px'
              }}
            >
              ×
            </button>
          </div>

          <div style={{ fontFamily: 'Monaco, monospace', fontSize: '13px' }}>
            {/* Response Timing */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ color: '#f85149', fontWeight: 'bold', marginBottom: '8px' }}>⏱️ Response Timing</div>
              <div>⏱️  Response Time: <span style={{ color: '#7ee787' }}>{response_time_seconds} seconds</span></div>
              <div>📝 Prompt Length: <span style={{ color: '#7ee787' }}>{prompt_length} chars</span></div>
              <div>💬 Response Length: <span style={{ color: '#7ee787' }}>{response_length} chars</span></div>
            </div>

            <div style={{ borderTop: '1px solid #30363d', margin: '16px 0', paddingTop: '16px' }}>
              {/* CPU and Memory */}
              <div style={{ color: '#f85149', fontWeight: 'bold', marginBottom: '8px' }}>💾 System Resources</div>
              <div>💾 Memory Usage: <span style={{ color: '#7ee787' }}>{system_metrics.current_memory_gb?.toFixed(2)} GB ({system_metrics.current_memory_percent?.toFixed(1)}%)</span></div>
              <div>📊 Average Memory: <span style={{ color: '#7ee787' }}>{system_metrics.average_memory_percent?.toFixed(1)}%</span></div>
              <div>⚡ CPU Usage: <span style={{ color: '#7ee787' }}>{system_metrics.current_cpu_percent?.toFixed(1)}%</span></div>
              <div>📈 Average CPU: <span style={{ color: '#7ee787' }}>{system_metrics.average_cpu_percent?.toFixed(1)}%</span></div>
            </div>

            <div style={{ borderTop: '1px solid #30363d', margin: '16px 0', paddingTop: '16px' }}>
              {/* GPU Information */}
              <div style={{ color: '#f85149', fontWeight: 'bold', marginBottom: '8px' }}>🎮 GPU Information</div>
              <div>🎮 CUDA Available: <span style={{ color: gpu_metrics.cuda_available ? '#7ee787' : '#f85149' }}>
                {gpu_metrics.cuda_available ? '✅ Yes' : '❌ No'}
              </span></div>
              <div>🔢 GPU Count: <span style={{ color: '#7ee787' }}>{gpu_metrics.gpu_count}</span></div>
              <div>🏷️  GPU Model: <span style={{ color: '#7ee787' }}>{gpu_metrics.gpu_model}</span></div>
              <div>🔧 Monitoring Method: <span style={{ color: '#7ee787' }}>{gpu_metrics.monitoring_method}</span></div>
              
              {gpu_metrics.cuda_available && gpu_metrics.gpu_count > 0 && (
                <>
                  <div>🔥 GPU Memory: <span style={{ color: '#7ee787' }}>
                    {gpu_metrics.gpu_memory_used_gb?.toFixed(2)} GB / {gpu_metrics.gpu_memory_total_gb?.toFixed(2)} GB ({gpu_metrics.gpu_memory_percent?.toFixed(1)}%)
                  </span></div>
                  <div>⚙️  GPU Utilization: <span style={{ color: '#7ee787' }}>{gpu_metrics.gpu_utilization_percent?.toFixed(1)}%</span></div>
                  {gpu_metrics.gpu_temperature > 0 && (
                    <div>🌡️  GPU Temperature: <span style={{ color: '#7ee787' }}>{gpu_metrics.gpu_temperature}°C</span></div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

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
    if (codeBlocksRef.current.length > 0) {
      console.log("Found code blocks:", codeBlocksRef.current.length);
    }
    codeBlocksRef.current.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      
      const canCompile = ['c', 'cpp', 'cuda', 'python'].includes(block.language.toLowerCase());
      
      const compileButton = canCompile ? 
        `<button class="compile-code-btn" data-code-index="${block.index}" 
                style="background: #76B900; border: none; color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 500; margin-left: 4px;" 
                title="Compile and run code">Compile</button>` : '';
      
      // Add System Info button if performance metrics exist (only at message level, not per code block)
      const systemInfoButton = '';
      
      const codeBlockHtml = `
        <div class="temp-code-block-container" style="margin: 12px 0; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; overflow: hidden;">
          <div class="temp-code-header" style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #161b22; border-bottom: 1px solid #30363d;">
            <span style="color: #1f6feb; font-size: 12px; font-weight: 500;">${block.language.toUpperCase()}</span>
            <div style="display: flex; gap: 4px; align-items: center;">
              <button class="copy-code-btn" data-code-index="${block.index}" 
                      style="background: none; border: none; color: #7d8590; cursor: pointer; font-size: 12px; padding: 4px;" 
                      title="Copy code">Copy</button>
              <button class="edit-code-btn" data-code-index="${block.index}" 
                      style="background: #238636; border: none; color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 500;" 
                      title="Edit in code panel">Edit</button>
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
        button.textContent = 'Opening...';
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
    <>
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
          
          {/* Add System Info button at message level for assistant messages with performance metrics */}
          {!isUser && message.performanceMetrics && (
            <div style={{
              marginTop: '8px',
              paddingTop: '8px',
              borderTop: '1px solid #30363d',
              display: 'flex',
              justifyContent: 'flex-end'
            }}>
              <button
                className="system-info-btn message-level"
                onClick={() => setShowSystemModal(true)}
                style={{
                  background: '#DA7A00',
                  border: 'none',
                  color: 'white',
                  padding: '6px 12px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
                title="View GPU and system metrics for this response"
              >
                🎮 System Info
              </button>
            </div>
          )}
        </div>
      </div>

      {/* System Info Modal */}
      {showSystemModal && (
        <SystemInfoModal 
          performanceMetrics={message.performanceMetrics} 
          onClose={() => setShowSystemModal(false)} 
        />
      )}
    </>
  );
};

export default Message;