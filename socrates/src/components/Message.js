import React, { useEffect, useRef, useState } from 'react';
import '../css/Message.css'; 
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Quiz from './Quiz';
import QuizFeedback from './QuizFeedback';

const Message = ({ 
  message, 
  onSendMessage, 
  isLoading, 
  onOpenCodeEditor, 
  splitPaneMode = false,
  tutoringMode = false,
  currentChatId // ADD THIS PARAMETER
}) => {
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
        zIndex: 1000,
        backdropFilter: 'blur(2px)'
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
          lineHeight: '1.6',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)',
          animation: 'modalFadeIn 0.2s ease-out'
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

// BEST SOLUTION: Updated handleQuizSubmit function in Message.js

const handleQuizSubmit = async (quizData, userAnswers) => {
  console.log('Quiz submission started:', { quizData, userAnswers });
  
  if (!onSendMessage) {
    console.error('onSendMessage not available');
    return;
  }

  try {
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';
    
    const requestBody = {
      quiz_data: quizData,
      user_answers: userAnswers,
      session_id: 'current_session',
      chat_id: currentChatId,
      tutoring_mode: tutoringMode
    };
    
    const response = await fetch(`${API_BASE_URL}/api/evaluate-quiz`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Quiz evaluation response:', data);
    
    if (data.success && data.evaluation) {
      // Send the quiz feedback (this will render as QuizFeedback component)
      const feedbackText = `QUIZ_FEEDBACK:${JSON.stringify(data.evaluation)}`;
      onSendMessage(feedbackText);
      
      // FIXED: If there's AI analysis, send it as a special AI_ANALYSIS message
      if (data.ai_analysis && tutoringMode) {
        setTimeout(() => {
          // Send AI analysis with special prefix so it gets handled as assistant message
          const analysisText = `AI_ANALYSIS:${data.ai_analysis}`;
          onSendMessage(analysisText);
        }, 1500); // Small delay so feedback appears first
      }
    }
    
  } catch (error) {
    console.error('Error submitting quiz:', error);
    onSendMessage('There was an error evaluating your quiz. Please try again.');
  }
};

  // Handle action button clicks
  const handleCopyPrompt = () => {
    if (message.content) {
      navigator.clipboard.writeText(message.content).then(() => {
        console.log('Prompt copied to clipboard');
      }).catch(err => {
        console.error('Failed to copy prompt:', err);
      });
    }
  };

  const handleThumbsUp = () => {
    console.log('Thumbs up clicked for message:', message.id);
    // Add your thumbs up logic here
  };

  const handleThumbsDown = () => {
    console.log('Thumbs down clicked for message:', message.id);
    // Add your thumbs down logic here
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
    // Check if this is a quiz message
    if (message.content && message.content.startsWith('QUIZ_DATA:')) {
      try {
        const quizDataString = message.content.replace('QUIZ_DATA:', '');
        const quizData = JSON.parse(quizDataString);
        
        return (
          <div className="quiz-message">
            <Quiz 
              quizData={quizData}
              onSubmit={(userAnswers) => handleQuizSubmit(quizData, userAnswers)}
              isLoading={isLoading}
            />
          </div>
        );
      } catch (error) {
        console.error('Error parsing quiz data:', error);
        return <div className="message-text">Error loading quiz. Please try again.</div>;
      }
    }
    
    // Check if this is quiz feedback
    if (message.quizEvaluation) {
      return (
        <div className="quiz-feedback-message">
          <QuizFeedback evaluation={message.quizEvaluation} />
        </div>
      );
    }
    
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
    <>
      <div ref={messageRef} className={`message ${isUser ? 'user' : 'assistant'} ${splitPaneMode ? 'split-mode' : ''}`}>
        <div className="message-content">
          {renderContent()}
          {message.isError && (
            <div className="message-error-indicator">
              <span className="error-icon">⚠️</span>
              <span className="error-text">This message contains an error</span>
            </div>
          )}
          
          {/* Action buttons for assistant messages */}
          {!isUser && (
            <div className="message-actions" style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginTop: '12px',
              paddingTop: '8px',
              borderTop: '1px solid rgba(48, 54, 61, 0.3)',
              justifyContent: 'flex-start'
            }}>
              {/* System Info Button - Always show for assistant messages */}
              <button
                onClick={() => {
                  if (message.performanceMetrics) {
                    setShowSystemModal(true);
                  } else {
                    console.log('No performance metrics available for this message');
                    // You could show a toast/notification here instead
                  }
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: message.performanceMetrics ? '#7d8590' : '#484f58',
                  cursor: message.performanceMetrics ? 'pointer' : 'not-allowed',
                  padding: '6px',
                  borderRadius: '4px',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '16px',
                  opacity: message.performanceMetrics ? 1 : 0.5
                }}
                title={message.performanceMetrics ? "View system metrics" : "No metrics available"}
                onMouseEnter={(e) => {
                  if (message.performanceMetrics) {
                    e.target.style.color = '#f0f6fc';
                    e.target.style.background = 'rgba(110, 118, 129, 0.1)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (message.performanceMetrics) {
                    e.target.style.color = '#7d8590';
                    e.target.style.background = 'none';
                  }
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M3 3v18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M18 9l-5 5-4-4-3 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              
              {/* Copy Prompt Button */}
              <button
                onClick={handleCopyPrompt}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#7d8590',
                  cursor: 'pointer',
                  padding: '6px',
                  borderRadius: '4px',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '16px'
                }}
                title="Copy message"
                onMouseEnter={(e) => {
                  e.target.style.color = '#f0f6fc';
                  e.target.style.background = 'rgba(110, 118, 129, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.color = '#7d8590';
                  e.target.style.background = 'none';
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" stroke="currentColor" strokeWidth="2"/>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" stroke="currentColor" strokeWidth="2"/>
                </svg>
              </button>
              
              {/* Thumbs Up Button */}
              <button
                onClick={handleThumbsUp}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#7d8590',
                  cursor: 'pointer',
                  padding: '6px',
                  borderRadius: '4px',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '16px'
                }}
                title="Good response"
                onMouseEnter={(e) => {
                  e.target.style.color = '#22c55e';
                  e.target.style.background = 'rgba(110, 118, 129, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.color = '#7d8590';
                  e.target.style.background = 'none';
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              
              {/* Thumbs Down Button */}
              <button
                onClick={handleThumbsDown}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#7d8590',
                  cursor: 'pointer',
                  padding: '6px',
                  borderRadius: '4px',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '16px'
                }}
                title="Poor response"
                onMouseEnter={(e) => {
                  e.target.style.color = '#f85149';
                  e.target.style.background = 'rgba(110, 118, 129, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.color = '#7d8590';
                  e.target.style.background = 'none';
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
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