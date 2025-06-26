import React, { useState, useRef, useEffect } from 'react';
import '../css/ImprovedCodeEditor.css';

const ImprovedCodeEditor = ({ 
  initialCode = '', 
  language = 'c', 
  onSendForReview,
  isLoading = false,
  title = 'Code Editor',
  onClose,
  showCloseButton = false
}) => {
  const [code, setCode] = useState(initialCode);
  const [isModified, setIsModified] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const textareaRef = useRef(null);

  // Update code when initialCode changes
  useEffect(() => {
    if (initialCode !== code) {
      setCode(initialCode);
      setIsModified(false);
    }
  }, [initialCode]);

  const handleCodeChange = (e) => {
    const value = e.target.value;
    setCode(value);
    setIsModified(value !== initialCode);
  };

  const handleSendForReview = () => {
    if (onSendForReview && code.trim()) {
      const reviewPrompt = `Please review and provide feedback on this ${language.toUpperCase()} code:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nPlease check for:\n- Correctness and potential bugs\n- Performance optimizations\n- Best practices\n- Memory management\n- Any improvements you'd suggest`;
      
      onSendForReview(reviewPrompt);
      setIsModified(false);
    }
  };

  const handleCopyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
      console.log('Code copied to clipboard');
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const handleReset = () => {
    setCode(initialCode);
    setIsModified(false);
  };

  const getLanguageDisplayName = (lang) => {
    const languages = {
      'cuda': 'CUDA',
      'cpp': 'C++',
      'c': 'C',
      'python': 'Python',
      'javascript': 'JavaScript',
      'typescript': 'TypeScript'
    };
    return languages[lang] || lang.toUpperCase();
  };

  const handleKeyDown = (e) => {
    // Handle Tab key for indentation
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      
      if (e.shiftKey) {
        // Shift+Tab: Remove indentation
        const lines = code.substring(0, start).split('\n');
        const currentLine = lines[lines.length - 1];
        if (currentLine.startsWith('  ')) {
          const newCode = code.substring(0, start - 2) + code.substring(start);
          setCode(newCode);
          setTimeout(() => {
            e.target.selectionStart = e.target.selectionEnd = start - 2;
          }, 0);
        }
      } else {
        // Tab: Add indentation
        const newCode = code.substring(0, start) + '  ' + code.substring(end);
        setCode(newCode);
        setTimeout(() => {
          e.target.selectionStart = e.target.selectionEnd = start + 2;
        }, 0);
      }
      setIsModified(true);
    }
  };

  return (
    <div className={`improved-code-editor ${isFullscreen ? 'fullscreen' : ''}`}>
      {/* Header */}
      <div className="code-editor-header">
        <div className="header-left">
          <div className="editor-icon">💻</div>
          <div className="editor-title">
            <span className="title-text">{title}</span>
            <span className="language-badge">{getLanguageDisplayName(language)}</span>
          </div>
          {isModified && <span className="modified-indicator">●</span>}
        </div>
        
        <div className="header-actions">
          <button
            className="action-btn"
            onClick={() => setIsFullscreen(!isFullscreen)}
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? '🗗' : '🗖'}
          </button>
          
          <button
            className="action-btn"
            onClick={handleCopyCode}
            title="Copy code"
          >
            📋
          </button>
          
          {showCloseButton && (
            <button
              className="action-btn close-btn"
              onClick={onClose}
              title="Close editor"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Editor Content */}
      <div className="editor-content">
        <textarea
          ref={textareaRef}
          value={code}
          onChange={handleCodeChange}
          onKeyDown={handleKeyDown}
          className="code-textarea"
          placeholder={`Write your ${language.toUpperCase()} code here...`}
          spellCheck={false}
        />
      </div>

      {/* Footer */}
      <div className="editor-footer">
        <div className="footer-stats">
          <span className="stat">{code.split('\n').length} lines</span>
          <span className="stat">{code.length} chars</span>
          {isModified && <span className="stat modified">Modified</span>}
        </div>
        
        <div className="footer-actions">
          {isModified && (
            <button
              className="footer-btn secondary"
              onClick={handleReset}
              title="Reset to original"
            >
              ↺ Reset
            </button>
          )}
          
          <button
            className="footer-btn primary"
            onClick={handleSendForReview}
            disabled={!code.trim() || isLoading}
            title="Ask AI to review this code"
          >
            {isLoading ? (
              <>⏳ Reviewing...</>
            ) : (
              <>🔍 Ask AI to Review</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImprovedCodeEditor;