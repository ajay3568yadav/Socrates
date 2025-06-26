import React, { useState, useEffect } from 'react';

const SimpleCodeEditor = ({ 
  initialCode = '', 
  language = 'cuda', 
  onSendForReview,
  isLoading = false,
  title = 'Code Editor'
}) => {
  const [code, setCode] = useState(initialCode);
  const [isModified, setIsModified] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

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

  return (
    <div className={`simple-code-editor-container ${isExpanded ? 'expanded' : ''}`}>
      <div className="code-editor-header">
        <div className="code-editor-title">
          <span className="code-icon">📝</span>
          <span className="title-text">{title}</span>
          <span className="language-badge">{getLanguageDisplayName(language)}</span>
          {isModified && <span className="modified-indicator">●</span>}
        </div>
        
        <div className="code-editor-actions">
          <button
            className="editor-action-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Minimize' : 'Expand'}
          >
            {isExpanded ? '🗗' : '🗖'}
          </button>
          
          <button
            className="editor-action-btn"
            onClick={handleCopyCode}
            title="Copy Code"
          >
            📋
          </button>
          
          <button
            className="editor-action-btn primary"
            onClick={handleSendForReview}
            disabled={!code.trim() || isLoading}
            title="Send code for AI review"
          >
            {isLoading ? '⏳' : '🔍'} Review
          </button>
        </div>
      </div>

      <div className="simple-code-editor-content">
        <textarea
          value={code}
          onChange={handleCodeChange}
          className="simple-code-textarea"
          style={{
            height: isExpanded ? "60vh" : "300px",
            width: '100%',
            background: '#0d1117',
            color: '#f0f6fc',
            border: 'none',
            padding: '16px',
            fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
            fontSize: '14px',
            lineHeight: '1.5',
            resize: 'vertical',
            outline: 'none'
          }}
          spellCheck={false}
        />
      </div>

      {code.trim() && (
        <div className="code-editor-footer">
          <div className="code-stats">
            <span>{code.split('\n').length} lines</span>
            <span>{code.length} characters</span>
            {isModified && <span className="modified-text">Modified</span>}
          </div>
          
          <div className="code-actions">
            {isModified && (
              <button
                className="action-btn secondary"
                onClick={() => {
                  setCode(initialCode);
                  setIsModified(false);
                }}
              >
                Reset
              </button>
            )}
            
            <button
              className="action-btn primary"
              onClick={handleSendForReview}
              disabled={!code.trim() || isLoading}
            >
              {isLoading ? 'Reviewing...' : 'Get AI Feedback'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleCodeEditor;