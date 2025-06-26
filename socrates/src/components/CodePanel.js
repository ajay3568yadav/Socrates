import React, { useState, useRef, useEffect } from 'react';
import '../css/CodePanel.css'; 
const CodePanel = ({ 
  isOpen, 
  onClose, 
  initialCode = '', 
  language = 'cuda', 
  onSendForReview,
  isLoading = false,
  title = 'Code Editor'
}) => {
  const [code, setCode] = useState(initialCode);
  const [isModified, setIsModified] = useState(false);
  const [panelWidth, setPanelWidth] = useState(50); // Percentage width
  const [isResizing, setIsResizing] = useState(false);
  const textareaRef = useRef(null);
  const panelRef = useRef(null);
  const resizeRef = useRef(null);

  // Update code when initialCode changes
  useEffect(() => {
    setCode(initialCode);
    setIsModified(false);
  }, [initialCode]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 600) + 'px';
    }
  }, [code]);

  // Focus textarea when panel opens
  useEffect(() => {
    if (isOpen && textareaRef.current) {
      setTimeout(() => {
        textareaRef.current.focus();
      }, 100);
    }
  }, [isOpen]);

  // Handle panel resizing
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return;
      
      const windowWidth = window.innerWidth;
      const newWidth = ((windowWidth - e.clientX) / windowWidth) * 100;
      
      // Constrain width between 20% and 80%
      const constrainedWidth = Math.min(Math.max(newWidth, 20), 80);
      setPanelWidth(constrainedWidth);
      
      // Update chat view width immediately
      document.documentElement.style.setProperty('--code-panel-width', `${constrainedWidth}%`);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizing) {
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  const handleResizeStart = (e) => {
    e.preventDefault();
    setIsResizing(true);
  };

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

  const handleClose = () => {
    console.log('Close button clicked!'); // Debug log
    // Clean up CSS variable
    document.documentElement.style.removeProperty('--code-panel-width');
    // Reset panel width to default
    setPanelWidth(50);
    
    if (onClose) {
      console.log('Calling onClose function'); // Debug log
      onClose();
    } else {
      console.log('onClose function not provided!'); // Debug log
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

  // Handle escape key to close panel
  useEffect(() => {
    const handleEscapeKey = (event) => {
      if (event.key === 'Escape' && isOpen) {
        handleClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscapeKey);
    }

    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [isOpen]);

  // Set CSS variable for panel width
  useEffect(() => {
    if (isOpen) {
      document.documentElement.style.setProperty('--code-panel-width', `${panelWidth}%`);
    }
  }, [isOpen, panelWidth]);

  // Handle clicking outside panel to close (optional)
  const handleOverlayClick = (e) => {
    // Only close if clicking directly on the overlay, not its children
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      ref={panelRef}
      className="code-panel-overlay" 
      style={{ width: `${panelWidth}%` }}
      onClick={handleOverlayClick} // Add this for click-outside-to-close
    >
      {/* Resize handle */}
      <div 
        ref={resizeRef}
        className="resize-handle"
        onMouseDown={handleResizeStart}
        title="Drag to resize panel"
      />
      
      <div className="code-panel">
        <div className="code-panel-header">
          <div className="code-panel-title">
            <span className="code-icon">💻</span>
            <span className="title-text">{title}</span>
            <span className="language-badge">{getLanguageDisplayName(language)}</span>
            {isModified && <span className="modified-indicator">●</span>}
          </div>
          
          <div className="code-panel-actions">
            <button
              className="panel-action-btn"
              onClick={handleCopyCode}
              title="Copy Code"
            >
              📋 Copy
            </button>
            
            {isModified && (
              <button
                className="panel-action-btn secondary"
                onClick={handleReset}
                title="Reset to original"
              >
                ↺ Reset
              </button>
            )}
            
            <button
              className="panel-action-btn close"
              onClick={handleClose}
              title="Close panel"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="code-panel-content">
          <textarea
            ref={textareaRef}
            value={code}
            onChange={handleCodeChange}
            className="code-textarea"
            placeholder={`Enter your ${language.toUpperCase()} code here...`}
            spellCheck={false}
          />
        </div>

        <div className="code-panel-footer">
          <div className="code-stats">
            <span>{code.split('\n').length} lines</span>
            <span>{code.length} characters</span>
            <span className="panel-width">{Math.round(panelWidth)}% width</span>
            {isModified && <span className="modified-text">Modified</span>}
          </div>
          
          <div className="code-panel-buttons">
            <button
              className="review-btn"
              onClick={handleSendForReview}
              disabled={!code.trim() || isLoading}
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
    </div>
  );
};

export default CodePanel;