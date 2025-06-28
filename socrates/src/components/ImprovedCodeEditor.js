// src/components/ImprovedCodeEditor.js
// Enhanced ImprovedCodeEditor with line numbers and syntax highlighting

import React, { useState, useRef, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
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
  const [isCompiling, setIsCompiling] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [compilationResult, setCompilationResult] = useState(null);
  const [executionResult, setExecutionResult] = useState(null);
  const [testScript, setTestScript] = useState('');
  const [isGeneratingTest, setIsGeneratingTest] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const textareaRef = useRef(null);
  const lineNumbersRef = useRef(null);

  // API Configuration
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';

  // Update code when initialCode changes
  useEffect(() => {
    if (initialCode !== code) {
      setCode(initialCode);
      setIsModified(false);
    }
  }, [initialCode]);

  // Update line numbers when code changes
  useEffect(() => {
    updateLineNumbers();
  }, [code]);

  const updateLineNumbers = () => {
    if (lineNumbersRef.current) {
      const lines = code.split('\n');
      const lineNumbers = lines.map((_, index) => index + 1).join('\n');
      lineNumbersRef.current.textContent = lineNumbers;
    }
  };

  const handleCodeChange = (e) => {
    const value = e.target.value;
    setCode(value);
    setIsModified(value !== initialCode);
  };

  const handleScroll = (e) => {
    // Sync scroll between textarea and line numbers
    const scrollTop = e.target.scrollTop;
    
    if (lineNumbersRef.current) {
      lineNumbersRef.current.scrollTop = scrollTop;
    }
  };

  const handleKeyDown = (e) => {
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

  const handleSendForReview = () => {
    if (onSendForReview && code.trim()) {
      const reviewPrompt = `Please review and provide feedback on this ${language.toUpperCase()} code:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nPlease check for:\n- Correctness and potential bugs\n- Performance optimizations\n- Best practices\n- Memory management\n- Any improvements you'd suggest`;
      
      onSendForReview(reviewPrompt);
      setIsModified(false);
    }
  };

  const handleGenerateTest = async () => {
    if (!code.trim()) return;

    setIsGeneratingTest(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          language: language
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setTestScript(data.testScript || '');
      setShowResults(true);
    } catch (error) {
      console.error('Error generating test script:', error);
      setTestScript(`// Error generating test script: ${error.message}`);
      setShowResults(true);
    } finally {
      setIsGeneratingTest(false);
    }
  };

  const handleCompileCode = async () => {
    if (!code.trim()) return;

    setIsCompiling(true);
    setCompilationResult(null);
    setExecutionResult(null);
    setShowResults(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/compile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          language: language,
          testScript: testScript
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setCompilationResult(data);
    } catch (error) {
      console.error('Error compiling code:', error);
      setCompilationResult({
        success: false,
        error: error.message,
        output: '',
        stderr: error.message
      });
    } finally {
      setIsCompiling(false);
    }
  };

  const handleRunCode = async () => {
    if (!compilationResult?.success) {
      await handleCompileCode();
      return;
    }

    setIsRunning(true);
    setExecutionResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          compilationId: compilationResult.compilationId,
          language: language,
          testScript: testScript
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setExecutionResult(data);
    } catch (error) {
      console.error('Error executing code:', error);
      setExecutionResult({
        success: false,
        error: error.message,
        output: '',
        stderr: error.message,
        exitCode: 1
      });
    } finally {
      setIsRunning(false);
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
    setCompilationResult(null);
    setExecutionResult(null);
    setTestScript('');
    setShowResults(false);
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

  const getResultStatus = () => {
    if (compilationResult && !compilationResult.success) {
      return 'error';
    }
    if (executionResult && !executionResult.success) {
      return 'error';
    }
    if (executionResult && executionResult.success) {
      return 'success';
    }
    if (compilationResult && compilationResult.success) {
      return 'compiled';
    }
    return 'none';
  };

  return (
    <div className={`improved-code-editor ${isFullscreen ? 'fullscreen' : ''}`}>
      {/* Header */}
      <div className="code-editor-header">
        <div className="header-left">
          <div className="editor-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9.5 9L7 6.5L9.5 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M14.5 4L17 6.5L14.5 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 2L10 22" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="editor-title">
            <span className="title-text">{title}</span>
            <span className="language-badge">{getLanguageDisplayName(language)}</span>
          </div>
          {isModified && <span className="modified-indicator">●</span>}
        </div>
        
        <div className="header-actions">
          <button
            className="action-btn"
            onClick={handleGenerateTest}
            disabled={isGeneratingTest || !code.trim()}
            title="Generate test script"
          >
            {isGeneratingTest ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="spinning">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeDasharray="31.416" strokeDashoffset="31.416">
                  <animate attributeName="stroke-dasharray" dur="2s" values="0 31.416;15.708 15.708;0 31.416" repeatCount="indefinite"/>
                  <animate attributeName="stroke-dashoffset" dur="2s" values="0;-15.708;-31.416" repeatCount="indefinite"/>
                </circle>
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 11H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M9 15H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M16.5 3.5C17.3284 3.5 18 4.17157 18 5V19C18 19.8284 17.3284 20.5 16.5 20.5H7.5C6.67157 20.5 6 19.8284 6 19V5C6 4.17157 6.67157 3.5 7.5 3.5H16.5Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M9 7H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
            <span>{isGeneratingTest ? 'Generating...' : 'Test'}</span>
          </button>

          <button
            className="action-btn compile-btn"
            onClick={handleCompileCode}
            disabled={isCompiling || !code.trim()}
            title="Compile code"
          >
            {isCompiling ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="spinning">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeDasharray="31.416" strokeDashoffset="31.416">
                  <animate attributeName="stroke-dasharray" dur="2s" values="0 31.416;15.708 15.708;0 31.416" repeatCount="indefinite"/>
                  <animate attributeName="stroke-dashoffset" dur="2s" values="0;-15.708;-31.416" repeatCount="indefinite"/>
                </circle>
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
            <span>{isCompiling ? 'Compiling...' : 'Compile'}</span>
          </button>

          <button
            className="action-btn run-btn"
            onClick={handleRunCode}
            disabled={isRunning || !code.trim()}
            title="Run code"
          >
            {isRunning ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="spinning">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeDasharray="31.416" strokeDashoffset="31.416">
                  <animate attributeName="stroke-dasharray" dur="2s" values="0 31.416;15.708 15.708;0 31.416" repeatCount="indefinite"/>
                  <animate attributeName="stroke-dashoffset" dur="2s" values="0;-15.708;-31.416" repeatCount="indefinite"/>
                </circle>
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <polygon points="5,3 19,12 5,21" fill="currentColor"/>
              </svg>
            )}
            <span>{isRunning ? 'Running...' : 'Run'}</span>
          </button>
          
          <button
            className="action-btn"
            onClick={() => setIsFullscreen(!isFullscreen)}
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M8 3V5H5V8H3V3H8Z" fill="currentColor"/>
                <path d="M3 16V21H8V19H5V16H3Z" fill="currentColor"/>
                <path d="M16 3H21V8H19V5H16V3Z" fill="currentColor"/>
                <path d="M21 16V21H16V19H19V16H21Z" fill="currentColor"/>
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 3H10V5H5V10H3V3Z" fill="currentColor"/>
                <path d="M3 14V21H10V19H5V14H3Z" fill="currentColor"/>
                <path d="M14 3H21V10H19V5H14V3Z" fill="currentColor"/>
                <path d="M21 14V21H14V19H19V14H21Z" fill="currentColor"/>
              </svg>
            )}
          </button>
          
          <button
            className="action-btn"
            onClick={handleCopyCode}
            title="Copy code"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" stroke="currentColor" strokeWidth="2" fill="none"/>
              <path d="M5 15H4C2.89543 15 2 14.1046 2 13V4C2 2.89543 2.89543 2 4 2H13C14.1046 2 15 2.89543 15 4V5" stroke="currentColor" strokeWidth="2" fill="none"/>
            </svg>
          </button>
          
          {showCloseButton && (
            <button
              className="action-btn close-btn"
              onClick={onClose}
              title="Close editor"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Editor Content */}
      <div className={`editor-content ${showResults ? 'with-results' : ''}`}>
        <div className="code-section">
          <div className="editor-container">
            {/* Toggle between editing and viewing modes */}
            {isEditing ? (
              <div className="code-editor-wrapper">
                {/* Line Numbers */}
                <div
                  ref={lineNumbersRef}
                  className="line-numbers"
                />

                {/* Code Editor Textarea */}
                <textarea
                  ref={textareaRef}
                  value={code}
                  onChange={handleCodeChange}
                  onKeyDown={handleKeyDown}
                  onScroll={handleScroll}
                  className="code-textarea"
                  placeholder={`Write your ${language.toUpperCase()} code here...`}
                  spellCheck={false}
                  onBlur={() => setIsEditing(false)}
                  autoFocus
                />
              </div>
            ) : (
              <div 
                className="syntax-highlighter-container"
                onClick={() => setIsEditing(true)}
              >
                <SyntaxHighlighter
                  language={language === 'cuda' ? 'cpp' : language}
                  style={vscDarkPlus}
                  customStyle={{
                    margin: 0,
                    padding: '16px',
                    background: 'transparent',
                    fontSize: '14px',
                    lineHeight: '1.6',
                    fontFamily: "'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', 'Courier New', monospace",
                    minHeight: '200px',
                    cursor: 'text'
                  }}
                  showLineNumbers={true}
                  lineNumberStyle={{
                    color: '#7d8590',
                    paddingRight: '16px',
                    paddingLeft: '16px',
                    backgroundColor: 'transparent',
                    userSelect: 'none',
                    borderRight: 'none',
                    minWidth: '50px',
                    textAlign: 'right',
                    fontSize: '14px',
                    fontFamily: "'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', 'Courier New', monospace"
                  }}
                  lineNumberContainerStyle={{
                    backgroundColor: 'transparent',
                    borderRight: 'none',
                    paddingRight: '0',
                    marginRight: '16px'
                  }}
                  wrapLongLines={false}
                >
                  {code || `Write your ${language.toUpperCase()} code here...`}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        {showResults && (
          <div className={`results-section ${getResultStatus()}`}>
            <div className="results-header">
              <h4 className="results-title">
                <span className="results-icon">
                  {getResultStatus() === 'success' ? '✅' : 
                   getResultStatus() === 'error' ? '❌' : 
                   getResultStatus() === 'compiled' ? '🔨' : '📝'}
                </span>
                Execution Results
              </h4>
              <button
                className="results-toggle"
                onClick={() => setShowResults(!showResults)}
                title="Hide results"
              >
                ✕
              </button>
            </div>

            <div className="results-content">
              {/* Test Script */}
              {testScript && (
                <div className="result-section">
                  <h5 className="result-label">Generated Test Script:</h5>
                  <pre className="result-output test-script">{testScript}</pre>
                </div>
              )}

              {/* Compilation Results */}
              {compilationResult && (
                <div className="result-section">
                  <h5 className="result-label">
                    Compilation: 
                    <span className={`status ${compilationResult.success ? 'success' : 'error'}`}>
                      {compilationResult.success ? 'SUCCESS' : 'FAILED'}
                    </span>
                  </h5>
                  {compilationResult.output && (
                    <pre className="result-output">{compilationResult.output}</pre>
                  )}
                  {compilationResult.stderr && (
                    <pre className="result-output error">{compilationResult.stderr}</pre>
                  )}
                </div>
              )}

              {/* Execution Results */}
              {executionResult && (
                <div className="result-section">
                  <h5 className="result-label">
                    Execution: 
                    <span className={`status ${executionResult.success ? 'success' : 'error'}`}>
                      {executionResult.success ? 'SUCCESS' : 'FAILED'}
                    </span>
                    {executionResult.exitCode !== undefined && (
                      <span className="exit-code">Exit Code: {executionResult.exitCode}</span>
                    )}
                  </h5>
                  {executionResult.output && (
                    <pre className="result-output">{executionResult.output}</pre>
                  )}
                  {executionResult.stderr && (
                    <pre className="result-output error">{executionResult.stderr}</pre>
                  )}
                  {executionResult.executionTime && (
                    <div className="execution-stats">
                      <span>Execution time: {executionResult.executionTime}ms</span>
                    </div>
                    )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="editor-footer">
        <div className="footer-stats">
          <span className="stat">{code.split('\n').length} lines</span>
          <span className="stat">{code.length} chars</span>
          {isModified && <span className="stat modified">Modified</span>}
          {compilationResult && (
            <span className={`stat ${compilationResult.success ? 'success' : 'error'}`}>
              {compilationResult.success ? 'Compiled ✓' : 'Compile Error ✗'}
            </span>
          )}
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
              <>Reviewing…</>
            ) : (
              <>Ask AI to Review</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImprovedCodeEditor;