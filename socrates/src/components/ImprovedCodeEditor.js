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
  const [isCompiling, setIsCompiling] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [compilationResult, setCompilationResult] = useState(null);
  const [executionResult, setExecutionResult] = useState(null);
  const [testScript, setTestScript] = useState('');
  const [isGeneratingTest, setIsGeneratingTest] = useState(false);
  const [showResults, setShowResults] = useState(false);
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

  // Handle textarea scroll to sync with line numbers
  const handleTextareaScroll = (e) => {
    if (lineNumbersRef.current) {
      lineNumbersRef.current.scrollTop = e.target.scrollTop;
    }
  };

  // Generate line numbers array
  const getLineNumbers = () => {
    const lines = code.split('\n');
    // Always show at least one line number, even for empty code
    const lineCount = Math.max(1, lines.length);
    return Array.from({ length: lineCount }, (_, i) => i + 1);
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
            onClick={handleGenerateTest}
            disabled={isGeneratingTest || !code.trim()}
            title="Generate test script"
          >
            {isGeneratingTest ? '⏳' : '🧪'} Test
          </button>

          <button
            className="action-btn compile-btn"
            onClick={handleCompileCode}
            disabled={isCompiling || !code.trim()}
            title="Compile code"
          >
            {isCompiling ? '⏳' : '🔨'} Compile
          </button>

          <button
            className="action-btn run-btn"
            onClick={handleRunCode}
            disabled={isRunning || !code.trim()}
            title="Run code"
          >
            {isRunning ? '⏳' : '▶️'} Run
          </button>
          
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
      <div className={`editor-content ${showResults ? 'with-results' : ''}`}>
        <div className="code-section">
          <div className="line-numbers" ref={lineNumbersRef}>
            {getLineNumbers().map(lineNumber => (
              <div key={lineNumber} className="line-number">
                {lineNumber}
              </div>
            ))}
          </div>
          <textarea
            ref={textareaRef}
            value={code}
            onChange={handleCodeChange}
            onKeyDown={handleKeyDown}
            onScroll={handleTextareaScroll}
            className="code-textarea"
            placeholder={`Write your ${language.toUpperCase()} code here...`}
            spellCheck={false}
          />
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