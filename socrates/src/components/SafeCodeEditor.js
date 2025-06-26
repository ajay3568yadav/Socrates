import React, { useState, useRef, useEffect, useCallback, Suspense } from 'react';
import SimpleCodeEditor from './SimpleCodeEditor';

// Lazy load Monaco Editor
const Editor = React.lazy(() => import('@monaco-editor/react'));

const SafeCodeEditor = ({ 
  initialCode = '', 
  language = 'cuda', 
  onSendForReview,
  isLoading = false,
  title = 'Code Editor'
}) => {
  const [code, setCode] = useState(initialCode);
  const [isModified, setIsModified] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [useMonaco, setUseMonaco] = useState(true);
  const editorRef = useRef(null);

  // Update code when initialCode changes
  useEffect(() => {
    if (initialCode !== code) {
      setCode(initialCode);
      setIsModified(false);
    }
  }, [initialCode]);

  const handleEditorDidMount = useCallback((editor, monaco) => {
    try {
      editorRef.current = editor;
      
      // Configure CUDA language support
      if (!monaco.languages.getLanguages().find(lang => lang.id === 'cuda')) {
        monaco.languages.register({ id: 'cuda' });
        monaco.languages.setMonarchTokensProvider('cuda', {
          tokenizer: {
            root: [
              [/__global__|__device__|__host__|__shared__|__constant__/, 'keyword.cuda'],
              [/threadIdx|blockIdx|blockDim|gridDim/, 'variable.cuda'],
              [/syncthreads|__syncthreads/, 'function.cuda'],
              [/\b\d+\b/, 'number'],
              [/".*?"/, 'string'],
              [/\/\/.*$/, 'comment'],
              [/\/\*[\s\S]*?\*\//, 'comment'],
              [/\b(int|float|double|char|void|if|else|for|while|return|include|define)\b/, 'keyword'],
            ]
          }
        });

        // Set CUDA syntax highlighting theme
        monaco.editor.defineTheme('cuda-dark', {
          base: 'vs-dark',
          inherit: true,
          rules: [
            { token: 'keyword.cuda', foreground: 'ff6b6b', fontStyle: 'bold' },
            { token: 'variable.cuda', foreground: '4ecdc4' },
            { token: 'function.cuda', foreground: 'ffe66d' },
          ],
          colors: {
            'editor.background': '#0d1117',
            'editor.foreground': '#f0f6fc',
          }
        });

        monaco.editor.setTheme('cuda-dark');
      }
    } catch (error) {
      console.error('Error setting up Monaco editor:', error);
      setUseMonaco(false);
    }
  }, []);

  const handleCodeChange = useCallback((value) => {
    setCode(value || '');
    setIsModified(value !== initialCode);
  }, [initialCode]);

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
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const handleFormatCode = () => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.formatDocument').run();
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

  // Error boundary fallback
  const MonacoErrorFallback = ({ error, resetErrorBoundary }) => (
    <SimpleCodeEditor
      initialCode={initialCode}
      language={language}
      onSendForReview={onSendForReview}
      isLoading={isLoading}
      title={title}
    />
  );

  // If Monaco should not be used, return simple editor
  if (!useMonaco) {
    return (
      <SimpleCodeEditor
        initialCode={initialCode}
        language={language}
        onSendForReview={onSendForReview}
        isLoading={isLoading}
        title={title}
      />
    );
  }

  return (
    <div className={`code-editor-container ${isExpanded ? 'expanded' : ''}`}>
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
            onClick={handleFormatCode}
            title="Format Code"
          >
            ✨
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

      <div className="code-editor-content">
        <Suspense fallback={
          <SimpleCodeEditor
            initialCode={initialCode}
            language={language}
            onSendForReview={onSendForReview}
            isLoading={isLoading}
            title={title}
          />
        }>
          <Editor
            height={isExpanded ? "60vh" : "300px"}
            language={language === 'cuda' ? 'cpp' : language}
            value={code}
            onChange={handleCodeChange}
            onMount={handleEditorDidMount}
            theme="cuda-dark"
            options={{
              minimap: { enabled: isExpanded },
              fontSize: 14,
              lineNumbers: 'on',
              roundedSelection: false,
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 2,
              insertSpaces: true,
              wordWrap: 'on',
              contextmenu: true,
              selectOnLineNumbers: true,
              lineDecorationsWidth: 10,
              lineNumbersMinChars: 3,
              glyphMargin: false,
              folding: true,
              foldingStrategy: 'indentation',
              showFoldingControls: 'always',
              disableLayerHinting: true,
              fixedOverflowWidgets: true,
            }}
          />
        </Suspense>
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

export default SafeCodeEditor;