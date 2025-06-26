import React, { useState, useRef, useEffect } from "react";
import Editor from "@monaco-editor/react";

const CodeEditor = ({
  initialCode = "",
  language = "cuda",
  onSendCode,
  onClose,
  isVisible = false,
  title = "Code Editor",
}) => {
  const [code, setCode] = useState(initialCode);
  const [isModified, setIsModified] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const editorRef = useRef(null);

  useEffect(() => {
    setCode(initialCode);
    setIsModified(false);
  }, [initialCode]);

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor;

    // Register CUDA language if not already registered
    if (!monaco.languages.getLanguages().find((lang) => lang.id === "cuda")) {
      monaco.languages.register({ id: "cuda" });

      // Set up CUDA syntax highlighting (based on C++)
      monaco.languages.setMonarchTokensProvider("cuda", {
        tokenizer: {
          root: [
            [
              /__global__|__device__|__host__|__shared__|__constant__|__texture__/,
              "keyword.cuda",
            ],
            [/threadIdx|blockIdx|blockDim|gridDim/, "variable.predefined.cuda"],
            [/syncthreads|__syncthreads/, "function.cuda"],
            [/[a-z_$][\w$]*/, "identifier"],
            [/[A-Z][\w\$]*/, "type.identifier"],
            [/".*?"/, "string"],
            [/'.*?'/, "string"],
            [/\/\/.*$/, "comment"],
            [/\/\*[\s\S]*?\*\//, "comment"],
            [/\d+/, "number"],
            [/[{}()\[\]]/, "bracket"],
            [/[<>]=?|[!=]=?|&&|\|\||[+\-*/%]/, "operator"],
          ],
        },
      });

      // Set up theme for CUDA
      monaco.editor.defineTheme("cuda-dark", {
        base: "vs-dark",
        inherit: true,
        rules: [
          { token: "keyword.cuda", foreground: "#569cd6", fontStyle: "bold" },
          { token: "variable.predefined.cuda", foreground: "#4ec9b0" },
          { token: "function.cuda", foreground: "#dcdcaa" },
        ],
        colors: {
          "editor.background": "#1e1e1e",
        },
      });
    }
  };

  const handleCodeChange = (value) => {
    setCode(value || "");
    setIsModified(value !== initialCode);
  };

  const handleSendCode = async () => {
    if (!code.trim() || isSending) return;

    setIsSending(true);
    try {
      await onSendCode(code, language);
      setIsModified(false);
    } catch (error) {
      console.error("Error sending code:", error);
    } finally {
      setIsSending(false);
    }
  };

  const handleCopyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
      // Could add a toast notification here
    } catch (error) {
      console.error("Failed to copy code:", error);
    }
  };

  const handleFormatCode = () => {
    if (editorRef.current) {
      editorRef.current.getAction("editor.action.formatDocument").run();
    }
  };

  if (!isVisible) return null;

  return (
    <div className="code-editor-overlay">
      <div className="code-editor-modal">
        <div className="code-editor-header">
          <div className="code-editor-title">
            <span className="editor-icon">⚡</span>
            <span>{title}</span>
            {isModified && <span className="modified-indicator">●</span>}
          </div>
          <div className="code-editor-actions">
            <button
              className="editor-action-btn"
              onClick={handleFormatCode}
              title="Format Code"
            >
              🎨
            </button>
            <button
              className="editor-action-btn"
              onClick={handleCopyCode}
              title="Copy Code"
            >
              📋
            </button>
            <button
              className="editor-action-btn close-btn"
              onClick={onClose}
              title="Close Editor"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="code-editor-content">
          <Editor
            height="60vh"
            language={language === "cuda" ? "cuda" : language}
            theme="cuda-dark"
            value={code}
            onChange={handleCodeChange}
            onMount={handleEditorDidMount}
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              lineNumbers: "on",
              roundedSelection: false,
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 2,
              insertSpaces: true,
              wordWrap: "on",
              contextmenu: true,
              selectOnLineNumbers: true,
              glyphMargin: true,
              folding: true,
              foldingHighlight: true,
              showFoldingControls: "always",
            }}
          />
        </div>

        <div className="code-editor-footer">
          <div className="editor-info">
            <span className="language-badge">{language.toUpperCase()}</span>
            <span className="lines-count">{code.split("\n").length} lines</span>
          </div>
          <div className="editor-controls">
            <button
              className="editor-btn secondary"
              onClick={() => setCode(initialCode)}
              disabled={!isModified}
            >
              Reset
            </button>
            <button
              className="editor-btn primary"
              onClick={handleSendCode}
              disabled={!code.trim() || isSending}
            >
              {isSending ? (
                <>
                  <span className="loading-spinner">⟳</span>
                  Analyzing...
                </>
              ) : (
                <>
                  <span>🚀</span>
                  Send for Analysis
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodeEditor;
