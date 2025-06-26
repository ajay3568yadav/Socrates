// Message.js
import React from "react";

const Message = ({ message, onOpenCodeEditor }) => {
  const isUser = message.role === "user";

  const extractCodeBlocks = (content = "") => {
    const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g;
    const codeBlocks = [];
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      codeBlocks.push({
        language: match[1] || "text",
        code: match[2].trim(),
        fullMatch: match[0],
      });
    }

    return codeBlocks;
  };

  const formatMessage = (content = "") => {
    const codeBlocks = extractCodeBlocks(content);

    // If there are code blocks and this is an assistant message, add editor buttons
    if (codeBlocks.length > 0 && !isUser) {
      let formattedContent = content;

      codeBlocks.forEach((block, index) => {
        const codeId = `code_${message.id}_${index}`;
        const encodedCode = encodeURIComponent(block.code);
        const encodedLanguage = encodeURIComponent(block.language);

        const codeWithButtons = `
          <div class="code-block-container">
            <div class="code-block-header">
              <span class="code-language">${block.language}</span>
              <div class="code-actions">
                <button 
                  class="code-action-btn copy-btn" 
                  onclick="copyCodeToClipboard('${encodedCode}', this)"
                  title="Copy code"
                >
                  📋
                </button>
                <button 
                  class="code-action-btn editor-btn" 
                  data-code="${encodedCode}"
                  data-language="${encodedLanguage}"
                  data-code-id="${codeId}"
                  title="Open in editor"
                >
                  ⚡ Editor
                </button>
              </div>
            </div>
            <pre><code>${block.code}</code></pre>
          </div>
        `;

        formattedContent = formattedContent.replace(
          block.fullMatch,
          codeWithButtons
        );
      });

      return formattedContent;
    }

    // Regular formatting for messages without code or user messages
    let formatted = content;

    // Convert code blocks (for user messages or when no editor is needed)
    formatted = formatted.replace(
      /```(\w+)?\n?([\s\S]*?)```/g,
      (match, language, code) => {
        const encodedCode = encodeURIComponent(code.trim());
        return `
        <div class="code-block-container">
          <div class="code-block-header">
            <span class="code-language">${language || "text"}</span>
            <button 
              class="code-action-btn copy-btn" 
              onclick="copyCodeToClipboard('${encodedCode}', this)"
              title="Copy code"
            >
              📋
            </button>
          </div>
          <pre><code>${code.trim()}</code></pre>
        </div>
      `;
      }
    );

    // Convert inline code
    formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Convert line breaks
    formatted = formatted.replace(/\n/g, "<br>");

    return formatted;
  };

  // Set up event delegation for editor buttons
  React.useEffect(() => {
    const handleEditorButtonClick = (event) => {
      if (event.target.matches(".code-action-btn.editor-btn")) {
        const encodedCode = event.target.getAttribute("data-code");
        const encodedLanguage = event.target.getAttribute("data-language");
        const codeId = event.target.getAttribute("data-code-id");

        if (onOpenCodeEditor && encodedCode && encodedLanguage) {
          const code = decodeURIComponent(encodedCode);
          const language = decodeURIComponent(encodedLanguage);
          onOpenCodeEditor(code, language, codeId);
        }
      }
    };

    // Add event listener to document for event delegation
    document.addEventListener("click", handleEditorButtonClick);

    // Keep the global function for backwards compatibility with copy buttons
    window.openCodeInEditor = (encodedCode, encodedLanguage, codeId) => {
      if (onOpenCodeEditor) {
        const code = decodeURIComponent(encodedCode);
        const language = decodeURIComponent(encodedLanguage);
        onOpenCodeEditor(code, language, codeId);
      }
    };

    return () => {
      document.removeEventListener("click", handleEditorButtonClick);
      delete window.openCodeInEditor;
    };
  }, [onOpenCodeEditor]);

  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-avatar">{isUser ? "👤" : "🤖"}</div>
      <div className="message-content">
        <div
          className="message-text"
          dangerouslySetInnerHTML={{
            __html: formatMessage(message.content),
          }}
        />
      </div>
    </div>
  );
};

export default Message;
