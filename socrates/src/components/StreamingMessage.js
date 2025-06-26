import React, { useState, useEffect } from "react";
import { highlightCudaCode } from "../utils/codeHighlighting";

const StreamingMessage = ({ message, onComplete, onOpenCodeEditor }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!message.text || currentIndex >= message.text.length) {
      setIsTyping(false);
      if (onComplete) onComplete();
      return;
    }

    const typingSpeed = message.text.length > 1000 ? 10 : 30; // Faster for long messages

    const timer = setTimeout(() => {
      setDisplayedText((prev) => prev + message.text[currentIndex]);
      setCurrentIndex((prev) => prev + 1);
    }, typingSpeed);

    return () => clearTimeout(timer);
  }, [currentIndex, message.text, onComplete]);

  // Reset when message changes
  useEffect(() => {
    setDisplayedText("");
    setCurrentIndex(0);
    setIsTyping(true);
  }, [message.id]);

  // Set up event delegation for editor buttons
  useEffect(() => {
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

    // Keep the global function for backwards compatibility
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

  const formatMessage = (content) => {
    const codeBlocks = extractCodeBlocks(content);

    // If there are code blocks, add editor buttons (since this is always an assistant message)
    if (codeBlocks.length > 0) {
      let formattedContent = content;

      codeBlocks.forEach((block, index) => {
        const codeId = `streaming_code_${message.id}_${index}`;
        const encodedCode = encodeURIComponent(block.code);
        const encodedLanguage = encodeURIComponent(block.language);
        const highlightedCode =
          block.language === "cuda" ||
          block.language === "cpp" ||
          block.language === "c"
            ? highlightCudaCode(block.code)
            : block.code;

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
            <pre class="code-block"><code>${highlightedCode}</code></pre>
          </div>
        `;

        formattedContent = formattedContent.replace(
          block.fullMatch,
          codeWithButtons
        );
      });

      // Convert inline code
      formattedContent = formattedContent.replace(
        /`([^`]+)`/g,
        '<code class="inline-code">$1</code>'
      );

      // Convert line breaks
      formattedContent = formattedContent.replace(/\n/g, "<br>");

      return formattedContent;
    }

    // Fallback to original formatting for content without code blocks
    // First, extract and replace code blocks with placeholders
    const codeBlocksForFallback = [];
    let codeBlockIndex = 0;

    // Extract code blocks
    content = content.replace(
      /```(\w+)?\n?([\s\S]*?)```/g,
      (match, lang, code) => {
        const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
        codeBlocksForFallback.push({
          language: lang || "text",
          code: code.trim(),
        });
        codeBlockIndex++;
        return placeholder;
      }
    );

    // Convert inline code
    content = content.replace(
      /`([^`]+)`/g,
      '<code class="inline-code">$1</code>'
    );

    // Convert line breaks
    content = content.replace(/\n/g, "<br>");

    // Restore code blocks with syntax highlighting
    codeBlocksForFallback.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      const highlightedCode =
        block.language === "cuda" ||
        block.language === "cpp" ||
        block.language === "c"
          ? highlightCudaCode(block.code)
          : block.code;

      const codeBlockHtml = `
        <div class="code-block-container">
          <div class="code-block-header">
            <span class="code-language">${block.language.toUpperCase()}</span>
            <button class="code-copy-btn" onclick="copyCodeToClipboard('${encodeURIComponent(
              block.code
            )}', this)" title="Copy code">📋</button>
          </div>
          <pre class="code-block"><code>${highlightedCode}</code></pre>
        </div>
      `;

      content = content.replace(placeholder, codeBlockHtml);
    });

    return content;
  };

  return (
    <div className={`message ${message.isUser ? "user" : "assistant"}`}>
      <div className="message-avatar">{message.isUser ? "👤" : "🤖"}</div>

      <div className="message-content">
        <div
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatMessage(displayedText) }}
        />
        {isTyping && <span className="typing-cursor">▋</span>}
      </div>
    </div>
  );
};

export default StreamingMessage;
