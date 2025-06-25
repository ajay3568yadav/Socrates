import './App.css';
import React, { useState, useEffect, useRef } from 'react';

// CSS Styles as a string (you can also put this in a separate .css file)
const styles = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #111111;
    color: white;
    height: 100vh;
    overflow: hidden;
  }

  .app-container {
    display: flex;
    height: 100vh;
    background: #111111;
  }

  /* Sidebar Styles */
  .sidebar {
    width: 288px;
    background: #1f1f1f;
    border-right: 1px solid #333333;
    display: flex;
    flex-direction: column;
    transition: transform 0.3s ease;
    position: relative; /* Changed from the mobile positioning */
  }

  /* Mobile-specific sidebar positioning */
  @media (max-width: 768px) {
    .sidebar {
      position: fixed;
      left: -288px;
      top: 0;
      height: 100vh;
      z-index: 50;
    }

    .sidebar.mobile-visible {
      left: 0;
    }
    
    .close-btn {
      display: block;
    }
  }

  /* Desktop sidebar - always visible, no transforms */
  @media (min-width: 769px) {
    .sidebar {
      position: relative;
      transform: none !important;
    }
    
    .sidebar.mobile-hidden,
    .sidebar.mobile-visible {
      transform: none !important;
      position: relative;
    }
  }

  .mobile-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 40;
    display: none;
  }

  .mobile-overlay.show {
    display: block;
  }

  /* Only show overlay on mobile devices */
  @media (max-width: 768px) {
    .mobile-overlay.show {
      display: block;
    }
  }

  /* Hide overlay on desktop */
  @media (min-width: 769px) {
    .mobile-overlay {
      display: none !important;
    }
  }

  .sidebar-header {
    padding: 16px;
    border-bottom: 1px solid #333333;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .app-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #4ade80, #22c55e);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
  }

  .app-title {
    font-weight: 600;
    flex: 1;
  }

  .close-btn {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    padding: 8px;
    border-radius: 6px;
    display: none;
  }

  .close-btn:hover {
    background: #333333;
  }

  .search-section {
    padding: 16px;
    border-bottom: 1px solid #333333;
  }

  .search-input {
    width: 100%;
    background: #333333;
    border: 1px solid #444444;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 14px;
    color: white;
    outline: none;
  }

  .search-input:focus {
    border-color: #22c55e;
  }

  .search-input::placeholder {
    color: #888888;
  }

  .sidebar-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px 0;
  }

  .folder-section {
    margin-bottom: 24px;
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px 8px 20px;
    font-size: 14px;
    color: #888888;
    font-weight: 500;
  }

  .section-actions {
    display: flex;
    gap: 8px;
  }

  .section-action {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    font-size: 12px;
    padding: 2px;
  }

  .section-action:hover {
    color: white;
  }

  .folder-item, .chat-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 20px;
    margin: 2px 0;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .folder-item:hover, .chat-item:hover {
    background: #333333;
  }

  .folder-item.active, .chat-item.active {
    background: #333333;
    border-left: 3px solid #22c55e;
  }

  .folder-icon, .chat-avatar {
    color: #888888;
    flex-shrink: 0;
  }

  .chat-avatar {
    width: 20px;
    height: 20px;
    background: #444444;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    margin-top: 2px;
  }

  .folder-name, .chat-title {
    font-size: 14px;
    color: #dddddd;
    font-weight: 500;
  }

  .chat-content {
    flex: 1;
    min-width: 0;
  }

  .chat-preview {
    font-size: 12px;
    color: #888888;
    line-height: 1.4;
    margin-top: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .item-menu {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.2s;
    flex-shrink: 0;
  }

  .folder-item:hover .item-menu,
  .chat-item:hover .item-menu {
    opacity: 1;
  }

  .new-chat-btn {
    margin: 16px;
    background: #22c55e;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: background-color 0.2s;
  }

  .new-chat-btn:hover {
    background: #16a34a;
  }

  /* Main Content Styles */
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #111111;
  }

  .chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid #333333;
    display: flex;
    align-items: center;
    gap: 12px;
    background: #1a1a1a;
  }

  .back-btn {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    font-size: 18px;
    padding: 8px;
    border-radius: 6px;
    transition: background-color 0.2s;
  }

  .back-btn:hover {
    background: #333333;
  }

  .chat-title-section {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
  }

  .chat-title-main {
    font-size: 16px;
    font-weight: 600;
  }

  .chat-status {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
  }

  .chat-status.online {
    background: #1a5c1a;
    color: #4ade80;
  }

  .chat-status.limited {
    background: #5c4a1a;
    color: #fbbf24;
  }

  .chat-status.offline {
    background: #5c1a1a;
    color: #f87171;
  }

  .chat-actions {
    display: flex;
    gap: 8px;
  }

  .chat-action {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    font-size: 18px;
    padding: 8px;
    border-radius: 6px;
    transition: background-color 0.2s;
  }

  .chat-action:hover {
    background: #333333;
  }

  /* Welcome View Styles */
  .welcome-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 24px;
    text-align: center;
  }

  .welcome-icon {
    width: 80px;
    height: 80px;
    background: linear-gradient(135deg, #4ade80, #22c55e);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
    margin-bottom: 24px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }

  .welcome-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #4ade80, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .welcome-subtitle {
    font-size: 16px;
    color: #888888;
    margin-bottom: 40px;
    line-height: 1.5;
    max-width: 600px;
  }

  .quick-actions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin-bottom: 40px;
    width: 100%;
    max-width: 800px;
  }

  .quick-action {
    background: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    color: inherit;
  }

  .quick-action:hover {
    background: #333333;
    border-color: #22c55e;
    transform: translateY(-2px);
  }

  .quick-action-icon {
    font-size: 24px;
    margin-bottom: 12px;
    color: #22c55e;
  }

  .quick-action-title {
    font-size: 14px;
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
  }

  .quick-action-desc {
    font-size: 12px;
    color: #888888;
    line-height: 1.4;
  }

  .input-section {
    width: 100%;
    max-width: 600px;
  }

  .input-tabs {
    display: flex;
    gap: 24px;
    margin-bottom: 24px;
    justify-content: center;
  }

  .input-tab {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    font-size: 14px;
    padding: 8px 0;
    position: relative;
    transition: color 0.2s;
  }

  .input-tab.active {
    color: #22c55e;
  }

  .input-tab.active::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: #22c55e;
  }

  .input-container {
    background: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    align-items: flex-end;
    gap: 12px;
    transition: border-color 0.2s;
  }

  .input-container:focus-within {
    border-color: #22c55e;
  }

  .input-icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #4ade80, #22c55e);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }

  .input-field {
    flex: 1;
    background: none;
    border: none;
    color: white;
    font-size: 16px;
    resize: none;
    min-height: 24px;
    max-height: 120px;
    outline: none;
    font-family: inherit;
  }

  .input-field::placeholder {
    color: #888888;
  }

  .input-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .input-action {
    background: none;
    border: none;
    color: #888888;
    cursor: pointer;
    font-size: 18px;
    padding: 6px;
    border-radius: 6px;
    transition: all 0.2s;
  }

  .input-action:hover {
    background: #333333;
    color: white;
  }

  .send-btn {
    background: #22c55e;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.2s;
  }

  .send-btn:hover {
    background: #16a34a;
  }

  .send-btn:disabled {
    background: #333333;
    cursor: not-allowed;
  }

  .disclaimer {
    margin-top: 24px;
    font-size: 12px;
    color: #666666;
    text-align: center;
  }

  /* Chat View Styles */
  .chat-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden; /* Prevent parent overflow issues */
    position: relative;
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 24px;
    scroll-behavior: smooth;
    /* Ensure smooth scrolling on all browsers */
    -webkit-scroll-behavior: smooth;
    position: relative;
  }

  .messages-wrapper {
    max-width: 800px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 24px;
    min-height: 100%; /* Ensure wrapper takes full height */
    padding-bottom: 20px; /* Extra space at bottom */
  }

  .scroll-to-bottom-btn {
    position: absolute;
    bottom: 80px;
    right: 24px;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #22c55e;
    color: white;
    border: none;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.2s;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .scroll-to-bottom-btn:hover {
    background: #16a34a;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
  }

  .scroll-to-bottom-btn:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }

  .message {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .message.user {
    flex-direction: row-reverse;
  }

  .message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }

  .message.user .message-avatar {
    background: #2563eb;
  }

  .message.assistant .message-avatar {
    background: linear-gradient(135deg, #4ade80, #22c55e);
  }

  .message-content {
    max-width: 70%;
    background: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 16px 20px;
  }

  .message.user .message-content {
    background: #2563eb;
    border-color: #2563eb;
  }

  .typing-cursor {
    color: #22c55e;
    animation: blink 1s infinite;
    font-weight: bold;
    margin-left: 2px;
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }

  /* Enhanced message text for streaming */
  .message-text {
    font-size: 14px;
    line-height: 1.5;
    color: white;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  /* Enhanced Code Block Styles */
  .code-block-container {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin: 12px 0;
    overflow: hidden;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  .code-block-header {
    background: #161b22;
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #30363d;
  }

  .code-language {
    font-size: 11px;
    font-weight: 600;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .code-copy-btn {
    background: none;
    border: none;
    color: #7d8590;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    transition: all 0.2s;
  }

  .code-copy-btn:hover {
    background: #30363d;
    color: #f0f6fc;
  }

  .code-block {
    background: #0d1117;
    padding: 16px;
    margin: 0;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.45;
    color: #f0f6fc;
  }

  .code-block code {
    background: none;
    padding: 0;
    color: inherit;
    font-size: inherit;
  }

  /* Syntax Highlighting Colors */
  .code-keyword {
    color: #ff7b72; /* Red - for keywords like __global__, if, for */
  }

  .code-function {
    color: #ffa657; /* Orange - for CUDA functions */
  }

  .code-function-call {
    color: #79c0ff; /* Light blue - for function calls */
  }

  .code-string {
    color: #a5d6ff; /* Light cyan - for strings */
  }

  .code-comment {
    color: #8b949e; /* Gray - for comments */
    font-style: italic;
  }

  .code-number {
    color: #79c0ff; /* Blue - for numbers */
  }

  .code-preprocessor {
    color: #ffa657; /* Orange - for #include, #define */
  }

  .code-operator {
    color: #ff7b72; /* Red - for operators */
  }

  .code-bracket {
    color: #f0f6fc; /* White - for brackets */
  }

  .inline-code {
    background: rgba(110, 118, 129, 0.4);
    color: #79c0ff;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  /* Old code block styles - keep for backwards compatibility */
  .message-content pre {
    background: #0d1117;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    overflow-x: auto;
    font-size: 13px;
    border: 1px solid #30363d;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  .message-content code {
    background: rgba(110, 118, 129, 0.4);
    color: #79c0ff;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  .typing-indicator {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .typing-content {
    background: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 16px 20px;
  }

  .typing-text {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #888888;
  }

  .typing-dots {
    display: flex;
    gap: 4px;
  }

  .typing-dot {
    width: 6px;
    height: 6px;
    background: #888888;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
  }

  .typing-dot:nth-child(1) { animation-delay: -0.32s; }
  .typing-dot:nth-child(2) { animation-delay: -0.16s; }

  @keyframes typing {
    0%, 80%, 100% { 
      transform: scale(0.5); 
      opacity: 0.5; 
    }
    40% { 
      transform: scale(1); 
      opacity: 1; 
    }
  }

  .chat-input-area {
    padding: 20px 24px;
    border-top: 1px solid #333333;
    background: #1a1a1a;
  }

  .chat-input-container {
    max-width: 800px;
    margin: 0 auto;
  }

  /* Connection Status Indicator */
  .connection-status {
    padding: 8px 16px;
    margin: 16px;
    border-radius: 8px;
    font-size: 12px;
    text-align: center;
  }

  .connection-status.connecting {
    background: #5c4a1a;
    color: #fbbf24;
    border: 1px solid #fbbf24;
  }

  .connection-status.connected {
    background: #1a5c1a;
    color: #4ade80;
    border: 1px solid #4ade80;
  }

  .connection-status.error {
    background: #5c1a1a;
    color: #f87171;
    border: 1px solid #f87171;
  }

  /* Mobile Responsive */
  @media (max-width: 768px) {
    .sidebar {
      position: fixed;
      left: -288px;
      top: 0;
      height: 100vh;
      z-index: 50;
    }

    .sidebar.mobile-visible {
      left: 0;
    }

    .close-btn {
      display: block;
    }

    .chat-header {
      padding: 12px 16px;
    }

    .welcome-container {
      padding: 20px 16px;
    }

    .quick-actions {
      grid-template-columns: 1fr;
    }

    .input-tabs {
      gap: 16px;
    }

    .welcome-title {
      font-size: 24px;
    }

    .welcome-icon {
      width: 60px;
      height: 60px;
      font-size: 28px;
    }

    .message-content {
      max-width: 85%;
    }

    /* Mobile overlay only visible on mobile */
    .mobile-overlay.show {
      display: block;
    }
  }

  /* Scrollbar Styling */
  .sidebar-content::-webkit-scrollbar,
  .messages-container::-webkit-scrollbar {
    width: 6px;
  }

  .sidebar-content::-webkit-scrollbar-track,
  .messages-container::-webkit-scrollbar-track {
    background: transparent;
  }

  .sidebar-content::-webkit-scrollbar-thumb,
  .messages-container::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 3px;
  }

  .sidebar-content::-webkit-scrollbar-thumb:hover,
  .messages-container::-webkit-scrollbar-thumb:hover {
    background: #444444;
  }
`;

// Backend API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-backend-domain.com' 
  : 'http://localhost:5001';

// Main App Component
const CudaTutorApp = () => {
  const [currentView, setCurrentView] = useState('welcome');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ 
    online: false, 
    limited: false, 
    connecting: true 
  });

  useEffect(() => {
    // Inject styles
    const styleSheet = document.createElement('style');
    styleSheet.type = 'text/css';
    styleSheet.innerText = styles;
    document.head.appendChild(styleSheet);

    // Add global copy function for code blocks
    window.copyCodeToClipboard = async (encodedCode, buttonElement) => {
      try {
        const code = decodeURIComponent(encodedCode);
        await navigator.clipboard.writeText(code);
        
        // Show temporary success feedback
        if (buttonElement) {
          const originalText = buttonElement.textContent;
          buttonElement.textContent = '✓';
          buttonElement.style.color = '#22c55e';
          
          setTimeout(() => {
            buttonElement.textContent = originalText;
            buttonElement.style.color = '';
          }, 2000);
        }
      } catch (err) {
        console.error('Failed to copy code:', err);
      }
    };

    // Add global highlight function
    window.highlightCudaCode = highlightCudaCode;

    // Check backend status on mount
    checkBackendStatus();

    // Periodic status check
    const statusInterval = setInterval(checkBackendStatus, 30000); // Check every 30 seconds

    return () => {
      document.head.removeChild(styleSheet);
      clearInterval(statusInterval);
      delete window.copyCodeToClipboard;
      delete window.highlightCudaCode;
    };
  }, []);

  const checkBackendStatus = async () => {
    try {
      setBackendStatus(prev => ({ ...prev, connecting: true }));
      
      const response = await fetch(`${API_BASE_URL}/api/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 10000, // 10 second timeout
      });
      
      if (response.ok) {
        const data = await response.json();
        setBackendStatus({
          online: data.rag_loaded && data.ollama_status === 'connected',
          limited: !data.rag_loaded || data.ollama_status !== 'connected',
          connecting: false,
          details: data
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Backend status check failed:', error);
      setBackendStatus({ 
        online: false, 
        limited: false, 
        connecting: false,
        error: error.message 
      });
    }
  };

  const startNewChat = () => {
    setCurrentChatId(`chat_${Date.now()}`);
    setCurrentView('welcome');
    setMessages([]);
    setSidebarOpen(false);
  };

  const sendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return;

    setCurrentView('chat');
    const userMessage = { id: Date.now(), text: messageText, isUser: true };
    setMessages(prev => [...prev, userMessage]);

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ 
          message: messageText,
          session_id: sessionId,
          chat_id: currentChatId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'error') {
        throw new Error(data.error || 'Unknown error from backend');
      }

      const assistantMessage = { 
        id: Date.now() + 1, 
        text: data.response, 
        isUser: false 
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Update backend status on successful message
      setBackendStatus(prev => ({ ...prev, online: true, limited: false }));

    } catch (error) {
      console.error('Error:', error);
      
      let errorMessage;
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMessage = {
          id: Date.now() + 1,
          text: `
            <div style="color: #fca5a5; margin-bottom: 16px;">
              <strong>🔌 Connection Error:</strong> Unable to connect to the CUDA Tutor backend at ${API_BASE_URL}.
            </div>
            <div style="background: #1a1a1a; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid #f87171;">
              <strong>Troubleshooting:</strong><br/>
              1. Make sure the Flask backend is running on port 5001<br/>
              2. Run: <code>python backend.py</code><br/>
              3. Check that the server started successfully<br/>
              4. Verify no firewall is blocking the connection
            </div>
            <div style="background: #1a1a1a; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid #22c55e;">
              <strong>🤖 Demo Response:</strong> CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows developers to use GPU cores for general-purpose computing, achieving significant speedups for parallelizable tasks.
            </div>
          `,
          isUser: false
        };
      } else {
        errorMessage = {
          id: Date.now() + 1,
          text: `
            <div style="color: #fca5a5; margin-bottom: 16px;">
              <strong>⚠️ Backend Error:</strong> ${error.message}
            </div>
            <div style="background: #1a1a1a; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid #22c55e;">
              <strong>🤖 Demo Response:</strong> I'm having trouble processing your request right now. CUDA is NVIDIA's parallel computing platform that enables developers to harness GPU power for general-purpose computing tasks.
            </div>
          `,
          isUser: false
        };
      }
      
      setMessages(prev => [...prev, errorMessage]);
      
      // Update backend status on error
      setBackendStatus(prev => ({ ...prev, online: false, limited: false, error: error.message }));
    }

    setIsLoading(false);
    
    // Force scroll after message is sent - with longer delay for async operations
    setTimeout(() => {
      const container = document.querySelector('.messages-container');
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    }, 300);
  };

  return (
    <div className="app-container">
      {/* Mobile Overlay - Only show on mobile */}
      {sidebarOpen && (
        <div 
          className={`mobile-overlay ${sidebarOpen ? 'show' : ''}`}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={startNewChat}
        backendStatus={backendStatus}
      />

      {/* Main Content */}
      <div className="main-content">
        <ChatHeader 
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          backendStatus={backendStatus}
        />
        
        {currentView === 'welcome' ? (
          <WelcomeView onSendMessage={sendMessage} />
        ) : (
          <ChatView 
            messages={messages}
            isLoading={isLoading}
            onSendMessage={sendMessage}
          />
        )}
      </div>
    </div>
  );
};

// Sidebar Component
const Sidebar = ({ isOpen, onClose, onNewChat, backendStatus }) => {
  const sidebarClass = `sidebar ${isOpen ? 'mobile-visible' : 'mobile-hidden'}`;

  const getConnectionStatusClass = () => {
    if (backendStatus.connecting) return 'connecting';
    if (backendStatus.online) return 'connected';
    return 'error';
  };

  const getConnectionStatusText = () => {
    if (backendStatus.connecting) return '🔄 Connecting to backend...';
    if (backendStatus.online) return '✅ Backend connected';
    if (backendStatus.limited) return '⚠️ Backend limited functionality';
    return '❌ Backend disconnected';
  };

  return (
    <div className={sidebarClass}>
      <div className="sidebar-header">
        <div className="app-logo">🚀</div>
        <div className="app-title">CUDA Tutor</div>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      <div className={`connection-status ${getConnectionStatusClass()}`}>
        {getConnectionStatusText()}
      </div>

      <div className="search-section">
        <input 
          type="text" 
          placeholder="Search conversations..."
          className="search-input"
        />
      </div>

      <div className="sidebar-content">
        <FolderSection />
        <ChatsSection />
      </div>

      <button className="new-chat-btn" onClick={onNewChat}>
        <span>+</span>
        <span>New chat</span>
      </button>
    </div>
  );
};

// Folder Section Component
const FolderSection = () => {
  const folders = [
    { name: 'CUDA Basics', icon: '📁', active: true },
    { name: 'Memory Optimization', icon: '📁', active: false },
    { name: 'Kernel Development', icon: '📁', active: false },
    { name: 'Performance Tuning', icon: '📁', active: false }
  ];

  return (
    <div className="folder-section">
      <div className="section-header">
        <span>Topics</span>
        <div className="section-actions">
          <button className="section-action">⋯</button>
        </div>
      </div>
      
      <div>
        {folders.map((folder, index) => (
          <div 
            key={index}
            className={`folder-item ${folder.active ? 'active' : ''}`}
          >
            <span className="folder-icon">{folder.icon}</span>
            <span className="folder-name">{folder.name}</span>
            <button className="item-menu">⋯</button>
          </div>
        ))}
      </div>
    </div>
  );
};

// Chats Section Component
const ChatsSection = () => {
  const chats = [
    { 
      title: 'Getting Started with CUDA', 
      preview: 'Basic CUDA concepts and your first kernel...', 
      avatar: '🤖',
      active: true 
    },
    { 
      title: 'Memory Optimization Tips', 
      preview: 'Coalesced memory access patterns and shared memory usage...', 
      avatar: '📊',
      active: false 
    },
    { 
      title: 'Kernel Performance Analysis', 
      preview: 'Profiling tools and optimization strategies...', 
      avatar: '⚡',
      active: false 
    }
  ];

  return (
    <div className="folder-section">
      <div className="section-header">
        <span>Recent Chats</span>
        <div className="section-actions">
          <button className="section-action">⋯</button>
        </div>
      </div>
      
      <div>
        {chats.map((chat, index) => (
          <div 
            key={index}
            className={`chat-item ${chat.active ? 'active' : ''}`}
          >
            <div className="chat-avatar">{chat.avatar}</div>
            <div className="chat-content">
              <div className="chat-title">{chat.title}</div>
              <div className="chat-preview">{chat.preview}</div>
            </div>
            <button className="item-menu">⋯</button>
          </div>
        ))}
      </div>
    </div>
  );
};

// Chat Header Component
const ChatHeader = ({ onToggleSidebar, backendStatus }) => {
  const getStatusClass = () => {
    if (backendStatus.online) return 'online';
    if (backendStatus.limited) return 'limited';
    return 'offline';
  };

  const getStatusText = () => {
    if (backendStatus.online) return 'CUDA GPT 3.0 ✓';
    if (backendStatus.limited) return 'CUDA GPT 3.0 ⚠';
    return 'CUDA GPT 3.0 ✗';
  };

  return (
    <div className="chat-header">
      <button className="back-btn" onClick={onToggleSidebar}>←</button>
      
      <div className="chat-title-section">
        <div className="chat-title-main">CUDA Tutor</div>
        <span className={`chat-status ${getStatusClass()}`}>
          {getStatusText()}
        </span>
      </div>
      
      <div className="chat-actions">
        <button className="chat-action" title="Share">📤</button>
        <button className="chat-action" title="Save">💾</button>
      </div>
    </div>
  );
};

// Welcome View Component
const WelcomeView = ({ onSendMessage }) => {
  const [inputValue, setInputValue] = useState('');
  const [activeTab, setActiveTab] = useState('All');
  const textareaRef = useRef(null);

  const tabs = ['All', 'Text', 'Image', 'Video', 'Music', 'Analytics'];
  
  const quickActions = [
    {
      icon: '📚',
      title: 'CUDA Basics',
      description: 'Learn the fundamentals of CUDA programming',
      question: 'What is CUDA and why is it useful?'
    },
    {
      icon: '⚡',
      title: 'Kernel Examples',
      description: 'Get started with your first CUDA kernel',
      question: 'Show me a simple CUDA kernel example'
    },
    {
      icon: '🧠',
      title: 'Memory Optimization',
      description: 'Master efficient memory usage patterns',
      question: 'How do I optimize memory access in CUDA?'
    },
    {
      icon: '🏆',
      title: 'Best Practices',
      description: 'Professional CUDA development tips',
      question: 'What are the best practices for CUDA programming?'
    }
  ];

  const handleSubmit = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleQuickAction = (question) => {
    onSendMessage(question);
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [inputValue]);

  return (
    <div className="welcome-container">
      <div className="welcome-icon">🚀</div>

      <h1 className="welcome-title">How can I help you today?</h1>
      
      <p className="welcome-subtitle">
        Ask me anything about CUDA programming, GPU optimization, or parallel computing.
        I'm here to help you master GPU development!
      </p>

      <div className="quick-actions">
        {quickActions.map((action, index) => (
          <div 
            key={index}
            onClick={() => handleQuickAction(action.question)}
            className="quick-action"
          >
            <div className="quick-action-icon">{action.icon}</div>
            <div className="quick-action-title">{action.title}</div>
            <div className="quick-action-desc">{action.description}</div>
          </div>
        ))}
      </div>

      <div className="input-section">
        <div className="input-tabs">
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`input-tab ${activeTab === tab ? 'active' : ''}`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="input-container">
          <div className="input-icon">🚀</div>
          
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your prompt here..."
            className="input-field"
            rows={1}
          />
          
          <div className="input-actions">
            <button className="input-action">🎤</button>
            <button 
              onClick={handleSubmit}
              disabled={!inputValue.trim()}
              className="send-btn"
            >
              →
            </button>
          </div>
        </div>

        <p className="disclaimer">
          CUDA Tutor can make mistakes. Consider checking important information.
        </p>
      </div>
    </div>
  );
};

// Chat View Component
const ChatView = ({ messages, isLoading, onSendMessage }) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState(null);

  const scrollToBottom = () => {
    // Multiple scroll methods for better compatibility
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
    
    // Fallback: scroll container to bottom
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom);
    }
  };

  const handleStreamingComplete = () => {
    setStreamingMessageId(null);
  };

  useEffect(() => {
    // Scroll immediately when messages change
    scrollToBottom();
    
    // Set streaming message ID for the latest assistant message
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && !lastMessage.isUser && !isLoading) {
      setStreamingMessageId(lastMessage.id);
    }
    
    // Also scroll after a short delay to handle dynamic content loading
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(timeoutId);
  }, [messages, isLoading]);

  // Force scroll when component mounts
  useEffect(() => {
    scrollToBottom();
  }, []);

  // Scroll during streaming
  useEffect(() => {
    if (streamingMessageId) {
      const interval = setInterval(scrollToBottom, 100);
      return () => clearInterval(interval);
    }
  }, [streamingMessageId]);

  return (
    <div className="chat-view">
      <div 
        className="messages-container" 
        ref={messagesContainerRef}
        onScroll={handleScroll}
      >
        <div className="messages-wrapper">
          {messages.map((message, index) => {
            // Check if this is the latest assistant message and should be streamed
            const shouldStream = !message.isUser && 
                               message.id === streamingMessageId && 
                               index === messages.length - 1;
                               
            return shouldStream ? (
              <StreamingMessage 
                key={message.id} 
                message={message} 
                onComplete={handleStreamingComplete}
              />
            ) : (
              <Message key={message.id} message={message} />
            );
          })}
          
          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} style={{ height: '1px', marginTop: '20px' }} />
        </div>
        
        {/* Scroll to bottom button */}
        {showScrollButton && (
          <button 
            className="scroll-to-bottom-btn"
            onClick={scrollToBottom}
            title="Scroll to bottom"
          >
            ↓
          </button>
        )}
      </div>

      <ChatInput onSendMessage={onSendMessage} isLoading={isLoading} />
    </div>
  );
};

// Syntax highlighting function for CUDA/C++ code
const highlightCudaCode = (code) => {
  // CUDA/C++ keywords
  const keywords = [
    '__global__', '__device__', '__host__', '__shared__', '__constant__',
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue', 'return',
    'int', 'float', 'double', 'char', 'bool', 'void', 'auto', 'const', 'static', 'extern',
    'struct', 'class', 'enum', 'typedef', 'sizeof', 'namespace', 'using',
    'cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaDeviceSynchronize', 'cudaGetLastError',
    'blockIdx', 'blockDim', 'threadIdx', 'gridDim', '__syncthreads'
  ];

  // CUDA functions and types
  const cudaFunctions = [
    'cudaEventCreate', 'cudaEventRecord', 'cudaEventSynchronize', 'cudaEventElapsedTime',
    'cudaEventDestroy', 'cudaMallocManaged', 'cudaMemcpyAsync', 'cudaStreamCreate',
    'cudaStreamSynchronize', 'cudaStreamDestroy', 'cudaGetDeviceProperties',
    'dim3', 'cudaError_t', 'cudaStream_t', 'cudaEvent_t'
  ];

  // Numbers and operators
  let highlighted = code
    // Comments (green)
    .replace(/(\/\/.*$)/gm, '<span class="code-comment">$1</span>')
    .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="code-comment">$1</span>')
    
    // Strings (yellow)
    .replace(/(".*?")/g, '<span class="code-string">$1</span>')
    .replace(/('.*?')/g, '<span class="code-string">$1</span>')
    
    // Numbers (cyan)
    .replace(/\b(\d+\.?\d*[fF]?)\b/g, '<span class="code-number">$1</span>')
    
    // Preprocessor directives (magenta)
    .replace(/(#\w+)/g, '<span class="code-preprocessor">$1</span>')
    
    // CUDA-specific keywords (bright blue)
    .replace(new RegExp(`\\b(${keywords.join('|')})\\b`, 'g'), '<span class="code-keyword">$1</span>')
    
    // CUDA functions (orange)
    .replace(new RegExp(`\\b(${cudaFunctions.join('|')})\\b`, 'g'), '<span class="code-function">$1</span>')
    
    // Function calls (light blue)
    .replace(/\b(\w+)(\s*\()/g, '<span class="code-function-call">$1</span>$2')
    
    // Operators (white/gray)
    .replace(/([+\-*/%=<>!&|^~])/g, '<span class="code-operator">$1</span>')
    
    // Brackets and parentheses (bright white)
    .replace(/([(){}\[\]])/g, '<span class="code-bracket">$1</span>');

  return highlighted;
};

// Enhanced code block component
const CodeBlock = ({ code, language = 'cuda' }) => {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const highlightedCode = language === 'cuda' || language === 'cpp' || language === 'c'
    ? highlightCudaCode(code)
    : code;

  return (
    <div className="code-block-container">
      <div className="code-block-header">
        <span className="code-language">{language.toUpperCase()}</span>
        <button 
          className="code-copy-btn"
          onClick={copyToClipboard}
          title="Copy code"
        >
          {copied ? '✓' : '📋'}
        </button>
      </div>
      <pre className="code-block">
        <code dangerouslySetInnerHTML={{ __html: highlightedCode }} />
      </pre>
    </div>
  );
};
const Message = ({ message }) => {
  const formatMessage = (content) => {
    // Convert code blocks
    content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    
    // Convert inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    return content;
  };

  return (
    <div className={`message ${message.isUser ? 'user' : 'assistant'}`}>
      <div className="message-avatar">
        {message.isUser ? '👤' : '🤖'}
      </div>
      
      <div className="message-content">
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }}
        />
      </div>
    </div>
  );
};

// Streaming Message Component with typing effect
const StreamingMessage = ({ message, onComplete }) => {
  const [displayedText, setDisplayedText] = useState('');
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
      setDisplayedText(prev => prev + message.text[currentIndex]);
      setCurrentIndex(prev => prev + 1);
    }, typingSpeed);

    return () => clearTimeout(timer);
  }, [currentIndex, message.text, onComplete]);

  // Reset when message changes
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
    setIsTyping(true);
  }, [message.id]);

  const formatMessage = (content) => {
    // First, extract and replace code blocks with placeholders
    const codeBlocks = [];
    let codeBlockIndex = 0;
    
    // Extract code blocks
    content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
      const placeholder = `__CODE_BLOCK_${codeBlockIndex}__`;
      codeBlocks.push({ language: lang || 'text', code: code.trim() });
      codeBlockIndex++;
      return placeholder;
    });
    
    // Convert inline code
    content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    // Restore code blocks with syntax highlighting
    codeBlocks.forEach((block, index) => {
      const placeholder = `__CODE_BLOCK_${index}__`;
      const highlightedCode = (block.language === 'cuda' || block.language === 'cpp' || block.language === 'c') 
        ? highlightCudaCode(block.code) 
        : block.code;
      
      const codeBlockHtml = `
        <div class="code-block-container">
          <div class="code-block-header">
            <span class="code-language">${block.language.toUpperCase()}</span>
            <button class="code-copy-btn" onclick="copyCodeToClipboard('${encodeURIComponent(block.code)}', this)" title="Copy code">📋</button>
          </div>
          <pre class="code-block"><code>${highlightedCode}</code></pre>
        </div>
      `;
      
      content = content.replace(placeholder, codeBlockHtml);
    });
    
    return content;
  };

  return (
    <div className={`message ${message.isUser ? 'user' : 'assistant'}`}>
      <div className="message-avatar">
        {message.isUser ? '👤' : '🤖'}
      </div>
      
      <div className="message-content">
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatMessage(displayedText) }}
        />
        {isTyping && (
          <span className="typing-cursor">▋</span>
        )}
      </div>
    </div>
  );
};

// Typing Indicator Component
const TypingIndicator = () => {
  return (
    <div className="typing-indicator">
      <div className="message-avatar">🤖</div>
      <div className="typing-content">
        <div className="typing-text">
          <span>Thinking</span>
          <div className="typing-dots">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Chat Input Component
const ChatInput = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [inputValue]);

  return (
    <div className="chat-input-area">
      <div className="chat-input-container">
        <div className="input-container">
          <div className="input-icon">🚀</div>
          
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="input-field"
            rows={1}
            disabled={isLoading}
          />
          
          <div className="input-actions">
            <button className="input-action">🎤</button>
            <button 
              onClick={handleSubmit}
              disabled={!inputValue.trim() || isLoading}
              className="send-btn"
            >
              →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Export the main App component
function App() {
  return <CudaTutorApp />;
}

export default App;