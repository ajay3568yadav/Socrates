import React from 'react';
import '../css/ChatHeader.css';

const ChatHeader = ({ 
  currentChat, 
  selectedModule, 
  user, 
  onToggleSidebar, 
  isSidebarVisible,
  splitPaneMode = false,
  onExitSplitMode
}) => {
  return (
    <div className="chat-header">
      {/* Sidebar Toggle Button */}
      <button 
        className="sidebar-toggle-btn" 
        onClick={onToggleSidebar}
        title={isSidebarVisible ? "Hide sidebar" : "Show sidebar"}
      >
        {isSidebarVisible ? '←' : '→'}
      </button>

      {/* Split Pane Mode Indicator */}
      {splitPaneMode && (
        <div className="split-mode-indicator">
          <span className="split-icon">⚡</span>
          <span className="split-text">Split View</span>
        </div>
      )}

      {/* Chat/Module Title */}
      <div className="chat-title-section">
        <h1 className="chat-title-main">
          {currentChat 
            ? currentChat.title 
            : selectedModule 
              ? selectedModule.name 
              : 'CUDA Tutor'
          }
        </h1>
        
        {selectedModule && (
          <span className="chat-status online">
            CUDA GPT 3.0
          </span>
        )}
      </div>

      {/* User Info */}
      {user && (
        <div className="header-user-info">
          <span className="user-greeting">
            Welcome, {user.name || user.email}!
          </span>
        </div>
      )}

      {/* Action Buttons */}
      <div className="chat-actions">
        {splitPaneMode && (
          <button 
            className="chat-action exit-split" 
            onClick={onExitSplitMode}
            title="Exit split view"
          >
            ✕
          </button>
        )}
        <button className="chat-action" title="Settings">
          ⚙️
        </button>
        <button className="chat-action" title="More options">
          🗑️
        </button>
      </div>
    </div>
  );
};

export default ChatHeader;