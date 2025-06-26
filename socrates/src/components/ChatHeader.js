import React from 'react';
import '../css/ChatHeader.css';

const ChatHeader = ({ 
  currentChat, 
  selectedModule, 
  user, 
  onToggleSidebar, 
  isSidebarVisible 
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

      {/* Back Button (for mobile) */}
      <button className="back-btn">
        ←
      </button>

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