import React from "react";
import "../css/ChatHeader.css";

const ChatHeader = ({
  currentChat,
  selectedModule,
  user,
  onToggleSidebar,
  isSidebarVisible,
  splitPaneMode = false,
  onExitSplitMode,
  onLogout,
}) => {
  return (
    <div className="chat-header">
      {/* Sidebar Toggle Button */}
      <button
        className="sidebar-toggle-btn"
        onClick={onToggleSidebar}
        title={isSidebarVisible ? "Hide sidebar" : "Show sidebar"}
      >
        {isSidebarVisible ? "←" : "→"}
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
            : "Socratic AI"}
        </h1>

        {selectedModule && !currentChat && (
          <span className="chat-status online">CUDA GPT 3.0</span>
        )}
      </div>

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
        {user && (
          <>
            <button className="chat-action" title="Settings">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 0 2.4l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1 0-2.4l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </button>
            <button className="chat-action" onClick={onLogout} title="Logout">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default ChatHeader;
