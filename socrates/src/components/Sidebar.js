import React from "react";
import FolderSection from "./FolderSection";
import ChatsSection from "./ChatsSection";
import "../css/Sidebar.css";

const Sidebar = ({
  isOpen,
  onClose,
  onNewChat,
  onNavigateHome,
  onSelectChat,
  chats,
  loadingChats,
  currentChatId,
  backendStatus,
  onRefreshBackend,
  onSelectModule,
  selectedModuleId,
}) => {
  const sidebarClass = `sidebar ${isOpen ? "mobile-visible" : "mobile-hidden"}`;

  const getConnectionStatusClass = () => {
    if (backendStatus.connecting) return "connecting";
    if (backendStatus.online) return "connected";
    return "error";
  };

  const getConnectionStatusText = () => {
    if (backendStatus.connecting) return " Connecting to backend...";
    if (backendStatus.online) return "Backend connected";
    if (backendStatus.limited) return "Backend limited functionality";
    if (backendStatus.error === "Connection timeout")
      return "⏱️ Backend connection timeout";
    return "Backend disconnected";
  };

  const handleStatusClick = () => {
    if (!backendStatus.connecting && onRefreshBackend) {
      onRefreshBackend();
    }
  };

  return (
    <div className={sidebarClass}>
      <div
        className="sidebar-header"
        onClick={onNavigateHome}
        title="Go to Home"
      >
        <div className="app-logo">
          <img src="/socratic-logo.png" alt="Socratic AI Logo" />
        </div>
        <div className="app-title">Socratic AI</div>
        <button className="close-btn" onClick={onClose}>
          ✕
        </button>
      </div>

      <div
        className={`connection-status ${getConnectionStatusClass()} ${
          !backendStatus.connecting ? "clickable" : ""
        }`}
        onClick={handleStatusClick}
        title={
          !backendStatus.connecting ? "Click to refresh backend status" : ""
        }
      >
        {getConnectionStatusText()}
        {!backendStatus.connecting && <span className="refresh-hint">↻</span>}
      </div>
      <div className="search-section">
        <input
          type="text"
          placeholder="Search conversations..."
          className="search-input"
        />
      </div>
      <div className="sidebar-content">
        <FolderSection
          onSelectModule={onSelectModule}
          selectedModuleId={selectedModuleId}
        />
        <ChatsSection
          chats={chats}
          loadingChats={loadingChats}
          currentChatId={currentChatId}
          onSelectChat={onSelectChat}
        />
      </div>
      <div className="sidebar-footer">
        <button className="new-chat-btn" onClick={onNewChat}>
          <span>+</span>
          <span>New chat</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
