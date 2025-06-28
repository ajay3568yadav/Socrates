import React from "react";
import FolderSection from "./FolderSection";
import ChatsSection from "./ChatsSection";
import "../css/Sidebar.css";

const Sidebar = ({
  isOpen,
  isCollapsed,
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
  user,
  onLogout,
}) => {
  // Get sidebar CSS classes
  const getSidebarClass = () => {
    let classes = "sidebar";
    
    if (isCollapsed && window.innerWidth > 768) {
      classes += " collapsed";
    }
    
    if (window.innerWidth <= 768) {
      classes += isOpen ? " mobile-visible" : " mobile-hidden";
    } else {
      classes += isOpen ? " desktop-visible" : " desktop-hidden";
    }
    
    return classes;
  };

  // Backend status helpers
  const getStatusClass = () => {
    if (backendStatus?.connecting) return "connecting";
    if (backendStatus?.online) return "connected";
    return "error";
  };

  const getStatusText = () => {
    if (backendStatus?.connecting) return "Connecting...";
    if (backendStatus?.online) return "Connected";
    return "Disconnected";
  };

  const getUserInitials = () => {
    if (!user?.email) return "U";
    return user.email.charAt(0).toUpperCase();
  };

  // Collapsed navigation for desktop
  const renderCollapsedNav = () => (
    <div className="collapsed-nav">
      {/* Home */}
      <button
        className="collapsed-item"
        onClick={onNavigateHome}
        title="Home"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </button>
      
      {/* Modules */}
      <button
        className={`collapsed-item ${selectedModuleId ? 'active' : ''}`}
        onClick={() => onSelectModule?.(selectedModuleId)}
        title="Modules"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" stroke="currentColor" strokeWidth="2"/>
          <path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </button>
      
      {/* Chats */}
      <button
        className={`collapsed-item ${currentChatId ? 'active' : ''}`}
        onClick={() => chats?.[0] && onSelectChat(chats[0].chat_id)}
        title={`Chats (${chats?.length || 0})`}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </button>
      
      {/* Status */}
      <button
        className="collapsed-item"
        onClick={onRefreshBackend}
        title={getStatusText()}
      >
        <div className={`status-dot ${getStatusClass()}`}></div>
      </button>
      
      {/* User */}
      {user && (
        <button
          className="collapsed-item user-item"
          onClick={onLogout}
          title="Logout"
        >
          {getUserInitials()}
        </button>
      )}
      
      {/* New Chat */}
      <button
        className="collapsed-new-chat"
        onClick={onNewChat}
        title="New Chat"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2"/>
          <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </button>
    </div>
  );

  return (
    <div className={getSidebarClass()}>
      {/* Header */}
      <div className="sidebar-header" onClick={onNavigateHome}>
        <div className="app-logo">
          <img src="/socratic-logo.png" alt="Socratic AI" />
        </div>
        <div className="app-title">Socratic AI</div>
        <button className="close-btn" onClick={onClose}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2"/>
            <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2"/>
          </svg>
        </button>
      </div>

      {/* Collapsed view */}
      {isCollapsed && window.innerWidth > 768 ? (
        renderCollapsedNav()
      ) : (
        <>
          {/* New Chat Button */}
          <div className="new-chat-section">
            <button className="new-chat-btn" onClick={onNewChat}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M14.5 4h-5L7 7H4a2 2 0 00-2 2v6a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2h-3l-2.5-3z" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <span>New chat</span>
            </button>
          </div>

          {/* Search */}
          <div className="search-section">
            <div className="search-container">
              <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
                <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
                <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <input
                type="text"
                placeholder="Search chats"
                className="search-input"
              />
            </div>
          </div>

          {/* Content */}
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
              selectedModuleId={selectedModuleId}
            />
          </div>

          {/* Footer */}
          <div className="sidebar-footer">
            <div
              className={`status-bar ${getStatusClass()}`}
              onClick={onRefreshBackend}
              title="Click to refresh"
            >
              <div className="status-indicator">
                <div className="status-dot"></div>
                <span>{getStatusText()}</span>
              </div>
              <svg className="refresh-icon" width="14" height="14" viewBox="0 0 24 24" fill="none">
                <polyline points="23,4 23,10 17,10" stroke="currentColor" strokeWidth="2"/>
                <path d="M20.49,9A9,9,0,0,0,5.64,5.64L1,10m22,4L18.36,18.36A9,9,0,0,1,3.51,15" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Sidebar;