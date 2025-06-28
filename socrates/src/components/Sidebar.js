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
  const getSidebarClass = () => {
    let classes = "sidebar";
    
    // Add collapsed class for desktop
    if (isCollapsed && window.innerWidth > 768) {
      classes += " collapsed";
    }
    
    // Add visibility classes
    if (window.innerWidth <= 768) {
      classes += isOpen ? " mobile-visible" : " mobile-hidden";
    } else {
      classes += isOpen ? " desktop-visible" : " desktop-hidden";
    }
    
    return classes;
  };

  const getConnectionStatusClass = () => {
    if (backendStatus.connecting) return "connecting";
    if (backendStatus.online) return "connected";
    return "error";
  };

  const getConnectionStatusText = () => {
    if (backendStatus.connecting) return "Connecting...";
    if (backendStatus.online) return "Backend connected";
    if (backendStatus.limited) return "Limited functionality";
    return "Backend disconnected";
  };

  const handleStatusClick = () => {
    if (!backendStatus.connecting && onRefreshBackend) {
      onRefreshBackend();
    }
  };

  const getUserInitials = () => {
    if (!user?.email) return "U";
    return user.email.charAt(0).toUpperCase();
  };

  // Render collapsed navigation for desktop
  const renderCollapsedNav = () => (
    <div className="collapsed-nav">
      <button
        className="collapsed-nav-item"
        onClick={onNavigateHome}
        data-tooltip="Home"
        title="Home"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <polyline points="9,22 9,12 15,12 15,22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      
      <button
        className={`collapsed-nav-item ${selectedModuleId ? 'active' : ''}`}
        onClick={() => onSelectModule && onSelectModule(selectedModuleId)}
        data-tooltip="Course Modules"
        title="Course Modules"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      
      <button
        className={`collapsed-nav-item ${currentChatId ? 'active' : ''}`}
        onClick={() => {
          if (chats && chats.length > 0) {
            onSelectChat(chats[0].chat_id);
          }
        }}
        data-tooltip={`Chats (${chats ? chats.length : 0})`}
        title={`Chats (${chats ? chats.length : 0})`}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      
      <button
        className="collapsed-nav-item"
        onClick={handleStatusClick}
        data-tooltip={getConnectionStatusText()}
        title={getConnectionStatusText()}
      >
        <div className={`status-dot-small ${getConnectionStatusClass()}`}></div>
      </button>
      
      {user && (
        <button
          className="collapsed-nav-item"
          onClick={onLogout}
          data-tooltip="Logout"
          title="Logout"
          style={{ marginTop: "auto", marginBottom: "8px" }}
        >
          {getUserInitials()}
        </button>
      )}
      
      <button
        className="collapsed-new-chat"
        onClick={onNewChat}
        data-tooltip="New Chat"
        title="New Chat"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      </button>
    </div>
  );

  return (
    <div className={getSidebarClass()}>
      {/* Header - always visible */}
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
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      {/* Collapsed navigation for desktop */}
      {isCollapsed && window.innerWidth > 768 ? (
        renderCollapsedNav()
      ) : (
        <>
          {/* Connection Status - Clean minimal design */}
          <div
            className={`connection-status ${getConnectionStatusClass()} ${
              !backendStatus.connecting ? "clickable" : ""
            }`}
            onClick={handleStatusClick}
            title={
              !backendStatus.connecting ? "Click to refresh backend status" : ""
            }
          >
            <div className={`status-indicator-icon ${getConnectionStatusClass()}`}>
              <div className="status-dot"></div>
            </div>
            <span className="status-text">{getConnectionStatusText()}</span>
            {!backendStatus.connecting && <span className="refresh-hint">↻</span>}
          </div>

          {/* Search Section */}
          <div className="search-section">
            <div className="search-container">
              <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
                <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <input
                type="text"
                placeholder="Search conversations..."
                className="search-input"
              />
            </div>
          </div>

          {/* Main Content */}
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
            <button className="new-chat-btn" onClick={onNewChat}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              <span>New chat</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default Sidebar;