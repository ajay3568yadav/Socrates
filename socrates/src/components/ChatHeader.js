import React from 'react';
import '../css/ChatHeader.css';

const ChatHeader = ({
  currentChat,
  selectedModule,
  user,
  onToggleSidebar,
  isSidebarVisible,
  isSidebarCollapsed,
  backendStatus,
  splitPaneMode,
  onExitSplitMode,
  onLogout,
}) => {
  // Get the appropriate icon for the sidebar toggle button
  const getSidebarToggleIcon = () => {
    if (!isSidebarVisible) {
      // Sidebar is hidden - show hamburger menu
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      );
    } else if (isSidebarCollapsed) {
      // Sidebar is collapsed - show expand icon (double chevron right)
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M13 17l5-5-5-5M6 17l5-5-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    } else {
      // Sidebar is expanded - show collapse icon (double chevron left)
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M11 17l-5-5 5-5M18 17l-5-5 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    }
  };

  const getSidebarToggleTooltip = () => {
    if (!isSidebarVisible) {
      return "Show sidebar";
    } else if (isSidebarCollapsed) {
      return "Expand sidebar";
    } else {
      return "Collapse sidebar";
    }
  };

  const getUserInitials = () => {
    if (!user?.email) return "U";
    const email = user.email;
    if (user.user_metadata?.full_name) {
      return user.user_metadata.full_name
        .split(' ')
        .map(name => name.charAt(0))
        .join('')
        .toUpperCase()
        .slice(0, 2);
    }
    return email.charAt(0).toUpperCase();
  };

  const getBackendStatusIcon = () => {
    if (backendStatus?.connecting) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="rotating">
          <path d="M21 12a9 9 0 11-6.219-8.56" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      );
    }
    return <div className="status-dot"></div>;
  };

  const getBackendStatusText = () => {
    if (backendStatus?.connecting) return "Connecting...";
    if (backendStatus?.online) return "Connected";
    if (backendStatus?.limited) return "Limited";
    return "Disconnected";
  };

  const getBackendStatusClass = () => {
    if (backendStatus?.connecting) return "connecting";
    if (backendStatus?.online) return "online";
    if (backendStatus?.limited) return "limited";
    return "offline";
  };

  return (
    <div className="chat-header">
      <div className="header-left">
        {/* Sidebar toggle button */}
        <button
          className="sidebar-toggle-btn"
          onClick={onToggleSidebar}
          title={getSidebarToggleTooltip()}
          aria-label={getSidebarToggleTooltip()}
        >
          {getSidebarToggleIcon()}
        </button>

        {/* Chat/Module info */}
        <div className="header-info">
          {currentChat ? (
            <div className="chat-info">
              <h1 className="chat-title" title={currentChat.heading}>
                {currentChat.heading}
              </h1>
              {selectedModule && (
                <span className="module-badge" title={`Module: ${selectedModule.name}`}>
                  {selectedModule.name}
                </span>
              )}
            </div>
          ) : (
            <div className="default-info">
              <h1 className="app-title">Socratic AI</h1>
              {selectedModule && (
                <span className="module-badge" title={`Module: ${selectedModule.name}`}>
                  {selectedModule.name}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="header-right">
        {/* Split pane controls */}
        {splitPaneMode && (
          <button
            className="exit-split-btn"
            onClick={onExitSplitMode}
            title="Exit split view"
            aria-label="Exit split view"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        )}

        {/* Backend status indicator */}
        {backendStatus && (
          <div 
            className={`status-indicator ${getBackendStatusClass()}`}
            title={`Backend status: ${getBackendStatusText()}`}
          >
            {getBackendStatusIcon()}
            <span className="status-text">
              {getBackendStatusText()}
            </span>
          </div>
        )}

        {/* User menu */}
        {user && (
          <div className="user-menu">
            <button
              className="user-menu-btn"
              onClick={onLogout}
              title={`Logout ${user.email}`}
              aria-label={`User menu for ${user.email}`}
            >
              <div className="user-avatar">
                {getUserInitials()}
              </div>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatHeader;