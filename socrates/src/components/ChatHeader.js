import React, { useState } from 'react';
import ModelSelector from './ModelSelector';
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
  selectedModel,
  onModelChange,
  tutoringMode = false // NEW: Tutoring mode prop
}) => {
  const [showUserMenu, setShowUserMenu] = useState(false);

  const getModuleName = () => {
    const moduleNames = {
      "c801ac6c-1232-4c96-89b1-c4eadf41026c": "CUDA Basics",
      "d26ccd91-cdf9-45e3-990f-a484d764bb9d": "Memory Optimization",
      "ff7d63fc-8646-4d9a-be5d-41a249beff02": "Kernel Development",
      "22107ce-5027-42bf-9941-6d00117da9ae": "Performance Tuning",
    };
    return selectedModule ? moduleNames[selectedModule.id] || selectedModule.name : 'CUDA';
  };

  const getChatTitle = () => {
    if (tutoringMode) {
      return `🎓 Tutoring: ${getModuleName()}`;
    }
    
    if (currentChat) {
      return currentChat.heading;
    }
    
    return getModuleName();
  };

  const getChatSubtitle = () => {
    if (tutoringMode) {
      return "Interactive AI tutoring session";
    }
    
    if (splitPaneMode) {
      return "Split view with code editor";
    }
    
    return "AI-powered CUDA programming assistant";
  };

  const getStatusColor = () => {
    if (backendStatus?.connecting) return '#fbbf24';
    if (backendStatus?.online) return '#22c55e';
    return '#ef4444';
  };

  const getStatusText = () => {
    if (backendStatus?.connecting) return 'Connecting...';
    if (backendStatus?.online) return 'Online';
    return 'Offline';
  };

  const getUserInitials = () => {
    if (!user?.email) return 'U';
    return user.email.charAt(0).toUpperCase();
  };

  return (
    <header className={`chat-header ${splitPaneMode ? 'split-mode' : ''} ${tutoringMode ? 'tutoring-mode' : ''}`}>
      <div className="header-left">
        <button 
          className="sidebar-toggle" 
          onClick={onToggleSidebar}
          title={isSidebarVisible ? (isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar') : 'Open sidebar'}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <line x1="3" y1="6" x2="21" y2="6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="3" y1="12" x2="21" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="3" y1="18" x2="21" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>

        <div className="chat-info">
          <div className="chat-title-container">
            <h1 className="chat-title">{getChatTitle()}</h1>
            {/* NEW: Tutoring mode indicator */}
            {tutoringMode && (
              <div className="tutoring-indicator">
                <div className="tutoring-dot"></div>
                <span>Live</span>
              </div>
            )}
          </div>
          <div className="chat-subtitle">{getChatSubtitle()}</div>
        </div>
      </div>

      <div className="header-center">
        {splitPaneMode && (
          <button 
            className="exit-split-btn"
            onClick={onExitSplitMode}
            title="Exit split view"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Exit Split View</span>
          </button>
        )}
      </div>

      <div className="header-right">
        {/* Model Selector - disabled in tutoring mode for consistency */}
        <ModelSelector 
          selectedModel={selectedModel}
          onModelChange={onModelChange}
          disabled={tutoringMode} // NEW: Disable model switching in tutoring mode
        />

        {/* Status Indicator */}
        <div className="status-indicator" title={`Backend: ${getStatusText()}`}>
          <div 
            className="status-dot" 
            style={{ backgroundColor: getStatusColor() }}
          ></div>
          <span className="status-text">{getStatusText()}</span>
        </div>

        {/* User Menu */}
        <div className="user-menu-container">
          <button 
            className="user-avatar"
            onClick={() => setShowUserMenu(!showUserMenu)}
            title="User menu"
          >
            {getUserInitials()}
          </button>
          
          {showUserMenu && (
            <div className="user-menu">
              <div className="user-info">
                <div className="user-email">{user?.email}</div>
                <div className="user-role">Student</div>
              </div>
              <div className="menu-divider"></div>
              <button className="menu-item" onClick={() => setShowUserMenu(false)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="2"/>
                  <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
                </svg>
                Profile
              </button>
              <button className="menu-item" onClick={() => setShowUserMenu(false)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                  <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" stroke="currentColor" strokeWidth="2"/>
                </svg>
                Settings
              </button>
              <div className="menu-divider"></div>
              <button 
                className="menu-item logout-item" 
                onClick={() => {
                  setShowUserMenu(false);
                  onLogout();
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" stroke="currentColor" strokeWidth="2"/>
                  <polyline points="16,17 21,12 16,7" stroke="currentColor" strokeWidth="2"/>
                  <line x1="21" y1="12" x2="9" y2="12" stroke="currentColor" strokeWidth="2"/>
                </svg>
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Click outside to close user menu */}
      {showUserMenu && (
        <div 
          className="user-menu-overlay" 
          onClick={() => setShowUserMenu(false)}
        ></div>
      )}
    </header>
  );
};

export default ChatHeader;