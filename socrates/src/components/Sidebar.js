import React from 'react';
import FolderSection from './FolderSection';
import ChatsSection from './ChatsSection';
import '../css/Sidebar.css'; 

const Sidebar = ({ 
  isOpen, 
  onClose, 
  onNewChat, 
  onSelectChat,
  chats,
  loadingChats,
  currentChatId,
  backendStatus, 
  user, 
  onLogout, 
  onRefreshBackend ,
  onSelectModule,
  selectedModuleId
}) => {
  const sidebarClass = `sidebar ${isOpen ? 'mobile-visible' : 'mobile-hidden'}`;

  const getConnectionStatusClass = () => {
    if (backendStatus.connecting) return 'connecting';
    if (backendStatus.online) return 'connected';
    return 'error';
  };

  const getConnectionStatusText = () => {
    if (backendStatus.connecting) return ' Connecting to backend...';
    if (backendStatus.online) return 'Backend connected';
    if (backendStatus.limited) return 'Backend limited functionality';
    if (backendStatus.error === 'Connection timeout') return '⏱️ Backend connection timeout';
    return 'Backend disconnected';
  };

  const handleStatusClick = () => {
    if (!backendStatus.connecting && onRefreshBackend) {
      onRefreshBackend();
    }
  };

  return (
    <div className={sidebarClass}>
      <div className="sidebar-header">
        <div className="app-logo">🚀</div>
        <div className="app-title">CUDA Tutor</div>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      {/* User Info Section */}
      {user && (
        <div className="user-info-section">
          <div className="user-info">
            <div className="user-avatar">
              {user.email?.charAt(0).toUpperCase() || '👤'}
            </div>
            <div className="user-details">
              <div className="user-name">
                {user.user_metadata?.full_name || user.email?.split('@')[0] || 'User'}
              </div>
              <div className="user-email">{user.email}</div>
            </div>
          </div>
          <button className="logout-btn" onClick={onLogout} title="Logout">
            🚪
          </button>
        </div>
      )}

      <div 
        className={`connection-status ${getConnectionStatusClass()} ${!backendStatus.connecting ? 'clickable' : ''}`}
        onClick={handleStatusClick}
        title={!backendStatus.connecting ? 'Click to refresh backend status' : ''}
      >
        {getConnectionStatusText()}
        {!backendStatus.connecting && (
          <span className="refresh-hint">↻</span>
        )}
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

      <button className="new-chat-btn" onClick={onNewChat}>
        <span>+</span>
        <span>New chat</span>
      </button>
    </div>
  );
};

export default Sidebar;