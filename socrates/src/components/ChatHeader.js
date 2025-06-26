import React from 'react';

const ChatHeader = ({ onToggleSidebar, backendStatus, user }) => {
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
      
      {user && (
        <div className="header-user-info">
          <span className="user-greeting">
            Welcome, {user.user_metadata?.full_name || user.email?.split('@')[0] || 'User'}!
          </span>
        </div>
      )}
      
      <div className="chat-actions">
        <button className="chat-action" title="Share">📤</button>
        <button className="chat-action" title="Save">💾</button>
      </div>
    </div>
  );
};

export default ChatHeader;