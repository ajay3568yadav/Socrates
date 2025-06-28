import React from "react";
import "../css/ChatsSection.css";

const ChatsSection = ({
  chats,
  loadingChats,
  currentChatId,
  onSelectChat,
  selectedModuleId,
}) => {
  // Group chats by time periods
  const groupChatsByTime = (chats) => {
    const now = new Date();
    const groups = {
      today: [],
      yesterday: [],
      previous7Days: [],
      previous30Days: [],
      older: []
    };

    chats.forEach(chat => {
      if (!chat.timestamp) {
        groups.older.push(chat);
        return;
      }

      const chatDate = new Date(chat.timestamp);
      const diffTime = Math.abs(now - chatDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays <= 1) {
        groups.today.push(chat);
      } else if (diffDays <= 2) {
        groups.yesterday.push(chat);
      } else if (diffDays <= 7) {
        groups.previous7Days.push(chat);
      } else if (diffDays <= 30) {
        groups.previous30Days.push(chat);
      } else {
        groups.older.push(chat);
      }
    });

    return groups;
  };

  // Truncate chat title to make it cleaner
  const truncateTitle = (title, maxLength = 30) => {
    if (!title) return "New Chat";
    if (title.length <= maxLength) return title;
    return title.substring(0, maxLength).trim() + "...";
  };

  // Handle chat menu click
  const handleChatMenu = (e, chatId) => {
    e.stopPropagation();
    console.log("Chat menu clicked for:", chatId);
    // TODO: Implement chat menu functionality (delete, rename, etc.)
  };

  // Render chat group
  const renderChatGroup = (title, chats, showDivider = true) => {
    if (chats.length === 0) return null;

    return (
      <div className="chat-group" key={title}>
        {showDivider && <div className="chat-group-divider">{title}</div>}
        <div className="chat-group-items">
          {chats.map((chat) => (
            <div
              key={chat.chat_id}
              className={`chat-item ${
                currentChatId === chat.chat_id ? "active" : ""
              }`}
              onClick={() => onSelectChat(chat.chat_id)}
            >
              <div className="chat-content">
                <div className="chat-title" title={chat.heading}>
                  {truncateTitle(chat.heading)}
                </div>
              </div>
              <button
                className="chat-menu"
                onClick={(e) => handleChatMenu(e, chat.chat_id)}
                title="More options"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="1" fill="currentColor"/>
                  <circle cx="19" cy="12" r="1" fill="currentColor"/>
                  <circle cx="5" cy="12" r="1" fill="currentColor"/>
                </svg>
              </button>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loadingChats) {
    return (
      <div className="chats-section">
        <div className="section-header">
          <div className="section-title">
            <svg className="section-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Chats</span>
          </div>
        </div>
        <div className="loading-state">
          <div className="loading-item">
            <div className="loading-spinner"></div>
            <span>Loading chats...</span>
          </div>
        </div>
      </div>
    );
  }

  if (chats.length === 0) {
    return (
      <div className="chats-section">
        <div className="section-header">
          <div className="section-title">
            <svg className="section-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Chats</span>
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-message">No chats yet</div>
          <div className="empty-description">Start your first conversation!</div>
        </div>
      </div>
    );
  }

  const groupedChats = groupChatsByTime(chats);

  return (
    <div className="chats-section">
      <div className="section-header">
        <div className="section-title">
          <svg className="section-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>Chats</span>
        </div>
        <div className="section-actions">
          <button
            className="section-action"
            title="Refresh chats"
            onClick={() => window.location.reload()}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <polyline points="23,4 23,10 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="1,20 1,14 7,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M20.49,9A9,9,0,0,0,5.64,5.64L1,10m22,4L18.36,18.36A9,9,0,0,1,3.51,15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button 
            className="section-action" 
            title="Chat options"
            onClick={(e) => handleChatMenu(e, 'all')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="1" fill="currentColor"/>
              <circle cx="19" cy="12" r="1" fill="currentColor"/>
              <circle cx="5" cy="12" r="1" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>

      <div className="chats-list">
        {renderChatGroup("Today", groupedChats.today, groupedChats.today.length > 0)}
        {renderChatGroup("Yesterday", groupedChats.yesterday, groupedChats.yesterday.length > 0)}
        {renderChatGroup("Previous 7 days", groupedChats.previous7Days, groupedChats.previous7Days.length > 0)}
        {renderChatGroup("Previous 30 days", groupedChats.previous30Days, groupedChats.previous30Days.length > 0)}
        {renderChatGroup("Older", groupedChats.older, groupedChats.older.length > 0)}
      </div>
    </div>
  );
};

export default ChatsSection;