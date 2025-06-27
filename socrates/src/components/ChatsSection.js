import React from "react";
import "../css/ChatsSection.css";

const ChatsSection = ({
  chats,
  loadingChats,
  currentChatId,
  onSelectChat,
  selectedModuleId,
}) => {
  // Format timestamp for display
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "";

    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffTime = Math.abs(now - date);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays === 1) {
        return "Today";
      } else if (diffDays === 2) {
        return "Yesterday";
      } else if (diffDays <= 7) {
        return `${diffDays - 1} days ago`;
      } else {
        return date.toLocaleDateString();
      }
    } catch (error) {
      console.error("Error formatting timestamp:", error);
      return "Recently";
    }
  };

  // Get chat preview text
  const getChatPreview = (chat) => {
    if (
      chat.description &&
      chat.description !== "Start a new conversation about CUDA programming" &&
      chat.description !== "Chat conversation"
    ) {
      return chat.description;
    }
    return "Click to start chatting...";
  };

  // Get chat avatar based on status or content
  const getChatAvatar = (chat) => {
    if (chat.status === "active") return "💬";
    if (chat.status === "completed") return "✅";
    if (chat.status === "archived") return "📁";
    return "📝";
  };

  // Handle chat menu click
  const handleChatMenu = (e, chatId) => {
    e.stopPropagation();
    console.log("Chat menu clicked for:", chatId);
    // TODO: Implement chat menu functionality (delete, rename, etc.)
  };

  return (
    <div className="chats-section">
      <div className="section-header">
        <span>
          {selectedModuleId ? "Module Chats" : "Recent Chats"} ({chats.length})
        </span>
        <div className="section-actions">
          <button
            className="section-action"
            title="Refresh chats"
            onClick={() => window.location.reload()}
          >
            ↻
          </button>
          <button className="section-action" title="Chat options">
            ⋯
          </button>
        </div>
      </div>

      <div>
        {loadingChats ? (
          <div className="loading-chats">
            <div className="chat-item">
              <div className="chat-avatar">⏳</div>
              <div className="chat-content">
                <div className="chat-title">Loading chats...</div>
                <div className="chat-preview">
                  {selectedModuleId
                    ? "Loading module chats..."
                    : "Please wait while we load your conversations"}
                </div>
              </div>
            </div>
          </div>
        ) : chats.length === 0 ? (
          <div className="empty-chats">
            <div className="chat-item">
              <div className="chat-avatar">💬</div>
              <div className="chat-content">
                <div className="chat-title">
                  {selectedModuleId
                    ? "No chats in this module"
                    : "No chats yet"}
                </div>
                <div className="chat-preview">
                  {selectedModuleId
                    ? "Start your first conversation in this module!"
                    : "Start your first conversation!"}
                </div>
              </div>
            </div>
          </div>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.chat_id}
              className={`chat-item ${
                currentChatId === chat.chat_id ? "active" : ""
              }`}
              onClick={() => onSelectChat(chat.chat_id)}
              style={{ cursor: "pointer" }}
            >
              <div className="chat-avatar">{getChatAvatar(chat)}</div>
              <div className="chat-content">
                <div className="chat-title" title={chat.heading}>
                  {chat.heading}
                </div>
                <div className="chat-preview" title={getChatPreview(chat)}>
                  {getChatPreview(chat)}
                </div>
                <div className="chat-timestamp">
                  {formatTimestamp(chat.timestamp)}
                </div>
              </div>
              <button
                className="item-menu"
                onClick={(e) => handleChatMenu(e, chat.chat_id)}
                title="Chat options"
              >
                ⋯
              </button>
            </div>
          ))
        )}
      </div>

      {/* Summary section */}
      {chats.length > 0 && (
        <div className="modules-summary">
          <div className="summary-text">
            {chats.filter((chat) => chat.status === "active").length} active •{" "}
            {chats.length} total
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatsSection;
