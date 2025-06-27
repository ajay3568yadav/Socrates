#!/usr/bin/env python3
"""
Session management for conversation tracking
"""

import time
from typing import Dict, List, Any, Optional
from config import get_config

config = get_config()

class SessionManager:
    """Manages conversation sessions with context tracking"""
    
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.max_sessions = config.MAX_SESSIONS
        self.max_exchanges_per_session = config.MAX_EXCHANGES_PER_SESSION
        self.context_exchanges = config.CONVERSATION_CONTEXT_EXCHANGES
        
    def get_conversation_context(self, session_id: str, max_exchanges: Optional[int] = None) -> str:
        """Build conversation context from session history"""
        if max_exchanges is None:
            max_exchanges = self.context_exchanges
            
        if session_id not in self.sessions or not self.sessions[session_id]:
            return ""
        
        # Get the most recent exchanges
        session_history = self.sessions[session_id]
        recent_exchanges = session_history[-max_exchanges:]
        context_parts = []
        
        for exchange in recent_exchanges:
            # Truncate long messages for context
            user_msg = exchange['user'][:150] + "..." if len(exchange['user']) > 150 else exchange['user']
            assistant_msg = exchange['assistant'][:200] + "..." if len(exchange['assistant']) > 200 else exchange['assistant']
            
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {assistant_msg}")
        
        return "\n".join(context_parts)
    
    def detect_follow_up_question(self, current_query: str, last_response: str = "") -> bool:
        """Detect if current query is a follow-up to previous conversation"""
        follow_up_indicators = [
            # Direct references to previous conversation
            'last question', 'previous question', 'what did i ask', 'what was my question',
            'you just said', 'you mentioned', 'earlier you said', 'before you said',
            
            # Continuation indicators
            'show me', 'give me', 'can you', 'what about', 'also', 'optimize', 
            'improve', 'explain that', 'those examples', 'the code', 'that kernel',
            'tell me more', 'more details', 'expand on', 'elaborate',
            
            # Pronouns that indicate reference to previous content
            'this', 'that', 'it', 'them', 'these', 'those'
        ]
        
        query_lower = current_query.lower()
        return any(indicator in query_lower for indicator in follow_up_indicators)
    
    def add_exchange(self, session_id: str, user_message: str, assistant_response: str, is_follow_up: bool = False) -> None:
        """Add a new exchange to the session"""
        # Clean up old sessions if we have too many
        if len(self.sessions) > self.max_sessions:
            self._cleanup_old_sessions()
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Add the new exchange
        exchange = {
            'user': user_message,
            'assistant': assistant_response,
            'timestamp': time.time(),
            'is_follow_up': is_follow_up
        }
        
        self.sessions[session_id].append(exchange)
        
        # Keep only the last N exchanges per session
        if len(self.sessions[session_id]) > self.max_exchanges_per_session:
            self.sessions[session_id] = self.sessions[session_id][-self.max_exchanges_per_session:]
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session"""
        if session_id not in self.sessions:
            return {'exists': False}
        
        session_data = self.sessions[session_id]
        return {
            'exists': True,
            'session_id': session_id,
            'message_count': len(session_data),
            'last_activity': session_data[-1]['timestamp'] if session_data else None,
            'first_activity': session_data[0]['timestamp'] if session_data else None,
            'duration_minutes': (session_data[-1]['timestamp'] - session_data[0]['timestamp']) / 60 if len(session_data) > 1 else 0,
            'follow_up_count': sum(1 for ex in session_data if ex.get('is_follow_up', False)),
            'status': 'active'
        }
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions with summary info"""
        current_time = time.time()
        sessions_info = {}
        
        for session_id, session_data in self.sessions.items():
            if session_data:
                last_activity = session_data[-1]['timestamp']
                age_minutes = (current_time - last_activity) / 60
                
                sessions_info[session_id] = {
                    'message_count': len(session_data),
                    'last_activity': last_activity,
                    'age_minutes': age_minutes,
                    'follow_up_count': sum(1 for ex in session_data if ex.get('is_follow_up', False))
                }
        
        return {
            'total_sessions': len(sessions_info),
            'sessions': sessions_info
        }
    
    def _cleanup_old_sessions(self) -> int:
        """Clean up old conversation sessions"""
        current_time = time.time()
        cutoff_time = current_time - (config.SESSION_CLEANUP_HOURS * 3600)
        
        sessions_to_remove = []
        for session_id, session_data in self.sessions.items():
            if session_data and len(session_data) > 0:
                last_timestamp = session_data[-1].get('timestamp', 0)
                if last_timestamp < cutoff_time:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            print(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")
        
        return len(sessions_to_remove)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session statistics"""
        if not self.sessions:
            return {
                'total_sessions': 0,
                'total_exchanges': 0,
                'average_exchanges_per_session': 0,
                'active_sessions_last_hour': 0
            }
        
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        total_exchanges = sum(len(session) for session in self.sessions.values())
        active_last_hour = sum(1 for session in self.sessions.values() 
                              if session and session[-1]['timestamp'] > one_hour_ago)
        
        return {
            'total_sessions': len(self.sessions),
            'total_exchanges': total_exchanges,
            'average_exchanges_per_session': total_exchanges / len(self.sessions) if self.sessions else 0,
            'active_sessions_last_hour': active_last_hour,
            'max_sessions_limit': self.max_sessions,
            'max_exchanges_per_session': self.max_exchanges_per_session
        }