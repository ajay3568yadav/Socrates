import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import AuthPage from './components/AuthPage';
import Sidebar from './components/Sidebar';
import ChatHeader from './components/ChatHeader';
import WelcomeView from './components/WelcomeView';
import ChatView from './components/ChatView';
import supabase from './config/supabaseClient';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';

// Main App Component with Authentication and Chat Management
const CudaTutorApp = () => {
  // UI State
  const [currentView, setCurrentView] = useState('welcome');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Chat Management State
  const [currentChatId, setCurrentChatId] = useState(null);
  const [chats, setChats] = useState([]);
  const [loadingChats, setLoadingChats] = useState(false);
  
  // Authentication State
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Backend Status State
  const [backendStatus, setBackendStatus] = useState({ 
    online: false, 
    limited: false, 
    connecting: false 
  });
  
  // Session Management
  const [sessionId] = useState(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  
  // Refs for preventing duplicate requests
  const statusCheckInProgress = useRef(false);
  const statusIntervalRef = useRef(null);

  // ==================== AUTHENTICATION FUNCTIONS ====================
  
  useEffect(() => {
    // Check initial auth state
    const checkAuth = async () => {
      try {
        const { data: { session }, error } = await supabase.auth.getSession();
        if (error) {
          console.error('Error getting session:', error);
        } else {
          setUser(session?.user || null);
        }
      } catch (error) {
        console.error('Auth check error:', error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();

    // Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      console.log('Auth state changed:', event, session);
      setUser(session?.user || null);
      setLoading(false);
    });

    // Add global copy function for code blocks
    window.copyCodeToClipboard = async (encodedCode, buttonElement) => {
      try {
        const code = decodeURIComponent(encodedCode);
        await navigator.clipboard.writeText(code);
        
        // Show temporary success feedback
        if (buttonElement) {
          const originalText = buttonElement.textContent;
          buttonElement.textContent = '✓';
          buttonElement.style.color = '#22c55e';
          
          setTimeout(() => {
            buttonElement.textContent = originalText;
            buttonElement.style.color = '';
          }, 2000);
        }
      } catch (err) {
        console.error('Failed to copy code:', err);
      }
    };

    return () => {
      subscription.unsubscribe();
      delete window.copyCodeToClipboard;
      // Clear any existing interval
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }
    };
  }, []); // Empty dependency array - run only once

  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        console.error('Logout error:', error);
      } else {
        // Clear app state on logout
        setMessages([]);
        setCurrentView('welcome');
        setCurrentChatId(null);
        setChats([]);
        setSidebarOpen(false);
      }
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // ==================== BACKEND STATUS FUNCTIONS ====================

  const checkBackendStatus = async () => {
    // Prevent multiple simultaneous requests
    if (statusCheckInProgress.current) {
      console.log('Status check already in progress, skipping...');
      return;
    }

    try {
      statusCheckInProgress.current = true;
      setBackendStatus(prev => ({ ...prev, connecting: true }));
      
      console.log('Checking backend status...');
      
      // Add a timeout wrapper to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch(`${API_BASE_URL}/api/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Backend status response:', data);
        setBackendStatus({
          online: data.rag_loaded && data.ollama_status === 'connected',
          limited: !data.rag_loaded || data.ollama_status !== 'connected',
          connecting: false,
          details: data
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Backend status check failed:', error);
      
      let errorMessage = error.message;
      if (error.name === 'AbortError') {
        errorMessage = 'Connection timeout';
      }
      
      setBackendStatus({ 
        online: false, 
        limited: false, 
        connecting: false,
        error: errorMessage 
      });
    } finally {
      statusCheckInProgress.current = false;
    }
  };

  // Backend status checking when user is authenticated
  useEffect(() => {
    // Clear any existing interval first
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
      statusIntervalRef.current = null;
    }

    if (user) {
      console.log('User authenticated, starting backend status checks');
      
      // Load user's chats
      loadChats();
      
      // Initial check
      checkBackendStatus();
      
      // Set up periodic checks
      statusIntervalRef.current = setInterval(() => {
        console.log('Periodic backend status check');
        checkBackendStatus();
      }, 120000); // Check every 2 minutes
    }

    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }
    };
  }, [user]); // Only re-run when user changes

  // ==================== CHAT MANAGEMENT FUNCTIONS ====================

  // Load user's chats from Supabase
  const loadChats = async () => {
    if (!user) return;
    
    try {
      setLoadingChats(true);
      // Use the correct table name 'Chats' with capital C
      const { data, error } = await supabase
        .from('Chats')
        .select('*')
        .eq('user_id', user.id)
        .order('timestamp', { ascending: false });

      if (error) {
        console.error('Error loading chats:', error);
        
        // Handle specific error for missing table
        if (error.code === '42P01') {
          console.warn('Chats table does not exist. Please create it in Supabase first.');
          setChats([]); // Set empty array instead of causing app crash
          return;
        }
      } else {
        setChats(data || []);
        console.log('Loaded chats:', data);
      }
    } catch (error) {
      console.error('Error loading chats:', error);
      setChats([]); // Fallback to empty array
    } finally {
      setLoadingChats(false);
    }
  };

  // Create a new chat
  const createNewChat = async () => {
    if (!user) return;

    try {
      const newChat = {
        user_id: user.id,
        heading: 'New Chat',
        description: 'Start a new conversation about CUDA programming',
        timestamp: new Date().toISOString(),
        course_id: '1e44eb02-8daa-44a0-a7ee-28f88ce6863f', // Default CUDA Basics course
        module_id: 'c801ac6c-1232-4c96-89b1-c4eadf41026c', // Default CUDA Basics module
        status: 'active'
      };

      const { data, error } = await supabase
        .from('Chats')
        .insert([newChat])
        .select()
        .single();

      if (error) {
        console.error('Error creating chat:', error);
        return;
      }

      // Add the new chat to the list
      setChats(prevChats => [data, ...prevChats]);
      
      // Switch to the new chat
      setCurrentChatId(data.chat_id);
      setMessages([]);
      setCurrentView('chat');
      
      console.log('New chat created:', data);
    } catch (error) {
      console.error('Error creating chat:', error);
    }
  };

  // Select a chat from the sidebar
  const selectChat = async (chatId) => {
    setCurrentChatId(chatId);
    setCurrentView('chat');
    // For now, we'll start with empty messages
    // Later you can load messages from the Messages table
    setMessages([]);
    console.log('Selected chat:', chatId);
  };

  // Update chat heading based on first message
  const updateChatTitle = async (chatId, firstMessage) => {
    if (!chatId || !firstMessage) return;

    try {
      // Generate a heading from the first message (first 50 characters)
      const heading = firstMessage.length > 50 
        ? firstMessage.substring(0, 50) + '...' 
        : firstMessage;

      const { error } = await supabase
        .from('Chats')
        .update({ 
          heading: heading,
          timestamp: new Date().toISOString()
        })
        .eq('chat_id', chatId);

      if (error) {
        console.error('Error updating chat heading:', error);
      } else {
        // Update local state
        setChats(prevChats => 
          prevChats.map(chat => 
            chat.chat_id === chatId 
              ? { ...chat, heading: heading, timestamp: new Date().toISOString() }
              : chat
          )
        );
        console.log('Chat heading updated:', heading);
      }
    } catch (error) {
      console.error('Error updating chat heading:', error);
    }
  };

  // Start a new chat
  const startNewChat = () => {
    createNewChat();
  };

  // ==================== MESSAGE HANDLING FUNCTIONS ====================

  const sendMessage = async (message) => {
    // Update chat title if this is the first message
    if (currentChatId && messages.length === 0) {
      updateChatTitle(currentChatId, message);
    }

    const userMessage = {
      id: Date.now() + '_user',
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setCurrentView('chat');

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
          chat_id: currentChatId,
          stream: false  // Explicitly set to false for now
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle non-streaming response
      const data = await response.json();
      
      const assistantMessage = {
        id: Date.now() + '_assistant',
        role: 'assistant',
        content: data.response || 'Sorry, I received an empty response.',
        timestamp: new Date().toISOString(),
        isStreaming: false
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = {
        id: Date.now() + '_error',
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}. Please check if the backend is running and try again.`,
        timestamp: new Date().toISOString(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // ==================== RENDER FUNCTIONS ====================

  // Show loading spinner while checking auth
  if (loading) {
    return (
      <div className="app-container">
        <div className="welcome-container">
          <div className="welcome-icon">🚀</div>
          <h1 className="welcome-title">Loading...</h1>
          <p className="welcome-subtitle">Checking authentication status...</p>
        </div>
      </div>
    );
  }

  // Show AuthPage if user is not authenticated
  if (!user) {
    return <AuthPage />;
  }

  console.log('Current user:', user);

  // Main authenticated app
  return (
    <div className="app-container">
      {/* Mobile Overlay */}
      <div 
        className={`mobile-overlay ${sidebarOpen ? 'show' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={startNewChat}
        onSelectChat={selectChat}
        chats={chats}
        loadingChats={loadingChats}
        currentChatId={currentChatId}
        backendStatus={backendStatus}
        user={user}
        onLogout={handleLogout}
        onRefreshBackend={checkBackendStatus}
      />

      {/* Main Content */}
      <div className="main-content">
        <ChatHeader 
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          backendStatus={backendStatus}
          user={user}
        />
        
        {currentView === 'welcome' ? (
          <WelcomeView onSendMessage={sendMessage} user={user} />
        ) : (
          <ChatView 
            messages={messages}
            isLoading={isLoading}
            onSendMessage={sendMessage}
          />
        )}
      </div>
    </div>
  );
};

// Export the main app
export default CudaTutorApp;
