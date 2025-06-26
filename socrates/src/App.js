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
  const [selectedModuleId, setSelectedModuleId] = useState(null);

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
        setSelectedModuleId(null);
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
      
      // Load user's chats (without module filter initially)
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

  // ==================== MESSAGE PERSISTENCE FUNCTIONS ====================

  // Save a message to the Messages table
  const saveMessage = async (chatId, sender, content, orderIndex) => {
    if (!chatId) {
      console.error('Cannot save message: chatId is null');
      return null;
    }

    try {
      const messageData = {
        chat_id: chatId,
        sender: sender, // 'user' or 'assistant'
        content: content,
        order_index: orderIndex,
        timestamp: new Date().toISOString()
      };

      console.log('Saving message:', messageData);

      const { data, error } = await supabase
        .from('Messages')
        .insert([messageData])
        .select()
        .single();

      if (error) {
        console.error('Error saving message:', error);
        return null;
      }

      console.log('Message saved successfully:', data);
      return data;
    } catch (error) {
      console.error('Error saving message:', error);
      return null;
    }
  };

  // Load messages for a specific chat
  const loadChatMessages = async (chatId) => {
    if (!chatId) {
      console.error('Cannot load messages: chatId is null');
      return [];
    }

    try {
      console.log('Loading messages for chat:', chatId);

      const { data, error } = await supabase
        .from('Messages')
        .select('*')
        .eq('chat_id', chatId)
        .order('order_index', { ascending: true });

      if (error) {
        console.error('Error loading messages:', error);
        return [];
      }

      // Convert database messages to app message format
      const formattedMessages = data.map(msg => ({
        id: `${msg.message_id}_${msg.sender}`,
        role: msg.sender,
        content: msg.content,
        timestamp: msg.timestamp,
        messageId: msg.message_id,
        orderIndex: msg.order_index
      }));

      console.log('Loaded messages for chat:', chatId, formattedMessages);
      return formattedMessages;
    } catch (error) {
      console.error('Error loading messages:', error);
      return [];
    }
  };

  // Create a new chat when user sends first message
  const createNewChatForMessage = async (userMessage) => {
    if (!user) {
      console.error('Cannot create chat: user is null');
      return null;
    }

    try {
      // Generate a title from the first message (first 50 characters)
      const heading = userMessage.length > 50 
        ? userMessage.substring(0, 50) + '...' 
        : userMessage;

      const newChat = {
        user_id: user.id,
        heading: heading,
        description: 'Chat conversation',
        timestamp: new Date().toISOString(),
        course_id: '1e44eb02-8daa-44a0-a7ee-28f88ce6863f', // Default CUDA Basics course
        module_id: selectedModuleId || 'c801ac6c-1232-4c96-89b1-c4eadf41026c', // Use selected module or default
        status: 'active'
      };

      console.log('Creating new chat for message:', newChat);

      const { data, error } = await supabase
        .from('Chats')
        .insert([newChat])
        .select()
        .single();

      if (error) {
        console.error('Error creating new chat:', error);
        return null;
      }

      // Add the new chat to the list and make it active
      setChats(prevChats => [data, ...prevChats]);
      setCurrentChatId(data.chat_id);
      
      console.log('New chat created for message:', data);
      return data;
    } catch (error) {
      console.error('Error creating new chat:', error);
      return null;
    }
  };

  // ==================== CHAT MANAGEMENT FUNCTIONS ====================

  // Load user's chats from Supabase (filtered by module if selected)
  const loadChats = async (moduleId = null) => {
    if (!user) return;
    
    try {
      setLoadingChats(true);
      const filterModuleId = moduleId || selectedModuleId;
      
      console.log('Loading chats for user:', user.id, 'module:', filterModuleId);

      let query = supabase
        .from('Chats')
        .select('*')
        .eq('user_id', user.id);

      // If a module is selected, filter chats by module_id
      if (filterModuleId) {
        query = query.eq('module_id', filterModuleId);
        console.log('Filtering chats by module_id:', filterModuleId);
      }

      const { data, error } = await query.order('timestamp', { ascending: false });

      if (error) {
        console.error('Error loading chats:', error);
        
        // Handle specific error for missing table
        if (error.code === '42P01') {
          console.warn('Chats table does not exist. Please create it in Supabase first.');
          setChats([]);
          return;
        }
      } else {
        setChats(data || []);
        console.log('Loaded chats for module:', filterModuleId, 'count:', data?.length || 0);
      }
    } catch (error) {
      console.error('Error loading chats:', error);
      setChats([]);
    } finally {
      setLoadingChats(false);
    }
  };

  // Create a new empty chat
  const createNewChat = async () => {
    if (!user) return;

    try {
      const newChat = {
        user_id: user.id,
        heading: 'New Chat',
        description: 'Start a new conversation about CUDA programming',
        timestamp: new Date().toISOString(),
        course_id: '1e44eb02-8daa-44a0-a7ee-28f88ce6863f',
        module_id: selectedModuleId || 'c801ac6c-1232-4c96-89b1-c4eadf41026c',
        status: 'active'
      };

      console.log('Creating new empty chat:', newChat);

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
      
      // Switch to the new chat with empty messages
      setCurrentChatId(data.chat_id);
      setMessages([]);
      setCurrentView('chat');
      
      console.log('New empty chat created:', data);
    } catch (error) {
      console.error('Error creating chat:', error);
    }
  };

  // Select a chat from the sidebar and load its messages
  const selectChat = async (chatId) => {
    console.log('Selecting chat:', chatId);
    setCurrentChatId(chatId);
    setCurrentView('chat');
    setIsLoading(true);

    try {
      // Load messages for this chat
      const chatMessages = await loadChatMessages(chatId);
      setMessages(chatMessages);
      console.log('Chat selected and messages loaded:', chatId);
    } catch (error) {
      console.error('Error selecting chat:', error);
      setMessages([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Update chat heading and timestamp
  const updateChatTitle = async (chatId, firstMessage) => {
    if (!chatId || !firstMessage) return;

    try {
      // Generate a heading from the first message (first 50 characters)
      const heading = firstMessage.length > 50 
        ? firstMessage.substring(0, 50) + '...' 
        : firstMessage;

      console.log('Updating chat title:', chatId, heading);

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

  // Handle module selection and load associated chats
  const handleSelectModule = async (moduleId) => {
    console.log('Module selected:', moduleId);
    setSelectedModuleId(moduleId);
    
    // Clear current chat if it doesn't belong to the selected module
    if (currentChatId) {
      const currentChat = chats.find(chat => chat.chat_id === currentChatId);
      if (currentChat && currentChat.module_id !== moduleId) {
        setCurrentChatId(null);
        setMessages([]);
        setCurrentView('welcome');
      }
    }
    
    // Load chats for the selected module
    await loadChats(moduleId);
  };

  // Start a new chat
  const startNewChat = () => {
    createNewChat();
  };

  // Re-load chats when selectedModuleId changes
  useEffect(() => {
    if (user && selectedModuleId) {
      console.log('Selected module changed, reloading chats for:', selectedModuleId);
      loadChats(selectedModuleId);
    }
  }, [selectedModuleId, user]); // loadChats is stable, no need to include it

  // ==================== MESSAGE HANDLING FUNCTIONS ====================

  const sendMessage = async (message) => {
    console.log('Sending message:', message);
    let chatId = currentChatId;
    let orderIndex = messages.length;

    // If no current chat, create a new one
    if (!chatId) {
      console.log('No current chat, creating new chat...');
      const newChat = await createNewChatForMessage(message);
      if (!newChat) {
        console.error('Failed to create new chat');
        return;
      }
      chatId = newChat.chat_id;
      orderIndex = 0; // First message in new chat
    }

    // Create user message object
    const userMessage = {
      id: Date.now() + '_user',
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };

    // Add user message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setCurrentView('chat');

    // Save user message to database
    const savedUserMessage = await saveMessage(chatId, 'user', message, orderIndex);
    if (savedUserMessage) {
      console.log('User message saved to database');
    }

    // Update chat title if this is the first message
    if (orderIndex === 0) {
      await updateChatTitle(chatId, message);
    }

    try {
      // Send message to backend API
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
          chat_id: chatId,
          module_id: selectedModuleId,
          stream: false
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const assistantMessage = {
        id: Date.now() + '_assistant',
        role: 'assistant',
        content: data.response || 'Sorry, I received an empty response.',
        timestamp: new Date().toISOString(),
        isStreaming: false
      };

      // Add assistant message to UI
      setMessages(prev => [...prev, assistantMessage]);

      // Save assistant message to database
      const savedAssistantMessage = await saveMessage(chatId, 'assistant', assistantMessage.content, orderIndex + 1);
      if (savedAssistantMessage) {
        console.log('Assistant message saved to database');
      }

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
      
      // Save error message to database
      await saveMessage(chatId, 'assistant', errorMessage.content, orderIndex + 1);
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
  console.log('Current chat ID:', currentChatId);
  console.log('Selected module ID:', selectedModuleId);

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
        onSelectModule={handleSelectModule}
        selectedModuleId={selectedModuleId}
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