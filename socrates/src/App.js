import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import AuthPage from "./components/AuthPage";
import Sidebar from "./components/Sidebar";
import ChatHeader from "./components/ChatHeader";
import WelcomeView from "./components/WelcomeView";
import ChatView from "./components/ChatView";
import SplitPaneLayout from "./components/SplitPaneLayout";
import ImprovedCodeEditor from "./components/ImprovedCodeEditor";
import supabase from "./config/supabaseClient";

// API Configuration
const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:5001";

// Main App Component with Authentication and Chat Management
const CudaTutorApp = () => {
  // UI State
  const [currentView, setCurrentView] = useState("welcome");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [selectedModuleId, setSelectedModuleId] = useState(null);

  // Split Pane State
  const [splitPaneMode, setSplitPaneMode] = useState(false);
  const [splitPaneWidth, setSplitPaneWidth] = useState(60);
  const [codeEditorContent, setCodeEditorContent] = useState("");
  const [codeEditorLanguage, setCodeEditorLanguage] = useState("c");

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
    connecting: false,
  });

  // Session Management
  const [sessionId] = useState(
    `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  );

  // Refs for preventing duplicate requests
  const statusCheckInProgress = useRef(false);
  const statusIntervalRef = useRef(null);

  // ==================== SPLIT PANE FUNCTIONS ====================

  const handleEnterSplitMode = (code, language = "c") => {
    setCodeEditorContent(code);
    setCodeEditorLanguage(language);
    setSplitPaneMode(true);
  };

  const handleExitSplitMode = () => {
    setSplitPaneMode(false);
    setCodeEditorContent("");
  };

  const handleSplitPaneWidthChange = (width) => {
    setSplitPaneWidth(width);
  };

  const handleCodeReviewFromSplit = (reviewPrompt) => {
    sendMessage(reviewPrompt);
  };

  const handleOpenCodeEditor = (code, language = "c") => {
    handleEnterSplitMode(code, language);
  };

  // ==================== MOBILE DETECTION ====================

  useEffect(() => {
    const checkMobile = () => {
      const isMobileDevice = window.innerWidth <= 768;
      setIsMobile(isMobileDevice);

      if (isMobileDevice && sidebarOpen) {
        setSidebarOpen(false);
      }
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, [sidebarOpen]);

  // ==================== SIDEBAR TOGGLE FUNCTIONS ====================

  const toggleSidebar = () => {
    setSidebarOpen((prev) => !prev);
  };

  const handleOverlayClick = () => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  const handleMobileItemSelect = () => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  // ==================== AUTHENTICATION FUNCTIONS ====================

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();
        if (error) {
          console.error("Error getting session:", error);
        } else {
          setUser(session?.user || null);
        }
      } catch (error) {
        console.error("Auth check error:", error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      console.log("Auth state changed:", event, session);
      setUser(session?.user || null);
      setLoading(false);
    });

    // Global function for copying code
    window.copyCodeToClipboard = async (encodedCode, buttonElement) => {
      try {
        const code = decodeURIComponent(encodedCode);
        await navigator.clipboard.writeText(code);

        if (buttonElement) {
          const originalText = buttonElement.textContent;
          buttonElement.textContent = "✓";
          buttonElement.style.color = "#22c55e";

          setTimeout(() => {
            buttonElement.textContent = originalText;
            buttonElement.style.color = "";
          }, 2000);
        }
      } catch (err) {
        console.error("Failed to copy code:", err);
      }
    };

    return () => {
      subscription.unsubscribe();
      delete window.copyCodeToClipboard;
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }
    };
  }, []);

  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        console.error("Logout error:", error);
      } else {
        setMessages([]);
        setCurrentView("welcome");
        setCurrentChatId(null);
        setChats([]);
        setSidebarOpen(!isMobile);
        setSelectedModuleId(null);
        setSplitPaneMode(false);
        setCodeEditorContent("");
      }
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  // ==================== BACKEND STATUS FUNCTIONS ====================

  const checkBackendStatus = async () => {
    if (statusCheckInProgress.current) {
      console.log("Status check already in progress, skipping...");
      return;
    }

    try {
      statusCheckInProgress.current = true;
      setBackendStatus((prev) => ({ ...prev, connecting: true }));

      console.log("Checking backend status...");

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${API_BASE_URL}/api/status`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        console.log("Backend status response:", data);
        setBackendStatus({
          online: data.rag_loaded && data.ollama_status === "connected",
          limited: !data.rag_loaded || data.ollama_status !== "connected",
          connecting: false,
          details: data,
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error("Backend status check failed:", error);

      let errorMessage = error.message;
      if (error.name === "AbortError") {
        errorMessage = "Connection timeout";
      }

      setBackendStatus({
        online: false,
        limited: false,
        connecting: false,
        error: errorMessage,
      });
    } finally {
      statusCheckInProgress.current = false;
    }
  };

  useEffect(() => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
      statusIntervalRef.current = null;
    }

    if (user) {
      console.log("User authenticated, starting backend status checks");

      if (!selectedModuleId) {
        console.log("No module selected, setting default CUDA Basics module");
        setSelectedModuleId("c801ac6c-1232-4c96-89b1-c4eadf41026c");
      }

      checkBackendStatus();

      statusIntervalRef.current = setInterval(() => {
        console.log("Periodic backend status check");
        checkBackendStatus();
      }, 120000);
    }

    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }
    };
  }, [user, selectedModuleId]);

  // ==================== MESSAGE PERSISTENCE FUNCTIONS ====================

  const saveMessage = async (chatId, sender, content, orderIndex, performanceMetrics = null) => {
    if (!chatId) {
      console.error("Cannot save message: chatId is null");
      return null;
    }

    try {
      const messageData = {
        chat_id: chatId,
        sender: sender,
        content: content,
        order_index: orderIndex,
        timestamp: new Date().toISOString(),
      };
      
      // Only add performance_metrics if they exist (for backward compatibility)
      if (performanceMetrics) {
        try {
          messageData.performance_metrics = JSON.stringify(performanceMetrics);
        } catch (e) {
          console.log("Could not save performance metrics:", e);
          // Continue without performance metrics if there's an error
        }
      }

      console.log("Saving message:", messageData);

      const { data, error } = await supabase
        .from("Messages")
        .insert([messageData])
        .select()
        .single();

      if (error) {
        console.error("Error saving message:", error);
        
        // If error is due to performance_metrics column not existing, try without it
        if (error.message?.includes('performance_metrics') && messageData.performance_metrics) {
          console.log("Retrying save without performance_metrics...");
          const { performance_metrics, ...messageDataWithoutMetrics } = messageData;
          
          const { data: retryData, error: retryError } = await supabase
            .from("Messages")
            .insert([messageDataWithoutMetrics])
            .select()
            .single();
            
          if (retryError) {
            console.error("Retry save also failed:", retryError);
            return null;
          }
          return retryData;
        }
        
        return null;
      }

      console.log("Message saved successfully:", data);
      return data;
    } catch (error) {
      console.error("Error saving message:", error);
      return null;
    }
  };

  const loadChatMessages = async (chatId) => {
    if (!chatId) {
      console.error("Cannot load messages: chatId is null");
      return [];
    }

    try {
      console.log("Loading messages for chat:", chatId);

      const { data, error } = await supabase
        .from("Messages")
        .select("*")
        .eq("chat_id", chatId)
        .order("order_index", { ascending: true });

      if (error) {
        console.error("Error loading messages:", error);
        return [];
      }

      const formattedMessages = data.map((msg) => {
        let performanceMetrics = null;
        
        // Safely parse performance metrics
        if (msg.performance_metrics) {
          try {
            performanceMetrics = typeof msg.performance_metrics === 'string' ? 
              JSON.parse(msg.performance_metrics) : 
              msg.performance_metrics;
          } catch (e) {
            console.log("Could not parse performance metrics for message:", msg.message_id);
          }
        }
        
        return {
          id: `${msg.message_id}_${msg.sender}`,
          role: msg.sender,
          content: msg.content,
          timestamp: msg.timestamp,
          messageId: msg.message_id,
          orderIndex: msg.order_index,
          performanceMetrics: performanceMetrics,
        };
      });

      console.log("Loaded messages for chat:", chatId, formattedMessages);
      return formattedMessages;
    } catch (error) {
      console.error("Error loading messages:", error);
      return [];
    }
  };

  const createNewChatForMessage = async (userMessage) => {
    if (!user) {
      console.error("Cannot create chat: user is null");
      return null;
    }

    try {
      const heading =
        userMessage.length > 50
          ? userMessage.substring(0, 50) + "..."
          : userMessage;

      const newChat = {
        user_id: user.id,
        heading: heading,
        description: "Chat conversation",
        timestamp: new Date().toISOString(),
        course_id: "1e44eb02-8daa-44a0-a7ee-28f88ce6863f",
        module_id: selectedModuleId || "c801ac6c-1232-4c96-89b1-c4eadf41026c",
        status: "active",
      };

      console.log("Creating new chat for message:", newChat);

      const { data, error } = await supabase
        .from("Chats")
        .insert([newChat])
        .select()
        .single();

      if (error) {
        console.error("Error creating new chat:", error);
        return null;
      }

      setChats((prevChats) => [data, ...prevChats]);
      setCurrentChatId(data.chat_id);

      console.log("New chat created for message:", data);
      return data;
    } catch (error) {
      console.error("Error creating new chat:", error);
      return null;
    }
  };

  // ==================== CHAT MANAGEMENT FUNCTIONS ====================

  const loadChats = async (moduleId = null) => {
    if (!user) return;

    try {
      setLoadingChats(true);
      const filterModuleId = moduleId || selectedModuleId;

      console.log(
        "Loading chats for user:",
        user.id,
        "module:",
        filterModuleId
      );

      let query = supabase.from("Chats").select("*").eq("user_id", user.id);

      if (filterModuleId) {
        query = query.eq("module_id", filterModuleId);
        console.log("Filtering chats by module_id:", filterModuleId);
      }

      const { data, error } = await query.order("timestamp", {
        ascending: false,
      });

      if (error) {
        console.error("Error loading chats:", error);

        if (error.code === "42P01") {
          console.warn(
            "Chats table does not exist. Please create it in Supabase first."
          );
          setChats([]);
          return;
        }
      } else {
        setChats(data || []);
        console.log(
          "Loaded chats for module:",
          filterModuleId,
          "count:",
          data?.length || 0
        );
      }
    } catch (error) {
      console.error("Error loading chats:", error);
      setChats([]);
    } finally {
      setLoadingChats(false);
    }
  };

  const createNewChat = async () => {
    if (!user) return;

    try {
      const newChat = {
        user_id: user.id,
        heading: "New Chat",
        description: "Start a new conversation about CUDA programming",
        timestamp: new Date().toISOString(),
        course_id: "1e44eb02-8daa-44a0-a7ee-28f88ce6863f",
        module_id: selectedModuleId || "c801ac6c-1232-4c96-89b1-c4eadf41026c",
        status: "active",
      };

      console.log("Creating new empty chat:", newChat);

      const { data, error } = await supabase
        .from("Chats")
        .insert([newChat])
        .select()
        .single();

      if (error) {
        console.error("Error creating chat:", error);
        return;
      }

      setChats((prevChats) => [data, ...prevChats]);

      setCurrentChatId(data.chat_id);
      setMessages([]);
      setCurrentView("chat");

      handleMobileItemSelect();

      console.log("New empty chat created:", data);
    } catch (error) {
      console.error("Error creating chat:", error);
    }
  };

  const selectChat = async (chatId) => {
    console.log("Selecting chat:", chatId);
    setCurrentChatId(chatId);
    setCurrentView("chat");
    setIsLoading(true);

    handleMobileItemSelect();

    try {
      const chatMessages = await loadChatMessages(chatId);
      setMessages(chatMessages);
      console.log("Chat selected and messages loaded:", chatId);
    } catch (error) {
      console.error("Error selecting chat:", error);
      setMessages([]);
    } finally {
      setIsLoading(false);
    }
  };

  const updateChatTitle = async (chatId, firstMessage) => {
    if (!chatId || !firstMessage) return;

    try {
      const heading =
        firstMessage.length > 50
          ? firstMessage.substring(0, 50) + "..."
          : firstMessage;

      console.log("Updating chat title:", chatId, heading);

      const { error } = await supabase
        .from("Chats")
        .update({
          heading: heading,
          timestamp: new Date().toISOString(),
        })
        .eq("chat_id", chatId);

      if (error) {
        console.error("Error updating chat heading:", error);
      } else {
        setChats((prevChats) =>
          prevChats.map((chat) =>
            chat.chat_id === chatId
              ? {
                  ...chat,
                  heading: heading,
                  timestamp: new Date().toISOString(),
                }
              : chat
          )
        );
        console.log("Chat heading updated:", heading);
      }
    } catch (error) {
      console.error("Error updating chat heading:", error);
    }
  };

  const handleSelectModule = async (moduleId) => {
    console.log("Module selected:", moduleId);
    setSelectedModuleId(moduleId);

    handleMobileItemSelect();

    if (currentChatId) {
      const currentChat = chats.find((chat) => chat.chat_id === currentChatId);
      if (currentChat && currentChat.module_id !== moduleId) {
        setCurrentChatId(null);
        setMessages([]);
        setCurrentView("welcome");
      }
    }

    await loadChats(moduleId);
  };

  const navigateToWelcome = () => {
    setCurrentView("welcome");
    setCurrentChatId(null);
    setMessages([]);
    handleMobileItemSelect();
  };

  const startNewChat = () => {
    createNewChat();
  };

  useEffect(() => {
    if (user && selectedModuleId) {
      console.log(
        "User and module ready, loading chats for module:",
        selectedModuleId
      );
      loadChats(selectedModuleId);
    }
  }, [selectedModuleId, user]);

  // ==================== MESSAGE HANDLING FUNCTIONS ====================

  const sendMessage = async (message) => {
    console.log("Sending message:", message);
    let chatId = currentChatId;
    let orderIndex = messages.length;

    // ========== SPECIAL HANDLING FOR QUIZ FEEDBACK ==========
    if (message.startsWith('QUIZ_FEEDBACK:')) {
      console.log("Processing quiz feedback message");
      const evaluationJson = message.replace('QUIZ_FEEDBACK:', '');
      
      try {
        const evaluation = JSON.parse(evaluationJson);
        console.log("Parsed quiz evaluation:", evaluation);
        
        // Create feedback message with quizEvaluation property
        const feedbackMessage = {
          id: Date.now() + '_quiz_feedback',
          role: 'assistant',
          content: '', // Empty content since we're using quizEvaluation
          quizEvaluation: evaluation, // This will trigger QuizFeedback component
          timestamp: new Date().toISOString(),
        };
        
        // Add feedback message to chat
        setMessages(prev => [...prev, feedbackMessage]);
        setIsLoading(false);
        
        // Save to database with a summary
        if (chatId) {
          await saveMessage(
            chatId,
            'assistant',
            `Quiz Results: ${evaluation.score}/${evaluation.total} (${evaluation.percentage.toFixed(0)}%) - ${evaluation.overall_message}`,
            orderIndex
          );
        }
        
        console.log("Quiz feedback message added successfully");
        return; // Exit early for quiz feedback
        
      } catch (error) {
        console.error('Error parsing quiz feedback:', error);
        // If parsing fails, treat as regular message
      }
    }

    // ========== REGULAR MESSAGE HANDLING ==========
    
    // Create new chat if needed
    if (!chatId) {
      console.log("No current chat, creating new chat...");
      const newChat = await createNewChatForMessage(message);
      if (!newChat) {
        console.error("Failed to create new chat");
        return;
      }
      chatId = newChat.chat_id;
      orderIndex = 0;
    }

    // Add user message to UI
    const userMessage = {
      id: Date.now() + "_user",
      role: "user",
      content: message,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setCurrentView("chat");

    // Save user message to database
    const savedUserMessage = await saveMessage(
      chatId,
      "user",
      message,
      orderIndex
    );
    if (savedUserMessage) {
      console.log("User message saved to database");
    }

    // Update chat title if this is the first message
    if (orderIndex === 0) {
      await updateChatTitle(chatId, message);
    }

    // Send message to backend
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
          chat_id: chatId,
          module_id: selectedModuleId,
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Backend response:", data);

      // Create assistant message
      const assistantMessage = {
        id: Date.now() + "_assistant",
        role: "assistant",
        content: data.response || "Sorry, I received an empty response.",
        timestamp: new Date().toISOString(),
        isStreaming: false,
        performanceMetrics: data.performance_metrics || null,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Save assistant message to database
      const savedAssistantMessage = await saveMessage(
        chatId,
        "assistant",
        assistantMessage.content,
        orderIndex + 1,
        assistantMessage.performanceMetrics
      );
      if (savedAssistantMessage) {
        console.log("Assistant message saved to database");
      }
      
    } catch (error) {
      console.error("Error sending message:", error);

      const errorMessage = {
        id: Date.now() + "_error",
        role: "assistant",
        content: `Sorry, I encountered an error: ${error.message}. Please check if the backend is running and try again.`,
        timestamp: new Date().toISOString(),
        isError: true,
      };

      setMessages((prev) => [...prev, errorMessage]);

      // Save error message to database
      await saveMessage(
        chatId,
        "assistant",
        errorMessage.content,
        orderIndex + 1
      );
    } finally {
      setIsLoading(false);
    }
  };

  // ==================== RENDER FUNCTIONS ====================

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

  if (!user) {
    return <AuthPage />;
  }

  const getSidebarClasses = () => {
    let classes = "sidebar";

    if (isMobile) {
      classes += sidebarOpen ? " mobile-visible" : " mobile-hidden";
    } else {
      classes += sidebarOpen ? " desktop-visible" : " desktop-hidden";
    }

    return classes;
  };

  return (
    <div className="app-container">
      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div className="mobile-overlay show" onClick={handleOverlayClick} />
      )}

      {/* Sidebar */}
      <div className={getSidebarClasses()}>
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onNewChat={startNewChat}
          onNavigateHome={navigateToWelcome}
          onSelectChat={selectChat}
          chats={chats || []}
          loadingChats={loadingChats}
          currentChatId={currentChatId}
          backendStatus={
            backendStatus || {
              online: false,
              limited: false,
              connecting: false,
            }
          }
          user={user}
          onLogout={handleLogout}
          onRefreshBackend={checkBackendStatus}
          onSelectModule={handleSelectModule}
          selectedModuleId={selectedModuleId}
          isMobile={isMobile}
        />
      </div>

      {/* Main Content */}
      <div className="main-content">
        <ChatHeader
          currentChat={chats.find((c) => c.chat_id === currentChatId)}
          selectedModule={
            selectedModuleId ? { id: selectedModuleId, name: "CUDA" } : null
          }
          user={user}
          onToggleSidebar={toggleSidebar}
          isSidebarVisible={sidebarOpen}
          backendStatus={backendStatus}
          splitPaneMode={splitPaneMode}
          onExitSplitMode={handleExitSplitMode}
          onLogout={handleLogout}
        />

        {splitPaneMode ? (
          /* Claude-like Split Pane Layout */
          <SplitPaneLayout
            initialLeftWidth={splitPaneWidth}
            onWidthChange={handleSplitPaneWidthChange}
            leftPane={
              currentView === "welcome" ? (
                <WelcomeView onSendMessage={sendMessage} user={user} />
              ) : (
                <ChatView
                  messages={messages}
                  isLoading={isLoading}
                  onSendMessage={sendMessage}
                  onOpenCodeEditor={handleOpenCodeEditor}
                  splitPaneMode={true}
                />
              )
            }
            rightPane={
              <ImprovedCodeEditor
                initialCode={codeEditorContent}
                language={codeEditorLanguage}
                onSendForReview={handleCodeReviewFromSplit}
                isLoading={isLoading}
                title={`${codeEditorLanguage.toUpperCase()} Code Editor`}
                onClose={handleExitSplitMode}
                showCloseButton={true}
              />
            }
          />
        ) : (
          /* Normal Single Pane Layout */
          <>
            {currentView === "welcome" ? (
              <WelcomeView onSendMessage={sendMessage} user={user} />
            ) : (
              <ChatView
                messages={messages}
                isLoading={isLoading}
                onSendMessage={sendMessage}
                onOpenCodeEditor={handleOpenCodeEditor}
                splitPaneMode={false}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default CudaTutorApp;