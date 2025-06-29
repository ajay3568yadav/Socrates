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

// Add this after the API_BASE_URL constant
const MODULE_TO_COURSE = {
  "c801ac6c-1232-4c96-89b1-c4eadf41026c": "CUDA Basics",
  "d26ccd91-cdf9-45e3-990f-a484d764bb9d": "Memory Optimization",
  "ff7d63fc-8646-4d9a-be5d-41a249beff02": "Kernel Development",
  "22107ce-5027-42bf-9941-6d00117da9ae": "Performance Tuning",
};

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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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

  // Model Selection State
  const [selectedModel, setSelectedModel] = useState('deepseek-r1');
  const [availableModels, setAvailableModels] = useState([]);

  // NEW: Tutoring Mode State
  const [tutoringMode, setTutoringMode] = useState(false);
  const [tutoringChatId, setTutoringChatId] = useState(null);
  const [originalChats, setOriginalChats] = useState([]); // Store original chats when entering tutoring mode

  const toggleSidebarCollapse = () => {
    // Only allow collapse on desktop
    if (window.innerWidth > 768) {
      setSidebarCollapsed(prev => !prev);
    }
  };

  // Session Management
  const [sessionId] = useState(
    `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  );

  // Refs for preventing duplicate requests
  const statusCheckInProgress = useRef(false);
  const statusIntervalRef = useRef(null);

  // ==================== TUTORING MODE FUNCTIONS - NEW ====================

  // Toggle tutoring mode
  const toggleTutoringMode = async () => {
    if (!selectedModuleId) {
      console.error("No module selected for tutoring mode");
      return;
    }

    if (!tutoringMode) {
      // Entering tutoring mode
      console.log("Entering tutoring mode for module:", selectedModuleId);
      
      // Store current chats
      setOriginalChats([...chats]);
      
      // Create a new tutoring session
      const tutoringChat = await createTutoringSession();
      if (tutoringChat) {
        setTutoringMode(true);
        setTutoringChatId(tutoringChat.chat_id);
        setCurrentChatId(tutoringChat.chat_id);
        setCurrentView("chat");
        setMessages([]);
        
        // Clear regular chats from view
        setChats([]);
        
        // Start tutoring session with AI
        await startTutoringSession(tutoringChat.chat_id);
      }
    } else {
      // Exiting tutoring mode
      console.log("Exiting tutoring mode");
      
      setTutoringMode(false);
      setTutoringChatId(null);
      
      // Restore original chats
      setChats(originalChats);
      setOriginalChats([]);
      
      // Clear current chat and messages
      setCurrentChatId(null);
      setMessages([]);
      setCurrentView("welcome");
    }
  };

  // Create a tutoring session chat
  const createTutoringSession = async () => {
    if (!user || !selectedModuleId) return null;

    try {
      const moduleNames = {
        "c801ac6c-1232-4c96-89b1-c4eadf41026c": "CUDA Basics",
        "d26ccd91-cdf9-45e3-990f-a484d764bb9d": "Memory Optimization",
        "ff7d63fc-8646-4d9a-be5d-41a249beff02": "Kernel Development",
        "22107ce-5027-42bf-9941-6d00117da9ae": "Performance Tuning",
      };

      const moduleName = moduleNames[selectedModuleId] || "CUDA Module";

      const newChat = {
        user_id: user.id,
        heading: `🎓 Tutoring: ${moduleName}`,
        description: `Interactive tutoring session for ${moduleName}`,
        timestamp: new Date().toISOString(),
        course_id: "1e44eb02-8daa-44a0-a7ee-28f88ce6863f",
        module_id: selectedModuleId,
        status: "tutoring", // Special status for tutoring sessions
      };

      console.log("Creating tutoring session:", newChat);

      const { data, error } = await supabase
        .from("Chats")
        .insert([newChat])
        .select()
        .single();

      if (error) {
        console.error("Error creating tutoring session:", error);
        return null;
      }

      console.log("Tutoring session created:", data);
      return data;
    } catch (error) {
      console.error("Error creating tutoring session:", error);
      return null;
    }
  };

  // Start the tutoring session with AI
  const startTutoringSession = async (chatId) => {
    const moduleNames = {
      "c801ac6c-1232-4c96-89b1-c4eadf41026c": "CUDA Basics",
      "d26ccd91-cdf9-45e3-990f-a484d764bb9d": "Memory Optimization", 
      "ff7d63fc-8646-4d9a-be5d-41a249beff02": "Kernel Development",
      "22107ce-5027-42bf-9941-6d00117da9ae": "Performance Tuning",
    };

    const moduleName = moduleNames[selectedModuleId] || "CUDA Programming";
    
    const tutoringPrompt = `Hello! I'm your CUDA programming tutor, and I'm excited to help you learn about ${moduleName}! 

🎓 **Welcome to Interactive Tutoring Mode**

In this session, I'll:
- Teach you the key concepts step by step
- Ask you questions to test your understanding
- Provide hands-on examples and exercises
- Give you immediate feedback on your answers

Let's start with the fundamentals. Are you ready to begin learning about ${moduleName}? 

First, let me ask: What's your current experience level with CUDA programming?
A) Complete beginner - I'm new to GPU programming
B) Some basic knowledge - I understand the basics but need practice
C) Intermediate level - I can write simple kernels
D) Advanced - I'm looking to optimize and master techniques

Please type your answer (A, B, C, or D) and I'll tailor the session to your level!`;

    // Add the tutoring welcome message
    const welcomeMessage = {
      id: Date.now() + "_tutor_welcome",
      role: "assistant",
      content: tutoringPrompt,
      timestamp: new Date().toISOString(),
      tutoring: true,
    };

    setMessages([welcomeMessage]);

    // Save the tutoring welcome message to database
    await saveMessage(
      chatId,
      "assistant",
      tutoringPrompt,
      0
    );
  };

  // Check if we're in tutoring mode for the current module
  const isInTutoringMode = () => {
    return tutoringMode && selectedModuleId;
  };

  // ==================== MODEL SELECTION FUNCTIONS ====================

  // Load model preference from localStorage
  useEffect(() => {
    const savedModel = localStorage.getItem('selectedModel');
    if (savedModel) {
      setSelectedModel(savedModel);
      console.log('Loaded model preference from localStorage:', savedModel);
    }
  }, []);

  // Handle model selection change
  const handleModelChange = (modelId) => {
    console.log('Model changed from', selectedModel, 'to', modelId);
    setSelectedModel(modelId);
    
    // Save preference to localStorage
    localStorage.setItem('selectedModel', modelId);
    
    console.log(`✅ Switched to ${getModelDisplayName(modelId)} model`);
  };

  // Get model display name helper function
  const getModelDisplayName = (modelId) => {
    const modelMap = {
      'deepseek-r1': 'DeepSeek R1',
      'qwen-2.5': 'Qwen 2.5',
      'mixtral': 'Mixtral',
      'llama-3.2': 'Llama 3.2'
    };
    return modelMap[modelId] || modelId;
  };

  // Load available models from backend (optional)
  const loadAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.models || []);
        console.log('Loaded available models:', data.models);
        
        // If the current selected model is not available, switch to default
        if (data.models && !data.models.find(m => m.id === selectedModel)) {
          const defaultModel = data.default_model || 'deepseek-r1';
          handleModelChange(defaultModel);
        }
      }
    } catch (error) {
      console.error('Error loading available models:', error);
      // Continue with default models if backend doesn't support this endpoint
    }
  };

  // Clear model preference on logout
  const clearModelPreference = () => {
    localStorage.removeItem('selectedModel');
    setSelectedModel('deepseek-r1'); // Reset to default
  };

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
    if (window.innerWidth <= 768) {
      // Mobile: toggle open/close
      setSidebarOpen(prev => !prev);
    } else {
      // Desktop: toggle collapsed/expanded if open, or just open if closed
      if (!sidebarOpen) {
        setSidebarOpen(true);
        setSidebarCollapsed(false);
      } else {
        toggleSidebarCollapse();
      }
    }
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
        
        // Clear tutoring mode state on logout
        setTutoringMode(false);
        setTutoringChatId(null);
        setOriginalChats([]);
        
        // Clear model preference on logout
        clearModelPreference();
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
      
      // Load available models when user is authenticated
      loadAvailableModels();

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
      .select("*, quiz_data")  // Include quiz_data column
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

      const baseMessage = {
        id: `${msg.message_id}_${msg.sender}`,
        role: msg.sender,
        content: msg.content,
        timestamp: msg.timestamp,
        messageId: msg.message_id,
        orderIndex: msg.order_index,
        performanceMetrics: performanceMetrics,
        tutoring: msg.sender === "assistant" && msg.content.includes("tutoring") // Mark tutoring messages
      };

      // NEW: Check if this message has full quiz data
      if (msg.quiz_data) {
        try {
          const quizEvaluation = JSON.parse(msg.quiz_data);
          return {
            ...baseMessage,
            content: "", // Empty content since we're using quizEvaluation
            quizEvaluation: quizEvaluation // This will trigger QuizFeedback component
          };
        } catch (e) {
          console.log("Could not parse quiz_data for message:", msg.message_id);
        }
      }

      // FALLBACK: Check if this is a quiz result message and reconstruct it
      if (msg.sender === "assistant" && msg.content.startsWith("Quiz Results:")) {
        // This is a quiz result summary - we need to reconstruct the quiz data
        // Parse the summary: "Quiz Results: 2/3 (67%) - Good job! You scored..."
        const summaryMatch = msg.content.match(/Quiz Results: (\d+)\/(\d+) \((\d+)%\) - (.+)/);
        
        if (summaryMatch) {
          const [, score, total, percentage, overallMessage] = summaryMatch;
          
          // Create a reconstructed quiz evaluation
          const reconstructedEvaluation = {
            score: parseInt(score),
            total: parseInt(total),
            percentage: parseInt(percentage),
            overall_message: overallMessage,
            detailed_feedback: [], // We don't have the detailed feedback anymore
            topic: "CUDA Programming", // Default topic
            reconstructed: true // Flag to indicate this was reconstructed
          };

          return {
            ...baseMessage,
            content: "", // Empty content since we're using quizEvaluation
            quizEvaluation: reconstructedEvaluation // This will trigger QuizFeedback component
          };
        }
      }

      return baseMessage;
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

      // Only add to chats if not in tutoring mode
      if (!tutoringMode) {
        setChats((prevChats) => [data, ...prevChats]);
      }
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

      // Exclude tutoring sessions from regular chat list
      query = query.neq("status", "tutoring");

      const { data, error } = await query.order("timestamp", {
        ascending: false,
      });

      if (error) {
        console.error("Error loading chats:", error);

        if (error.code === "42P01") {
          console.warn(
            "Chats table does not exist. Please create it in Supabase first."
          );
          if (!tutoringMode) {
            setChats([]);
          }
          return;
        }
      } else {
        // Only update chats if not in tutoring mode
        if (!tutoringMode) {
          setChats(data || []);
          console.log(
            "Loaded chats for module:",
            filterModuleId,
            "count:",
            data?.length || 0
          );
        }
      }
    } catch (error) {
      console.error("Error loading chats:", error);
      if (!tutoringMode) {
        setChats([]);
      }
    } finally {
      setLoadingChats(false);
    }
  };

  const createNewChat = async () => {
    if (!user) return;

    // Don't create regular chats in tutoring mode
    if (tutoringMode) {
      console.log("Cannot create regular chat in tutoring mode");
      return;
    }

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
        // Only update chats state if not in tutoring mode
        if (!tutoringMode) {
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
        }
        console.log("Chat heading updated:", heading);
      }
    } catch (error) {
      console.error("Error updating chat heading:", error);
    }
  };

  const handleSelectModule = async (moduleId) => {
    console.log("Module selected:", moduleId);
    
    // If switching modules while in tutoring mode, exit tutoring mode first
    if (tutoringMode && selectedModuleId !== moduleId) {
      await toggleTutoringMode(); // This will exit tutoring mode
    }
    
    setSelectedModuleId(moduleId);

    handleMobileItemSelect();

    if (currentChatId && !tutoringMode) {
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
    // If in tutoring mode, exit it first
    if (tutoringMode) {
      toggleTutoringMode();
    }
    
    setCurrentView("welcome");
    setCurrentChatId(null);
    setMessages([]);
    handleMobileItemSelect();
  };

  const startNewChat = () => {
    createNewChat();
  };

  // Handle start module action
  const handleStartModule = (courseName, initialMessage) => {
    console.log("Starting module:", courseName);

    // Create new chat for the course
    createNewChat().then(() => {
      // Send the initial course message
      if (initialMessage) {
        // Add the course welcome message as an assistant message
        const welcomeMessage = {
          id: Date.now() + "_course_welcome",
          role: "assistant",
          content: initialMessage,
          timestamp: new Date().toISOString(),
        };

        setMessages([welcomeMessage]);
        setCurrentView("chat");
      }
    });
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
  console.log("Sending message:", message, "using model:", selectedModel, "tutoring mode:", tutoringMode);
  let chatId = currentChatId;
  let orderIndex = messages.length;

  // Handle tutoring mode messages
  if (tutoringMode && tutoringChatId) {
    chatId = tutoringChatId;
  }

  // ========== SPECIAL HANDLING FOR QUIZ FEEDBACK ==========
  if (message.startsWith("QUIZ_FEEDBACK:")) {
    console.log("Processing quiz feedback message");
    const evaluationJson = message.replace("QUIZ_FEEDBACK:", "");

    try {
      const evaluation = JSON.parse(evaluationJson);
      console.log("Parsed quiz evaluation:", evaluation);

      // Create feedback message with quizEvaluation property
      const feedbackMessage = {
        id: Date.now() + "_quiz_feedback",
        role: "assistant",
        content: "", 
        quizEvaluation: evaluation,
        timestamp: new Date().toISOString(),
        tutoring: tutoringMode,
      };

      // Add feedback message to chat
      setMessages((prev) => [...prev, feedbackMessage]);
      setIsLoading(false);

      // Save quiz feedback to database
      if (chatId) {
        try {
          const quizMessageData = {
            chat_id: chatId,
            sender: "assistant", 
            content: `Quiz Results: ${evaluation.score}/${evaluation.total} (${evaluation.percentage.toFixed(0)}%) - ${evaluation.overall_message}`,
            order_index: orderIndex,
            timestamp: new Date().toISOString(),
            quiz_data: JSON.stringify(evaluation)
          };

          const { data, error } = await supabase
            .from("Messages")
            .insert([quizMessageData])
            .select()
            .single();

          if (error) {
            console.error("Error saving quiz message:", error);
            await saveMessage(chatId, "assistant", quizMessageData.content, orderIndex);
          } else {
            console.log("Quiz message with full data saved successfully");
          }
        } catch (error) {
          console.error("Error saving enhanced quiz message:", error);
          await saveMessage(
            chatId,
            "assistant",
            `Quiz Results: ${evaluation.score}/${evaluation.total} (${evaluation.percentage.toFixed(0)}%) - ${evaluation.overall_message}`,
            orderIndex
          );
        }
      }

      console.log("Quiz feedback message added successfully");
      return; // Exit early for quiz feedback
    } catch (error) {
      console.error("Error parsing quiz feedback:", error);
    }
  }

  // ========== NEW: SPECIAL HANDLING FOR AI ANALYSIS ==========
  if (message.startsWith("AI_ANALYSIS:")) {
    console.log("Processing AI analysis message");
    const analysisContent = message.replace("AI_ANALYSIS:", "");

    // Create AI analysis message as assistant message
    const analysisMessage = {
      id: Date.now() + "_ai_analysis",
      role: "assistant",
      content: analysisContent,
      timestamp: new Date().toISOString(),
      tutoring: tutoringMode,
    };

    // Add analysis message to chat
    setMessages((prev) => [...prev, analysisMessage]);
    setIsLoading(false);

    // Save AI analysis to database
    if (chatId) {
      const savedAnalysisMessage = await saveMessage(
        chatId,
        "assistant",
        analysisContent,
        orderIndex + 1 // +1 because quiz feedback was just saved
      );
      if (savedAnalysisMessage) {
        console.log("AI analysis message saved to database");
      }
    }

    console.log("AI analysis message added successfully");
    return; // Exit early for AI analysis
  }

  // ========== REGULAR MESSAGE HANDLING ==========

  // Create new chat if needed (but not in tutoring mode - tutoring chat already exists)
  if (!chatId && !tutoringMode) {
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
    tutoring: tutoringMode, // Mark as tutoring if in tutoring mode
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

  // Update chat title if this is the first message and not in tutoring mode
  if (orderIndex === 0 && !tutoringMode) {
    await updateChatTitle(chatId, message);
  }

  // UPDATED: Send message to backend with chat_id for proper context and tutoring mode
  try {
    const requestBody = {
      message: message,
      session_id: sessionId,    // Keep for compatibility
      chat_id: chatId,          // NEW: Pass actual chat_id for database context
      module_id: selectedModuleId,
      model: selectedModel,     // Include selected model
      stream: false,
      tutoring_mode: tutoringMode, // Include tutoring mode flag
    };

    console.log("Sending request to backend with chat_id and tutoring mode:", requestBody);

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
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
      modelUsed: data.model_used || selectedModel,
      modelName: data.model_name || getModelDisplayName(selectedModel),
      tutoring: tutoringMode, // Mark as tutoring if in tutoring mode
      contextSource: data.context_source || 'unknown', // NEW: Track context source
    };

    setMessages((prev) => [...prev, assistantMessage]);

    // Log model usage and tutoring context
    if (tutoringMode) {
      console.log(`🎓 Tutoring response generated using ${assistantMessage.modelName} for ${MODULE_TO_COURSE[selectedModuleId] || 'module'}`);
      console.log(`📚 Context source: ${assistantMessage.contextSource}`);
    } else {
      console.log(`✅ Response generated using ${assistantMessage.modelName} (${assistantMessage.modelUsed})`);
      console.log(`📚 Context source: ${assistantMessage.contextSource}`);
    }

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
      tutoring: tutoringMode, // Mark as tutoring if in tutoring mode
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
      if (sidebarCollapsed && sidebarOpen) {
        classes += " collapsed";
      }
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
          isCollapsed={sidebarCollapsed}
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
          tutoringMode={tutoringMode} // NEW: Pass tutoring mode state
          onToggleTutoringMode={toggleTutoringMode} // NEW: Pass tutoring mode toggle function
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
          isSidebarCollapsed={sidebarCollapsed}
          backendStatus={backendStatus}
          splitPaneMode={splitPaneMode}
          onExitSplitMode={handleExitSplitMode}
          onLogout={handleLogout}
          selectedModel={selectedModel}
          onModelChange={handleModelChange}
          tutoringMode={tutoringMode} // NEW: Pass tutoring mode state
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
                  splitPaneMode={splitPaneMode}
                  tutoringMode={tutoringMode}
                  currentChatId={currentChatId} // ADD THIS LINE
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
                tutoringMode={tutoringMode} // NEW: Pass tutoring mode state
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default CudaTutorApp;