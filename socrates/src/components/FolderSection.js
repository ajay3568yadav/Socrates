import React, { useState, useEffect } from "react";
import supabase from "../config/supabaseClient";
import "../css/FolderSection.css"; // Ensure you have the correct CSS file

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:5001";

const FolderSection = ({
  onSelectModule,
  selectedModuleId,
  onStartModule,
  sessionId,
}) => {
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [courseInfo, setCourseInfo] = useState(null);
  const [courseProgress, setCourseProgress] = useState(null);
  const [loadingCourseData, setLoadingCourseData] = useState(false);

  // Default course ID for CUDA Basics
  const CUDA_COURSE_ID = "1e44eb02-8daa-44a0-a7ee-28f88ce6863f";

  // Fallback static data based on your screenshot
  const fallbackModules = [
    {
      module_id: "22107ce-5027-42bf-9941-6d00117da9ae",
      course_id: CUDA_COURSE_ID,
      module_name: "Performance Tuning",
      status: "in-progress",
      timestamp: "2025-06-26T01:46:03.929264+00:00",
    },
    {
      module_id: "c801ac6c-1232-4c96-89b1-c4eadf41026c",
      course_id: CUDA_COURSE_ID,
      module_name: "CUDA Basics",
      status: "in-progress",
      timestamp: "2025-06-26T01:43:23.497186+00:00",
    },
    {
      module_id: "d26ccd91-cdf9-45e3-990f-a484d764bb9d",
      course_id: CUDA_COURSE_ID,
      module_name: "Memory Optimization",
      status: "in-progress",
      timestamp: "2025-06-26T01:44:09.052017+00:00",
    },
    {
      module_id: "ff7d63fc-8646-4d9a-be5d-41a249beff02",
      course_id: CUDA_COURSE_ID,
      module_name: "Kernel Development",
      status: "in-progress",
      timestamp: "2025-06-26T01:45:41.274865+00:00",
    },
  ];

  // Load modules from Supabase with fallback
  const loadModules = async () => {
    try {
      setLoading(true);

      // Try to load from modules table first
      const { data, error } = await supabase
        .from("Modules")
        .select("*")
        .eq("course_id", CUDA_COURSE_ID)
        .order("timestamp", { ascending: true });

      if (error) {
        console.error("Error loading modules:", error);

        // If modules table doesn't exist, use fallback data
        if (error.code === "42P01") {
          console.warn("Modules table does not exist. Using fallback data.");
          setModules(fallbackModules);
          // Don't auto-select here - let the parent handle it
          return;
        }
      } else {
        setModules(data || []);
        console.log("Loaded modules from database:", data);
        // Don't auto-select here - let the parent handle it
      }
    } catch (error) {
      console.error("Error loading modules:", error);
      // Use fallback data on any error
      console.log("Using fallback module data");
      setModules(fallbackModules);
      // Don't auto-select here - let the parent handle it
    } finally {
      setLoading(false);
    }
  };

  // Load modules on component mount
  useEffect(() => {
    loadModules();
  }, []); // Empty dependency array is correct here

  // Map module IDs to course names
  const MODULE_TO_COURSE = {
    "c801ac6c-1232-4c96-89b1-c4eadf41026c": "CUDA Basics",
    "d26ccd91-cdf9-45e3-990f-a484d764bb9d": "Memory Optimization",
    "ff7d63fc-8646-4d9a-be5d-41a249beff02": "Kernel Development",
    "22107ce-5027-42bf-9941-6d00117da9ae": "Performance Tuning",
  };

  // Load course information and progress
  const loadCourseData = async (moduleId) => {
    const courseName = MODULE_TO_COURSE[moduleId];
    if (!courseName || !sessionId) return;

    setLoadingCourseData(true);
    try {
      // Load course info
      const courseInfoResponse = await fetch(
        `${API_BASE_URL}/api/course-info/${encodeURIComponent(courseName)}`
      );
      if (courseInfoResponse.ok) {
        const courseInfoData = await courseInfoResponse.json();
        setCourseInfo(courseInfoData);
      }

      // Load course progress
      const progressResponse = await fetch(
        `${API_BASE_URL}/api/course-progress/${sessionId}/${encodeURIComponent(
          courseName
        )}`
      );
      if (progressResponse.ok) {
        const progressData = await progressResponse.json();
        setCourseProgress(progressData);
      }
    } catch (error) {
      console.error("Error loading course data:", error);
    } finally {
      setLoadingCourseData(false);
    }
  };

  // Handle module selection
  const handleModuleClick = (moduleId) => {
    console.log("Module clicked:", moduleId);
    if (onSelectModule) {
      onSelectModule(moduleId);
    }
    // Load course data when module is selected
    loadCourseData(moduleId);
  };

  // Handle start module action
  const handleStartModule = async () => {
    const courseName = MODULE_TO_COURSE[selectedModuleId];
    if (!courseName || !sessionId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/start-module`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          course_name: courseName,
          session_id: sessionId,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (onStartModule) {
          onStartModule(courseName, data.response);
        }
        // Reload progress after starting
        loadCourseData(selectedModuleId);
      } else {
        console.error("Failed to start module");
      }
    } catch (error) {
      console.error("Error starting module:", error);
    }
  };

  // Get appropriate icon based on module name
  const getModuleIcon = (moduleName) => {
    const name = moduleName.toLowerCase();
    if (name.includes("performance") || name.includes("tuning")) return "⚡";
    if (name.includes("memory") || name.includes("optimization")) return "🧠";
    if (name.includes("kernel") || name.includes("development")) return "⚙️";
    if (name.includes("basic") || name.includes("introduction")) return "📚";
    return "📁";
  };

  return (
    <div className="folder-section">
      <div className="section-header">
        <span>Course Modules</span>
        <div className="section-actions">
          <button
            className="section-action"
            onClick={loadModules}
            title="Refresh modules"
          >
            ↻
          </button>
          <button className="section-action" title="Module options">
            ⋯
          </button>
        </div>
      </div>

      <div>
        {loading ? (
          <div className="loading-modules">
            <div className="folder-item">
              <span className="folder-icon">⏳</span>
              <span className="folder-name">Loading modules...</span>
            </div>
          </div>
        ) : modules.length === 0 ? (
          <div className="empty-modules">
            <div className="folder-item">
              <span className="folder-icon">📚</span>
              <span className="folder-name">No modules found</span>
            </div>
          </div>
        ) : (
          modules.map((module) => (
            <div
              key={module.module_id}
              className={`folder-item ${
                selectedModuleId === module.module_id ? "active" : ""
              }`}
              onClick={() => handleModuleClick(module.module_id)}
              style={{ cursor: "pointer" }}
              title={`Click to view ${module.module_name} chats`}
            >
              <span className="folder-icon">
                {getModuleIcon(module.module_name)}
              </span>
              <div className="folder-content" style={{ flex: 1, minWidth: 0 }}>
                <span className="folder-name">{module.module_name}</span>
                {selectedModuleId === module.module_id && (
                  <div
                    style={{
                      fontSize: "11px",
                      color: "#888",
                      marginTop: "2px",
                    }}
                  >
                    Currently selected
                  </div>
                )}
              </div>
              <div className="module-status">
                <span className={`status-badge ${module.status}`}>
                  {module.status === "in-progress"
                    ? "🔄"
                    : module.status === "completed"
                    ? "✅"
                    : module.status === "not-started"
                    ? "⏸️"
                    : "📋"}
                </span>
              </div>
              <button
                className="item-menu"
                onClick={(e) => {
                  e.stopPropagation();
                  console.log("Module menu clicked:", module.module_id);
                }}
                title="Module options"
              >
                ⋯
              </button>
            </div>
          ))
        )}
      </div>

      {modules.length > 0 && (
        <div className="modules-summary">
          <div className="summary-text">
            {modules.filter((m) => m.status === "completed").length} of{" "}
            {modules.length} completed
          </div>
        </div>
      )}

      {/* Course Information Section */}
      {selectedModuleId && (
        <div className="course-info-section">
          <div className="section-header">
            <span>Course Details</span>
          </div>

          {loadingCourseData ? (
            <div className="loading-course-data">
              <div className="folder-item">
                <span className="folder-icon">⏳</span>
                <span className="folder-name">Loading course data...</span>
              </div>
            </div>
          ) : courseInfo ? (
            <div className="course-details">
              <div className="course-name">
                <h3>{courseInfo.course_name}</h3>
              </div>

              <div className="course-stats">
                <div className="stat-item">
                  <span className="stat-label">Topics:</span>
                  <span className="stat-value">{courseInfo.total_topics}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Practice Questions:</span>
                  <span className="stat-value">
                    {courseInfo.practice_questions_count}
                  </span>
                </div>
                {courseProgress && (
                  <div className="stat-item">
                    <span className="stat-label">Progress:</span>
                    <span className="stat-value">
                      {courseProgress.topics_completed}/
                      {courseProgress.total_topics}(
                      {Math.round(courseProgress.completion_percentage)}%)
                    </span>
                  </div>
                )}
              </div>

              {courseProgress && (
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${courseProgress.completion_percentage}%`,
                    }}
                  ></div>
                </div>
              )}

              <div className="course-actions">
                <button
                  className="start-module-btn"
                  onClick={handleStartModule}
                  disabled={!sessionId}
                >
                  {courseProgress && courseProgress.progress.course_completed
                    ? "Review Module"
                    : courseProgress && courseProgress.topics_completed > 0
                    ? "Continue Module"
                    : "Start Module"}
                </button>
              </div>

              <div className="prerequisites">
                <h4>Prerequisites:</h4>
                <ul>
                  {courseInfo.prerequisites.map((prereq, index) => (
                    <li key={index}>{prereq}</li>
                  ))}
                </ul>
              </div>

              <div className="topics-list">
                <h4>Topics Covered:</h4>
                <ul>
                  {courseInfo.topics.map((topic, index) => (
                    <li
                      key={index}
                      className={
                        courseProgress &&
                        courseProgress.progress.topics_covered.includes(index)
                          ? "completed"
                          : courseProgress &&
                            index ===
                              courseProgress.progress.current_topic_index
                          ? "current"
                          : ""
                      }
                    >
                      {courseProgress &&
                        courseProgress.progress.topics_covered.includes(
                          index
                        ) &&
                        "✅ "}
                      {courseProgress &&
                        index === courseProgress.progress.current_topic_index &&
                        !courseProgress.progress.topics_covered.includes(
                          index
                        ) &&
                        "🔄 "}
                      {topic}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};

export default FolderSection;
