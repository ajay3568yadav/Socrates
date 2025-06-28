import React, { useState, useEffect } from "react";
import supabase from "../config/supabaseClient";
import "../css/FolderSection.css";

const FolderSection = ({ onSelectModule, selectedModuleId }) => {
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);

  // Default course ID for CUDA Basics
  const CUDA_COURSE_ID = "1e44eb02-8daa-44a0-a7ee-28f88ce6863f";

  // Fallback static data
  const fallbackModules = [
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
    {
      module_id: "22107ce-5027-42bf-9941-6d00117da9ae",
      course_id: CUDA_COURSE_ID,
      module_name: "Performance Tuning",
      status: "in-progress",
      timestamp: "2025-06-26T01:46:03.929264+00:00",
    },
  ];

  // Load modules from Supabase with fallback
  const loadModules = async () => {
    try {
      setLoading(true);

      const { data, error } = await supabase
        .from("Modules")
        .select("*")
        .eq("course_id", CUDA_COURSE_ID)
        .order("timestamp", { ascending: true });

      if (error) {
        console.error("Error loading modules:", error);
        if (error.code === "42P01") {
          console.warn("Modules table does not exist. Using fallback data.");
          setModules(fallbackModules);
          return;
        }
      } else {
        setModules(data || []);
        console.log("Loaded modules from database:", data);
      }
    } catch (error) {
      console.error("Error loading modules:", error);
      console.log("Using fallback module data");
      setModules(fallbackModules);
    } finally {
      setLoading(false);
    }
  };

  // Load modules on component mount
  useEffect(() => {
    loadModules();
  }, []);

  // Handle module selection
  const handleModuleClick = (moduleId) => {
    console.log("Module clicked:", moduleId);
    if (onSelectModule) {
      onSelectModule(moduleId);
    }
  };

  // Get appropriate icon based on module name
  const getModuleIcon = (moduleName) => {
    const name = moduleName.toLowerCase();
    if (name.includes("performance") || name.includes("tuning")) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <polygon points="13,2 3,14 12,14 11,22 21,10 12,10 13,2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    }
    if (name.includes("memory") || name.includes("optimization")) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" stroke="currentColor" strokeWidth="2"/>
          <polyline points="22,6 12,13 2,6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    }
    if (name.includes("kernel") || name.includes("development")) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
          <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" stroke="currentColor" strokeWidth="2"/>
        </svg>
      );
    }
    // Default for basics/introduction
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "completed":
        return (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polyline points="20,6 9,17 4,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        );
      case "in-progress":
        return (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
            <polyline points="12,6 12,12 16,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        );
      case "not-started":
        return (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
            <rect x="10" y="10" width="4" height="4" fill="currentColor"/>
          </svg>
        );
      default:
        return (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
          </svg>
        );
    }
  };

  return (
    <div className="folder-section">
      <div className="section-header">
        <div className="section-title">
          <svg className="section-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>Course Modules</span>
        </div>
        <div className="section-actions">
          <button
            className="section-action"
            onClick={loadModules}
            title="Refresh modules"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <polyline points="23,4 23,10 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="1,20 1,14 7,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M20.49,9A9,9,0,0,0,5.64,5.64L1,10m22,4L18.36,18.36A9,9,0,0,1,3.51,15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button className="section-action" title="Module options">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="1" stroke="currentColor" strokeWidth="2" fill="currentColor"/>
              <circle cx="19" cy="12" r="1" stroke="currentColor" strokeWidth="2" fill="currentColor"/>
              <circle cx="5" cy="12" r="1" stroke="currentColor" strokeWidth="2" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>

      <div className="modules-list">
        {loading ? (
          <div className="loading-state">
            <div className="loading-item">
              <div className="loading-spinner"></div>
              <span>Loading modules...</span>
            </div>
          </div>
        ) : modules.length === 0 ? (
          <div className="empty-state">
            <div className="empty-message">No modules found</div>
          </div>
        ) : (
          modules.map((module) => (
            <div
              key={module.module_id}
              className={`module-item ${
                selectedModuleId === module.module_id ? "active" : ""
              }`}
              onClick={() => handleModuleClick(module.module_id)}
              title={`Click to view ${module.module_name} chats`}
            >
              <div className="module-icon">
                {getModuleIcon(module.module_name)}
              </div>
              <div className="module-content">
                <div className="module-name">{module.module_name}</div>
              </div>
              <div className="module-status">
                <div className={`status-icon ${module.status}`}>
                  {getStatusIcon(module.status)}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {modules.length > 0 && (
        <div className="modules-summary">
          <div className="summary-text">
            {modules.filter((m) => m.status === "completed").length} of {modules.length} completed
          </div>
        </div>
      )}
    </div>
  );
};

export default FolderSection;