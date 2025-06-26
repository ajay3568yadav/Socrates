import React, { useState, useEffect } from 'react';
import supabase from '../config/supabaseClient';

const FolderSection = ({ onSelectModule, selectedModuleId }) => {
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Default course ID for CUDA Basics
  const CUDA_COURSE_ID = '1e44eb02-8daa-44a0-a7ee-28f88ce6863f';

  // Fallback static data based on your screenshot
  const fallbackModules = [
    {
      module_id: '22107ce-5027-42bf-9941-6d00117da9ae',
      course_id: CUDA_COURSE_ID,
      module_name: 'Performance Tuning',
      status: 'in-progress',
      timestamp: '2025-06-26T01:46:03.929264+00:00'
    },
    {
      module_id: 'c801ac6c-1232-4c96-89b1-c4eadf41026c',
      course_id: CUDA_COURSE_ID,
      module_name: 'CUDA Basics',
      status: 'in-progress',
      timestamp: '2025-06-26T01:43:23.497186+00:00'
    },
    {
      module_id: 'd26ccd91-cdf9-45e3-990f-a484d764bb9d',
      course_id: CUDA_COURSE_ID,
      module_name: 'Memory Optimization',
      status: 'in-progress',
      timestamp: '2025-06-26T01:44:09.052017+00:00'
    },
    {
      module_id: 'ff7d63fc-8646-4d9a-be5d-41a249beff02',
      course_id: CUDA_COURSE_ID,
      module_name: 'Kernel Development',
      status: 'in-progress',
      timestamp: '2025-06-26T01:45:41.274865+00:00'
    }
  ];

  // Load modules from Supabase with fallback
  const loadModules = async () => {
    try {
      setLoading(true);
      
      // Try to load from modules table first
      const { data, error } = await supabase
        .from('Modules')
        .select('*')
        .eq('course_id', CUDA_COURSE_ID)
        .order('timestamp', { ascending: true });

      if (error) {
        console.error('Error loading modules:', error);
        
        // If modules table doesn't exist, use fallback data
        if (error.code === '42P01') {
          console.warn('Modules table does not exist. Using fallback data.');
          setModules(fallbackModules);
          // Set first module as selected by default if none selected
          if (!selectedModuleId && onSelectModule) {
            onSelectModule(fallbackModules[0]?.module_id);
          }
          return;
        }
      } else {
        setModules(data || []);
        console.log('Loaded modules from database:', data);
        
        // Set first module as selected by default if none selected
        if (!selectedModuleId && data && data.length > 0 && onSelectModule) {
          onSelectModule(data[0].module_id);
        }
      }
    } catch (error) {
      console.error('Error loading modules:', error);
      // Use fallback data on any error
      console.log('Using fallback module data');
      setModules(fallbackModules);
      if (!selectedModuleId && onSelectModule) {
        onSelectModule(fallbackModules[0]?.module_id);
      }
    } finally {
      setLoading(false);
    }
  };

  // Load modules on component mount
  useEffect(() => {
    loadModules();
  }, []); // Empty dependency array is correct here

  // Handle module selection
  const handleModuleClick = (moduleId) => {
    console.log('Module clicked:', moduleId);
    if (onSelectModule) {
      onSelectModule(moduleId);
    }
  };

  // Get appropriate icon based on module name
  const getModuleIcon = (moduleName) => {
    const name = moduleName.toLowerCase();
    if (name.includes('performance') || name.includes('tuning')) return '⚡';
    if (name.includes('memory') || name.includes('optimization')) return '🧠';
    if (name.includes('kernel') || name.includes('development')) return '⚙️';
    if (name.includes('basic') || name.includes('introduction')) return '📚';
    return '📁';
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
          <button className="section-action" title="Module options">⋯</button>
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
              className={`folder-item ${selectedModuleId === module.module_id ? 'active' : ''}`}
              onClick={() => handleModuleClick(module.module_id)}
              style={{ cursor: 'pointer' }}
              title={`Click to view ${module.module_name} chats`}
            >
              <span className="folder-icon">
                {getModuleIcon(module.module_name)}
              </span>
              <div className="folder-content" style={{ flex: 1, minWidth: 0 }}>
                <span className="folder-name">{module.module_name}</span>
                {selectedModuleId === module.module_id && (
                  <div style={{ fontSize: '11px', color: '#888', marginTop: '2px' }}>
                    Currently selected
                  </div>
                )}
              </div>
              <div className="module-status">
                <span className={`status-badge ${module.status}`}>
                  {module.status === 'in-progress' ? '🔄' : 
                   module.status === 'completed' ? '✅' : 
                   module.status === 'not-started' ? '⏸️' : '📋'}
                </span>
              </div>
              <button 
                className="item-menu" 
                onClick={(e) => {
                  e.stopPropagation();
                  console.log('Module menu clicked:', module.module_id);
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
            {selectedModuleId ? (
              <>Selected: {modules.find(m => m.module_id === selectedModuleId)?.module_name || 'Unknown'}</>
            ) : (
              <>{modules.filter(m => m.status === 'completed').length} of {modules.length} completed</>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FolderSection;