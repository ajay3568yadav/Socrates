import React, { useState, useRef, useEffect, useCallback } from 'react';
import '../css/SplitPaneLayout.css';

const SplitPaneLayout = ({ 
  leftPane, 
  rightPane, 
  initialLeftWidth = 60,
  minLeftWidth = 30,
  maxLeftWidth = 80,
  onWidthChange
}) => {
  const [leftWidth, setLeftWidth] = useState(initialLeftWidth);
  const [isResizing, setIsResizing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);
  const resizerRef = useRef(null);

  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    setIsResizing(true);
    setIsDragging(true);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!isResizing || !containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;
    
    // Constrain the width
    const constrainedWidth = Math.min(Math.max(newLeftWidth, minLeftWidth), maxLeftWidth);
    
    setLeftWidth(constrainedWidth);
    
    if (onWidthChange) {
      onWidthChange(constrainedWidth);
    }
  }, [isResizing, minLeftWidth, maxLeftWidth, onWidthChange]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
    setIsDragging(false);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp]);

  // Handle double-click to reset to default
  const handleDoubleClick = () => {
    setLeftWidth(initialLeftWidth);
    if (onWidthChange) {
      onWidthChange(initialLeftWidth);
    }
  };

  return (
    <div 
      ref={containerRef}
      className={`split-pane-container ${isResizing ? 'resizing' : ''}`}
    >
      {/* Left Pane - Chat */}
      <div 
        className="split-pane-left"
        style={{ width: `${leftWidth}%` }}
      >
        {leftPane}
      </div>

      {/* Resizer */}
      <div 
        ref={resizerRef}
        className={`split-pane-resizer ${isDragging ? 'dragging' : ''}`}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        title="Drag to resize • Double-click to reset"
      >
        <div className="resizer-handle">
          <div className="resizer-line"></div>
          <div className="resizer-dots">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
          <div className="resizer-line"></div>
        </div>
      </div>

      {/* Right Pane - Code Editor */}
      <div 
        className="split-pane-right"
        style={{ width: `${100 - leftWidth}%` }}
      >
        {rightPane}
      </div>
    </div>
  );
};

export default SplitPaneLayout;