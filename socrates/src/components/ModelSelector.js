import React, { useState, useRef, useEffect } from 'react';
import '../css/ModelSelector.css';

const ModelSelector = ({ selectedModel, onModelChange, disabled = false }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const models = [
    {
      id: 'deepseek-r1',
      name: 'DeepSeek R1',
      description: 'Advanced reasoning model',
      icon: '🧠'
    },
    {
      id: 'qwen-2.5',
      name: 'Qwen 2.5',
      description: 'Multilingual language model',
      icon: '🌐'
    },
    {
      id: 'mixtral',
      name: 'Mixtral',
      description: 'Mixture of experts model',
      icon: '⚡'
    },
    {
      id: 'llama-3.2',
      name: 'Llama 3.2',
      description: 'Meta\'s latest language model',
      icon: '🦙'
    }
  ];

  const currentModel = models.find(model => model.id === selectedModel) || models[0];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  // Close dropdown on escape key
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen]);

  const handleToggle = () => {
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  };

  const handleModelSelect = (model) => {
    onModelChange(model.id);
    setIsOpen(false);
  };

  const handleKeyDown = (event, model) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleModelSelect(model);
    }
  };

  return (
    <div className={`model-selector ${disabled ? 'disabled' : ''}`} ref={dropdownRef}>
      <button
        className={`model-selector-button ${isOpen ? 'open' : ''}`}
        onClick={handleToggle}
        disabled={disabled}
        aria-label={`Current model: ${currentModel.name}. Click to select a different model.`}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        title={`Current model: ${currentModel.name} - ${currentModel.description}`}
      >
        <div className="model-selector-current">
          <span className="model-icon">{currentModel.icon}</span>
          <span className="model-name">{currentModel.name}</span>
        </div>
        <svg
          className={`dropdown-arrow ${isOpen ? 'rotated' : ''}`}
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M6 9l6 6 6-6"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {isOpen && (
        <div className="model-selector-dropdown" role="listbox" aria-label="Select a model">
          {models.map((model) => (
            <div
              key={model.id}
              className={`model-option ${model.id === selectedModel ? 'selected' : ''}`}
              onClick={() => handleModelSelect(model)}
              onKeyDown={(e) => handleKeyDown(e, model)}
              role="option"
              aria-selected={model.id === selectedModel}
              tabIndex={0}
            >
              <div className="model-option-content">
                <div className="model-option-header">
                  <span className="model-icon">{model.icon}</span>
                  <span className="model-name">{model.name}</span>
                  {model.id === selectedModel && (
                    <svg
                      className="check-icon"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        d="M20 6L9 17l-5-5"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}
                </div>
                <div className="model-description">{model.description}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelSelector;