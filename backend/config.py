#!/usr/bin/env python3
"""
Configuration management for CUDA Tutor Backend with Model Selection
"""

import os
from pathlib import Path
import tempfile

class Config:
    """Base configuration"""
    
    # Flask settings
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5001))
    
    # CORS settings
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:3001"
    ]
    
    # Compilation settings
    COMPILATION_TIMEOUT = int(os.getenv('COMPILATION_TIMEOUT', 30))  # seconds
    EXECUTION_TIMEOUT = int(os.getenv('EXECUTION_TIMEOUT', 10))     # seconds
    MAX_OUTPUT_SIZE = int(os.getenv('MAX_OUTPUT_SIZE', 10000))      # characters
    
    # Temporary directory for compilations
    TEMP_DIR = Path(tempfile.gettempdir()) / "cuda_tutor_compile"
    
    # Session management
    MAX_SESSIONS = int(os.getenv('MAX_SESSIONS', 50))
    SESSION_CLEANUP_HOURS = int(os.getenv('SESSION_CLEANUP_HOURS', 1))
    MAX_EXCHANGES_PER_SESSION = int(os.getenv('MAX_EXCHANGES_PER_SESSION', 10))
    CONVERSATION_CONTEXT_EXCHANGES = int(os.getenv('CONVERSATION_CONTEXT_EXCHANGES', 3))
    
    # Ollama/AI settings
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')  # Default fallback model
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))
    
    # Model Selection Settings - NEW
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'deepseek-r1')
    ENABLE_MODEL_FALLBACK = os.getenv('ENABLE_MODEL_FALLBACK', 'True').lower() == 'true'
    MODEL_TIMEOUT = int(os.getenv('MODEL_TIMEOUT', 120))
    
    # Model-specific settings - NEW
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:latest')
    QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen2.5:latest')
    MIXTRAL_MODEL = os.getenv('MIXTRAL_MODEL', 'mixtral:latest')
    LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama3.2:latest')
    
    # RAG settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    DATASET_LIMIT = int(os.getenv('DATASET_LIMIT', 50))
    RAG_TOP_K = int(os.getenv('RAG_TOP_K', 2))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', None)  # None means log to console only
    
    # Cleanup settings
    CLEANUP_INTERVAL_HOURS = int(os.getenv('CLEANUP_INTERVAL_HOURS', 1))
    COMPILATION_CLEANUP_HOURS = int(os.getenv('COMPILATION_CLEANUP_HOURS', 1))
    
    # Model Configuration Helper - NEW
    @classmethod
    def get_model_configs(cls):
        """Get model configurations for all available models"""
        return {
            'deepseek-r1': {
                'name': 'DeepSeek R1',
                'ollama_model': cls.DEEPSEEK_MODEL,
                'max_tokens': 4096,
                'temperature': 0.7,
                'description': 'Advanced reasoning model optimized for complex problem solving',
                'fallback': cls.LLAMA_MODEL if cls.ENABLE_MODEL_FALLBACK else None
            },
            'qwen-2.5': {
                'name': 'Qwen 2.5',
                'ollama_model': cls.QWEN_MODEL,
                'max_tokens': 4096,
                'temperature': 0.7,
                'description': 'Multilingual language model with strong performance across languages',
                'fallback': cls.LLAMA_MODEL if cls.ENABLE_MODEL_FALLBACK else None
            },
            'mixtral': {
                'name': 'Mixtral',
                'ollama_model': cls.MIXTRAL_MODEL,
                'max_tokens': 4096,
                'temperature': 0.7,
                'description': 'Mixture of experts model for efficient and powerful responses',
                'fallback': cls.LLAMA_MODEL if cls.ENABLE_MODEL_FALLBACK else None
            },
            'llama-3.2': {
                'name': 'Llama 3.2',
                'ollama_model': cls.LLAMA_MODEL,
                'max_tokens': 4096,
                'temperature': 0.7,
                'description': 'Meta\'s latest open-source language model',
                'fallback': None  # This is our fallback model
            }
        }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Enable model fallback in development
    ENABLE_MODEL_FALLBACK = True
    
    # Shorter timeouts for development
    MODEL_TIMEOUT = 60
    OLLAMA_TIMEOUT = 60

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production-specific model settings
    ENABLE_MODEL_FALLBACK = True
    MODEL_TIMEOUT = 180
    OLLAMA_TIMEOUT = 180
    
    # Longer session limits for production
    MAX_SESSIONS = 100
    MAX_EXCHANGES_PER_SESSION = 20
    CONVERSATION_CONTEXT_EXCHANGES = 5

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    
    # Shorter timeouts for testing
    COMPILATION_TIMEOUT = 5
    EXECUTION_TIMEOUT = 3
    MODEL_TIMEOUT = 10
    OLLAMA_TIMEOUT = 10
    
    # Disable model fallback in testing for predictable behavior
    ENABLE_MODEL_FALLBACK = False
    
    # smaller limits for testing
    DATASET_LIMIT = 10
    MAX_SESSIONS = 10

# Configuration mapping
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    return config_mapping.get(config_name, DevelopmentConfig)