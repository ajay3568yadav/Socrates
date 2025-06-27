#!/usr/bin/env python3
"""
Configuration management for CUDA Tutor Backend
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
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))
    
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

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    # Add production-specific settings here

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    COMPILATION_TIMEOUT = 5  # Shorter timeouts for testing
    EXECUTION_TIMEOUT = 3

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