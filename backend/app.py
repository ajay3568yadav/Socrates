#!/usr/bin/env python3
"""
Main Flask application for CUDA Tutor Backend
Modular version with proper separation of concerns
"""

import sys
import traceback
import requests
import subprocess
import platform
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from flask_cors import CORS

from config import get_config
from utils.cleanup import start_cleanup_service
from models.rag import initialize_rag_system
from compiler.enhanced import EnhancedCodeCompiler
from compiler.base import CodeCompiler

# Get configuration
config = get_config()

print("🔧 Starting Modular Flask Server with RAG and Code Compilation...")

# Create Flask app
try:
    app = Flask(__name__)
    app.config.from_object(config)
    print("✅ Flask app created successfully")
except Exception as e:
    print(f"❌ Error creating Flask app: {e}")
    sys.exit(1)

# Configure CORS
try:
    CORS(app, origins=config.CORS_ORIGINS)
    print("✅ CORS configured successfully")
except Exception as e:
    print(f"❌ Error configuring CORS: {e}")
    sys.exit(1)

# Initialize systems
def initialize_systems():
    """Initialize all backend systems"""
    systems = {}
    
    # Initialize RAG system
    try:
        systems['rag'] = initialize_rag_system()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ RAG system initialization failed: {e}")
        systems['rag'] = None
    
    # Initialize compiler system
    try:
        systems['compiler'] = EnhancedCodeCompiler()
        print("✅ Enhanced code compilation system initialized")
    except Exception as e:
        print(f"❌ Enhanced compiler initialization failed: {e}")
        try:
            # Fallback to basic compiler
            systems['compiler'] = CodeCompiler()
            print("✅ Basic code compilation system initialized as fallback")
        except Exception as e2:
            print(f"❌ Fallback compiler also failed: {e2}")
            systems['compiler'] = None
    
    return systems

# Initialize all systems
app_systems = initialize_systems()

# Register blueprints/routes
def register_routes():
    """Register all application routes"""
    try:
        # Import and register route blueprints
        from routes.chat import create_chat_blueprint
        from routes.compilation import create_compilation_blueprint
        from backend.compiler.dependencies import create_dependencies_blueprint
        from routes.status import create_status_blueprint
        
        # Create blueprints with system dependencies
        chat_bp = create_chat_blueprint(app_systems['rag'])
        compilation_bp = create_compilation_blueprint(app_systems['compiler'])
        dependencies_bp = create_dependencies_blueprint(app_systems['compiler'])
        status_bp = create_status_blueprint(app_systems)
        
        # Register blueprints
        app.register_blueprint(chat_bp, url_prefix='/api')
        app.register_blueprint(compilation_bp, url_prefix='/api')
        app.register_blueprint(dependencies_bp, url_prefix='/api')
        app.register_blueprint(status_bp, url_prefix='/api')
        
        print("✅ All route blueprints registered successfully")
        
    except Exception as e:
        print(f"❌ Error registering routes: {e}")
        traceback.print_exc()
        sys.exit(1)

# Register all routes
register_routes()

# Add root route
@app.route('/')
def index():
    """Main API info endpoint"""
    try:
        return {
            'message': 'CUDA Tutor Backend is running!',
            'status': 'healthy',
            'version': '2.0.0-modular',
            'systems': {
                'rag': 'loaded' if app_systems['rag'] else 'failed',
                'compiler': 'enhanced' if (app_systems['compiler'] and hasattr(app_systems['compiler'], 'dep_manager')) 
                           else ('basic' if app_systems['compiler'] else 'failed')
            },
            'endpoints': {
                'chat': '/api/chat',
                'status': '/api/status',
                'health': '/api/health',
                'compile': '/api/compile',
                'execute': '/api/execute',
                'analyze_dependencies': '/api/analyze-dependencies',
                'install_dependency': '/api/install-dependency'
            }
        }
    except Exception as e:
        print(f"Error in index route: {e}")
        return {'error': str(e)}, 500

# Health check endpoint
@app.route('/health')
def health():
    """Simple health check"""
    try:
        return {
            'status': 'healthy',
            'service': 'CUDA Tutor Backend (Modular)',
            'version': '2.0.0',
            'systems': {
                'rag': app_systems['rag'] is not None,
                'compiler': app_systems['compiler'] is not None,
                'enhanced_compiler': app_systems['compiler'] and hasattr(app_systems['compiler'], 'dep_manager')
            }
        }
    except Exception as e:
        print(f"Error in health route: {e}")
        return {'error': str(e)}, 500

def check_system_requirements():
    """Check and display system requirements"""
    print()
    print(" 🔧 System Requirements Check:")
    
    # Check Ollama
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}", timeout=3)
        print("   ✅ Ollama is running")
    except:
        print("   ❌ Ollama not found - install and run: ollama serve")
    
    # Determine Python command based on OS
    python_cmd = 'python' if platform.system() == 'Windows' else 'python3'
    
    # Check compilers
    compilers_to_check = {
        'gcc': 'C compiler',
        'g++': 'C++ compiler', 
        'nvcc': 'CUDA compiler',
        python_cmd: 'Python interpreter'
    }
    
    for cmd, name in compilers_to_check.items():
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, timeout=3)
            if result.returncode == 0:
                print(f"   ✅ {name} is available")
            else:
                print(f"   ❌ {name} found but may have issues")
        except FileNotFoundError:
            status = "⚠️ " if cmd == 'nvcc' else "❌"
            print(f"   {status} {name} not found")
        except Exception as e:
            print(f"   ❌ Error checking {name}: {e}")

def start_background_services():
    """Start background services"""
    try:
        # Start cleanup service
        cleanup_service = start_cleanup_service(app_systems['compiler'])
        print("✅ Background cleanup service started")
        return cleanup_service
    except Exception as e:
        print(f"⚠️ Could not start background services: {e}")
        return None

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*80)
    print(" 🚀 STARTING MODULAR FLASK SERVER")
    print("="*80)
    print(" 🗣️  Chat & Conversation:")
    print(f"   • http://localhost:{config.PORT}/                    - API Info")
    print(f"   • http://localhost:{config.PORT}/api/chat            - Chat API (with context)")
    print(f"   • http://localhost:{config.PORT}/api/status          - System status")
    print(f"   • http://localhost:{config.PORT}/api/clear-session   - Clear conversation")
    print(f"   • http://localhost:{config.PORT}/health              - Health check")
    print()
    print(" 💻 Code Compilation & Execution:")
    print(f"   • http://localhost:{config.PORT}/api/compile         - Compile code")
    print(f"   • http://localhost:{config.PORT}/api/execute         - Execute compiled code")
    print(f"   • http://localhost:{config.PORT}/api/generate-test   - Generate test scripts")
    print()
    print(" 🔬 Enhanced Dependency Management:")
    print(f"   • http://localhost:{config.PORT}/api/analyze-dependencies - Analyze code dependencies")
    print(f"   • http://localhost:{config.PORT}/api/install-dependency   - Install missing packages")
    print()
    print(" 🎯 System Status:")
    print(f"   ✅ RAG System: {'Loaded' if app_systems['rag'] else 'Failed'}")
    enhanced_status = 'Enhanced' if (app_systems['compiler'] and hasattr(app_systems['compiler'], 'dep_manager')) else ('Basic' if app_systems['compiler'] else 'Failed')
    print(f"   ✅ Compiler: {enhanced_status}")
    
    check_system_requirements()
    
    print()
    print("="*80)
    print(f" 🚀 FLASK SERVER RUNNING ON http://localhost:{config.PORT}")
    print(f" 📊 Visit /api/status to check all system components")
    print(" 🛑 Press Ctrl+C to stop the server")
    print("="*80)

if __name__ == '__main__':
    # Start background services
    background_services = start_background_services()
    
    # Print startup information
    print_startup_info()
    
    try:
        # Start the Flask server
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            use_reloader=False,  # Disable reloader in modular version
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if background_services:
            print("🧹 Cleaning up background services...")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)