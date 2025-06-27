#!/usr/bin/env python3
"""
Status and health monitoring routes
"""

import subprocess
import requests
import traceback
from flask import Blueprint, jsonify
from config import get_config

config = get_config()

def create_status_blueprint(app_systems):
    """Create status blueprint with system dependencies"""
    
    bp = Blueprint('status', __name__)
    
    @bp.route('/status')
    def status():
        """Comprehensive system status check"""
        try:
            rag_system = app_systems.get('rag')
            compiler = app_systems.get('compiler')
            
            status_info = {
                'service': 'CUDA Tutor Backend',
                'version': '2.0.0-modular',
                'rag_loaded': rag_system is not None,
                'knowledge_count': len(rag_system.knowledge) if rag_system else 0,
                'compiler_available': compiler is not None,
                'enhanced_compiler': hasattr(compiler, 'dep_manager') if compiler else False,
                'active_compilations': len(compiler.active_compilations) if compiler else 0,
                'ollama_status': 'unknown'
            }
            
            # Test Ollama connection
            try:
                response = requests.get(config.OLLAMA_BASE_URL, timeout=5)
                status_info['ollama_status'] = 'connected' if response.status_code == 200 else 'error'
                status_info['ollama_url'] = config.OLLAMA_BASE_URL
            except requests.exceptions.ConnectionError:
                status_info['ollama_status'] = 'disconnected'
            except Exception as e:
                status_info['ollama_status'] = f'error: {str(e)}'
            
            # Check compilers
            if compiler:
                compiler_status = {}
                for lang in ['c', 'cpp', 'cuda', 'python']:
                    config_info = compiler.get_compiler_config(lang)
                    if config_info['compiler']:
                        try:
                            subprocess.run([config_info['compiler'], '--version'], 
                                         capture_output=True, timeout=3)
                            compiler_status[lang] = 'available'
                        except (subprocess.TimeoutExpired, FileNotFoundError):
                            compiler_status[lang] = 'not_found'
                        except Exception:
                            compiler_status[lang] = 'error'
                    else:
                        compiler_status[lang] = 'interpreter'
                
                status_info['compilers'] = compiler_status
            
            # Add configuration info
            status_info['configuration'] = {
                'compilation_timeout': config.COMPILATION_TIMEOUT,
                'execution_timeout': config.EXECUTION_TIMEOUT,
                'max_output_size': config.MAX_OUTPUT_SIZE,
                'temp_dir': str(config.TEMP_DIR),
                'debug_mode': config.DEBUG
            }
            
            return jsonify(status_info)
            
        except Exception as e:
            print(f"❌ Error in status route: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/health')
    def health():
        """Simple health check endpoint"""
        try:
            rag_system = app_systems.get('rag')
            compiler = app_systems.get('compiler')
            
            return jsonify({
                'status': 'healthy',
                'service': 'CUDA Tutor Backend (Modular)',
                'version': '2.0.0',
                'timestamp': int(__import__('time').time()),
                'systems': {
                    'rag': rag_system is not None,
                    'compiler': compiler is not None,
                    'enhanced_compiler': compiler and hasattr(compiler, 'dep_manager')
                }
            })
            
        except Exception as e:
            print(f"❌ Error in health route: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500
    
    @bp.route('/system-info')
    def system_info():
        """Detailed system information"""
        try:
            import platform
            import psutil
            import sys
            
            rag_system = app_systems.get('rag')
            compiler = app_systems.get('compiler')
            
            info = {
                'platform': {
                    'system': platform.system(),
                    'platform': platform.platform(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'python_version': sys.version
                },
                'resources': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                    'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2)
                },
                'cuda_tutor': {
                    'rag_system': 'loaded' if rag_system else 'failed',
                    'compiler_system': 'enhanced' if (compiler and hasattr(compiler, 'dep_manager')) 
                                    else ('basic' if compiler else 'failed'),
                    'active_compilations': len(compiler.active_compilations) if compiler else 0,
                    'temp_directory': str(config.TEMP_DIR)
                }
            }
            
            # Check CUDA availability
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['cuda'] = {
                        'nvidia_smi_available': True,
                        'driver_info': result.stdout.split('\n')[2:4]  # First few lines with driver info
                    }
                else:
                    info['cuda'] = {'nvidia_smi_available': False}
            except (subprocess.TimeoutExpired, FileNotFoundError):
                info['cuda'] = {'nvidia_smi_available': False, 'note': 'NVIDIA drivers may not be installed'}
            
            return jsonify(info)
            
        except Exception as e:
            print(f"❌ Error getting system info: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/compilation-stats')
    def compilation_stats():
        """Get compilation system statistics"""
        try:
            compiler = app_systems.get('compiler')
            
            if not compiler:
                return jsonify({
                    'available': False,
                    'error': 'Compiler system not loaded'
                }), 500
            
            stats = {
                'available': True,
                'enhanced': hasattr(compiler, 'dep_manager'),
                'active_compilations': compiler.list_active_compilations(),
                'supported_languages': ['c', 'cpp', 'cuda', 'python'],
                'configuration': {
                    'compilation_timeout': config.COMPILATION_TIMEOUT,
                    'execution_timeout': config.EXECUTION_TIMEOUT,
                    'max_output_size': config.MAX_OUTPUT_SIZE
                }
            }
            
            # Add dependency manager stats if available
            if hasattr(compiler, 'dep_manager'):
                dep_manager = compiler.dep_manager
                stats['dependency_manager'] = {
                    'python_packages_tracked': len(dep_manager.python_packages),
                    'system_packages_tracked': len(dep_manager.system_packages)
                }
            
            return jsonify(stats)
            
        except Exception as e:
            print(f"❌ Error getting compilation stats: {e}")
            return jsonify({'error': str(e)}), 500
    
    return bp