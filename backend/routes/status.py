#!/usr/bin/env python3
"""
Status and health monitoring routes with model information
"""

import subprocess
import requests
import traceback
import time
from flask import Blueprint, jsonify
from config import get_config
from utils.gpu_monitor import get_performance_tracker

config = get_config()

def create_status_blueprint(app_systems):
    """Create status blueprint with system dependencies"""
    
    bp = Blueprint('status', __name__)
    
    @bp.route('/status')
    def status():
        """Comprehensive system status check including model availability"""
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
                'ollama_status': 'unknown',
                'model_support': True,  # NEW: Indicate model selection is supported
                'available_models': []  # NEW: List of available models
            }
            
            # Test Ollama connection
            try:
                response = requests.get(config.OLLAMA_BASE_URL, timeout=5)
                status_info['ollama_status'] = 'connected' if response.status_code == 200 else 'error'
                status_info['ollama_url'] = config.OLLAMA_BASE_URL
                
                # Check available models in Ollama - NEW
                if response.status_code == 200:
                    models_response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                    if models_response.status_code == 200:
                        ollama_models = models_response.json().get('models', [])
                        installed_model_names = [m['name'] for m in ollama_models]
                        
                        # Check each configured model
                        model_configs = config.get_model_configs()
                        for model_id, model_config in model_configs.items():
                            model_available = any(
                                model_config['ollama_model'].split(':')[0] in installed_name 
                                for installed_name in installed_model_names
                            )
                            
                            fallback_available = False
                            if model_config.get('fallback'):
                                fallback_available = any(
                                    model_config['fallback'].split(':')[0] in installed_name 
                                    for installed_name in installed_model_names
                                )
                            
                            status_info['available_models'].append({
                                'id': model_id,
                                'name': model_config['name'],
                                'available': model_available,
                                'fallback_available': fallback_available,
                                'ollama_model': model_config['ollama_model'],
                                'fallback_model': model_config.get('fallback')
                            })
                        
                        status_info['ollama_models_count'] = len(installed_model_names)
                        status_info['configured_models_count'] = len(model_configs)
                        
            except requests.exceptions.ConnectionError:
                status_info['ollama_status'] = 'disconnected'
                status_info['available_models'] = []
            except Exception as e:
                status_info['ollama_status'] = f'error: {str(e)}'
                status_info['available_models'] = []
            
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
            
            # Add GPU monitoring information
            try:
                performance_tracker = get_performance_tracker()
                gpu_info = performance_tracker.gpu_monitor.get_gpu_info()
                system_metrics = performance_tracker.system_monitor.get_system_metrics()
                
                status_info['gpu_monitoring'] = {
                    'cuda_available': gpu_info['cuda_available'],
                    'gpu_count': gpu_info['gpu_count'],
                    'gpu_model': gpu_info['gpu_model'],
                    'monitoring_method': gpu_info['monitoring_method'],
                    'gpu_memory_used_gb': gpu_info['gpu_memory_used_gb'],
                    'gpu_memory_total_gb': gpu_info['gpu_memory_total_gb'],
                    'gpu_memory_percent': gpu_info['gpu_memory_percent'],
                    'gpu_utilization_percent': gpu_info['gpu_utilization_percent']
                }
                
                status_info['system_monitoring'] = {
                    'memory_gb': system_metrics['memory_gb'],
                    'memory_percent': system_metrics['memory_percent'],
                    'cpu_percent': system_metrics['cpu_percent'],
                    'average_memory_percent': performance_tracker.system_monitor.get_average_memory_usage(),
                    'average_cpu_percent': performance_tracker.system_monitor.get_average_cpu_usage()
                }
            except Exception as e:
                print(f"⚠️ Error getting GPU/system monitoring info: {e}")
                status_info['gpu_monitoring'] = {'error': str(e)}
                status_info['system_monitoring'] = {'error': str(e)}
            
            # Add configuration info with model settings - NEW
            status_info['configuration'] = {
                'compilation_timeout': config.COMPILATION_TIMEOUT,
                'execution_timeout': config.EXECUTION_TIMEOUT,
                'max_output_size': config.MAX_OUTPUT_SIZE,
                'temp_dir': str(config.TEMP_DIR),
                'debug_mode': config.DEBUG,
                'default_model': config.DEFAULT_MODEL,
                'model_fallback_enabled': config.ENABLE_MODEL_FALLBACK,
                'model_timeout': config.MODEL_TIMEOUT
            }
            
            return jsonify(status_info)
            
        except Exception as e:
            print(f"❌ Error in status route: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/models', methods=['GET'])
    def get_models():
        """Get available AI models - NEW ENDPOINT"""
        try:
            model_configs = config.get_model_configs()
            models = []
            
            for model_id, model_config in model_configs.items():
                models.append({
                    'id': model_id,
                    'name': model_config['name'],
                    'description': model_config['description'],
                    'max_tokens': model_config['max_tokens'],
                    'temperature': model_config['temperature'],
                    'ollama_model': model_config['ollama_model'],
                    'has_fallback': model_config.get('fallback') is not None
                })
            
            return jsonify({
                'models': models,
                'default_model': config.DEFAULT_MODEL,
                'total_models': len(models),
                'fallback_enabled': config.ENABLE_MODEL_FALLBACK
            })
            
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/model/status/<model_id>', methods=['GET'])
    def check_model_status(model_id):
        """Check if a specific model is available - NEW ENDPOINT"""
        try:
            model_configs = config.get_model_configs()
            
            if model_id not in model_configs:
                return jsonify({
                    'available': False,
                    'error': 'Model not found in configuration'
                }), 404
            
            model_config = model_configs[model_id]
            
            # Try to check if model is available in Ollama
            try:
                response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                
                if response.status_code == 200:
                    models_data = response.json()
                    installed_models = [m['name'] for m in models_data.get('models', [])]
                    
                    # Check if the specific model is installed
                    model_available = any(
                        model_config['ollama_model'].split(':')[0] in installed_model 
                        for installed_model in installed_models
                    )
                    
                    # Check fallback model if main model not available
                    fallback_available = False
                    if not model_available and model_config.get('fallback'):
                        fallback_available = any(
                            model_config['fallback'].split(':')[0] in installed_model 
                            for installed_model in installed_models
                        )
                    
                    return jsonify({
                        'model_id': model_id,
                        'model_name': model_config['name'],
                        'available': model_available,
                        'fallback_available': fallback_available,
                        'ollama_model': model_config['ollama_model'],
                        'fallback_model': model_config.get('fallback'),
                        'installed_models': installed_models
                    })
                else:
                    return jsonify({
                        'available': False,
                        'error': 'Ollama service not responding'
                    })
                    
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'available': False,
                    'error': f'Connection error: {str(e)}'
                })
                
        except Exception as e:
            print(f"❌ Error checking model status: {e}")
            return jsonify({
                'available': False,
                'error': str(e)
            }), 500
    
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
                    'enhanced_compiler': compiler and hasattr(compiler, 'dep_manager'),
                    'model_selection': True  # NEW: Indicate model selection support
                },
                'default_model': config.DEFAULT_MODEL  # NEW
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
                    'temp_directory': str(config.TEMP_DIR),
                    'model_selection_enabled': True,  # NEW
                    'available_models': len(config.get_model_configs())  # NEW
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
    
    @bp.route('/gpu-debug')
    def gpu_debug():
        """Detailed GPU debugging information"""
        try:
            performance_tracker = get_performance_tracker()
            debug_info = performance_tracker.gpu_monitor.get_detailed_gpu_debug()
            
            return jsonify(debug_info)
            
        except Exception as e:
            print(f"❌ Error getting GPU debug info: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/performance-metrics')
    def performance_metrics():
        """Get current performance metrics without processing a prompt"""
        try:
            performance_tracker = get_performance_tracker()
            
            # Get current metrics
            system_metrics = performance_tracker.system_monitor.get_system_metrics()
            gpu_info = performance_tracker.gpu_monitor.get_gpu_info()
            
            # Get averages
            avg_cpu = performance_tracker.system_monitor.get_average_cpu_usage()
            avg_memory = performance_tracker.system_monitor.get_average_memory_usage()
            
            metrics = {
                'timestamp': time.time(),
                'system_metrics': {
                    'current_memory_gb': system_metrics['memory_gb'],
                    'current_memory_percent': system_metrics['memory_percent'],
                    'average_memory_percent': avg_memory,
                    'current_cpu_percent': system_metrics['cpu_percent'],
                    'average_cpu_percent': avg_cpu
                },
                'gpu_metrics': gpu_info
            }
            
            return jsonify(metrics)
            
        except Exception as e:
            print(f"❌ Error getting performance metrics: {e}")
            return jsonify({'error': str(e)}), 500
    
    return bp