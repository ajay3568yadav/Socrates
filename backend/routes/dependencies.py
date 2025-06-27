#!/usr/bin/env python3
"""
Enhanced dependency management routes
"""

import traceback
from flask import Blueprint, request, jsonify

def create_dependencies_blueprint(compiler):
    """Create dependencies blueprint with compiler dependency"""
    
    bp = Blueprint('dependencies', __name__)
    
    @bp.route('/analyze-dependencies', methods=['POST', 'OPTIONS'])
    def analyze_dependencies():
        """Analyze code dependencies and provide installation guidance"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            code = data.get('code', '')
            language = data.get('language', 'python')
            
            if not code.strip():
                return jsonify({'error': 'No code provided'}), 400
            
            if not compiler or not hasattr(compiler, 'analyze_and_prepare_code'):
                return jsonify({
                    'success': False,
                    'error': 'Enhanced compilation system not available',
                    'analysis': {'has_missing_deps': False, 'can_execute': True}
                }), 500
            
            print(f"🔍 Analyzing dependencies for {language} code...")
            analysis = compiler.analyze_and_prepare_code(code, language)
            
            return jsonify({
                'success': True,
                'analysis': analysis
            })
            
        except Exception as e:
            print(f"❌ Error analyzing dependencies: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'analysis': {'has_missing_deps': False, 'can_execute': True}
            }), 500
    
    @bp.route('/install-dependency', methods=['POST', 'OPTIONS'])
    def install_dependency():
        """Attempt to install a missing dependency"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            package_name = data.get('package_name', '')
            
            if not package_name:
                return jsonify({'error': 'No package name provided'}), 400
            
            if not compiler or not hasattr(compiler, 'dep_manager'):
                return jsonify({
                    'success': False,
                    'error': 'Dependency manager not available',
                    'output': '',
                    'stderr': 'Enhanced compilation system not loaded'
                }), 500
            
            print(f"📦 Installing dependency: {package_name}")
            success, stdout, stderr = compiler.dep_manager.install_python_package(package_name)
            
            return jsonify({
                'success': success,
                'package': package_name,
                'output': stdout,
                'error': stderr if not success else '',
                'message': f"{'✅ Successfully installed' if success else '❌ Failed to install'} {package_name}"
            })
            
        except Exception as e:
            print(f"❌ Error installing dependency: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'output': '',
                'stderr': f'Installation error: {str(e)}'
            }), 500
    
    @bp.route('/create-conda-script', methods=['POST', 'OPTIONS'])
    def create_conda_script():
        """Create a conda environment setup script"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            missing_deps = data.get('missing_deps', [])
            
            if not compiler or not hasattr(compiler, 'create_conda_environment_script'):
                return jsonify({
                    'success': False,
                    'error': 'Conda script generation not available'
                }), 500
            
            script = compiler.create_conda_environment_script(missing_deps)
            
            return jsonify({
                'success': True,
                'script': script,
                'filename': 'setup_cuda_environment.sh'
            })
            
        except Exception as e:
            print(f"❌ Error creating conda script: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @bp.route('/dependency-status', methods=['POST', 'OPTIONS'])
    def get_dependency_status():
        """Get comprehensive dependency status for code"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            code = data.get('code', '')
            language = data.get('language', 'python')
            
            if not code.strip():
                return jsonify({'error': 'No code provided'}), 400
            
            if not compiler or not hasattr(compiler, 'get_dependency_status'):
                # Fallback for basic compiler
                return jsonify({
                    'missing_dependencies': [],
                    'available_python_packages': [],
                    'available_system_commands': [],
                    'can_execute': True,
                    'requires_installation': False,
                    'fallback': True
                })
            
            status = compiler.get_dependency_status(code, language)
            
            return jsonify(status)
            
        except Exception as e:
            print(f"❌ Error getting dependency status: {e}")
            return jsonify({
                'error': str(e),
                'can_execute': False
            }), 500
    
    @bp.route('/execute-enhanced', methods=['POST', 'OPTIONS'])
    def execute_code_enhanced():
        """Execute code with dependency checking"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            compilation_id = data.get('compilationId', '')
            input_data = data.get('input', '')
            
            if not compilation_id:
                return jsonify({'error': 'No compilation ID provided'}), 400
            
            if not compiler:
                return jsonify({
                    'success': False,
                    'error': 'Code execution system not available',
                    'output': '',
                    'stderr': 'Execution system not initialized'
                }), 500
            
            print(f"🚀 Enhanced execution for compilation ID: {compilation_id}")
            
            # Use enhanced execution if available, otherwise fallback to basic
            if hasattr(compiler, 'execute_code_with_dependency_check'):
                result = compiler.execute_code_with_dependency_check(compilation_id, input_data)
            else:
                result = compiler.execute_code(compilation_id, input_data)
                result['enhanced'] = False
            
            # Clean up after execution if successful
            if result.get('success', False):
                compiler.cleanup_compilation(compilation_id)
            
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error in enhanced execution: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'output': '',
                'stderr': f'Server error: {str(e)}'
            }), 500
    
    return bp