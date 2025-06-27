#!/usr/bin/env python3
"""
Code compilation and execution routes
"""

import traceback
from flask import Blueprint, request, jsonify

def create_compilation_blueprint(compiler):
    """Create compilation blueprint with compiler dependency"""
    
    bp = Blueprint('compilation', __name__)
    
    @bp.route('/generate-test', methods=['POST', 'OPTIONS'])
    def generate_test():
        """Generate test script for code using AI"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            code = data.get('code', '')
            language = data.get('language', 'c')
            
            if not code.strip():
                return jsonify({'error': 'No code provided'}), 400
            
            if not compiler:
                return jsonify({
                    'success': False,
                    'error': 'Code compilation system not available',
                    'testScript': '// Compilation system not initialized'
                }), 500
            
            print(f"🧪 Generating test script for {language} code...")
            test_script = compiler.generate_test_script_with_ai(code, language)
            
            return jsonify({
                'success': True,
                'testScript': test_script
            })
            
        except Exception as e:
            print(f"❌ Error generating test script: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'testScript': f'// Error generating test: {str(e)}'
            }), 500
    
    @bp.route('/compile', methods=['POST', 'OPTIONS'])
    def compile_code():
        """Compile code"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            code = data.get('code', '')
            language = data.get('language', 'c')
            test_script = data.get('testScript', '')
            
            if not code.strip():
                return jsonify({'error': 'No code provided'}), 400
            
            if not compiler:
                return jsonify({
                    'success': False,
                    'error': 'Code compilation system not available',
                    'output': '',
                    'stderr': 'Compilation system not initialized'
                }), 500
            
            print(f"🔨 Compiling {language} code...")
            result = compiler.compile_code(code, language, test_script)
            
            print(f"✅ Compilation result: {result['success']}")
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error compiling code: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'output': '',
                'stderr': f'Server error: {str(e)}'
            }), 500
    
    @bp.route('/execute', methods=['POST', 'OPTIONS'])
    def execute_code():
        """Execute compiled code with enhanced dependency checking"""
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
            
            print(f"▶️ Executing code for compilation ID: {compilation_id}")
            
            # Use enhanced execution if available, otherwise fallback to basic
            if hasattr(compiler, 'execute_code_with_mandatory_dependency_check'):
                result = compiler.execute_code_with_mandatory_dependency_check(compilation_id, input_data)
            elif hasattr(compiler, 'execute_code_with_dependency_check'):
                result = compiler.execute_code_with_dependency_check(compilation_id, input_data)
            else:
                result = compiler.execute_code(compilation_id, input_data)
                # Enhance basic errors
                result = _enhance_basic_execution_errors(result)
            
            # Clean up after execution if successful
            if result.get('success', False):
                compiler.cleanup_compilation(compilation_id)
            
            execution_time = result.get('executionTime', 0)
            print(f"✅ Execution result: {result['success']}, time: {execution_time}ms")
            
            # Log if dependency check caught the issue
            if result.get('dependency_check'):
                print("✅ Dependency check prevented execution - showing installation guide")
            
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error executing code: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'output': '',
                'stderr': f'Server error: {str(e)}'
            }), 500
    
    @bp.route('/cleanup-compilation/<compilation_id>', methods=['DELETE', 'OPTIONS'])
    def cleanup_compilation(compilation_id):
        """Clean up compilation files"""
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if compiler:
                success = compiler.cleanup_compilation(compilation_id)
                return jsonify({
                    'success': success,
                    'message': f'Compilation {compilation_id} cleaned up' if success else 'Compilation not found'
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': 'Compiler not available'
                }), 500
        except Exception as e:
            print(f"❌ Error cleaning up compilation: {e}")
            return jsonify({
                'success': False, 
                'error': str(e)
            }), 500
    
    @bp.route('/compilation-info/<compilation_id>')
    def get_compilation_info(compilation_id):
        """Get information about a specific compilation"""
        try:
            if not compiler:
                return jsonify({'error': 'Compiler not available'}), 500
            
            info = compiler.get_compilation_info(compilation_id)
            return jsonify(info)
            
        except Exception as e:
            print(f"❌ Error getting compilation info: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/active-compilations')
    def list_active_compilations():
        """List all active compilations"""
        try:
            if not compiler:
                return jsonify({
                    'count': 0,
                    'compilations': {},
                    'error': 'Compiler not available'
                })
            
            compilations = compiler.list_active_compilations()
            return jsonify(compilations)
            
        except Exception as e:
            print(f"❌ Error listing compilations: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _enhance_basic_execution_errors(result):
        """Enhance error messages for basic compiler execution"""
        stderr = result.get('stderr', '')
        
        # Check for common dependency errors
        if 'ModuleNotFoundError' in stderr and 'pycuda' in stderr:
            result['stderr'] = f"""🚫 PYCUDA NOT INSTALLED

{stderr}

💡 **Quick Fix:**
```bash
pip install pycuda
```

🔧 **Alternative Solutions:**
1. Use CuPy: `pip install cupy-cuda12x`
2. Create conda environment with CUDA support
3. Use CPU-based alternatives for learning

❓ **Need help?** Ask the AI tutor: "How do I install PyCUDA?"
"""
            result['enhanced_error'] = True
        
        return result
    
    return bp