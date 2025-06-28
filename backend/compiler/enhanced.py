#!/usr/bin/env python3
"""
Enhanced code compiler with dependency management
"""

from typing import Dict, Any
from .base import CodeCompiler
from ..routes.dependencies import DependencyManager

class EnhancedCodeCompiler(CodeCompiler):
    """Enhanced compiler with dependency checking and management"""
    
    def __init__(self):
        super().__init__()
        self.dep_manager = DependencyManager()
        print("✅ Enhanced CodeCompiler with dependency management initialized")
    
    def analyze_and_prepare_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code for dependencies and provide setup guidance"""
        missing_deps = self.dep_manager.get_missing_dependencies(code, language)
        
        if missing_deps:
            installation_guide = self.dep_manager.create_installation_guide(missing_deps)
            return {
                'has_missing_deps': True,
                'missing_dependencies': missing_deps,
                'installation_guide': installation_guide,
                'can_execute': False
            }
        else:
            return {
                'has_missing_deps': False,
                'missing_dependencies': [],
                'installation_guide': '',
                'can_execute': True
            }
    
    def execute_code_with_dependency_check(self, compilation_id: str, input_data: str = "") -> Dict[str, Any]:
        """Execute code with enhanced dependency checking"""
        if compilation_id not in self.active_compilations:
            return {
                'success': False,
                'output': '',
                'stderr': 'Compilation ID not found',
                'exitCode': -1
            }
        
        comp_info = self.active_compilations[compilation_id]
        
        # Read the source code to check dependencies
        try:
            with open(comp_info['source_file'], 'r') as f:
                source_code = f.read()
            
            # Check dependencies
            dep_analysis = self.analyze_and_prepare_code(source_code, comp_info['language'])
            
            if dep_analysis['has_missing_deps']:
                # Return helpful error message with installation guide
                error_msg = self._create_dependency_error_message(dep_analysis)
                
                return {
                    'success': False,
                    'output': '',
                    'stderr': error_msg,
                    'exitCode': -1,
                    'missing_dependencies': dep_analysis['missing_dependencies'],
                    'installation_guide': dep_analysis['installation_guide'],
                    'dependency_check': True
                }
        except Exception as e:
            print(f"Error checking dependencies: {e}")
        
        # If no dependency issues, proceed with normal execution
        return super().execute_code(compilation_id, input_data)
    
    def execute_code_with_mandatory_dependency_check(self, compilation_id: str, input_data: str = "") -> Dict[str, Any]:
        """Execute code with MANDATORY dependency checking - enhanced version"""
        
        if compilation_id not in self.active_compilations:
            return {
                'success': False,
                'output': '',
                'stderr': 'Compilation ID not found',
                'exitCode': -1
            }
        
        comp_info = self.active_compilations[compilation_id]
        
        # ALWAYS check dependencies before execution
        try:
            with open(comp_info['source_file'], 'r') as f:
                source_code = f.read()
            
            # Check for missing dependencies
            missing_deps = self.dep_manager.get_missing_dependencies(source_code, comp_info['language'])
            
            if missing_deps:
                # Create installation guide
                installation_guide = self.dep_manager.create_installation_guide(missing_deps)
                
                # Create user-friendly error message
                error_msg = self._create_enhanced_dependency_error_message(missing_deps, installation_guide)
                
                return {
                    'success': False,
                    'output': '',
                    'stderr': error_msg,
                    'exitCode': -1,
                    'missing_dependencies': missing_deps,
                    'installation_guide': installation_guide,
                    'dependency_check': True
                }
                
        except Exception as e:
            print(f"Error checking dependencies: {e}")
            # If dependency check fails, still show a helpful message
            if comp_info['language'] == 'python' and 'pycuda' in source_code.lower():
                return {
                    'success': False,
                    'output': '',
                    'stderr': self._create_fallback_cuda_error_message(str(e)),
                    'exitCode': -1,
                    'dependency_check': False
                }
        
        # If no dependency issues, proceed with normal execution but enhance errors
        result = super().execute_code(compilation_id, input_data)
        
        # Enhance error messages for common dependency issues
        if not result.get('success', True) and result.get('stderr'):
            result = self._enhance_execution_error_message(result, comp_info)
        
        return result
    
    def _create_dependency_error_message(self, dep_analysis: Dict[str, Any]) -> str:
        """Create a user-friendly dependency error message"""
        missing_deps = dep_analysis['missing_dependencies']
        installation_guide = dep_analysis['installation_guide']
        
        error_msg = "🚫 MISSING DEPENDENCIES DETECTED!\n\n"
        error_msg += "Your code requires packages that aren't installed on this system.\n\n"
        error_msg += installation_guide
        
        return error_msg
    
    def _create_enhanced_dependency_error_message(self, missing_deps, installation_guide: str) -> str:
        """Create an enhanced user-friendly error message"""
        
        # Check for critical dependencies
        has_cuda = any(dep['name'] in ['pycuda', 'cupy', 'nvcc'] for dep in missing_deps)
        has_python_packages = any(dep['type'] == 'python' for dep in missing_deps)
        
        error_msg = """🚫 MISSING DEPENDENCIES DETECTED!

Your code requires packages that aren't installed on this system.

"""
        
        error_msg += installation_guide
        
        if has_cuda:
            error_msg += """

💡 QUICK CUDA SOLUTIONS:

1. **Install PyCUDA (if you have NVIDIA GPU):**
   ```bash
   pip install pycuda
   ```

2. **Use CuPy (easier alternative):**
   ```bash
   pip install cupy-cuda12x
   ```

3. **Conda Environment (recommended):**
   ```bash
   conda create -n cuda_dev python=3.9
   conda activate cuda_dev
   conda install -c conda-forge pycuda
   ```

4. **For macOS users:**
   - PyCUDA requires NVIDIA GPU (not available on Apple Silicon)
   - Consider using simulation mode or CPU alternatives

🎯 **For Learning Purposes:**
You can modify your code to use NumPy for CPU-based learning:

```python
# Replace PyCUDA imports with:
import numpy as np
print("Running in CPU simulation mode")
```
"""
        
        error_msg += "\n❓ **Need Help?** Ask the AI tutor: \"How can I run CUDA code without a GPU?\""
        
        return error_msg
    
    def _create_fallback_cuda_error_message(self, error: str) -> str:
        """Create fallback error message for CUDA issues"""
        return f"""🚫 EXECUTION FAILED - Likely Missing Dependencies

Error: {error}

This appears to be a Python script that uses CUDA libraries.

💡 **Try installing PyCUDA:**
```bash
pip install pycuda
```

Or ask the AI tutor for help with CUDA setup!"""
    
    def _enhance_execution_error_message(self, result: Dict[str, Any], comp_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance error messages for common execution issues"""
        stderr = result.get('stderr', '')
        
        # Check for common dependency errors
        if 'ModuleNotFoundError' in stderr:
            if 'pycuda' in stderr:
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

❓ **Need help?** Ask: "How do I install PyCUDA on macOS?"
"""
                result['enhanced_error'] = True
            
            elif 'cupy' in stderr:
                result['stderr'] = f"""🚫 CUPY NOT INSTALLED

{stderr}

💡 **Quick Fix:**
```bash
pip install cupy-cuda12x
```

For CUDA 11.x users:
```bash
pip install cupy-cuda11x
```
"""
                result['enhanced_error'] = True
            
            elif any(pkg in stderr.lower() for pkg in ['numpy', 'torch', 'tensorflow']):
                # Extract package name
                import re
                match = re.search(r"No module named '(\w+)'", stderr)
                if match:
                    package = match.group(1)
                    result['stderr'] = f"""🚫 MISSING PYTHON PACKAGE: {package.upper()}

{stderr}

💡 **Quick Fix:**
```bash
pip install {package}
```
"""
                    result['enhanced_error'] = True
        
        elif 'CUDA' in stderr and 'error' in stderr.lower():
            result['stderr'] = f"""🚫 CUDA RUNTIME ERROR

{stderr}

💡 **Possible Solutions:**
1. Check if NVIDIA drivers are installed
2. Verify CUDA toolkit installation
3. Try running with CPU fallback
4. Ask the AI tutor: "How do I fix CUDA runtime errors?"
"""
            result['enhanced_error'] = True
        
        return result
    
    def create_conda_environment_script(self, missing_deps) -> str:
        """Create a conda environment setup script"""
        return self.dep_manager.create_conda_environment_script(missing_deps)
    
    def install_missing_dependency(self, package_name: str) -> Dict[str, Any]:
        """Install a missing Python dependency"""
        success, stdout, stderr = self.dep_manager.install_python_package(package_name)
        
        return {
            'success': success,
            'package': package_name,
            'output': stdout,
            'error': stderr,
            'message': f"{'✅ Successfully installed' if success else '❌ Failed to install'} {package_name}"
        }
    
    def get_dependency_status(self, code: str, language: str) -> Dict[str, Any]:
        """Get comprehensive dependency status for code"""
        missing_deps = self.dep_manager.get_missing_dependencies(code, language)
        
        # Get available packages/compilers
        available_python = []
        available_system = []
        
        for package in self.dep_manager.python_packages.keys():
            if self.dep_manager.check_python_package(package):
                available_python.append(package)
        
        for command in self.dep_manager.system_packages.keys():
            if self.dep_manager.check_system_command(command):
                available_system.append(command)
        
        return {
            'missing_dependencies': missing_deps,
            'available_python_packages': available_python,
            'available_system_commands': available_system,
            'can_execute': len(missing_deps) == 0,
            'requires_installation': len(missing_deps) > 0
        }