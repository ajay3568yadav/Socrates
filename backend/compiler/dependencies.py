#!/usr/bin/env python3
"""
Dependency management for code compilation and execution
"""

import subprocess
import sys
from typing import List, Dict, Any

class DependencyManager:
    """Manages and installs dependencies for code execution"""
    
    def __init__(self):
        self.python_packages = {
            'pycuda': 'pip install pycuda',
            'cupy': 'pip install cupy-cuda12x',  # For CUDA 12.x
            'numba': 'pip install numba',
            'numpy': 'pip install numpy',
            'torch': 'pip install torch',
            'tensorflow': 'pip install tensorflow',
            'scipy': 'pip install scipy',
            'matplotlib': 'pip install matplotlib',
            'pandas': 'pip install pandas'
        }
        
        self.system_packages = {
            'nvcc': 'NVIDIA CUDA Toolkit',
            'gcc': 'GNU C Compiler',
            'g++': 'GNU C++ Compiler',
            'clang': 'Clang Compiler',
            'make': 'GNU Make'
        }
    
    def check_python_package(self, package_name: str) -> bool:
        """Check if a Python package is installed"""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    def check_system_command(self, command: str) -> bool:
        """Check if a system command/compiler is available"""
        try:
            result = subprocess.run(
                [command, '--version'], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    def install_python_package(self, package_name: str) -> tuple[bool, str, str]:
        """Install a Python package"""
        if package_name not in self.python_packages:
            return False, "", f"Unknown package: {package_name}"
        
        install_cmd = self.python_packages[package_name]
        try:
            print(f"Installing {package_name}...")
            result = subprocess.run(
                install_cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Installation of {package_name} timed out"
        except Exception as e:
            return False, "", str(e)
    
    def get_missing_dependencies(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Analyze code and return missing dependencies"""
        missing = []
        
        if language == 'python':
            missing.extend(self._check_python_dependencies(code))
        elif language in ['cuda', 'cu']:
            missing.extend(self._check_cuda_dependencies())
        elif language in ['c', 'cpp']:
            missing.extend(self._check_c_cpp_dependencies(language))
        
        return missing
    
    def _check_python_dependencies(self, code: str) -> List[Dict[str, Any]]:
        """Check Python-specific dependencies"""
        missing = []
        
        # Define import patterns to check
        import_patterns = {
            'pycuda': ['import pycuda', 'from pycuda'],
            'cupy': ['import cupy', 'from cupy'],
            'numba': ['import numba', 'from numba', '@cuda.jit', '@jit'],
            'numpy': ['import numpy', 'from numpy', 'np.'],
            'torch': ['import torch', 'from torch'],
            'tensorflow': ['import tensorflow', 'from tensorflow', 'import tf'],
            'scipy': ['import scipy', 'from scipy'],
            'matplotlib': ['import matplotlib', 'from matplotlib', 'plt.'],
            'pandas': ['import pandas', 'from pandas', 'pd.']
        }
        
        for package, patterns in import_patterns.items():
            if any(pattern in code for pattern in patterns):
                if not self.check_python_package(package):
                    missing.append({
                        'type': 'python',
                        'name': package,
                        'install_cmd': self.python_packages.get(package, f'pip install {package}'),
                        'description': f'Python package {package} is required',
                        'priority': self._get_package_priority(package)
                    })
        
        return missing
    
    def _check_cuda_dependencies(self) -> List[Dict[str, Any]]:
        """Check CUDA-specific dependencies"""
        missing = []
        
        if not self.check_system_command('nvcc'):
            missing.append({
                'type': 'system',
                'name': 'nvcc',
                'install_cmd': 'Download from https://developer.nvidia.com/cuda-toolkit',
                'description': 'NVIDIA CUDA Toolkit is required for CUDA compilation',
                'priority': 'critical'
            })
        
        return missing
    
    def _check_c_cpp_dependencies(self, language: str) -> List[Dict[str, Any]]:
        """Check C/C++ dependencies"""
        missing = []
        
        compiler = 'gcc' if language == 'c' else 'g++'
        if not self.check_system_command(compiler):
            install_cmd = self._get_compiler_install_command(compiler)
            missing.append({
                'type': 'system',
                'name': compiler,
                'install_cmd': install_cmd,
                'description': f'{compiler.upper()} compiler is required',
                'priority': 'critical'
            })
        
        return missing
    
    def _get_package_priority(self, package: str) -> str:
        """Get installation priority for packages"""
        critical_packages = ['numpy', 'pycuda', 'cupy']
        important_packages = ['torch', 'tensorflow', 'numba']
        
        if package in critical_packages:
            return 'critical'
        elif package in important_packages:
            return 'important'
        else:
            return 'optional'
    
    def _get_compiler_install_command(self, compiler: str) -> str:
        """Get platform-specific compiler installation command"""
        if sys.platform.startswith('linux'):
            return f'sudo apt install {compiler} build-essential'
        elif sys.platform == 'darwin':  # macOS
            return f'brew install {compiler}' if compiler != 'gcc' else 'xcode-select --install'
        elif sys.platform.startswith('win'):
            return 'Install Visual Studio Build Tools or MinGW'
        else:
            return f'Install {compiler} for your platform'
    
    def create_installation_guide(self, missing_deps: List[Dict[str, Any]]) -> str:
        """Create a comprehensive installation guide"""
        if not missing_deps:
            return "✅ All dependencies are available!"
        
        # Sort by priority
        priority_order = {'critical': 0, 'important': 1, 'optional': 2}
        missing_deps.sort(key=lambda x: priority_order.get(x.get('priority', 'optional'), 2))
        
        guide = "# 🔧 Missing Dependencies Installation Guide\n\n"
        
        python_deps = [dep for dep in missing_deps if dep['type'] == 'python']
        system_deps = [dep for dep in missing_deps if dep['type'] == 'system']
        
        if python_deps:
            guide += "## 🐍 Python Packages\n\n"
            guide += "**Quick install all at once:**\n"
            package_names = [dep['name'] for dep in python_deps]
            guide += f"```bash\npip install {' '.join(package_names)}\n```\n\n"
            
            guide += "**Individual installation:**\n"
            for dep in python_deps:
                priority_emoji = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}.get(dep.get('priority', 'optional'), '🟢')
                guide += f"{priority_emoji} **{dep['name']}**: {dep['description']}\n"
                guide += f"```bash\n{dep['install_cmd']}\n```\n\n"
        
        if system_deps:
            guide += "## 🖥️ System Dependencies\n\n"
            for dep in system_deps:
                guide += f"**{dep['name']}**: {dep['description']}\n"
                guide += f"```bash\n{dep['install_cmd']}\n```\n\n"
        
        # Add special CUDA installation guide
        if any(dep['name'] in ['pycuda', 'nvcc'] for dep in missing_deps):
            guide += self._create_cuda_installation_guide()
        
        return guide
    
    def _create_cuda_installation_guide(self) -> str:
        """Create CUDA-specific installation guide"""
        return """## 🚀 CUDA Setup Instructions

### Option 1: PyCUDA (Direct CUDA Programming)
```bash
# 1. Install CUDA Toolkit first
# Download from: https://developer.nvidia.com/cuda-toolkit

# 2. Install PyCUDA
pip install pycuda

# 3. Verify installation
python -c "import pycuda.autoinit; print('CUDA ready!')"
```

### Option 2: CuPy (Easier Alternative)
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x  
pip install cupy-cuda11x

# Verify
python -c "import cupy; print('CuPy ready!')"
```

### Option 3: Conda Environment (Recommended)
```bash
# Create environment with CUDA support
conda create -n cuda_dev python=3.9
conda activate cuda_dev
conda install -c conda-forge pycuda cudatoolkit

# Or with CuPy
conda install -c conda-forge cupy
```

### 📝 Platform-Specific Notes:

**macOS:**
- Apple Silicon Macs: CUDA not supported, use Metal or CPU alternatives
- Intel Macs with NVIDIA: Follow Linux instructions

**Linux:**
- Recommended: Use package manager (apt, yum) for CUDA toolkit
- Ensure NVIDIA drivers are installed first

**Windows:**
- Download CUDA toolkit from NVIDIA
- Use Visual Studio for compilation support

### 🎯 For Learning Without GPU:
```python
# CPU simulation mode
import numpy as np
print("Running in CPU simulation mode")
# Use NumPy arrays instead of GPU arrays
```

"""
    
    def create_conda_environment_script(self, missing_deps: List[Dict[str, Any]]) -> str:
        """Create a conda environment setup script"""
        python_deps = [dep for dep in missing_deps if dep['type'] == 'python']
        
        script = """#!/bin/bash
# Conda Environment Setup Script for CUDA Development

echo "🚀 Creating CUDA development environment..."

# Create new conda environment
conda create -n cuda_dev python=3.9 -y
conda activate cuda_dev

# Install CUDA toolkit via conda (recommended)
conda install -c conda-forge cudatoolkit-dev -y

echo "📦 Installing Python packages..."
"""
        
        for dep in python_deps:
            if dep['name'] == 'pycuda':
                script += "conda install -c conda-forge pycuda -y\n"
            elif dep['name'] == 'cupy':
                script += "conda install -c conda-forge cupy -y\n"
            elif dep['name'] in ['numpy', 'scipy', 'matplotlib', 'pandas']:
                script += f"conda install -c conda-forge {dep['name']} -y\n"
            else:
                script += f"pip install {dep['name']}\n"
        
        script += """
echo "✅ Environment setup complete!"
echo ""
echo "🎯 To use this environment:"
echo "  conda activate cuda_dev"
echo ""
echo "🧪 To test CUDA:"
echo '  python -c "import pycuda.autoinit; print(\\"CUDA ready!\\")"'
echo ""
"""
        return script