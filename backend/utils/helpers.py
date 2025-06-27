#!/usr/bin/env python3
"""
Helper functions for CUDA Tutor backend
"""

import platform
import subprocess
import sys
from typing import Dict, Any, Optional

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    try:
        info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture()
            },
            'python': {
                'version': sys.version,
                'version_info': {
                    'major': sys.version_info.major,
                    'minor': sys.version_info.minor,
                    'micro': sys.version_info.micro
                },
                'executable': sys.executable
            }
        }
        
        # Try to get additional system info
        try:
            import psutil
            info['resources'] = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_bytes': psutil.virtual_memory().total,
                'memory_available_bytes': psutil.virtual_memory().available,
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'free': psutil.disk_usage('/').free,
                    'used': psutil.disk_usage('/').used
                }
            }
        except ImportError:
            info['resources'] = {'note': 'psutil not available for detailed resource info'}
        
        # Check for CUDA/GPU info
        info['cuda'] = check_cuda_availability()
        
        return info
        
    except Exception as e:
        return {'error': f'Failed to get system info: {str(e)}'}

def check_cuda_availability() -> Dict[str, Any]:
    """Check CUDA and GPU availability"""
    cuda_info = {
        'nvidia_smi_available': False,
        'cuda_toolkit_available': False,
        'gpu_count': 0,
        'gpus': []
    }
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            cuda_info['nvidia_smi_available'] = True
            gpu_lines = result.stdout.strip().split('\n')
            cuda_info['gpu_count'] = len(gpu_lines)
            
            for i, line in enumerate(gpu_lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    cuda_info['gpus'].append({
                        'id': i,
                        'name': parts[0].strip(),
                        'memory_mb': int(parts[1].strip()),
                        'driver_version': parts[2].strip()
                    })
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    
    # Check CUDA toolkit
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cuda_info['cuda_toolkit_available'] = True
            # Extract CUDA version from nvcc output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    cuda_info['cuda_version'] = line.strip()
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return cuda_info

def format_error_message(error: Exception, context: Optional[str] = None) -> str:
    """Format error messages in a user-friendly way"""
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Common error patterns and user-friendly messages
    if 'ModuleNotFoundError' in error_type:
        if 'pycuda' in error_msg.lower():
            return """🚫 PyCUDA Not Installed

The code requires PyCUDA, but it's not installed on your system.

💡 Quick Solutions:
1. Install PyCUDA: `pip install pycuda`
2. Use CuPy instead: `pip install cupy-cuda12x`
3. Ask the AI tutor: "How do I install PyCUDA?"

❓ Need help with installation? The AI tutor can provide step-by-step guidance!"""
        
        elif 'cupy' in error_msg.lower():
            return """🚫 CuPy Not Installed

The code requires CuPy, but it's not installed on your system.

💡 Quick Solution:
Install CuPy: `pip install cupy-cuda12x`

For CUDA 11.x: `pip install cupy-cuda11x`"""
        
        else:
            # Extract module name
            import re
            match = re.search(r"No module named '(\w+)'", error_msg)
            module = match.group(1) if match else 'unknown'
            return f"""🚫 Missing Python Package: {module.upper()}

The code requires the '{module}' package.

💡 Quick Solution:
Install the package: `pip install {module}`"""
    
    elif 'FileNotFoundError' in error_type:
        if any(compiler in error_msg.lower() for compiler in ['gcc', 'g++', 'nvcc']):
            return """🚫 Compiler Not Found

Your system is missing required compilers.

💡 Solutions:
• For C/C++: Install build tools for your platform
• For CUDA: Install NVIDIA CUDA Toolkit
• macOS: `xcode-select --install`
• Ubuntu: `sudo apt install build-essential`"""
        
    elif 'TimeoutError' in error_type or 'timeout' in error_msg.lower():
        return """⏱️ Operation Timed Out

The operation took too long to complete.

💡 Possible causes:
• Code has infinite loop
• System is under heavy load
• Network connectivity issues

Try simplifying your code or ask the AI tutor for optimization tips!"""
    
    # Default formatting
    formatted_msg = f"❌ {error_type}: {error_msg}"
    
    if context:
        formatted_msg = f"❌ Error in {context}: {error_msg}"
    
    formatted_msg += "\n\n💡 If you need help, ask the AI tutor to explain this error!"
    
    return formatted_msg

def check_compiler_availability() -> Dict[str, bool]:
    """Check availability of various compilers"""
    compilers = {
        'gcc': False,
        'g++': False,
        'nvcc': False,
        'python3': False,
        'python': False
    }
    
    for compiler in compilers.keys():
        try:
            result = subprocess.run([compiler, '--version'], 
                                  capture_output=True, timeout=3)
            compilers[compiler] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            compilers[compiler] = False
    
    return compilers

def get_package_info(package_name: str) -> Dict[str, Any]:
    """Get information about a Python package"""
    try:
        import importlib.metadata
        
        try:
            dist = importlib.metadata.distribution(package_name)
            return {
                'installed': True,
                'version': dist.version,
                'summary': dist.metadata.get('Summary', ''),
                'homepage': dist.metadata.get('Home-page', ''),
                'author': dist.metadata.get('Author', '')
            }
        except importlib.metadata.PackageNotFoundError:
            return {
                'installed': False,
                'error': f'Package {package_name} not found'
            }
    except ImportError:
        # Fallback for older Python versions
        try:
            __import__(package_name)
            return {
                'installed': True,
                'version': 'unknown',
                'note': 'Package installed but version info unavailable'
            }
        except ImportError:
            return {
                'installed': False,
                'error': f'Package {package_name} not installed'
            }

def validate_code_safety(code: str, language: str) -> Dict[str, Any]:
    """Basic validation to check for potentially unsafe code patterns"""
    warnings = []
    
    # Common unsafe patterns
    unsafe_patterns = {
        'python': [
            ('import os', 'Uses system operations'),
            ('import subprocess', 'Can execute system commands'),
            ('exec(', 'Dynamic code execution'),
            ('eval(', 'Dynamic code evaluation'),
            ('__import__', 'Dynamic imports'),
            ('open(', 'File operations')
        ],
        'c': [
            ('system(', 'System command execution'),
            ('exec', 'Process execution'),
            ('gets(', 'Unsafe input function'),
            ('strcpy(', 'Potential buffer overflow')
        ],
        'cpp': [
            ('system(', 'System command execution'),
            ('exec', 'Process execution'),
            ('gets(', 'Unsafe input function'),
            ('strcpy(', 'Potential buffer overflow')
        ],
        'cuda': [
            ('system(', 'System command execution'),
            ('exec', 'Process execution')
        ]
    }
    
    if language in unsafe_patterns:
        for pattern, description in unsafe_patterns[language]:
            if pattern in code:
                warnings.append({
                    'pattern': pattern,
                    'description': description,
                    'severity': 'warning'
                })
    
    return {
        'safe': len(warnings) == 0,
        'warnings': warnings,
        'message': 'Code appears safe' if len(warnings) == 0 else f'Found {len(warnings)} potential safety concerns'
    }

def format_compilation_output(stdout: str, stderr: str, success: bool) -> str:
    """Format compilation output for better readability"""
    if success:
        if stdout.strip():
            return f"✅ Compilation successful!\n\nOutput:\n{stdout}"
        else:
            return "✅ Compilation successful!"
    else:
        formatted_error = "❌ Compilation failed!\n\n"
        
        if stderr.strip():
            # Try to make error messages more readable
            error_lines = stderr.split('\n')
            formatted_lines = []
            
            for line in error_lines:
                if line.strip():
                    # Highlight error types
                    if 'error:' in line.lower():
                        formatted_lines.append(f"🔴 {line}")
                    elif 'warning:' in line.lower():
                        formatted_lines.append(f"🟡 {line}")
                    elif 'note:' in line.lower():
                        formatted_lines.append(f"💡 {line}")
                    else:
                        formatted_lines.append(f"   {line}")
            
            formatted_error += '\n'.join(formatted_lines)
        
        if stdout.strip():
            formatted_error += f"\n\nAdditional output:\n{stdout}"
        
        return formatted_error

def get_cuda_device_info() -> Dict[str, Any]:
    """Get detailed CUDA device information"""
    try:
        # Try using nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            devices = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                devices.append({
                    'id': i,
                    'name': name,
                    'memory_total': memory_info.total,
                    'memory_free': memory_info.free,
                    'memory_used': memory_info.used
                })
            
            return {
                'available': True,
                'device_count': device_count,
                'devices': devices,
                'method': 'pynvml'
            }
            
        except ImportError:
            # Fallback to nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                devices = []
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        devices.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': int(parts[2]),
                            'memory_free_mb': int(parts[3]),
                            'memory_used_mb': int(parts[4]),
                            'temperature_c': int(parts[5]) if parts[5] != '[Not Supported]' else None
                        })
                
                return {
                    'available': True,
                    'device_count': len(devices),
                    'devices': devices,
                    'method': 'nvidia-smi'
                }
    
    except Exception as e:
        pass
    
    return {
        'available': False,
        'error': 'CUDA devices not available or accessible'
    }