#!/usr/bin/env python3
"""
Enhanced Flask Backend for CUDA Chat Interface with Context Memory + Code Compilation
FIXED VERSION with correct class ordering
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
import os
import json
import time
import subprocess
import tempfile
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any
import threading
import signal
import traceback
import sys

print("🔧 Starting Flask server with RAG and Code Compilation...")

# Create Flask app first
try:
    app = Flask(__name__)
    print("✅ Flask app created successfully")
except Exception as e:
    print(f"❌ Error creating Flask app: {e}")
    sys.exit(1)

# Configure CORS properly - AFTER creating the app
try:
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"])
    print("✅ CORS configured successfully")
except Exception as e:
    print(f"❌ Error configuring CORS: {e}")
    sys.exit(1)

conversation_sessions = {}

# Configuration for code compilation
COMPILATION_TIMEOUT = 30  # seconds
EXECUTION_TIMEOUT = 10    # seconds
MAX_OUTPUT_SIZE = 10000   # characters

class DependencyManager:
    """Manages and installs dependencies for code execution"""
    
    def __init__(self):
        self.python_packages = {
            'pycuda': 'pip install pycuda',
            'cupy': 'pip install cupy-cuda12x',  # For CUDA 12.x
            'numba': 'pip install numba',
            'numpy': 'pip install numpy',
            'torch': 'pip install torch',
            'tensorflow': 'pip install tensorflow'
        }
        
        self.system_packages = {
            'nvcc': 'NVIDIA CUDA Toolkit',
            'gcc': 'GNU C Compiler',
            'g++': 'GNU C++ Compiler'
        }
    
    def check_python_package(self, package_name):
        """Check if a Python package is installed"""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    def check_system_command(self, command):
        """Check if a system command/compiler is available"""
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def install_python_package(self, package_name):
        """Install a Python package"""
        if package_name in self.python_packages:
            install_cmd = self.python_packages[package_name]
            try:
                print(f"Installing {package_name}...")
                result = subprocess.run(install_cmd.split(), 
                                      capture_output=True, text=True, timeout=300)
                return result.returncode == 0, result.stdout, result.stderr
            except Exception as e:
                return False, "", str(e)
        return False, "", "Unknown package"
    
    def get_missing_dependencies(self, code, language):
        """Analyze code and return missing dependencies"""
        missing = []
        
        if language == 'python':
            # Check for common imports
            import_patterns = {
                'pycuda': ['import pycuda', 'from pycuda'],
                'cupy': ['import cupy', 'from cupy'],
                'numba': ['import numba', 'from numba', '@cuda.jit'],
                'numpy': ['import numpy', 'from numpy', 'np.'],
                'torch': ['import torch', 'from torch'],
                'tensorflow': ['import tensorflow', 'from tensorflow']
            }
            
            for package, patterns in import_patterns.items():
                if any(pattern in code for pattern in patterns):
                    if not self.check_python_package(package):
                        missing.append({
                            'type': 'python',
                            'name': package,
                            'install_cmd': self.python_packages.get(package, f'pip install {package}'),
                            'description': f'Python package {package} is required'
                        })
        
        elif language in ['cuda', 'cu']:
            # Check for CUDA compiler
            if not self.check_system_command('nvcc'):
                missing.append({
                    'type': 'system',
                    'name': 'nvcc',
                    'install_cmd': 'Download from https://developer.nvidia.com/cuda-toolkit',
                    'description': 'NVIDIA CUDA Toolkit is required for CUDA compilation'
                })
        
        elif language in ['c', 'cpp']:
            # Check for C/C++ compilers
            compiler = 'gcc' if language == 'c' else 'g++'
            if not self.check_system_command(compiler):
                missing.append({
                    'type': 'system',
                    'name': compiler,
                    'install_cmd': f'sudo apt install {compiler}' if sys.platform.startswith('linux') else f'brew install {compiler}',
                    'description': f'{compiler.upper()} compiler is required'
                })
        
        return missing
    
    def create_installation_guide(self, missing_deps):
        """Create a comprehensive installation guide"""
        if not missing_deps:
            return "All dependencies are available!"
        
        guide = "# Missing Dependencies Installation Guide\n\n"
        
        python_deps = [dep for dep in missing_deps if dep['type'] == 'python']
        system_deps = [dep for dep in missing_deps if dep['type'] == 'system']
        
        if python_deps:
            guide += "## Python Packages\n"
            guide += "Run these commands to install missing Python packages:\n\n"
            for dep in python_deps:
                guide += f"```bash\n{dep['install_cmd']}\n```\n"
                guide += f"*{dep['description']}*\n\n"
        
        if system_deps:
            guide += "## System Dependencies\n"
            guide += "Install these system components:\n\n"
            for dep in system_deps:
                guide += f"**{dep['name']}**: {dep['description']}\n"
                guide += f"```bash\n{dep['install_cmd']}\n```\n\n"
        
        # Add platform-specific CUDA installation instructions
        if any(dep['name'] == 'pycuda' for dep in python_deps) or any(dep['name'] == 'nvcc' for dep in system_deps):
            guide += "## CUDA Setup Instructions\n\n"
            guide += "### For PyCUDA (Python CUDA):\n"
            guide += "1. **Install NVIDIA CUDA Toolkit** (if not already installed):\n"
            guide += "   - Download from: https://developer.nvidia.com/cuda-toolkit\n"
            guide += "   - Follow the installation guide for your OS\n\n"
            guide += "2. **Install PyCUDA**:\n"
            guide += "```bash\n"
            guide += "# On Ubuntu/Debian:\n"
            guide += "sudo apt update\n"
            guide += "sudo apt install nvidia-cuda-toolkit python3-dev\n"
            guide += "pip install pycuda\n\n"
            guide += "# On macOS:\n"
            guide += "# Note: CUDA is not supported on Apple Silicon Macs\n"
            guide += "# For Intel Macs with NVIDIA GPUs:\n"
            guide += "brew install cuda\n"
            guide += "pip install pycuda\n\n"
            guide += "# Alternative: Use CuPy (easier installation)\n"
            guide += "pip install cupy-cuda12x  # For CUDA 12.x\n"
            guide += "# or\n"
            guide += "pip install cupy-cuda11x  # For CUDA 11.x\n"
            guide += "```\n\n"
            guide += "3. **Verify Installation**:\n"
            guide += "```python\n"
            guide += "import pycuda.autoinit\n"
            guide += "import pycuda.driver as cuda\n"
            guide += "print(f'CUDA Devices: {cuda.Device.count()}')\n"
            guide += "```\n\n"
        
        return guide

class CodeCompiler:
    def __init__(self):
        try:
            self.temp_dir = Path(tempfile.gettempdir()) / "cuda_tutor_compile"
            self.temp_dir.mkdir(exist_ok=True)
            self.active_compilations = {}
            print("✅ CodeCompiler initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing CodeCompiler: {e}")
            raise
        
    def get_compiler_config(self, language: str) -> Dict[str, Any]:
        """Get compiler configuration for different languages"""
        configs = {
            'c': {
                'extension': '.c',
                'compiler': 'gcc',
                'compile_args': ['-o', '{output}', '{source}', '-std=c99', '-Wall'],
                'run_command': './{executable}'
            },
            'cpp': {
                'extension': '.cpp',
                'compiler': 'g++',
                'compile_args': ['-o', '{output}', '{source}', '-std=c++17', '-Wall'],
                'run_command': './{executable}'
            },
            'cuda': {
                'extension': '.cu',
                'compiler': 'nvcc',
                'compile_args': ['-o', '{output}', '{source}', '-std=c++14'],
                'run_command': './{executable}'
            },
            'python': {
                'extension': '.py',
                'compiler': None,  # Interpreted
                'compile_args': [],
                'run_command': 'python3 {source}'
            }
        }
        return configs.get(language, configs['c'])
    
    def generate_test_script_with_ai(self, code: str, language: str) -> str:
        """Generate a test script using the existing AI model"""
        try:
            # Create a prompt for test generation
            prompt = f"""Generate a comprehensive test script for this {language.upper()} code. 

CODE TO TEST:
```{language}
{code}
```

Create a complete, compilable test script that:
1. Tests the main functionality
2. Includes edge cases
3. Has clear output showing test results
4. Uses proper error checking

Generate ONLY the test code, ready to compile and run:"""

            # Use the existing RAG system to generate test
            if 'rag_system' in globals() and rag_system:
                test_script = rag_system.generate_response(prompt, "", stream=False)
                
                # Clean up the response - extract code blocks
                if '```' in test_script:
                    # Extract code from markdown
                    parts = test_script.split('```')
                    for i, part in enumerate(parts):
                        if i % 2 == 1:  # Odd indices are code blocks
                            # Remove language identifier if present
                            lines = part.strip().split('\n')
                            if lines[0].strip().lower() in ['c', 'cpp', 'cuda', 'python']:
                                return '\n'.join(lines[1:])
                            return part.strip()
                
                return test_script
            else:
                return self._generate_fallback_test(code, language)
                
        except Exception as e:
            print(f"Error generating AI test script: {e}")
            return self._generate_fallback_test(code, language)
    
    def _generate_fallback_test(self, code: str, language: str) -> str:
        """Generate a basic fallback test script"""
        
        if language in ['c', 'cpp']:
            if 'int main' in code:
                return """// Test script - Main function detected
// This code has its own main function
// It will be executed directly for testing

// Test case: Run the program and check output
// Expected: Program should compile and run without errors
"""
            else:
                return """#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Test script for the provided functions
int main() {
    printf("Running tests for provided code...\\n");
    
    // TODO: Add specific tests based on the functions in your code
    printf("Test 1: Basic functionality\\n");
    
    printf("All tests completed successfully!\\n");
    return 0;
}"""

        elif language == 'cuda':
            return """#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        printf("CUDA error: %s\\n", cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
}

int main() {
    printf("Testing CUDA code...\\n");
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Found %d CUDA device(s)\\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found. Skipping GPU tests.\\n");
        return 0;
    }
    
    // Basic memory test
    float *d_test;
    CUDA_CHECK(cudaMalloc(&d_test, sizeof(float) * 100));
    CUDA_CHECK(cudaFree(d_test));
    printf("Basic CUDA memory test passed\\n");
    
    // TODO: Add specific tests for your CUDA kernels
    
    printf("CUDA tests completed!\\n");
    return 0;
}"""

        elif language == 'python':
            return """#!/usr/bin/env python3
# Test script for Python code

def test_code():
    print("Running Python tests...")
    
    try:
        # TODO: Add specific tests for your Python code
        print("Test 1: Basic execution")
        
        print("All tests passed!")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_code()
    exit(0 if success else 1)
"""

        return f"// No test script available for language: {language}"

    def compile_code(self, code: str, language: str, test_script: str = "") -> Dict[str, Any]:
        """Compile the given code"""
        compilation_id = str(uuid.uuid4())
        config = self.get_compiler_config(language)
        
        # Create compilation directory
        comp_dir = self.temp_dir / compilation_id
        comp_dir.mkdir(exist_ok=True)
        
        try:
            # Write source file
            source_file = comp_dir / f"main{config['extension']}"
            with open(source_file, 'w') as f:
                f.write(code)
            
            # Write test script if provided
            if test_script:
                test_file = comp_dir / f"test{config['extension']}"
                with open(test_file, 'w') as f:
                    f.write(test_script)
            
            # For interpreted languages, no compilation needed
            if config['compiler'] is None:
                self.active_compilations[compilation_id] = {
                    'language': language,
                    'source_file': str(source_file),
                    'executable': str(source_file),
                    'directory': str(comp_dir)
                }
                return {
                    'success': True,
                    'compilationId': compilation_id,
                    'output': 'Python code ready for execution',
                    'stderr': ''
                }
            
            # Check if compiler exists
            try:
                subprocess.run([config['compiler'], '--version'], 
                             capture_output=True, timeout=5)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return {
                    'success': False,
                    'compilationId': compilation_id,
                    'output': '',
                    'stderr': f'Compiler {config["compiler"]} not found. Please install it first.',
                    'returnCode': -1
                }
            
            # Compile the code
            executable = comp_dir / "main"
            compile_cmd = [config['compiler']] + [
                arg.format(
                    output=str(executable),
                    source=str(source_file)
                ) for arg in config['compile_args']
            ]
            
            print(f"Compiling with command: {' '.join(compile_cmd)}")
            
            result = subprocess.run(
                compile_cmd,
                cwd=comp_dir,
                capture_output=True,
                text=True,
                timeout=COMPILATION_TIMEOUT
            )
            
            # Store compilation info
            self.active_compilations[compilation_id] = {
                'language': language,
                'source_file': str(source_file),
                'executable': str(executable),
                'directory': str(comp_dir)
            }
            
            return {
                'success': result.returncode == 0,
                'compilationId': compilation_id,
                'output': result.stdout[:MAX_OUTPUT_SIZE],
                'stderr': result.stderr[:MAX_OUTPUT_SIZE],
                'returnCode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'compilationId': compilation_id,
                'output': '',
                'stderr': f'Compilation timed out after {COMPILATION_TIMEOUT} seconds',
                'returnCode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'compilationId': compilation_id,
                'output': '',
                'stderr': f'Compilation error: {str(e)}',
                'returnCode': -1
            }
    
    def execute_code(self, compilation_id: str, input_data: str = "") -> Dict[str, Any]:
        """Execute the compiled code"""
        if compilation_id not in self.active_compilations:
            return {
                'success': False,
                'output': '',
                'stderr': 'Compilation ID not found',
                'exitCode': -1
            }
        
        comp_info = self.active_compilations[compilation_id]
        config = self.get_compiler_config(comp_info['language'])
        
        try:
            # Prepare execution command
            if config['compiler'] is None:  # Python
                run_cmd = ['python3', comp_info['source_file']]
            else:
                executable_name = os.path.basename(comp_info['executable'])
                run_cmd = config['run_command'].format(executable=executable_name).split()
            
            print(f"Executing with command: {' '.join(run_cmd)}")
            
            start_time = time.time()
            
            # Set environment variables for CUDA if needed
            env = os.environ.copy()
            if comp_info['language'] == 'cuda':
                env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            
            result = subprocess.run(
                run_cmd,
                cwd=comp_info['directory'],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT,
                env=env
            )
            
            execution_time = int((time.time() - start_time) * 1000)  # milliseconds
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout[:MAX_OUTPUT_SIZE],
                'stderr': result.stderr[:MAX_OUTPUT_SIZE],
                'exitCode': result.returncode,
                'executionTime': execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'stderr': f'Execution timed out after {EXECUTION_TIMEOUT} seconds',
                'exitCode': -1,
                'executionTime': EXECUTION_TIMEOUT * 1000
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'stderr': f'Execution error: {str(e)}',
                'exitCode': -1,
                'executionTime': 0
            }
    
    def cleanup_compilation(self, compilation_id: str):
        """Clean up compilation files"""
        if compilation_id in self.active_compilations:
            comp_info = self.active_compilations[compilation_id]
            try:
                shutil.rmtree(comp_info['directory'])
            except:
                pass
            del self.active_compilations[compilation_id]

# NOW define EnhancedCodeCompiler AFTER CodeCompiler
class EnhancedCodeCompiler(CodeCompiler):
    def __init__(self):
        super().__init__()
        self.dep_manager = DependencyManager()
    
    def analyze_and_prepare_code(self, code: str, language: str):
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
                error_msg = "Missing Dependencies Detected!\n\n"
                error_msg += dep_analysis['installation_guide']
                
                return {
                    'success': False,
                    'output': '',
                    'stderr': error_msg,
                    'exitCode': -1,
                    'missing_dependencies': dep_analysis['missing_dependencies'],
                    'installation_guide': dep_analysis['installation_guide']
                }
        except Exception as e:
            print(f"Error checking dependencies: {e}")
        
        # If no dependency issues, proceed with normal execution
        return super().execute_code(compilation_id, input_data)
    
    def create_conda_environment_script(self, missing_deps):
        """Create a conda environment setup script"""
        script = """#!/bin/bash
# Conda Environment Setup Script for CUDA Development

echo "Creating CUDA development environment..."

# Create new conda environment
conda create -n cuda_dev python=3.9 -y
conda activate cuda_dev

# Install CUDA toolkit via conda (recommended)
conda install -c conda-forge cudatoolkit-dev -y

# Install Python packages
"""
        
        for dep in missing_deps:
            if dep['type'] == 'python':
                if dep['name'] == 'pycuda':
                    script += "conda install -c conda-forge pycuda -y\n"
                elif dep['name'] == 'cupy':
                    script += "conda install -c conda-forge cupy -y\n"
                else:
                    script += f"pip install {dep['name']}\n"
        
        script += """
echo "Environment setup complete!"
echo "To activate: conda activate cuda_dev"
echo "To test: python -c 'import pycuda.autoinit; print(\"CUDA ready!\")"
"""
        return script

class SimpleRAG:
    def __init__(self):
        print("🔧 Initializing RAG system...")
        
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
            
            # Load CUDA dataset
            print(" Loading CUDA dataset...")
            try:
                dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
                self.df = dataset["level_1"].to_pandas().head(50)  # Limit for speed
                print("✅ CUDA dataset loaded from online")
            except Exception as e:
                print(f"⚠️ Could not load dataset: {e}")
                # Create dummy data for demo
                self.df = pd.DataFrame({
                    'Op_Name': ['Matrix Multiply', 'Vector Add', 'Convolution'],
                    'CUDA_Code': ['__global__ void matmul(...)', '__global__ void vecadd(...)', '__global__ void conv(...)'],
                    'CUDA_Speedup_Native': [15.2, 8.5, 12.3]
                })
                print("✅ Using fallback dummy data")
            
            # Create knowledge base
            self.knowledge = []
            for _, row in self.df.iterrows():
                if pd.notna(row.get('CUDA_Code')) and pd.notna(row.get('Op_Name')):
                    self.knowledge.append({
                        'operation': row['Op_Name'],
                        'cuda_code': str(row['CUDA_Code'])[:500],
                        'speedup': row.get('CUDA_Speedup_Native', 'N/A'),
                        'text': f"Operation: {row['Op_Name']} | CUDA Code: {str(row['CUDA_Code'])[:300]}"
                    })
            
            if self.knowledge:
                # Create embeddings
                print(" Creating embeddings...")
                texts = [item['text'] for item in self.knowledge]
                self.embeddings = self.embedding_model.encode(texts)
                print(f" Ready! Loaded {len(self.knowledge)} CUDA examples")
            else:
                print(" No knowledge loaded, using fallback responses")
                self.embeddings = None
                
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            traceback.print_exc()
            raise
    
    def search(self, query, top_k=2):
        """Search for relevant examples"""
        if self.embeddings is None or not self.knowledge:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [self.knowledge[i] for i in top_indices]
        return results
    
    def generate_response(self, query, conversation_context="", stream=False):
        """Generate response using RAG with conversation context"""
        
        # 1. Retrieve relevant examples
        examples = self.search(query, top_k=2)
        
        # 2. Create context from examples
        if examples:
            example_context = "\n".join([
                f"Example: {ex['operation']} (Speedup: {ex['speedup']}x)\nCode: {ex['cuda_code'][:200]}..."
                for ex in examples
            ])
        else:
            example_context = "General CUDA programming knowledge."
        
        if conversation_context:
            prompt = f"""You are a helpful CUDA programming tutor. Build on our previous conversation.

Previous conversation:
{conversation_context}

Relevant CUDA examples:
{example_context}

Current question: {query}

Answer the current question, referencing our previous discussion when relevant. Be conversational and remember what we've discussed:"""
        else:
            prompt = f"""You are a helpful CUDA programming tutor. Answer concisely and clearly.

Relevant examples:
{example_context}

Student question: {query}

Provide a helpful answer:"""
        
        # 4. Generate response with improved settings
        try:
            response = requests.post("http://localhost:11434/api/generate", 
                json={
                    "model": "deepseek-r1:latest",
                    "prompt": prompt,
                    "stream": stream,  # Enable streaming if requested
                    "options": {
                        "num_predict": 10000,  # Reasonable token limit
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stop": ["Student question:", "Question:", "Human:", "User:", "Previous conversation:", "Current question:"],
                        "num_ctx": 4096,  # Larger context window for conversation memory
                        "repeat_penalty": 1.1
                    }
                }, 
                timeout=120,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response (for future implementation)
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    full_response += chunk['response']
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response
                else:
                    result = response.json()
                    generated_text = result.get("response", "").strip()
                    
                    # Clean up the response - remove any truncated sentences
                    if generated_text:
                        # If response ends abruptly (no punctuation), try to clean it up
                        if not generated_text.endswith(('.', '!', '?', ':', '```')):
                            # Find the last complete sentence
                            sentences = generated_text.split('.')
                            if len(sentences) > 1:
                                generated_text = '.'.join(sentences[:-1]) + '.'
                            # If no complete sentences, keep as is but add note
                            elif len(generated_text) > 50:
                                generated_text += "..."
                    
                    return generated_text
            else:
                return f"I'm having trouble connecting to the AI model. Status: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "The AI model is taking too long to respond. Please try a simpler question."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to the AI model. Please make sure Ollama is running with: `ollama serve`"
        except Exception as e:
            return f"An error occurred: {str(e)}"

def get_conversation_context(session_history, max_exchanges=3):
    """Build conversation context from session history"""
    if not session_history:
        return ""
    
    # Get the most recent exchanges
    recent_exchanges = session_history[-max_exchanges:]
    context_parts = []
    
    for exchange in recent_exchanges:
        # Truncate long messages for context
        user_msg = exchange['user'][:150] + "..." if len(exchange['user']) > 150 else exchange['user']
        assistant_msg = exchange['assistant'][:200] + "..." if len(exchange['assistant']) > 200 else exchange['assistant']
        
        context_parts.append(f"User: {user_msg}")
        context_parts.append(f"Assistant: {assistant_msg}")
    
    return "\n".join(context_parts)

def detect_follow_up_question(current_query, last_response=""):
    """Detect if current query is a follow-up to previous conversation - ENHANCED"""
    follow_up_indicators = [
        # Direct references to previous conversation
        'last question', 'previous question', 'what did i ask', 'what was my question',
        'you just said', 'you mentioned', 'earlier you said', 'before you said',
        
        # Continuation indicators
        'show me', 'give me', 'can you', 'what about', 'also', 'optimize', 
        'improve', 'explain that', 'those examples', 'the code', 'that kernel',
        'tell me more', 'more details', 'expand on', 'elaborate',
        
        # Pronouns that indicate reference to previous content
        'this', 'that', 'it', 'them', 'these', 'those'
    ]
    
    query_lower = current_query.lower()
    return any(indicator in query_lower for indicator in follow_up_indicators)

def clean_old_sessions():
    """Clean up old conversation sessions (older than 1 hour)"""
    current_time = time.time()
    cutoff_time = current_time - 3600  # 1 hour ago
    
    sessions_to_remove = []
    for session_id, session_data in conversation_sessions.items():
        if session_data and len(session_data) > 0:
            last_timestamp = session_data[-1].get('timestamp', 0)
            if last_timestamp < cutoff_time:
                sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del conversation_sessions[session_id]

# Initialize systems
try:
    rag_system = SimpleRAG()
    print("✅ RAG system initialized successfully")
except Exception as e:
    print(f"❌ RAG system initialization failed: {e}")
    rag_system = None

try:
    compiler = EnhancedCodeCompiler()  # Use the enhanced version
    print("✅ Enhanced code compilation system initialized")
except Exception as e:
    print(f"❌ Code compilation system initialization failed: {e}")
    try:
        # Fallback to basic compiler
        compiler = CodeCompiler()
        print("✅ Basic code compilation system initialized as fallback")
    except Exception as e2:
        print(f"❌ Fallback compiler also failed: {e2}")
        compiler = None

# Enhanced compilation routes
def setup_enhanced_compilation_routes(app):
    """Setup enhanced compilation routes with dependency management"""
    
    @app.route('/api/analyze-dependencies', methods=['POST', 'OPTIONS'])
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
            
            analysis = compiler.analyze_and_prepare_code(code, language)
            
            return jsonify({
                'success': True,
                'analysis': analysis
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'analysis': {'has_missing_deps': False, 'can_execute': True}
            }), 500
    
    @app.route('/api/install-dependency', methods=['POST', 'OPTIONS'])
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
            
            success, stdout, stderr = compiler.dep_manager.install_python_package(package_name)
            
            return jsonify({
                'success': success,
                'output': stdout,
                'error': stderr if not success else ''
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'output': '',
                'stderr': f'Installation error: {str(e)}'
            }), 500
    
    @app.route('/api/execute-enhanced', methods=['POST', 'OPTIONS'])
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
            
            # Use enhanced execution if available, otherwise fallback to basic
            if hasattr(compiler, 'execute_code_with_dependency_check'):
                result = compiler.execute_code_with_dependency_check(compilation_id, input_data)
            else:
                result = compiler.execute_code(compilation_id, input_data)
            
            # Clean up after execution if successful
            if result.get('success', False):
                compiler.cleanup_compilation(compilation_id)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'output': '',
                'stderr': f'Server error: {str(e)}'
            }), 500

# Define all routes
@app.route('/')
def index():
    """Serve the API info"""
    try:
        return jsonify({
            'message': 'CUDA Tutor Backend is running!',
            'status': 'healthy',
            'endpoints': {
                'chat': '/api/chat',
                'status': '/api/status',
                'health': '/health',
                'generate_test': '/api/generate-test',
                'compile': '/api/compile',
                'execute': '/api/execute',
                'analyze_dependencies': '/api/analyze-dependencies',
                'install_dependency': '/api/install-dependency',
                'execute_enhanced': '/api/execute-enhanced'
            }
        })
    except Exception as e:
        print(f"Error in index route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight CORS request
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')  # Get session ID
        stream = data.get('stream', False)  # Check if client wants streaming
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f" Session {session_id}: {message}")
        
        # Clean up old sessions periodically
        if len(conversation_sessions) > 50:  # Clean when we have too many sessions
            clean_old_sessions()
        
        # Get or create session
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        session_history = conversation_sessions[session_id]
        
        # Build conversation context - ALWAYS use context now
        conversation_context = get_conversation_context(session_history, max_exchanges=3)
        
        # Check if this is a follow-up question
        is_follow_up = detect_follow_up_question(message)
        
        # Generate response with context
        if rag_system:
            # Always use conversation context if available
            response = rag_system.generate_response(message, conversation_context, stream=stream)
        else:
            # Fallback response
            response = """I apologize, but the RAG system isn't fully initialized. Here's a basic response:

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It allows developers to use NVIDIA GPUs for general-purpose computing tasks, often achieving significant speedups for parallelizable problems.

Key benefits:
• Massive parallel processing (thousands of cores)
• High memory bandwidth
• Optimized for data-parallel computations
• Excellent for machine learning, scientific computing, and image processing

To get started with CUDA programming, you'll need:
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit installed
3. Understanding of parallel programming concepts"""
        
        # Save to session history
        session_history.append({
            'user': message,
            'assistant': response,
            'timestamp': time.time(),
            'is_follow_up': is_follow_up
        })
        
        # Keep only the last 10 exchanges per session to prevent memory bloat
        if len(session_history) > 10:
            conversation_sessions[session_id] = session_history[-10:]
        
        print(f" Session {session_id}: {response[:100]}...")
        print(f" Context used: {bool(conversation_context)}")
        print(f"🔗 Follow-up detected: {is_follow_up}")
        
        return jsonify({
            'response': response,
            'session_id': session_id,
            'is_follow_up': is_follow_up,
            'context_used': bool(conversation_context),
            'status': 'success'
        })
        
    except Exception as e:
        print(f" Error in chat route: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# CODE COMPILATION ENDPOINTS

@app.route('/api/generate-test', methods=['POST', 'OPTIONS'])
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
        print(f"Error generating test script: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'testScript': f'// Error generating test: {str(e)}'
        }), 500

@app.route('/api/compile', methods=['POST', 'OPTIONS'])
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
        
        print(f"Compilation result: {result['success']}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error compiling code: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'output': '',
            'stderr': f'Server error: {str(e)}'
        }), 500

@app.route('/api/execute', methods=['POST', 'OPTIONS'])
def execute_code():
    """Execute compiled code"""
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
        result = compiler.execute_code(compilation_id, input_data)
        
        # Clean up after execution
        compiler.cleanup_compilation(compilation_id)
        
        print(f"Execution result: {result['success']}, time: {result.get('executionTime', 0)}ms")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error executing code: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'output': '',
            'stderr': f'Server error: {str(e)}'
        }), 500

@app.route('/api/cleanup-compilation/<compilation_id>', methods=['DELETE', 'OPTIONS'])
def cleanup_compilation(compilation_id):
    """Clean up compilation files"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if compiler:
            compiler.cleanup_compilation(compilation_id)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Compiler not available'}), 500
    except Exception as e:
        print(f"Error cleaning up compilation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Check system status"""
    try:
        status_info = {
            'rag_loaded': rag_system is not None,
            'knowledge_count': len(rag_system.knowledge) if rag_system else 0,
            'active_sessions': len(conversation_sessions),
            'ollama_status': 'unknown',
            'compiler_available': compiler is not None,
            'enhanced_compiler': hasattr(compiler, 'dep_manager') if compiler else False,
            'active_compilations': len(compiler.active_compilations) if compiler else 0
        }
        
        # Test Ollama connection
        try:
            response = requests.get("http://localhost:11434", timeout=5)
            status_info['ollama_status'] = 'connected' if response.status_code == 200 else 'error'
        except:
            status_info['ollama_status'] = 'disconnected'
        
        # Check compilers
        if compiler:
            compiler_status = {}
            for lang, config in [('c', compiler.get_compiler_config('c')), 
                               ('cpp', compiler.get_compiler_config('cpp')), 
                               ('cuda', compiler.get_compiler_config('cuda'))]:
                if config['compiler']:
                    try:
                        subprocess.run([config['compiler'], '--version'], 
                                     capture_output=True, timeout=3)
                        compiler_status[lang] = 'available'
                    except:
                        compiler_status[lang] = 'not_found'
                else:
                    compiler_status[lang] = 'interpreter'
            
            status_info['compilers'] = compiler_status
        
        return jsonify(status_info)
    except Exception as e:
        print(f"Error in status route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Clear a specific conversation session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
            return jsonify({'message': f'Session {session_id} cleared', 'status': 'success'})
        else:
            return jsonify({'message': f'Session {session_id} not found', 'status': 'warning'})
    except Exception as e:
        print(f"Error clearing session: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/session_info/<session_id>')
def session_info(session_id):
    """Get information about a specific session"""
    try:
        if session_id in conversation_sessions:
            session_data = conversation_sessions[session_id]
            return jsonify({
                'session_id': session_id,
                'message_count': len(session_data),
                'last_activity': session_data[-1]['timestamp'] if session_data else None,
                'status': 'active'
            })
        else:
            return jsonify({
                'session_id': session_id,
                'status': 'not_found'
            }), 404
    except Exception as e:
        print(f"Error getting session info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Simple health check"""
    try:
        return jsonify({
            'status': 'healthy', 
            'service': 'CUDA Chat Backend with Context + Code Compilation',
            'active_sessions': len(conversation_sessions),
            'rag_status': 'loaded' if rag_system else 'not_loaded',
            'compiler_status': 'enhanced' if (compiler and hasattr(compiler, 'dep_manager')) else ('basic' if compiler else 'not_loaded')
        })
    except Exception as e:
        print(f"Error in health route: {e}")
        return jsonify({'error': str(e)}), 500

# Cleanup function to run periodically
def cleanup_old_compilations():
    """Clean up old compilation directories"""
    if compiler:
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour ago
        
        # Clean up temporary directories older than 1 hour
        for comp_id in list(compiler.active_compilations.keys()):
            try:
                comp_dir = Path(compiler.active_compilations[comp_id]['directory'])
                if comp_dir.exists():
                    stat = comp_dir.stat()
                    if stat.st_mtime < cutoff_time:
                        compiler.cleanup_compilation(comp_id)
                        print(f"🧹 Cleaned up old compilation: {comp_id}")
            except Exception as e:
                print(f"Error cleaning up compilation {comp_id}: {e}")

# Schedule cleanup every hour
def periodic_cleanup():
    while True:
        time.sleep(3600)  # 1 hour
        try:
            cleanup_old_compilations()
            clean_old_sessions()
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")

# Start cleanup thread
try:
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    print("✅ Cleanup thread started")
except Exception as e:
    print(f"⚠️ Could not start cleanup thread: {e}")

# Setup enhanced routes
setup_enhanced_compilation_routes(app)

# Test all routes are registered
print("🔍 Testing route registration...")
try:
    with app.app_context():
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(f"{rule.rule} -> {rule.endpoint}")
        print(f"✅ {len(routes)} routes registered successfully")
        for route in routes:
            print(f"   {route}")
except Exception as e:
    print(f"❌ Error testing routes: {e}")

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" STARTING ENHANCED FLASK SERVER")
    print("="*80)
    print(" 🗣️  Chat & Conversation:")
    print("   • http://localhost:5001/                    - API Info")
    print("   • http://localhost:5001/api/chat            - Chat API (with context)")
    print("   • http://localhost:5001/api/status          - System status")
    print("   • http://localhost:5001/api/clear_session   - Clear conversation")
    print("   • http://localhost:5001/api/session_info/<id> - Session details")
    print("   • http://localhost:5001/health              - Health check")
    print()
    print(" 💻 Code Compilation & Execution:")
    print("   • http://localhost:5001/api/generate-test   - Generate test scripts")
    print("   • http://localhost:5001/api/compile         - Compile code")
    print("   • http://localhost:5001/api/execute         - Execute compiled code")
    print("   • http://localhost:5001/api/cleanup-compilation/<id> - Cleanup files")
    print()
    print(" 🔬 Enhanced Dependency Management:")
    print("   • http://localhost:5001/api/analyze-dependencies - Analyze code dependencies")
    print("   • http://localhost:5001/api/install-dependency   - Install missing packages")
    print("   • http://localhost:5001/api/execute-enhanced     - Execute with dependency checks")
    print()
    print(" 🎯 System Status:")
    print(f"   ✅ RAG System: {'Loaded' if rag_system else 'Failed'}")
    enhanced_status = 'Enhanced' if (compiler and hasattr(compiler, 'dep_manager')) else ('Basic' if compiler else 'Failed')
    print(f"   ✅ Compiler: {enhanced_status}")
    print()
    print(" 🔧 System Requirements Check:")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434", timeout=3)
        print("   ✅ Ollama is running")
    except:
        print("   ❌ Ollama not found - install and run: ollama serve")
    
    # Check compilers
    compilers_to_check = {
        'gcc': 'C compiler',
        'g++': 'C++ compiler', 
        'nvcc': 'CUDA compiler',
        'python3': 'Python interpreter'
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
    
    print()
    print("="*80)
    print(" 🚀 STARTING FLASK SERVER ON http://localhost:5001")
    print(" 📊 Visit /api/status to check all system components")
    print(" 🛑 Press Ctrl+C to stop the server")
    print("="*80)
    
    try:
        # Start the Flask server with better error handling
        app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)