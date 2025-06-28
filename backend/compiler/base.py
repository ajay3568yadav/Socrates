#!/usr/bin/env python3
"""
Base code compiler for CUDA Tutor
"""

import os
import time
import uuid
import shutil
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Dict, Any

from config import get_config

# Get configuration
config = get_config()

class CodeCompiler:
    """Base code compiler class"""
    
    def __init__(self):
        try:
            self.temp_dir = config.TEMP_DIR
            self.temp_dir.mkdir(exist_ok=True)
            self.active_compilations = {}
            self.compilation_timeout = config.COMPILATION_TIMEOUT
            self.execution_timeout = config.EXECUTION_TIMEOUT
            self.max_output_size = config.MAX_OUTPUT_SIZE
            # Determine Python command based on OS
            if platform.system() == 'Windows':
                # On Windows, use the full path to Python if 'python' command doesn't work
                try:
                    # First try the python command
                    result = subprocess.run(['python', '--version'], capture_output=True, timeout=2)
                    if result.returncode == 0:
                        self.python_cmd = 'python'
                    else:
                        # If that fails, use the full path
                        self.python_cmd = r'C:\Users\Captain\AppData\Local\Programs\Python\Python310\python.exe'
                except:
                    # Use full path as fallback
                    self.python_cmd = r'C:\Users\Captain\AppData\Local\Programs\Python\Python310\python.exe'
            else:
                self.python_cmd = 'python3'
            print(f"✅ CodeCompiler initialized successfully (using {self.python_cmd})")
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
                'run_command': f'{self.python_cmd} {{source}}'
            }
        }
        return configs.get(language, configs['c'])
    
    def generate_test_script_with_ai(self, code: str, language: str) -> str:
        """Generate a test script using the existing AI model"""
        try:
            # Import here to avoid circular imports
            from models.rag import get_rag_system
            
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
            rag_system = get_rag_system()
            if rag_system:
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
                    'directory': str(comp_dir),
                    'timestamp': time.time()
                }
                return {
                    'success': True,
                    'compilationId': compilation_id,
                    'output': f'{language.title()} code ready for execution',
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
                timeout=self.compilation_timeout
            )
            
            # Store compilation info
            self.active_compilations[compilation_id] = {
                'language': language,
                'source_file': str(source_file),
                'executable': str(executable),
                'directory': str(comp_dir),
                'timestamp': time.time()
            }
            
            return {
                'success': result.returncode == 0,
                'compilationId': compilation_id,
                'output': result.stdout[:self.max_output_size],
                'stderr': result.stderr[:self.max_output_size],
                'returnCode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'compilationId': compilation_id,
                'output': '',
                'stderr': f'Compilation timed out after {self.compilation_timeout} seconds',
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
                run_cmd = [self.python_cmd, comp_info['source_file']]
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
                timeout=self.execution_timeout,
                env=env
            )
            
            execution_time = int((time.time() - start_time) * 1000)  # milliseconds
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout[:self.max_output_size],
                'stderr': result.stderr[:self.max_output_size],
                'exitCode': result.returncode,
                'executionTime': execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'stderr': f'Execution timed out after {self.execution_timeout} seconds',
                'exitCode': -1,
                'executionTime': self.execution_timeout * 1000
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'stderr': f'Execution error: {str(e)}',
                'exitCode': -1,
                'executionTime': 0
            }
    
    def cleanup_compilation(self, compilation_id: str) -> bool:
        """Clean up compilation files"""
        if compilation_id not in self.active_compilations:
            return False
            
        comp_info = self.active_compilations[compilation_id]
        try:
            shutil.rmtree(comp_info['directory'])
            print(f"🧹 Cleaned up compilation: {compilation_id}")
        except Exception as e:
            print(f"⚠️ Error cleaning up compilation {compilation_id}: {e}")
        finally:
            del self.active_compilations[compilation_id]
        
        return True
    
    def cleanup_old_compilations(self, max_age_hours: int = 1) -> int:
        """Clean up old compilation directories"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        cleaned_count = 0
        
        # Clean up temporary directories older than max_age_hours
        for comp_id in list(self.active_compilations.keys()):
            try:
                comp_info = self.active_compilations[comp_id]
                if comp_info.get('timestamp', 0) < cutoff_time:
                    if self.cleanup_compilation(comp_id):
                        cleaned_count += 1
            except Exception as e:
                print(f"Error cleaning up compilation {comp_id}: {e}")
        
        return cleaned_count
    
    def get_compilation_info(self, compilation_id: str) -> Dict[str, Any]:
        """Get information about a specific compilation"""
        if compilation_id not in self.active_compilations:
            return {'exists': False}
        
        comp_info = self.active_compilations[compilation_id]
        return {
            'exists': True,
            'language': comp_info['language'],
            'directory': comp_info['directory'],
            'timestamp': comp_info['timestamp'],
            'age_seconds': time.time() - comp_info['timestamp']
        }
    
    def list_active_compilations(self) -> Dict[str, Any]:
        """List all active compilations"""
        current_time = time.time()
        compilations = {}
        
        for comp_id, comp_info in self.active_compilations.items():
            compilations[comp_id] = {
                'language': comp_info['language'],
                'age_seconds': current_time - comp_info.get('timestamp', current_time),
                'directory_exists': Path(comp_info['directory']).exists()
            }
        
        return {
            'count': len(compilations),
            'compilations': compilations
        }