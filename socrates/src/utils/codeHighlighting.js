// Syntax highlighting function for CUDA/C++ code
export const highlightCudaCode = (code) => {
  // CUDA/C++ keywords
  const keywords = [
    '__global__', '__device__', '__host__', '__shared__', '__constant__',
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue', 'return',
    'int', 'float', 'double', 'char', 'bool', 'void', 'auto', 'const', 'static', 'extern',
    'struct', 'class', 'enum', 'typedef', 'sizeof', 'namespace', 'using',
    'cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaDeviceSynchronize', 'cudaGetLastError',
    'blockIdx', 'blockDim', 'threadIdx', 'gridDim', '__syncthreads'
  ];

  // CUDA functions and types
  const cudaFunctions = [
    'cudaEventCreate', 'cudaEventRecord', 'cudaEventSynchronize', 'cudaEventElapsedTime',
    'cudaEventDestroy', 'cudaMallocManaged', 'cudaMemcpyAsync', 'cudaStreamCreate',
    'cudaStreamSynchronize', 'cudaStreamDestroy', 'cudaGetDeviceProperties',
    'dim3', 'cudaError_t', 'cudaStream_t', 'cudaEvent_t'
  ];

  // Numbers and operators
  let highlighted = code
    // Comments (green)
    .replace(/(\/\/.*$)/gm, '<span class="code-comment">$1</span>')
    .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="code-comment">$1</span>')
    
    // Strings (yellow)
    .replace(/(".*?")/g, '<span class="code-string">$1</span>')
    .replace(/('.*?')/g, '<span class="code-string">$1</span>')
    
    // Numbers (cyan)
    .replace(/\b(\d+\.?\d*[fF]?)\b/g, '<span class="code-number">$1</span>')
    
    // Preprocessor directives (magenta)
    .replace(/(#\w+)/g, '<span class="code-preprocessor">$1</span>')
    
    // CUDA-specific keywords (bright blue)
    .replace(new RegExp(`\\b(${keywords.join('|')})\\b`, 'g'), '<span class="code-keyword">$1</span>')
    
    // CUDA functions (orange)
    .replace(new RegExp(`\\b(${cudaFunctions.join('|')})\\b`, 'g'), '<span class="code-function">$1</span>')
    
    // Function calls (light blue)
    .replace(/\b(\w+)(\s*\()/g, '<span class="code-function-call">$1</span>$2')
    
    // Operators (white/gray)
    .replace(/([+\-*/%=<>!&|^~])/g, '<span class="code-operator">$1</span>')
    
    // Brackets and parentheses (bright white)
    .replace(/([(){}\[\]])/g, '<span class="code-bracket">$1</span>');

  return highlighted;
};