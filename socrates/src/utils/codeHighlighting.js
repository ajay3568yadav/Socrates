// Simple CUDA code highlighting utility
export const highlightCudaCode = (code) => {
  if (!code) return '';

  // CUDA-specific keywords and functions
  const patterns = [
    // CUDA qualifiers
    { regex: /\b(__global__|__device__|__host__|__shared__|__constant__)\b/g, class: 'code-keyword' },
    
    // CUDA built-in variables
    { regex: /\b(threadIdx|blockIdx|blockDim|gridDim)\b/g, class: 'code-function-call' },
    
    // CUDA functions
    { regex: /\b(__syncthreads|__threadfence|__threadfence_block|__threadfence_system)\b/g, class: 'code-function' },
    
    // Memory functions
    { regex: /\b(cudaMalloc|cudaFree|cudaMemcpy|cudaMemset|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost)\b/g, class: 'code-function' },
    
    // Standard C/C++ keywords
    { regex: /\b(int|float|double|char|void|bool|if|else|for|while|do|switch|case|break|continue|return|sizeof|typedef|struct|union|enum|const|static|extern|inline)\b/g, class: 'code-keyword' },
    
    // Preprocessor directives
    { regex: /^#\s*(include|define|ifdef|ifndef|endif|pragma|undef|if|else|elif)\b.*$/gm, class: 'code-preprocessor' },
    
    // Numbers
    { regex: /\b\d+\.?\d*[fF]?\b/g, class: 'code-number' },
    
    // Strings
    { regex: /"([^"\\]|\\.)*"/g, class: 'code-string' },
    { regex: /'([^'\\]|\\.)*'/g, class: 'code-string' },
    
    // Comments
    { regex: /\/\/.*$/gm, class: 'code-comment' },
    { regex: /\/\*[\s\S]*?\*\//g, class: 'code-comment' },
    
    // Operators
    { regex: /[+\-*/%=<>!&|^~?:]/g, class: 'code-operator' },
    
    // Brackets and parentheses
    { regex: /[(){}\[\]]/g, class: 'code-bracket' },
  ];

  let highlightedCode = code;

  // Apply syntax highlighting
  patterns.forEach(pattern => {
    highlightedCode = highlightedCode.replace(
      pattern.regex,
      `<span class="${pattern.class}">$&</span>`
    );
  });

  return highlightedCode;
};