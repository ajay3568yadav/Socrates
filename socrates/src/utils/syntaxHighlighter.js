// src/utils/syntaxHighlighter.js
// Enhanced syntax highlighting utility for CUDA/C/C++ code

export const applySyntaxHighlighting = (code, language = 'c') => {
  if (!code) return '';

  // Escape HTML first
  let highlightedCode = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Define syntax patterns based on language with proper order
  const getPatterns = (lang) => {
    switch (lang.toLowerCase()) {
      case 'python':
        return [
          // Multi-line strings first (highest priority)
          { regex: /"""[\s\S]*?"""/g, class: 'string' },
          { regex: /'''[\s\S]*?'''/g, class: 'string' },
          
          // Single line comments
          { regex: /#.*$/gm, class: 'comment' },
          
          // Strings (double and single quoted)
          { regex: /"([^"\\]|\\.)*"/g, class: 'string' },
          { regex: /'([^'\\]|\\.)*'/g, class: 'string' },
          
          // Python keywords
          { regex: /\b(def|class|import|from|as|with|lambda|yield|async|await|try|except|finally|raise|assert|pass|del|global|nonlocal|and|or|not|in|is|if|else|elif|for|while|break|continue|return)\b/g, class: 'keyword' },
          
          // Python built-ins
          { regex: /\b(self|cls|True|False|None|print|len|range|str|int|float|list|dict|tuple|set)\b/g, class: 'builtin' },
          
          // Numbers
          { regex: /\b(\d+\.?\d*[jJ]?|\d+[eE][+-]?\d+)\b/g, class: 'number' },
          
          // Function calls
          { regex: /\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, class: 'function-call' },
          
          // Operators
          { regex: /[+\-*/%=<>!&|^~@]+/g, class: 'operator' },
          
          // Brackets and parentheses
          { regex: /[(){}\[\]]/g, class: 'bracket' },
        ];

      case 'javascript':
      case 'typescript':
        return [
          // Multi-line comments first
          { regex: /\/\*[\s\S]*?\*\//g, class: 'comment' },
          
          // Single line comments
          { regex: /\/\/.*$/gm, class: 'comment' },
          
          // Template literals
          { regex: /`([^`\\]|\\.)*`/g, class: 'string' },
          
          // Strings (double and single quoted)
          { regex: /"([^"\\]|\\.)*"/g, class: 'string' },
          { regex: /'([^'\\]|\\.)*'/g, class: 'string' },
          
          // JavaScript/TypeScript keywords
          { regex: /\b(function|var|let|const|class|extends|import|export|from|default|async|await|promise|try|catch|finally|throw|typeof|instanceof|new|this|super|if|else|for|while|do|switch|case|break|continue|return)\b/g, class: 'keyword' },
          
          // Built-ins
          { regex: /\b(true|false|null|undefined|NaN|Infinity|console|window|document)\b/g, class: 'builtin' },
          
          // Numbers
          { regex: /\b(0x[0-9a-fA-F]+|0b[01]+|\d+\.?\d*[eE][+-]?\d+|\d+\.?\d*)\b/g, class: 'number' },
          
          // Function calls
          { regex: /\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=\()/g, class: 'function-call' },
          
          // Operators
          { regex: /[+\-*/%=<>!&|^~?:]+/g, class: 'operator' },
          
          // Brackets and parentheses
          { regex: /[(){}\[\]]/g, class: 'bracket' },
        ];

      case 'cuda':
      case 'cpp':
      case 'c':
      default:
        return [
          // Multi-line comments first
          { regex: /\/\*[\s\S]*?\*\//g, class: 'comment' },
          
          // Single line comments
          { regex: /\/\/.*$/gm, class: 'comment' },
          
          // Preprocessor directives
          { regex: /^#\s*(include|define|ifdef|ifndef|endif|pragma|undef|if|else|elif|error|warning|line)\b.*$/gm, class: 'preprocessor' },
          
          // Strings (double and single quoted)
          { regex: /"([^"\\]|\\.)*"/g, class: 'string' },
          { regex: /'([^'\\]|\\.)*'/g, class: 'string' },
          
          // CUDA qualifiers
          { regex: /\b(__global__|__device__|__host__|__shared__|__constant__|__restrict__)\b/g, class: 'cuda-qualifier' },
          
          // CUDA built-in variables
          { regex: /\b(threadIdx|blockIdx|blockDim|gridDim|warpSize)\b/g, class: 'cuda-builtin' },
          
          // CUDA functions and memory functions
          { regex: /\b(__syncthreads|__threadfence|__threadfence_block|__threadfence_system|atomicAdd|atomicSub|atomicMax|atomicMin|atomicCAS|cudaMalloc|cudaFree|cudaMemcpy|cudaMemset|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|cudaDeviceSynchronize)\b/g, class: 'cuda-function' },
          
          // C/C++ keywords
          { regex: /\b(int|float|double|char|void|bool|short|long|unsigned|signed|const|static|extern|inline|volatile|register|auto|typedef|struct|union|enum|class|public|private|protected|virtual|override|if|else|for|while|do|switch|case|default|break|continue|return|goto|sizeof|new|delete|this|nullptr|true|false)\b/g, class: 'keyword' },
          
          // Numbers (integers, floats, hex, binary)
          { regex: /\b(0x[0-9a-fA-F]+[uUlL]*|0b[01]+[uUlL]*|\d+\.?\d*[fFlL]?[uUlL]*)\b/g, class: 'number' },
          
          // Function calls
          { regex: /\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, class: 'function-call' },
          
          // Operators
          { regex: /[+\-*/%=<>!&|^~?:]+/g, class: 'operator' },
          
          // Brackets and parentheses
          { regex: /[(){}\[\]]/g, class: 'bracket' },
        ];
    }
  };

  // Apply syntax highlighting with careful replacement to avoid nested spans
  const patterns = getPatterns(language);
  
  // Create a tokenization approach to avoid conflicts
  const tokens = [];
  let currentIndex = 0;
  
  // Find all matches first
  patterns.forEach((pattern, patternIndex) => {
    const regex = new RegExp(pattern.regex.source, pattern.regex.flags);
    let match;
    
    while ((match = regex.exec(highlightedCode)) !== null) {
      tokens.push({
        start: match.index,
        end: match.index + match[0].length,
        content: match[0],
        class: pattern.class,
        priority: patternIndex // Earlier patterns have higher priority
      });
      
      // Prevent infinite loop for zero-length matches
      if (match.index === regex.lastIndex) {
        regex.lastIndex++;
      }
    }
  });
  
  // Sort tokens by start position, then by priority (lower number = higher priority)
  tokens.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return a.priority - b.priority;
  });
  
  // Remove overlapping tokens (keep higher priority ones)
  const filteredTokens = [];
  for (let i = 0; i < tokens.length; i++) {
    const current = tokens[i];
    let hasOverlap = false;
    
    for (let j = 0; j < filteredTokens.length; j++) {
      const existing = filteredTokens[j];
      if (current.start < existing.end && current.end > existing.start) {
        hasOverlap = true;
        break;
      }
    }
    
    if (!hasOverlap) {
      filteredTokens.push(current);
    }
  }
  
  // Sort final tokens by start position (descending) for safe replacement
  filteredTokens.sort((a, b) => b.start - a.start);
  
  // Apply highlighting by replacing from end to start
  filteredTokens.forEach(token => {
    const before = highlightedCode.substring(0, token.start);
    const highlighted = `<span class="${token.class}">${token.content}</span>`;
    const after = highlightedCode.substring(token.end);
    highlightedCode = before + highlighted + after;
  });

  return highlightedCode;
};

// Generate line numbers for code blocks
export const generateLineNumbers = (code) => {
  const lines = code.split('\n');
  return lines.map((_, index) => index + 1).join('\n');
};

// Get CSS styles for syntax highlighting
export const getSyntaxHighlightingCSS = () => `
  .keyword { color: #569cd6; font-weight: 500; }
  .string { color: #ce9178; }
  .comment { color: #6a9955; font-style: italic; }
  .number { color: #b5cea8; }
  .function-call { color: #dcdcaa; }
  .cuda-qualifier { color: #c586c0; font-weight: bold; }
  .cuda-builtin { color: #4ec9b0; font-weight: 500; }
  .cuda-function { color: #dcdcaa; }
  .builtin { color: #4ec9b0; }
  .preprocessor { color: #9b9b9b; }
  .operator { color: #d4d4d4; }
  .bracket { color: #ffd700; }
`;

// Dark theme color palette
export const SYNTAX_COLORS = {
  background: '#1e1e1e',
  backgroundSecondary: '#252526',
  backgroundTertiary: '#2d2d30',
  foreground: '#d4d4d4',
  foregroundSecondary: '#858585',
  border: '#3e3e3e',
  keyword: '#569cd6',
  string: '#ce9178',
  comment: '#6a9955',
  number: '#b5cea8',
  functionCall: '#dcdcaa',
  cudaQualifier: '#c586c0',
  cudaBuiltin: '#4ec9b0',
  cudaFunction: '#dcdcaa',
  preprocessor: '#9b9b9b',
  operator: '#d4d4d4',
  bracket: '#ffd700',
  accent: '#007acc',
  success: '#22c55e',
  warning: '#f97316',
  error: '#f85149'
};