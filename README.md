# Socrates - CUDA Programming Tutor

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running ([Download here](https://ollama.ai))
3. **Git** (optional, for version control)

## 🚀 Quick Setup

### 1. Clone/Download Project
```bash
# If using git
git clone https://github.com/ajay3568yadav/Socrates.git
cd socrates

# Or download and extract the files to a folder
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv cuda_rag_env

# Activate it
# On macOS/Linux:
source cuda_rag_env/bin/activate
# On Windows:
cuda_rag_env\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you have a CUDA GPU (optional optimization):
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 4. Setup Ollama
```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:latest

# Test the model
ollama run llama3.2:latest "Hello, are you working?"
```

### 5. Prepare Project Files
Make sure you have these files in your project directory:
```
cuda-rag-chat/
├── requirements.txt
├── backend.py          # The enhanced Flask backend
├── index.html          # The chat interface
└── README.md           # This file
```

### 6. Run the Application
```bash
# Start the Flask backend
python backend.py

# Open your browser to:
# http://localhost:5001
```

## 📊 Datasets Used for RAG

### Current Dataset
Our CUDA tutor uses the **SakanaAI AI-CUDA-Engineer-Archive** dataset to provide intelligent responses and code examples:

#### **SakanaAI AI-CUDA-Engineer-Archive**
- **Source**: [Hugging Face - SakanaAI/AI-CUDA-Engineer-Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)
- **Size**: ~30,000 CUDA kernels
- **License**: CC-By-4.0
- **Content**: CUDA kernels generated by The AI CUDA Engineer
- **Features**:
  - PyTorch reference implementations
  - CUDA kernel code examples
  - Performance benchmarks and speedup metrics
  - NCU and Clang-tidy profiling data
  - Error messages and debugging information
  - Multi-level tasks (Level 1, 2, 3)

**Loading in Code:**
```python
from datasets import load_dataset
dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
df = dataset["level_1"].to_pandas()
```

### Other Publicly Available CUDA/GPU Programming Datasets

#### **1. KernelBench by Stanford Scaling Intelligence Lab**
- **Source**: [Hugging Face - ScalingIntelligence/KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)
- **Paper**: [KernelBench: Can LLMs Write GPU Kernels?](https://scalingintelligence.stanford.edu/blogs/kernelbench/)
- **Size**: 250 well-defined neural network tasks
- **Content**:
  - Foundational GPU operators
  - Simple fusion patterns
  - Full ML architectures
  - Performance benchmarks
  - Correctness verification tests

#### **2. NVIDIA CUDA Samples**
- **Source**: [GitHub - NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- **Content**:
  - Official CUDA programming examples
  - Performance benchmarks
  - Best practices demonstrations
  - Hardware feature showcases
  - Covers: cuBLAS, cuFFT, cuSPARSE, Thrust, etc.

#### **3. NVIDIA Deep Learning Examples**
- **Source**: [GitHub - NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)
- **Content**:
  - State-of-the-art deep learning implementations
  - Optimized CUDA kernels for AI workloads
  - Performance benchmarks
  - Multi-GPU training examples

#### **4. Rodinia Benchmark Suite**
- **Source**: [University of Virginia Rodinia](http://lava.cs.virginia.edu/Rodinia/) | [GitHub Fork](https://github.com/yuhc/gpu-rodinia)
- **Content**:
  - Heterogeneous computing benchmarks
  - CPU and GPU implementations
  - Diverse parallel computing patterns
  - Applications: BFS, K-means, Matrix multiplication, etc.
  - Multiple programming models: CUDA, OpenCL, OpenMP

#### **5. PolyBench/GPU**
- **Content**:
  - Collection of benchmarks for polyhedral compilation
  - CUDA and OpenCL implementations
  - Linear algebra and stencil computations
  - Scientific computing kernels

#### **6. GPU Benchmark Collections on Kaggle**
- **GPU Benchmarks Compilation**: [Kaggle Dataset](https://www.kaggle.com/datasets/alanjo/gpu-benchmarks)
  - Performance data for various GPUs
  - Benchmarking results and comparisons

#### **7. Specialized Research Datasets**
- **Machine Learning for Code**: [GitHub - CUHK-ARISE/ml4code-dataset](https://github.com/CUHK-ARISE/ml4code-dataset)
  - Code analysis and vulnerability detection
  - Source code datasets for ML applications
- **CUDA Learning Resources**: [GitHub - rkinas/cuda-learning](https://github.com/rkinas/cuda-learning)
  - Curated CUDA programming tutorials and examples
  - Learning path for GPU programming

### 🎯 Dataset Usage in Socrates

The RAG system in Socrates leverages these datasets to:

1. **Provide Code Examples**: Real CUDA kernels and implementations
2. **Performance Insights**: Speedup metrics and optimization techniques
3. **Error Handling**: Common CUDA programming errors and solutions
4. **Best Practices**: Industry-standard CUDA programming patterns
5. **Interactive Learning**: Context-aware responses based on user queries

### 📈 Performance Metrics from Our Dataset

From the SakanaAI dataset, our system can reference:
- **Operation Types**: Matrix multiplication, convolution, attention mechanisms
- **Speedup Ranges**: 1.2x to 381x performance improvements
- **Hardware Compatibility**: Support for various NVIDIA GPU architectures
- **Optimization Techniques**: Memory coalescing, shared memory usage, register optimization

## 🧪 Testing Your Setup

### Test 1: Check Backend Status
```bash
curl http://localhost:5001/health
```
Should return: `{"status": "healthy", ...}`

### Test 2: Test Chat API
```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is CUDA?", "session_id": "test"}'
```

### Test 3: Open Web Interface
Navigate to `http://localhost:5001` and try these questions:
1. "What is CUDA?"
2. "Show me examples"
3. "Optimize those examples"

## 📚 Learning Resources

### Additional CUDA Learning Materials
- **NVIDIA CUDA Programming Guide**: Official documentation
- **CUDA Zone**: [developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)
- **CUDA Toolkit**: [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
- **GPU Programming Tutorials**: Interactive examples and best practices

### Research Papers
- "The AI CUDA Engineer: Agentic CUDA Kernel Discovery and Optimization" - SakanaAI
- "KernelBench: Can LLMs Write GPU Kernels?" - Stanford Scaling Intelligence Lab
- "Rodinia: A benchmark suite for heterogeneous computing" - University of Virginia
