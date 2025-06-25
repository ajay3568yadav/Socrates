# Socrates

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
├── index.html     # The chat interface
└── README.md                      # This file
```

### 6. Run the Application
```bash
# Start the Flask backend
python backend.py

# Open your browser to:
# http://localhost:5001
```

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