#!/usr/bin/env python3
"""
Enhanced Flask Backend for CUDA Chat Interface with Context Memory - FIXED
Connects the web chat to your RAG system with conversation tracking
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
import psutil
import threading
from collections import deque

# GPU monitoring imports
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Create Flask app first
app = Flask(__name__)

# Configure CORS properly - AFTER creating the app
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"])

conversation_sessions = {}

# System monitoring variables
cpu_readings = deque(maxlen=10)  # Store last 10 CPU readings for averaging
memory_readings = deque(maxlen=10)  # Store last 10 memory readings for averaging

def get_system_metrics():
    """Get current system memory and CPU usage"""
    try:
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)  # Convert to MB
        memory_usage_gb = memory_usage_mb / 1024  # Convert to GB
        memory_percent = memory.percent
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'memory_mb': round(memory_usage_mb, 2),
            'memory_gb': round(memory_usage_gb, 2),
            'memory_percent': round(memory_percent, 1),
            'cpu_percent': round(cpu_percent, 1)
        }
    except Exception as e:
        print(f"⚠️ Error getting system metrics: {e}")
        return {
            'memory_mb': 0,
            'memory_gb': 0,
            'memory_percent': 0,
            'cpu_percent': 0
        }

def get_average_cpu_usage():
    """Get average CPU usage from recent readings"""
    if not cpu_readings:
        return 0
    return round(sum(cpu_readings) / len(cpu_readings), 1)

def get_average_memory_usage():
    """Get average memory usage from recent readings"""
    if not memory_readings:
        return 0
    return round(sum(memory_readings) / len(memory_readings), 1)

def update_system_readings():
    """Update system readings for averaging"""
    metrics = get_system_metrics()
    cpu_readings.append(metrics['cpu_percent'])
    memory_readings.append(metrics['memory_percent'])

def print_system_usage():
    """Print current system usage to terminal"""
    metrics = get_system_metrics()
    avg_cpu = get_average_cpu_usage()
    avg_memory = get_average_memory_usage()
    
    print("=" * 60)
    print("🖥️  SYSTEM USAGE METRICS")
    print("=" * 60)
    print(f"💾 Memory Usage: {metrics['memory_gb']:.2f} GB ({metrics['memory_percent']:.1f}%)")
    print(f"📊 Average Memory (last 10): {avg_memory:.1f}%")
    print(f"⚡ CPU Usage: {metrics['cpu_percent']:.1f}%")
    print(f"📈 Average CPU (last 10): {avg_cpu:.1f}%")
    print("=" * 60)

def get_gpu_info():
    """Get GPU information including CUDA availability and memory usage"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_model': 'None',
        'gpu_memory_used_gb': 0,
        'gpu_memory_total_gb': 0,
        'gpu_memory_percent': 0,
        'gpu_utilization_percent': 0
    }
    
    # First try GPUtil for comprehensive system-wide GPU info
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Get first GPU
                gpu_info['gpu_model'] = gpu.name
                gpu_info['gpu_utilization_percent'] = round(gpu.load * 100, 1)
                # Fix memory conversion - GPUtil gives MB, convert to GB
                gpu_info['gpu_memory_total_gb'] = round(gpu.memoryTotal / 1024, 2)
                gpu_info['gpu_memory_used_gb'] = round(gpu.memoryUsed / 1024, 2)
                gpu_info['gpu_memory_percent'] = round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1)
                gpu_info['gpu_count'] = len(gpus)
                print(f"🎮 GPUtil readings: {gpu_info['gpu_memory_used_gb']:.2f}/{gpu_info['gpu_memory_total_gb']:.2f} GB ({gpu_info['gpu_memory_percent']:.1f}%)")
        except Exception as e:
            print(f"⚠️ GPUtil error: {e}")
    
    # Check CUDA availability via PyTorch (but don't override GPUtil memory readings)
    if TORCH_AVAILABLE:
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            if gpu_info['gpu_count'] == 0:  # Only set if GPUtil didn't work
                gpu_info['gpu_count'] = torch.cuda.device_count()
            
            if gpu_info['gpu_count'] > 0:
                if gpu_info['gpu_model'] == 'None':  # Only set if GPUtil didn't work
                    gpu_info['gpu_model'] = torch.cuda.get_device_name(0)
                
                # Only use PyTorch memory if GPUtil failed
                if gpu_info['gpu_memory_total_gb'] == 0:
                    try:
                        # Get PyTorch memory info (includes cached memory for more accurate usage)
                        reserved = torch.cuda.memory_reserved(0) / (1024**3)  # Reserved memory (more accurate than allocated)
                        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # Actually allocated memory
                        
                        # Get total memory from device properties
                        gpu_memory = torch.cuda.get_device_properties(0)
                        total_gb = gpu_memory.total_memory / (1024**3)
                        
                        # Use reserved memory as it's more representative of actual usage
                        used_memory = max(reserved, allocated)
                        
                        gpu_info['gpu_memory_total_gb'] = round(total_gb, 2)
                        gpu_info['gpu_memory_used_gb'] = round(used_memory, 2)
                        gpu_info['gpu_memory_percent'] = round((used_memory / total_gb) * 100, 1)
                        
                        print(f"🔥 PyTorch readings: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Using reserved for accuracy")
                    except Exception as e:
                        print(f"⚠️ PyTorch memory error: {e}")
    
    return gpu_info

def print_enhanced_system_usage(response_time=None):
    """Print enhanced system usage including GPU info and response time"""
    metrics = get_system_metrics()
    gpu_info = get_gpu_info()
    avg_cpu = get_average_cpu_usage()
    avg_memory = get_average_memory_usage()
    
    print("=" * 70)
    print("🖥️  ENHANCED SYSTEM METRICS")
    print("=" * 70)
    
    # Response timing
    if response_time:
        print(f"⏱️  Response Time: {response_time:.2f} seconds")
        print("-" * 70)
    
    # CPU and Memory
    print(f"💾 Memory Usage: {metrics['memory_gb']:.2f} GB ({metrics['memory_percent']:.1f}%)")
    print(f"📊 Average Memory (last 10): {avg_memory:.1f}%")
    print(f"⚡ CPU Usage: {metrics['cpu_percent']:.1f}%")
    print(f"📈 Average CPU (last 10): {avg_cpu:.1f}%")
    print("-" * 70)
    
    # GPU Information - Enhanced with debugging
    print(f"🎮 CUDA Available: {'✅ Yes' if gpu_info['cuda_available'] else '❌ No'}")
    print(f"🔢 GPU Count: {gpu_info['gpu_count']}")
    print(f"🏷️  GPU Model: {gpu_info['gpu_model']}")
    
    if gpu_info['cuda_available'] and gpu_info['gpu_count'] > 0:
        print(f"🔥 GPU Memory: {gpu_info['gpu_memory_used_gb']:.2f} GB / {gpu_info['gpu_memory_total_gb']:.2f} GB ({gpu_info['gpu_memory_percent']:.1f}%)")
        print(f"⚙️  GPU Utilization: {gpu_info['gpu_utilization_percent']:.1f}%")
        
        # Add detailed GPU debugging info
        if gpu_info['gpu_memory_percent'] < 1.0:
            print("🔍 DEBUG: Low GPU memory usage detected")
            if GPU_AVAILABLE:
                print("   - GPUtil is available for system-wide monitoring")
            if TORCH_AVAILABLE:
                print("   - PyTorch is available for CUDA memory tracking")
    else:
        print("🔥 GPU Memory: Not available")
        print("⚙️  GPU Utilization: Not available")
    
    print("=" * 70)

class SimpleRAG:
    def __init__(self):
        print("🔧 Initializing RAG system...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load CUDA dataset
        print(" Loading CUDA dataset...")
        try:
            dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
            self.df = dataset["level_1"].to_pandas().head(50)  # Limit for speed
        except Exception as e:
            print(f"⚠️ Could not load dataset: {e}")
            # Create dummy data for demo
            self.df = pd.DataFrame({
                'Op_Name': ['Matrix Multiply', 'Vector Add', 'Convolution'],
                'CUDA_Code': ['__global__ void matmul(...)', '__global__ void vecadd(...)', '__global__ void conv(...)'],
                'CUDA_Speedup_Native': [15.2, 8.5, 12.3]
            })
        
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
                    "model": "qwen2.5-coder:14b",
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

# Initialize RAG system
print("🔧 Starting Flask server with RAG...")
try:
    rag_system = SimpleRAG()
except Exception as e:
    print(f" RAG initialization failed: {e}")
    rag_system = None

# Initialize system monitoring baseline
print("📊 Initializing system monitoring...")
for _ in range(5):  # Take 5 initial readings to establish baseline
    update_system_readings()
    time.sleep(0.2)
print("✅ System monitoring initialized")

# Initialize GPU detection
print("🎮 Detecting GPU capabilities...")
gpu_info = get_gpu_info()
if gpu_info['cuda_available']:
    print(f"✅ CUDA detected: {gpu_info['gpu_model']} ({gpu_info['gpu_memory_total_gb']:.1f} GB)")
else:
    print("❌ No CUDA-capable GPU detected")
print("✅ GPU detection completed")

@app.route('/')
def index():
    """Serve the chat interface"""
    return jsonify({
        'message': 'CUDA Tutor Backend is running!',
        'status': 'healthy',
        'endpoints': {
            'chat': '/api/chat',
            'status': '/api/status',
            'health': '/health'
        }
    })

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
        
        # Start timing the response
        start_time = time.time()
        
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
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f" Session {session_id}: {response[:100]}...")
        print(f" Context used: {bool(conversation_context)}")
        print(f"🔗 Follow-up detected: {is_follow_up}")
        
        # Update system readings and print enhanced usage metrics
        update_system_readings()
        print_enhanced_system_usage(response_time)
        
        return jsonify({
            'response': response,
            'session_id': session_id,
            'is_follow_up': is_follow_up,
            'context_used': bool(conversation_context),
            'status': 'success'
        })
        
    except Exception as e:
        print(f" Error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/status')
def status():
    """Check system status"""
    # Get current system metrics
    current_metrics = get_system_metrics()
    gpu_info = get_gpu_info()
    avg_cpu = get_average_cpu_usage()
    avg_memory = get_average_memory_usage()
    
    status_info = {
        'rag_loaded': rag_system is not None,
        'knowledge_count': len(rag_system.knowledge) if rag_system else 0,
        'active_sessions': len(conversation_sessions),
        'ollama_status': 'unknown',
        'system_metrics': {
            'current_memory_gb': current_metrics['memory_gb'],
            'current_memory_percent': current_metrics['memory_percent'],
            'average_memory_percent': avg_memory,
            'current_cpu_percent': current_metrics['cpu_percent'],
            'average_cpu_percent': avg_cpu
        },
        'gpu_metrics': gpu_info
    }
    
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        status_info['ollama_status'] = 'connected' if response.status_code == 200 else 'error'
    except:
        status_info['ollama_status'] = 'disconnected'
    
    return jsonify(status_info)

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
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/session_info/<session_id>')
def session_info(session_id):
    """Get information about a specific session"""
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

#Visit http://localhost:5001/api/gpu_debug to see detailed GPU information

@app.route('/api/gpu_debug')
def gpu_debug():
    """Detailed GPU debugging information"""
    debug_info = {
        'gpu_available_libs': {
            'torch_available': TORCH_AVAILABLE,
            'gputil_available': GPU_AVAILABLE
        },
        'gpu_info': get_gpu_info(),
        'detailed_readings': {}
    }
    
    # Get detailed PyTorch readings if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                debug_info['detailed_readings'][f'torch_device_{i}'] = {
                    'name': torch.cuda.get_device_name(i),
                    'allocated_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 3),
                    'reserved_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 3),
                    'cached_gb': round(torch.cuda.memory_cached(i) / (1024**3), 3) if hasattr(torch.cuda, 'memory_cached') else 'N/A',
                    'total_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                }
        except Exception as e:
            debug_info['detailed_readings']['torch_error'] = str(e)
    
    # Get detailed GPUtil readings if available
    if GPU_AVAILABLE:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                debug_info['detailed_readings'][f'gputil_device_{i}'] = {
                    'name': gpu.name,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_free_mb': gpu.memoryFree,
                    'memory_used_gb': round(gpu.memoryUsed / 1024, 2),
                    'memory_total_gb': round(gpu.memoryTotal / 1024, 2),
                    'memory_percent': round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1),
                    'utilization_percent': round(gpu.load * 100, 1),
                    'temperature': gpu.temperature
                }
        except Exception as e:
            debug_info['detailed_readings']['gputil_error'] = str(e)
    
    return jsonify(debug_info)

@app.route('/health')
def health():
    """Simple health check"""
    return jsonify({
        'status': 'healthy', 
        'service': 'CUDA Chat Backend with Context',
        'active_sessions': len(conversation_sessions),
        'rag_status': 'loaded' if rag_system else 'not_loaded'
    })

if __name__ == '__main__':
    print(" Starting Enhanced Flask server with Context Memory...")
    print(" Available endpoints:")
    print("   • http://localhost:5001/                    - API Info")
    print("   • http://localhost:5001/api/chat            - Chat API (with context)")
    print("   • http://localhost:5001/api/status          - System status")
    print("   • http://localhost:5001/api/clear_session   - Clear conversation")                        #DEBUGGING ENDPOINTS
    print("   • http://localhost:5001/api/session_info/<id> - Session details")
    print("   • http://localhost:5001/api/gpu_debug       - Detailed GPU debugging")
    print("   • http://localhost:5001/health              - Health check")
    print()
    print(" Features:")
    print("   ✅ Conversation memory per session")
    print("   ✅ Context-aware responses")
    print("   ✅ Follow-up question detection")
    print("   ✅ Automatic session cleanup")
    print("   ✅ Fixed CORS configuration")
    print("   🖥️  Real-time system monitoring (Memory & CPU)")
    print("   🎮 Enhanced GPU monitoring (FIXED memory detection)")
    print("   ⏱️  Response time tracking")
    print("   🔍 GPU debugging endpoint for troubleshooting")
    print()
    print("⚠️ Make sure Ollama is running: ollama serve")
    print(" Starting server on http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001)


