#!/usr/bin/env python3
"""
Simplified Flask Backend for CUDA Chat Interface - FIXED
Removes all generator-related issues and focuses on core functionality
"""

from flask import Flask, request, jsonify
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
import re

# Create Flask app first
app = Flask(__name__)

# Configure CORS properly
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"])

conversation_sessions = {}

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
    
    def generate_response(self, query, conversation_context=""):
        """Generate response using RAG with conversation context - SIMPLIFIED"""
        
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
        
        # 3. Create prompt
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
        
        # 4. Generate response
        try:
            response = requests.post("http://localhost:11434/api/generate", 
                json={
                    "model": "deepseek-coder:1.3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 10000,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stop": ["Student question:", "Question:", "Human:", "User:", "Previous conversation:", "Current question:"],
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1
                    }
                }, 
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                # Clean up the response
                if generated_text:
                    if not generated_text.endswith(('.', '!', '?', ':', '```')):
                        sentences = generated_text.split('.')
                        if len(sentences) > 1:
                            generated_text = '.'.join(sentences[:-1]) + '.'
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

def simple_code_detection(query):
    """Simple code detection without generators"""
    try:
        # Look for code blocks
        has_code_blocks = '```' in query
        
        # Look for analysis keywords
        analysis_keywords = ['analyze', 'feedback', 'review', 'optimize', 'improve', 'gpu', 'cuda']
        has_analysis_intent = any(keyword in query.lower() for keyword in analysis_keywords)
        
        return has_code_blocks and has_analysis_intent
    except:
        return False

# Initialize RAG system
print("🔧 Starting Flask server with RAG...")
try:
    rag_system = SimpleRAG()
except Exception as e:
    print(f" RAG initialization failed: {e}")
    rag_system = None

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
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"💬 Session {session_id}: {message}")
        
        # Get or create session
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        session_history = conversation_sessions[session_id]
        
        # Build conversation context
        conversation_context = get_conversation_context(session_history, max_exchanges=3)
        
        # Simple code analysis detection
        is_code_analysis = simple_code_detection(message)
        
        print(f"🔍 Code analysis detected: {is_code_analysis}")
        
        # Generate response
        if rag_system:
            try:
                print(f"🚀 Generating response...")
                response_content = rag_system.generate_response(message, conversation_context)
                print(f"✅ Response generated: {len(response_content)} chars")
            except Exception as e:
                print(f"❌ Error in RAG generation: {e}")
                response_content = f"Sorry, I encountered an error: {str(e)}"
        else:
            response_content = """I apologize, but the RAG system isn't fully initialized. Here's a basic response:

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
            'assistant': response_content,
            'timestamp': time.time()
        })
        
        # Keep only the last 10 exchanges per session
        if len(session_history) > 10:
            conversation_sessions[session_id] = session_history[-10:]
        
        print(f"✅ Response complete: {response_content[:100]}...")
        
        return jsonify({
            'response': response_content,
            'session_id': session_id,
            'is_code_analysis': is_code_analysis,
            'context_used': bool(conversation_context),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/status')
def status():
    """Check system status"""
    status_info = {
        'rag_loaded': rag_system is not None,
        'knowledge_count': len(rag_system.knowledge) if rag_system else 0,
        'active_sessions': len(conversation_sessions),
        'ollama_status': 'unknown'
    }
    
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        status_info['ollama_status'] = 'connected' if response.status_code == 200 else 'error'
    except:
        status_info['ollama_status'] = 'disconnected'
    
    return jsonify(status_info)

@app.route('/health')
def health():
    """Simple health check"""
    return jsonify({
        'status': 'healthy', 
        'service': 'CUDA Chat Backend - Fixed',
        'active_sessions': len(conversation_sessions),
        'rag_status': 'loaded' if rag_system else 'not_loaded'
    })

if __name__ == '__main__':
    print("🚀 Starting Fixed Flask server...")
    print("   • http://localhost:5001/                    - API Info")
    print("   • http://localhost:5001/api/chat            - Chat API")
    print("   • http://localhost:5001/api/status          - System status")
    print("   • http://localhost:5001/health              - Health check")
    print()
    print("⚠️ Make sure Ollama is running: ollama serve")
    print("🎯 Starting server on http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 