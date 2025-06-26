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
import re

# Create Flask app first
app = Flask(__name__)

# Configure CORS properly - AFTER creating the app
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
    
    def generate_response(self, query, conversation_context="", stream=False, is_code_analysis=False, code_blocks=None):
        """Generate response using RAG with conversation context"""
        
        # Use provided code analysis info or detect it
        if code_blocks is None:
            try:
                is_code_analysis, code_blocks = detect_code_analysis_request(query)
            except Exception as e:
                print(f"Error in code analysis detection inside RAG: {e}")
                is_code_analysis, code_blocks = False, []
        
        # Ensure code_blocks is always a list
        if code_blocks and not isinstance(code_blocks, list):
            try:
                code_blocks = list(code_blocks)
            except Exception as e:
                print(f"Error converting code_blocks to list: {e}")
                code_blocks = []
        
        # 1. Retrieve relevant examples
        examples = self.search(query, top_k=3 if is_code_analysis else 2)
        
        # 2. Create context from examples
        if examples:
            example_context = "\n".join([
                f"Example: {ex['operation']} (Speedup: {ex['speedup']}x)\nCode: {ex['cuda_code'][:200]}..."
                for ex in examples
            ])
        else:
            example_context = "General CUDA programming knowledge."
        
        # 3. Create specialized prompts for code analysis
        if is_code_analysis and code_blocks and len(code_blocks) > 0:
            # Safely get the first code block
            code_to_analyze = code_blocks[0] if isinstance(code_blocks, list) and len(code_blocks) > 0 else "No code found"
            
            code_analysis_prompt = f"""You are an expert CUDA performance analyst. Analyze the provided code and give specific optimization recommendations.

CUDA Performance Knowledge:
{example_context}

Code to analyze:
{code_to_analyze}

Analysis guidelines:
1. Identify potential performance bottlenecks
2. Suggest memory optimization techniques (coalesced access, shared memory usage)
3. Recommend thread block size and grid configuration improvements
4. Point out opportunities for better parallelization
5. Suggest CUDA-specific optimizations (__shared__, __constant__, etc.)
6. Provide concrete code improvements with explanations

Student's request: {query}

Provide a detailed analysis with specific, actionable recommendations:"""
            
            prompt = code_analysis_prompt
        elif conversation_context:
            prompt = f"""You are a helpful CUDA programming tutor. Build on our previous conversation.\n\nPrevious conversation:\n{conversation_context}\n\nRelevant CUDA examples:\n{example_context}\n\nCurrent question: {query}\n\nAnswer the current question, referencing our previous discussion when relevant. Be conversational and remember what we've discussed:"""
        else:
            prompt = f"""You are a helpful CUDA programming tutor. Answer concisely and clearly.\n\nRelevant examples:\n{example_context}\n\nStudent question: {query}\n\nProvide a helpful answer:"""
        
        # 4. Generate response with improved settings
        try:
            response = requests.post("http://localhost:11434/api/generate", 
                json={
                    "model": "deepseek-coder:1.3b",
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
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    yield chunk['response']
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
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
            if stream: yield "The AI model is taking too long to respond. Please try a simpler question."
            else: return "The AI model is taking too long to respond. Please try a simpler question."
        except requests.exceptions.ConnectionError:
            if stream: yield "Cannot connect to the AI model. Please make sure Ollama is running with: `ollama serve`"
            else: return "Cannot connect to the AI model. Please make sure Ollama is running with: `ollama serve`"
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

def detect_code_analysis_request(query):
    """Detect if the query contains code for analysis"""
    try:
        # Check for code blocks
        code_block_pattern = r'```[\w]*\n?([\s\S]*?)```'
        code_matches = re.findall(code_block_pattern, query)
        
        # Always ensure we return a proper list
        code_blocks = []
        if code_matches:
            for match in code_matches:
                if isinstance(match, (str, tuple)):
                    code_blocks.append(match)
        
        # Check for analysis request keywords
        analysis_keywords = [
            'analyze', 'feedback', 'review', 'optimize', 'improve', 'suggestions',
            'performance', 'gpu', 'cuda', 'parallel', 'efficiency', 'bottleneck',
            'memory', 'thread', 'block', 'kernel', 'speedup'
        ]
        
        query_lower = str(query).lower()
        has_analysis_intent = any(keyword in query_lower for keyword in analysis_keywords)
        
        # Always return a boolean and a list
        return bool(len(code_blocks) > 0 and has_analysis_intent), code_blocks
    except Exception as e:
        print(f"Error in detect_code_analysis_request: {e}")
        import traceback
        traceback.print_exc()
        return False, []

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
        
        # Clean up old sessions periodically
        if len(conversation_sessions) > 50:  # Clean when we have too many sessions
            clean_old_sessions()
        
        # Get or create session
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        session_history = conversation_sessions[session_id]
        
        # Build conversation context - ALWAYS use context now
        conversation_context = get_conversation_context(session_history, max_exchanges=3)
        
        # Check if this is a follow-up question or code analysis request
        is_follow_up = detect_follow_up_question(message)
        
        # Initialize with safe defaults
        is_code_analysis = False
        code_blocks = []
        
        try:
            print(f"🔍 Detecting code analysis for: {message[:50]}...")
            result = detect_code_analysis_request(message)
            
            if isinstance(result, tuple) and len(result) == 2:
                is_code_analysis, code_blocks = result
                # Force code_blocks to be a list
                if not isinstance(code_blocks, list):
                    code_blocks = list(code_blocks) if code_blocks else []
            else:
                print(f"Unexpected result format from detect_code_analysis_request: {result}")
                is_code_analysis, code_blocks = False, []
            
            print(f"✅ Code analysis detection successful: {is_code_analysis}, blocks: {len(code_blocks)}")
        except Exception as e:
            print(f"❌ Error in code analysis detection: {e}")
            import traceback
            traceback.print_exc()
            is_code_analysis, code_blocks = False, []
        
        # Generate response with context
        if rag_system:
            # Always use conversation context if available
            try:
                if stream:
                    def generate():
                        full_response_content = ""
                        try:
                            for chunk in rag_system.generate_response(message, conversation_context, stream=True, is_code_analysis=is_code_analysis, code_blocks=code_blocks):
                                full_response_content += chunk
                                yield chunk
                        except Exception as e:
                            print(f"❌ Error during streaming generation: {e}")
                            import traceback
                            traceback.print_exc()
                            yield f"Error generating response: {str(e)}"
                        
                        # Save to session history after streaming is complete
                        session_history.append({
                            'user': message,
                            'assistant': full_response_content,
                            'timestamp': time.time(),
                            'is_follow_up': is_follow_up,
                            'is_code_analysis': is_code_analysis
                        })
                        if len(session_history) > 10:
                            conversation_sessions[session_id] = session_history[-10:]
                        print(f" Session {session_id}: {full_response_content[:100]}...")
                        print(f" Context used: {bool(conversation_context)}")
                        print(f"🔗 Follow-up detected: {is_follow_up}")
                        print(f"⚡ Code analysis detected: {is_code_analysis}")
                        if is_code_analysis:
                            try:
                                code_blocks_count = len(list(code_blocks)) if code_blocks else 0
                                print(f"📝 Code blocks found: {code_blocks_count}")
                            except Exception as e:
                                print(f"Error getting code blocks count in streaming: {e}")

                    return app.response_class(generate(), mimetype='text/plain')
                else:
                    print(f"🚀 Generating non-streaming response...")
                    response_content = rag_system.generate_response(message, conversation_context, stream=False, is_code_analysis=is_code_analysis, code_blocks=code_blocks)
                    print(f"✅ Response generated successfully: {len(response_content) if response_content else 0} chars")
            except Exception as e:
                print(f"❌ Error in RAG system generation: {e}")
                import traceback
                traceback.print_exc()
                response_content = f"Sorry, I encountered an error processing your request: {str(e)}"
        else:
            # Fallback response
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
            'timestamp': time.time(),
            'is_follow_up': is_follow_up,
            'is_code_analysis': is_code_analysis
        })
        
        # Keep only the last 10 exchanges per session to prevent memory bloat
        if len(session_history) > 10:
            conversation_sessions[session_id] = session_history[-10:]
        
        print(f" Session {session_id}: {response_content[:100]}...")
        print(f" Context used: {bool(conversation_context)}")
        print(f"🔗 Follow-up detected: {is_follow_up}")
        print(f"⚡ Code analysis detected: {is_code_analysis}")
        if is_code_analysis:
            # Safely get length of code_blocks
            try:
                code_blocks_count = len(list(code_blocks)) if code_blocks else 0
                print(f"📝 Code blocks found: {code_blocks_count}")
            except Exception as e:
                print(f"Error getting code blocks count: {e}")
                code_blocks_count = 0
        else:
            code_blocks_count = 0
        
        return jsonify({
            'response': response_content,
            'session_id': session_id,
            'is_follow_up': is_follow_up,
            'is_code_analysis': is_code_analysis,
            'code_blocks_count': code_blocks_count,
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
    print("   • http://localhost:5001/api/clear_session   - Clear conversation")
    print("   • http://localhost:5001/api/session_info/<id> - Session details")
    print("   • http://localhost:5001/health              - Health check")
    print()
    print(" New Features:")
    print("   Conversation memory per session")
    print("   Context-aware responses")
    print("   Follow-up question detection")
    print("   Automatic session cleanup")
    print("   Fixed CORS configuration")
    print()
    print("⚠️ Make sure Ollama is running: ollama serve")
    print(" Starting server on http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001)