#!/usr/bin/env python3
"""
RAG system for CUDA programming assistance
"""

import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
import traceback
from config import get_config

config = get_config()

class SimpleRAG:
    def __init__(self):
        print("🔧 Initializing RAG system...")
        
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            print("✅ Embedding model loaded")
            
            # Load CUDA dataset
            print("📚 Loading CUDA dataset...")
            try:
                dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
                self.df = dataset["level_1"].to_pandas().head(config.DATASET_LIMIT)
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
                print("🧮 Creating embeddings...")
                texts = [item['text'] for item in self.knowledge]
                self.embeddings = self.embedding_model.encode(texts)
                print(f"✅ Ready! Loaded {len(self.knowledge)} CUDA examples")
            else:
                print("⚠️ No knowledge loaded, using fallback responses")
                self.embeddings = None
                
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            traceback.print_exc()
            raise
    
    def search(self, query, top_k=None):
        """Search for relevant examples"""
        if top_k is None:
            top_k = config.RAG_TOP_K
            
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
        examples = self.search(query)
        
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
        
        # 4. Generate response
        try:
            response = requests.post(f"{config.OLLAMA_BASE_URL}/api/generate", 
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "num_predict": 10000,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stop": ["Student question:", "Question:", "Human:", "User:", "Previous conversation:", "Current question:"],
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1
                    }
                }, 
                timeout=config.OLLAMA_TIMEOUT,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                import json
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

# Global RAG system instance
_rag_system = None

def initialize_rag_system():
    """Initialize the global RAG system"""
    global _rag_system
    try:
        _rag_system = SimpleRAG()
        return _rag_system
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return None

def get_rag_system():
    """Get the global RAG system instance"""
    return _rag_system