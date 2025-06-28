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
import json
import uuid
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
    
    def detect_quiz_request(self, query):
        """Detect if the user is requesting a quiz"""
        quiz_keywords = ['quiz', 'test me', 'question me', 'practice questions', 'mcq', 'multiple choice']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in quiz_keywords)

    def generate_quiz_data(self, topic="CUDA programming", conversation_context=""):
        """Generate quiz data with 3 questions"""
        # Create a prompt for quiz generation
        quiz_prompt = f"""Generate a CUDA programming quiz with exactly 3 questions based on the topic: {topic}

Based on conversation context:
{conversation_context}

Create a JSON response with this EXACT format:
{{
  "quiz_id": "unique_id_here",
  "topic": "{topic}",
  "questions": [
    {{
      "id": 1,
      "type": "mcq",
      "question": "What does CUDA stand for?",
      "options": [
        "Compute Unified Device Architecture",
        "Central Unit Device Architecture", 
        "Computer Universal Data Architecture",
        "Core Unified Development Architecture"
      ],
      "correct_answer": 0,
      "explanation": "CUDA stands for Compute Unified Device Architecture, NVIDIA's parallel computing platform."
    }},
    {{
      "id": 2,
      "type": "true_false",
      "question": "CUDA can only run on NVIDIA GPUs.",
      "options": ["True", "False"],
      "correct_answer": 0,
      "explanation": "True. CUDA is proprietary to NVIDIA and only works on NVIDIA GPUs."
    }},
    {{
      "id": 3,
      "type": "mcq", 
      "question": "Which memory type is fastest in CUDA?",
      "options": [
        "Global memory",
        "Shared memory", 
        "Register memory",
        "Texture memory"
      ],
      "correct_answer": 2,
      "explanation": "Register memory is the fastest memory type in CUDA, followed by shared memory."
    }}
  ]
}}

Guidelines:
- Always create exactly 3 questions
- Mix MCQ (4 options) and True/False (2 options) questions
- Include clear explanations for each answer
- Make questions educational and relevant to CUDA programming
- Correct_answer should be the index (0-based) of the correct option
- Keep questions at appropriate difficulty level

Return ONLY the JSON, no additional text:"""

        try:
            response = requests.post(f"{config.OLLAMA_BASE_URL}/api/generate", 
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": quiz_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 2000,
                        "temperature": 0.3,  # Lower temperature for more consistent JSON
                        "top_p": 0.9,
                        "stop": ["Human:", "User:", "Assistant:"],
                        "num_ctx": 4096
                    }
                }, 
                timeout=config.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                quiz_text = result.get("response", "").strip()
                
                # Try to parse the JSON response
                try:
                    # Clean up the response to extract JSON
                    if '```json' in quiz_text:
                        json_start = quiz_text.find('```json') + 7
                        json_end = quiz_text.find('```', json_start)
                        quiz_text = quiz_text[json_start:json_end].strip()
                    elif '{' in quiz_text:
                        # Find first { and last }
                        start = quiz_text.find('{')
                        end = quiz_text.rfind('}') + 1
                        quiz_text = quiz_text[start:end]
                    
                    quiz_data = json.loads(quiz_text)
                    
                    # Validate the quiz data structure
                    if self._validate_quiz_data(quiz_data):
                        # Ensure unique ID
                        quiz_data['quiz_id'] = str(uuid.uuid4())
                        return quiz_data
                    else:
                        return self._get_fallback_quiz()
                        
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    return self._get_fallback_quiz()
            else:
                return self._get_fallback_quiz()
                
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return self._get_fallback_quiz()

    def _validate_quiz_data(self, quiz_data):
        """Validate quiz data structure"""
        try:
            required_fields = ['quiz_id', 'topic', 'questions']
            if not all(field in quiz_data for field in required_fields):
                return False
                
            questions = quiz_data['questions']
            if len(questions) != 3:
                return False
                
            for q in questions:
                required_q_fields = ['id', 'type', 'question', 'options', 'correct_answer', 'explanation']
                if not all(field in q for field in required_q_fields):
                    return False
                    
                if q['type'] == 'mcq' and len(q['options']) != 4:
                    return False
                elif q['type'] == 'true_false' and len(q['options']) != 2:
                    return False
                    
                if not (0 <= q['correct_answer'] < len(q['options'])):
                    return False
                    
            return True
        except:
            return False

    def _get_fallback_quiz(self):
        """Fallback quiz if generation fails"""
        return {
            "quiz_id": str(uuid.uuid4()),
            "topic": "CUDA Programming Basics",
            "questions": [
                {
                    "id": 1,
                    "type": "mcq",
                    "question": "What does CUDA stand for?",
                    "options": [
                        "Compute Unified Device Architecture",
                        "Central Unit Device Architecture",
                        "Computer Universal Data Architecture", 
                        "Core Unified Development Architecture"
                    ],
                    "correct_answer": 0,
                    "explanation": "CUDA stands for Compute Unified Device Architecture, NVIDIA's parallel computing platform and programming model."
                },
                {
                    "id": 2,
                    "type": "true_false",
                    "question": "CUDA can only run on NVIDIA GPUs.",
                    "options": ["True", "False"],
                    "correct_answer": 0,
                    "explanation": "True. CUDA is proprietary to NVIDIA and only works on NVIDIA GPUs, unlike OpenCL which is cross-platform."
                },
                {
                    "id": 3,
                    "type": "mcq",
                    "question": "Which function synchronizes all threads in a CUDA block?",
                    "options": [
                        "__syncthreads()",
                        "__syncwarp()",
                        "cudaDeviceSynchronize()",
                        "__threadfence()"
                    ],
                    "correct_answer": 0,
                    "explanation": "__syncthreads() synchronizes all threads within a block, ensuring they reach this point before continuing execution."
                }
            ]
        }
    
    def generate_response(self, query, conversation_context="", stream=False):
        """Modified to handle quiz requests"""
        
        # Check if this is a quiz request
        if self.detect_quiz_request(query):
            # Extract topic from query if specified
            topic = "CUDA programming"
            if "about" in query.lower():
                # Try to extract topic after "about"
                parts = query.lower().split("about")
                if len(parts) > 1:
                    topic = parts[1].strip()[:50]  # Limit topic length
            
            # Generate quiz data
            quiz_data = self.generate_quiz_data(query, conversation_context)
            
            # Return quiz as JSON string with special marker
            return "QUIZ_DATA:" + json.dumps(quiz_data)
        
        # Regular response generation (existing code)
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