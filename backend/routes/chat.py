#!/usr/bin/env python3
"""
Chat and conversation routes for CUDA Tutor with Enhanced Tutoring Mode Support and Proper Chat Context
"""

import time
import traceback
from flask import Blueprint, request, jsonify
from models.session import SessionManager
from utils.gpu_monitor import get_performance_tracker

# Model configuration mapping
MODEL_CONFIGS = {
    'deepseek-r1': {
        'name': 'DeepSeek R1',
        'ollama_model': 'deepseek-r1:latest',
        'max_tokens': 4096,
        'temperature': 0.7,
        'description': 'Advanced reasoning model optimized for complex problem solving',
        'fallback': 'llama3.2:latest'
    },
    'qwen-2.5': {
        'name': 'Qwen 2.5',
        'ollama_model': 'qwen2.5:latest',
        'max_tokens': 4096,
        'temperature': 0.7,
        'description': 'Multilingual language model with strong performance across languages',
        'fallback': 'llama3.2:latest'
    },
    'mixtral': {
        'name': 'Mixtral',
        'ollama_model': 'mixtral:latest',
        'max_tokens': 4096,
        'temperature': 0.7,
        'description': 'Mixture of experts model for efficient and powerful responses',
        'fallback': 'llama3.2:latest'
    },
    'llama-3.2': {
        'name': 'Llama 3.2',
        'ollama_model': 'llama3.2:latest',
        'max_tokens': 4096,
        'temperature': 0.7,
        'description': 'Meta\'s latest open-source language model',
        'fallback': None
    }
}

DEFAULT_MODEL = 'deepseek-r1'

def create_chat_blueprint(rag_system):
    """Create chat blueprint with RAG system dependency"""
    
    bp = Blueprint('chat', __name__)
    session_manager = SessionManager()
    
    # NEW: Function to get chat-based conversation context from database
    def get_chat_conversation_context(chat_id, max_exchanges=5):
        """Get conversation context from Messages table based on chat_id"""
        if not chat_id:
            return ""
        
        supabase_client = None
        
        try:
            # Try to import Supabase client from the config directory
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from config.supabaseClient import supabase
                supabase_client = supabase
                print("✅ Using config supabase client")
            except ImportError:
                # If not available, create a simple client
                try:
                    from supabase import create_client
                    import os
                    
                    supabase_url = os.getenv('SUPABASE_URL') or os.getenv('REACT_APP_SUPABASE_URL')
                    supabase_key = os.getenv('SUPABASE_ANON_KEY') or os.getenv('REACT_APP_SUPABASE_KEY')
                    
                    if not supabase_url or not supabase_key:
                        print("❌ Supabase credentials not found for chat context")
                        return ""
                    
                    supabase_client = create_client(supabase_url, supabase_key)
                    print("✅ Created temporary supabase client")
                except Exception as setup_error:
                    print(f"❌ Failed to setup Supabase client: {setup_error}")
                    return ""
            
            if not supabase_client:
                print("❌ No Supabase client available")
                return ""
            
            # Query Messages table for this chat
            response = supabase_client.from_("Messages").select(
                "sender, content, order_index, timestamp"
            ).eq(
                "chat_id", chat_id
            ).order(
                "order_index", desc=True
            ).limit(
                max_exchanges * 2
            ).execute()
            
            if response.data:
                # Reverse to get chronological order (oldest first)
                messages = list(reversed(response.data))
                
                context_parts = []
                for msg in messages:
                    # Format each message
                    role = "User" if msg['sender'] == 'user' else "Assistant"
                    content = msg['content']
                    
                    # Truncate long messages for context
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    context_parts.append(f"{role}: {content}")
                
                context = "\n".join(context_parts)
                print(f"📚 Retrieved {len(messages)} messages for chat context (chat_id: {chat_id})")
                return context
            else:
                print(f"📚 No previous messages found for chat_id: {chat_id}")
                return ""
                
        except Exception as e:
            print(f"❌ Error retrieving chat context: {e}")
            traceback.print_exc()
            return ""
    
    @bp.route('/chat', methods=['POST', 'OPTIONS'])
    def chat():
        """Main chat endpoint with conversation context and enhanced tutoring mode support"""
        # Handle preflight CORS request
        if request.method == 'OPTIONS':
            return '', 200
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            message = data.get('message', '').strip()
            session_id = data.get('session_id', 'default')
            chat_id = data.get('chat_id')  # NEW: Get actual chat_id from request
            selected_model = data.get('model', DEFAULT_MODEL)
            stream = data.get('stream', False)
            tutoring_mode = data.get('tutoring_mode', False)  # NEW: Tutoring mode flag
            module_id = data.get('module_id')
            
            if not message:
                return jsonify({'error': 'No message provided'}), 400
            
            # Validate model selection
            if selected_model not in MODEL_CONFIGS:
                print(f"⚠️ Invalid model '{selected_model}', falling back to default")
                selected_model = DEFAULT_MODEL
            
            model_config = MODEL_CONFIGS[selected_model]
            print(f"💬 Chat {chat_id}: {message} (Model: {model_config['name']}, Tutoring: {tutoring_mode})")
            
            # Start performance tracking
            start_time = time.time()
            
            # UPDATED: Get conversation context from database using chat_id OR fallback to session-based
            if chat_id:
                conversation_context = get_chat_conversation_context(chat_id, max_exchanges=5)
                is_follow_up = len(conversation_context) > 0  # Simple follow-up detection based on context
                print(f"📝 Using database context for chat {chat_id}: {len(conversation_context)} chars")
                print(conversation_context)
                context_source = 'database'
            else:
                # Fallback to session-based context (legacy compatibility)
                conversation_context = session_manager.get_conversation_context(session_id)
                is_follow_up = session_manager.detect_follow_up_question(message)
                print(f"📝 Using session context for session {session_id}: {len(conversation_context)} chars")
                context_source = 'session'
            
            # NEW: Enhanced prompt generation for tutoring mode
            if tutoring_mode:
                enhanced_message = _create_tutoring_context_message(
                    message, 
                    conversation_context, 
                    module_id,
                    model_config
                )
                # For tutoring mode, use enhanced message with context but don't double-add context
                final_context = ""  # Context already embedded in enhanced_message
            else:
                enhanced_message = message
                final_context = conversation_context
            
            # Generate response with selected model
            if rag_system:
                if tutoring_mode:
                    # Use enhanced tutoring message with embedded context
                    response = rag_system.generate_response_with_model(
                        enhanced_message, 
                        final_context,  # Empty for tutoring as context is embedded
                        model_config,
                        stream=stream
                    )
                else:
                    # Regular chat mode with separate context
                    response = rag_system.generate_response_with_model(
                        enhanced_message, 
                        final_context, 
                        model_config,
                        stream=stream
                    )
            else:
                response = _get_fallback_response(model_config['name'])
            
            # End performance tracking
            end_time = time.time()
            
            # Track GPU usage and system performance
            performance_tracker = get_performance_tracker()
            performance_data = performance_tracker.track_prompt_performance(
                start_time=start_time,
                end_time=end_time,
                session_id=chat_id or session_id,  # Use chat_id for tracking if available
                prompt_length=len(message),
                response_length=len(response)
            )
            
            # UPDATED: Save to session history (keep for legacy compatibility)
            session_manager.add_exchange(session_id, message, response, is_follow_up)
            
            print(f"✅ Chat {chat_id}: Generated response using {model_config['name']} ({len(response)} chars)")
            print(f"🔗 Follow-up detected: {is_follow_up}")
            print(f"🎓 Tutoring mode: {tutoring_mode}")
            print(f"📚 Context source: {context_source}")
            
            # Print enhanced system usage with GPU stats
            performance_tracker.print_enhanced_system_usage(performance_data)
            
            return jsonify({
                'response': response,
                'session_id': session_id,  # Keep for compatibility
                'chat_id': chat_id,        # NEW: Include chat_id in response
                'model_used': selected_model,
                'model_name': model_config['name'],
                'is_follow_up': is_follow_up,
                'context_used': bool(conversation_context),
                'context_source': context_source,  # NEW: Indicate context source
                'tutoring_mode': tutoring_mode,
                'performance_metrics': {
                    'response_time_seconds': performance_data['response_time_seconds'],
                    'prompt_length': performance_data['prompt_length'],
                    'response_length': performance_data['response_length'],
                    'system_metrics': performance_data['system_metrics'],
                    'gpu_metrics': performance_data['gpu_metrics']
                },
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Error in chat route: {e}")
            traceback.print_exc()
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
    
    # Updated _create_tutoring_context_message function in chat.py

    def _create_tutoring_context_message(message, conversation_context, module_id, model_config):
        """Create enhanced message with comprehensive tutoring context and guidelines"""
        
        # Module information with enhanced details
        module_info = {
            "c801ac6c-1232-4c96-89b1-c4eadf41026c": {
                "name": "CUDA Basics",
                "topics": ["CUDA Architecture", "Thread Hierarchy", "Memory Model", "Basic Kernels", "GPU vs CPU"],
                "level": "Beginner",
                "learning_objectives": [
                    "Understand GPU architecture and parallel computing concepts",
                    "Learn CUDA thread hierarchy (grids, blocks, threads)",
                    "Master basic kernel launch syntax and execution model",
                    "Identify when to use GPU vs CPU for different tasks"
                ],
                "common_mistakes": [
                    "Confusing thread indexing calculations",
                    "Not understanding memory coalescing basics",
                    "Forgetting to synchronize after kernel launches"
                ]
            },
            "d26ccd91-cdf9-45e3-990f-a484d764bb9d": {
                "name": "Memory Optimization",
                "topics": ["Global Memory", "Shared Memory", "Memory Coalescing", "Memory Banks", "Texture Memory"],
                "level": "Intermediate",
                "learning_objectives": [
                    "Optimize memory access patterns for performance",
                    "Understand different CUDA memory hierarchies",
                    "Implement shared memory for data reuse",
                    "Avoid memory bank conflicts"
                ],
                "common_mistakes": [
                    "Non-coalesced global memory access",
                    "Shared memory bank conflicts",
                    "Inefficient memory transfer patterns"
                ]
            },
            "ff7d63fc-8646-4d9a-be5d-41a249beff02": {
                "name": "Kernel Development",
                "topics": ["Kernel Launch", "Thread Synchronization", "Performance Optimization", "Error Handling"],
                "level": "Intermediate",
                "learning_objectives": [
                    "Design efficient CUDA kernels",
                    "Implement proper thread synchronization",
                    "Handle edge cases and boundary conditions",
                    "Debug CUDA applications effectively"
                ],
                "common_mistakes": [
                    "Race conditions in kernel code",
                    "Improper grid/block size selection",
                    "Missing error checking"
                ]
            },
            "22107ce-5027-42bf-9941-6d00117da9ae": {
                "name": "Performance Tuning",
                "topics": ["Profiling", "Occupancy", "Memory Bandwidth", "Instruction Optimization", "Advanced Techniques"],
                "level": "Advanced",
                "learning_objectives": [
                    "Profile and analyze CUDA application performance",
                    "Optimize kernel occupancy and resource usage",
                    "Maximize memory bandwidth utilization",
                    "Apply advanced optimization techniques"
                ],
                "common_mistakes": [
                    "Optimizing without profiling first",
                    "Ignoring occupancy limitations",
                    "Over-optimization of minor bottlenecks"
                ]
            }
        }
        
        current_module = module_info.get(module_id, {
            "name": "CUDA Programming",
            "topics": ["General CUDA Concepts"],
            "level": "Intermediate",
            "learning_objectives": ["Understand CUDA programming fundamentals"],
            "common_mistakes": ["General CUDA programming errors"]
        })
        
        # UPDATED: Focus on teaching first, quizzes second
        tutoring_context = f"""
    You are an expert CUDA programming tutor conducting an interactive tutoring session. You are patient, encouraging, and focused on TEACHING concepts clearly.

    **CURRENT SESSION CONTEXT:**
    - Module: {current_module['name']} ({current_module['level']} level)
    - Topics: {', '.join(current_module['topics'])}
    - Model: {model_config['name']}

    **LEARNING OBJECTIVES FOR THIS MODULE:**
    {chr(10).join([f"• {obj}" for obj in current_module['learning_objectives']])}

    **COMMON STUDENT MISTAKES TO WATCH FOR:**
    {chr(10).join([f"• {mistake}" for mistake in current_module['common_mistakes']])}

    **PRIMARY TUTORING APPROACH - TEACH FIRST:**

    1. **Answer Questions Thoroughly:**
    - When students ask questions, provide comprehensive explanations
    - Break down complex concepts into simple, understandable parts
    - Use analogies and real-world examples to clarify abstract concepts
    - Show code examples with detailed line-by-line explanations

    2. **Teaching Strategy:**
    - Start with the fundamentals and build up complexity
    - Explain WHY things work the way they do, not just HOW
    - Connect new concepts to previously learned material
    - Use visual descriptions to help students understand GPU architecture

    3. **Interactive Learning Flow:**
    - Explain concept thoroughly FIRST
    - Check understanding with follow-up questions
    - Provide additional examples if needed
    - ONLY offer quizzes AFTER you've taught substantial material
    - Let students drive the conversation - answer what they ask

    4. **Quiz Guidelines (Use Sparingly):**
    - Only suggest quizzes AFTER you've explained concepts thoroughly
    - Ask permission: "Would you like to test your understanding with a quiz?"
    - Don't force quizzes - let students request them
    - Focus on teaching, not testing

    5. **Code Examples:**
    - Always provide complete, compilable examples
    - Explain each CUDA-specific keyword and function
    - Show both correct implementations and common mistakes
    - Include comments explaining the reasoning

    6. **Encouragement and Support:**
    - Celebrate when students ask good questions
    - Make students feel comfortable asking for clarification
    - Frame mistakes as learning opportunities
    - Be patient and supportive

    **CONVERSATION HISTORY:**
    {conversation_context if conversation_context else "This is the beginning of the tutoring session."}

    **CURRENT STUDENT MESSAGE:**
    {message}

    **RESPONSE INSTRUCTIONS:**
    - Focus on TEACHING and EXPLAINING concepts thoroughly
    - Answer the student's question with detailed explanations
    - Provide examples and analogies to aid understanding
    - Only suggest quizzes AFTER you've taught something substantial
    - Be encouraging and make learning enjoyable
    - If the student explicitly asks for a quiz, then provide one
    - Otherwise, focus on education and explanation

    Remember: Your primary role is to TEACH and EXPLAIN. Quizzes are secondary and should only be offered after substantial teaching has occurred or when explicitly requested by the student.
    """
        
        return tutoring_context
    
    
    @bp.route('/models', methods=['GET'])
    def get_available_models():
        """Get list of available AI models"""
        try:
            models = []
            for model_id, config in MODEL_CONFIGS.items():
                models.append({
                    'id': model_id,
                    'name': config['name'],
                    'description': config['description'],
                    'max_tokens': config['max_tokens'],
                    'temperature': config['temperature']
                })
            
            return jsonify({
                'models': models,
                'default_model': DEFAULT_MODEL,
                'total_models': len(models)
            })
            
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/model/status/<model_id>', methods=['GET'])
    def check_model_status(model_id):
        """Check if a specific model is available in Ollama"""
        try:
            if model_id not in MODEL_CONFIGS:
                return jsonify({
                    'available': False,
                    'error': 'Model not found in configuration'
                }), 404
            
            model_config = MODEL_CONFIGS[model_id]
            
            # Try to check if model is available in Ollama
            try:
                import requests
                from config import get_config
                
                config = get_config()
                
                # Check Ollama status
                response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                
                if response.status_code == 200:
                    models_data = response.json()
                    installed_models = [m['name'] for m in models_data.get('models', [])]
                    
                    # Check if the specific model is installed
                    model_available = any(
                        model_config['ollama_model'].split(':')[0] in installed_model 
                        for installed_model in installed_models
                    )
                    
                    # Check fallback model if main model not available
                    fallback_available = False
                    if not model_available and model_config.get('fallback'):
                        fallback_available = any(
                            model_config['fallback'].split(':')[0] in installed_model 
                            for installed_model in installed_models
                        )
                    
                    return jsonify({
                        'model_id': model_id,
                        'model_name': model_config['name'],
                        'available': model_available,
                        'fallback_available': fallback_available,
                        'ollama_model': model_config['ollama_model'],
                        'fallback_model': model_config.get('fallback'),
                        'installed_models': installed_models
                    })
                else:
                    return jsonify({
                        'available': False,
                        'error': 'Ollama service not responding'
                    })
                    
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'available': False,
                    'error': f'Connection error: {str(e)}'
                })
                
        except Exception as e:
            print(f"❌ Error checking model status: {e}")
            return jsonify({
                'available': False,
                'error': str(e)
            }), 500

    @bp.route('/evaluate-quiz', methods=['POST', 'OPTIONS'])
    def evaluate_quiz():
        """Evaluate quiz answers and provide feedback with AI analysis"""
        if request.method == 'OPTIONS':
            return '', 200
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            quiz_data = data.get('quiz_data', {})
            user_answers = data.get('user_answers', {})
            session_id = data.get('session_id', 'default')
            chat_id = data.get('chat_id')
            tutoring_mode = data.get('tutoring_mode', False)
            
            if not quiz_data or not user_answers:
                return jsonify({'error': 'Missing quiz data or answers'}), 400
            
            print(f"📝 Evaluating quiz for session {session_id}")
            
            # Calculate score and generate feedback
            total_questions = len(quiz_data.get('questions', []))
            correct_count = 0
            detailed_feedback = []
            weak_areas = []
            strong_areas = []
            
            for question in quiz_data.get('questions', []):
                q_id = str(question['id'])
                user_answer = user_answers.get(q_id)
                correct_answer = question['correct_answer']
                
                is_correct = user_answer == correct_answer
                if is_correct:
                    correct_count += 1
                    strong_areas.append(question.get('topic', question['question'][:50]))
                else:
                    weak_areas.append({
                        'topic': question.get('topic', question['question'][:50]),
                        'question': question['question'],
                        'explanation': question['explanation']
                    })
                
                feedback_item = {
                    'question_id': question['id'],
                    'question': question['question'],
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'explanation': question['explanation'],
                    'user_answer_text': question['options'][user_answer] if user_answer is not None and 0 <= user_answer < len(question['options']) else 'No answer',
                    'correct_answer_text': question['options'][correct_answer],
                    'topic': question.get('topic', 'General')
                }
                detailed_feedback.append(feedback_item)
            
            # Calculate percentage
            score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
            
            # Generate overall feedback message
            if score_percentage >= 80:
                overall_message = f"🎉 Excellent work! You scored {correct_count}/{total_questions} ({score_percentage:.0f}%). You have a strong understanding of CUDA concepts!"
            elif score_percentage >= 60:
                overall_message = f"👍 Good job! You scored {correct_count}/{total_questions} ({score_percentage:.0f}%). Review the explanations below to strengthen your understanding."
            else:
                overall_message = f"📚 Keep studying! You scored {correct_count}/{total_questions} ({score_percentage:.0f}%). Don't worry - CUDA takes practice. Review the explanations and try again!"
            
            evaluation_result = {
                'score': correct_count,
                'total': total_questions,
                'percentage': score_percentage,
                'overall_message': overall_message,
                'detailed_feedback': detailed_feedback,
                'topic': quiz_data.get('topic', 'CUDA Programming'),
                'weak_areas': weak_areas,
                'strong_areas': strong_areas
            }
            
            print(f"✅ Quiz evaluated: {correct_count}/{total_questions} ({score_percentage:.0f}%)")
            
            # Generate personalized AI feedback if in tutoring mode
            ai_analysis = None
            if tutoring_mode and rag_system and weak_areas:
                ai_analysis = _generate_quiz_analysis(evaluation_result, quiz_data, rag_system)
            
            response_data = {
                'success': True,
                'evaluation': evaluation_result,
                'session_id': session_id,
                'chat_id': chat_id,
                'ai_analysis': ai_analysis
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"❌ Error evaluating quiz: {e}")
            traceback.print_exc()
            return jsonify({
                'error': str(e),
                'success': False
            }), 500

    def _generate_quiz_analysis(evaluation, quiz_data, rag_system):
        """Generate AI-powered analysis of quiz performance"""
        try:
            weak_topics = [area['topic'] for area in evaluation['weak_areas']]
            
            analysis_prompt = f"""As a CUDA programming tutor, analyze this student's quiz performance and provide personalized feedback.

    **QUIZ RESULTS:**
    - Score: {evaluation['score']}/{evaluation['total']} ({evaluation['percentage']:.0f}%)
    - Topic: {evaluation['topic']}

    **AREAS WHERE STUDENT STRUGGLED:**
    {chr(10).join([f"- {area['topic']}: {area['explanation']}" for area in evaluation['weak_areas']])}

    **STRONG AREAS:**
    {chr(10).join([f"- {topic}" for topic in evaluation['strong_areas']])}

    **YOUR TASK:**
    Provide a personalized, encouraging response that:
    1. Acknowledges their performance
    2. Identifies 2-3 specific weak areas they need to focus on
    3. Suggests practical next steps for improvement
    4. Recommends specific CUDA concepts to study
    5. Offers to help with practice exercises

    Keep the tone supportive and motivating. Be specific about what they should study next.

    Respond as their tutor would in a conversational way:"""

            # Generate analysis using the RAG system
            analysis = rag_system.generate_response(analysis_prompt, "", stream=False)
            
            print(f"✅ Generated AI quiz analysis: {len(analysis)} chars")
            return analysis
            
        except Exception as e:
            print(f"❌ Error generating quiz analysis: {e}")
            return None
    
    @bp.route('/clear-session', methods=['POST'])
    def clear_session():
        """Clear a specific conversation session"""
        try:
            data = request.get_json()
            session_id = data.get('session_id', 'default')
            
            cleared = session_manager.clear_session(session_id)
            
            if cleared:
                return jsonify({
                    'message': f'Session {session_id} cleared',
                    'status': 'success'
                })
            else:
                return jsonify({
                    'message': f'Session {session_id} not found',
                    'status': 'warning'
                })
        except Exception as e:
            print(f"❌ Error clearing session: {e}")
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
    @bp.route('/session-info/<session_id>')
    def session_info(session_id):
        """Get information about a specific session"""
        try:
            info = session_manager.get_session_info(session_id)
            
            if info['exists']:
                return jsonify(info)
            else:
                return jsonify({
                    'session_id': session_id,
                    'status': 'not_found',
                    'exists': False
                }), 404
        except Exception as e:
            print(f"❌ Error getting session info: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/sessions')
    def list_sessions():
        """List all active sessions"""
        try:
            sessions = session_manager.list_sessions()
            return jsonify(sessions)
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _get_fallback_response(model_name="AI"):
        """Fallback response when RAG system is not available"""
        return f"""I apologize, but the RAG system isn't fully initialized. Here's a basic response from {model_name}:

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It allows developers to use NVIDIA GPUs for general-purpose computing tasks, often achieving significant speedups for parallelizable problems.

Key benefits:
• Massive parallel processing (thousands of cores)
• High memory bandwidth
• Optimized for data-parallel computations
• Excellent for machine learning, scientific computing, and image processing

To get started with CUDA programming, you'll need:
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit installed
3. Understanding of parallel programming concepts

Note: This response was generated using the {model_name} model."""
    
    return bp