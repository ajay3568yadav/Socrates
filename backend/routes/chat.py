#!/usr/bin/env python3
"""
Chat and conversation routes for CUDA Tutor with Model Selection
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
        'fallback': 'llama3.2:latest'  # Fallback if main model not available
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
        'fallback': None  # This is our fallback model
    }
}

DEFAULT_MODEL = 'deepseek-r1'

def create_chat_blueprint(rag_system):
    """Create chat blueprint with RAG system dependency"""
    
    bp = Blueprint('chat', __name__)
    session_manager = SessionManager()
    
    @bp.route('/chat', methods=['POST', 'OPTIONS'])
    def chat():
        """Main chat endpoint with conversation context and model selection"""
        # Handle preflight CORS request
        if request.method == 'OPTIONS':
            return '', 200
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            message = data.get('message', '').strip()
            session_id = data.get('session_id', 'default')
            selected_model = data.get('model', DEFAULT_MODEL)
            stream = data.get('stream', False)
            
            if not message:
                return jsonify({'error': 'No message provided'}), 400
            
            # Validate model selection
            if selected_model not in MODEL_CONFIGS:
                print(f"⚠️ Invalid model '{selected_model}', falling back to default")
                selected_model = DEFAULT_MODEL
            
            model_config = MODEL_CONFIGS[selected_model]
            print(f"💬 Session {session_id}: {message} (Model: {model_config['name']})")
            
            # Start performance tracking
            start_time = time.time()
            
            # Get conversation context
            conversation_context = session_manager.get_conversation_context(session_id)
            is_follow_up = session_manager.detect_follow_up_question(message)
            
            # Generate response with selected model
            if rag_system:
                response = rag_system.generate_response_with_model(
                    message, 
                    conversation_context, 
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
                session_id=session_id,
                prompt_length=len(message),
                response_length=len(response)
            )
            
            # Save to session history
            session_manager.add_exchange(session_id, message, response, is_follow_up)
            
            print(f"✅ Session {session_id}: Generated response using {model_config['name']} ({len(response)} chars)")
            print(f"🔗 Follow-up detected: {is_follow_up}")
            
            # Print enhanced system usage with GPU stats
            performance_tracker.print_enhanced_system_usage(performance_data)
            
            return jsonify({
                'response': response,
                'session_id': session_id,
                'model_used': selected_model,
                'model_name': model_config['name'],
                'is_follow_up': is_follow_up,
                'context_used': bool(conversation_context),
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
        """Evaluate quiz answers and provide feedback"""
        if request.method == 'OPTIONS':
            return '', 200
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            quiz_data = data.get('quiz_data', {})
            user_answers = data.get('user_answers', {})
            session_id = data.get('session_id', 'default')
            
            if not quiz_data or not user_answers:
                return jsonify({'error': 'Missing quiz data or answers'}), 400
            
            print(f"📝 Evaluating quiz for session {session_id}")
            
            # Calculate score and generate feedback
            total_questions = len(quiz_data.get('questions', []))
            correct_count = 0
            detailed_feedback = []
            
            for question in quiz_data.get('questions', []):
                q_id = str(question['id'])
                user_answer = user_answers.get(q_id)
                correct_answer = question['correct_answer']
                
                is_correct = user_answer == correct_answer
                if is_correct:
                    correct_count += 1
                
                feedback_item = {
                    'question_id': question['id'],
                    'question': question['question'],
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'explanation': question['explanation'],
                    'user_answer_text': question['options'][user_answer] if user_answer is not None and 0 <= user_answer < len(question['options']) else 'No answer',
                    'correct_answer_text': question['options'][correct_answer]
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
                'topic': quiz_data.get('topic', 'CUDA Programming')
            }
            
            print(f"✅ Quiz evaluated: {correct_count}/{total_questions} ({score_percentage:.0f}%)")
            
            return jsonify({
                'success': True,
                'evaluation': evaluation_result,
                'session_id': session_id
            })
            
        except Exception as e:
            print(f"❌ Error evaluating quiz: {e}")
            traceback.print_exc()
            return jsonify({
                'error': str(e),
                'success': False
            }), 500
    
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