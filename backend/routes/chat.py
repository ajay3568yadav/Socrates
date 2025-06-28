#!/usr/bin/env python3
"""
Chat and conversation routes for CUDA Tutor
"""

import time
import traceback
from flask import Blueprint, request, jsonify
from models.session import SessionManager

def create_chat_blueprint(rag_system):
    """Create chat blueprint with RAG system dependency"""
    
    bp = Blueprint('chat', __name__)
    session_manager = SessionManager()
    
    @bp.route('/chat', methods=['POST', 'OPTIONS'])
    def chat():
        """Main chat endpoint with conversation context"""
        # Handle preflight CORS request
        if request.method == 'OPTIONS':
            return '', 200
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            message = data.get('message', '').strip()
            session_id = data.get('session_id', 'default')
            stream = data.get('stream', False)
            
            if not message:
                return jsonify({'error': 'No message provided'}), 400
            
            print(f"💬 Session {session_id}: {message}")
            
            # Get conversation context
            conversation_context = session_manager.get_conversation_context(session_id)
            is_follow_up = session_manager.detect_follow_up_question(message)
            
            # Generate response
            if rag_system:
                response = rag_system.generate_response(message, conversation_context, stream=stream)
            else:
                response = _get_fallback_response()
            
            # Save to session history
            session_manager.add_exchange(session_id, message, response, is_follow_up)
            
            print(f"✅ Session {session_id}: Generated response ({len(response)} chars)")
            print(f"🔗 Follow-up detected: {is_follow_up}")
            
            return jsonify({
                'response': response,
                'session_id': session_id,
                'is_follow_up': is_follow_up,
                'context_used': bool(conversation_context),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Error in chat route: {e}")
            traceback.print_exc()
            return jsonify({
                'error': str(e),
                'status': 'error'
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
    
    def _get_fallback_response():
        """Fallback response when RAG system is not available"""
        return """I apologize, but the RAG system isn't fully initialized. Here's a basic response:

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
    
    return bp