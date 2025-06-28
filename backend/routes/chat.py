#!/usr/bin/env python3
"""
Chat and conversation routes for CUDA Tutor
"""

import time
import traceback
from flask import Blueprint, request, jsonify
from models.session import SessionManager
from utils.gpu_monitor import get_performance_tracker

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
            
            # Start performance tracking
            start_time = time.time()
            
            # Get conversation context
            conversation_context = session_manager.get_conversation_context(session_id)
            is_follow_up = session_manager.detect_follow_up_question(message)
            
            # Generate response
            if rag_system:
                response = rag_system.generate_response(message, conversation_context, stream=stream)
            else:
                response = _get_fallback_response()
            
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
            
            print(f"✅ Session {session_id}: Generated response ({len(response)} chars)")
            print(f"🔗 Follow-up detected: {is_follow_up}")
            
            # Print enhanced system usage with GPU stats
            performance_tracker.print_enhanced_system_usage(performance_data)
            
            return jsonify({
                'response': response,
                'session_id': session_id,
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