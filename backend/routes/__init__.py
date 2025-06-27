"""
Flask routes for CUDA Tutor API
"""

from .chat import create_chat_blueprint
from .compilation import create_compilation_blueprint
from .dependencies import create_dependencies_blueprint
from .status import create_status_blueprint

__all__ = [
    'create_chat_blueprint',
    'create_compilation_blueprint', 
    'create_dependencies_blueprint',
    'create_status_blueprint'
]