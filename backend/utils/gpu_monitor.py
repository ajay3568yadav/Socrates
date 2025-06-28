#!/usr/bin/env python3
"""
GPU Monitoring utilities for CUDA Tutor Backend
Provides comprehensive GPU usage statistics and monitoring
"""

import psutil
import time
from collections import deque
from typing import Dict, Optional, List

# GPU monitoring imports with fallbacks
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pynvml import nvmlInit
    nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except (ImportError, Exception):
    NVIDIA_ML_AVAILABLE = False

class SystemMonitor:
    """Monitor system resources including CPU, memory, and GPU"""
    
    def __init__(self, max_readings: int = 10):
        self.max_readings = max_readings
        self.cpu_readings = deque(maxlen=max_readings)
        self.memory_readings = deque(maxlen=max_readings)
        
        # Initialize with baseline readings
        for _ in range(3):
            self.update_system_readings()
            time.sleep(0.1)
    
    def get_system_metrics(self) -> Dict:
        """Get current system memory and CPU usage"""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            memory_usage_gb = memory_usage_mb / 1024
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
    
    def update_system_readings(self):
        """Update system readings for averaging"""
        metrics = self.get_system_metrics()
        self.cpu_readings.append(metrics['cpu_percent'])
        self.memory_readings.append(metrics['memory_percent'])
    
    def get_average_cpu_usage(self) -> float:
        """Get average CPU usage from recent readings"""
        if not self.cpu_readings:
            return 0
        return round(sum(self.cpu_readings) / len(self.cpu_readings), 1)
    
    def get_average_memory_usage(self) -> float:
        """Get average memory usage from recent readings"""
        if not self.memory_readings:
            return 0
        return round(sum(self.memory_readings) / len(self.memory_readings), 1)

class GPUMonitor:
    """Monitor GPU usage and statistics"""
    
    def __init__(self):
        self.gpu_util_available = GPU_UTIL_AVAILABLE
        self.torch_available = TORCH_AVAILABLE
        self.nvidia_ml_available = NVIDIA_ML_AVAILABLE
    
    def get_gpu_info(self) -> Dict:
        """Get comprehensive GPU information including CUDA availability and memory usage"""
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_model': 'None',
            'gpu_memory_used_gb': 0,
            'gpu_memory_total_gb': 0,
            'gpu_memory_percent': 0,
            'gpu_utilization_percent': 0,
            'gpu_temperature': 0,
            'monitoring_method': 'none'
        }
        
        # First try GPUtil for comprehensive system-wide GPU info
        if self.gpu_util_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    gpu_info['gpu_model'] = gpu.name
                    gpu_info['gpu_utilization_percent'] = round(gpu.load * 100, 1)
                    gpu_info['gpu_memory_total_gb'] = round(gpu.memoryTotal / 1024, 2)
                    gpu_info['gpu_memory_used_gb'] = round(gpu.memoryUsed / 1024, 2)
                    gpu_info['gpu_memory_percent'] = round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1)
                    gpu_info['gpu_count'] = len(gpus)
                    gpu_info['gpu_temperature'] = getattr(gpu, 'temperature', 0)
                    gpu_info['monitoring_method'] = 'gputil'
                    print(f"🎮 GPUtil readings: {gpu_info['gpu_memory_used_gb']:.2f}/{gpu_info['gpu_memory_total_gb']:.2f} GB ({gpu_info['gpu_memory_percent']:.1f}%)")
            except Exception as e:
                print(f"⚠️ GPUtil error: {e}")
        
        # Check CUDA availability via PyTorch
        if self.torch_available:
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
                            # Get PyTorch memory info
                            reserved = torch.cuda.memory_reserved(0) / (1024**3)
                            allocated = torch.cuda.memory_allocated(0) / (1024**3)
                            
                            # Get total memory from device properties
                            gpu_memory = torch.cuda.get_device_properties(0)
                            total_gb = gpu_memory.total_memory / (1024**3)
                            
                            # Use reserved memory as it's more representative
                            used_memory = max(reserved, allocated)
                            
                            gpu_info['gpu_memory_total_gb'] = round(total_gb, 2)
                            gpu_info['gpu_memory_used_gb'] = round(used_memory, 2)
                            gpu_info['gpu_memory_percent'] = round((used_memory / total_gb) * 100, 1)
                            gpu_info['monitoring_method'] = 'pytorch'
                            
                            print(f"🔥 PyTorch readings: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                        except Exception as e:
                            print(f"⚠️ PyTorch memory error: {e}")
        
        return gpu_info
    
    def get_detailed_gpu_debug(self) -> Dict:
        """Get detailed GPU debugging information"""
        debug_info = {
            'libraries_available': {
                'torch_available': self.torch_available,
                'gputil_available': self.gpu_util_available,
                'nvidia_ml_available': self.nvidia_ml_available
            },
            'gpu_info': self.get_gpu_info(),
            'detailed_readings': {}
        }
        
        # Get detailed PyTorch readings if available
        if self.torch_available and torch.cuda.is_available():
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
        if self.gpu_util_available:
            try:
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
                        'temperature': getattr(gpu, 'temperature', 'N/A')
                    }
            except Exception as e:
                debug_info['detailed_readings']['gputil_error'] = str(e)
        
        return debug_info

class PerformanceTracker:
    """Track performance metrics for prompt processing"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.gpu_monitor = GPUMonitor()
        
    def track_prompt_performance(self, start_time: float, end_time: float, 
                               session_id: str, prompt_length: int, 
                               response_length: int) -> Dict:
        """Track performance metrics for a prompt processing session"""
        
        # Update system readings
        self.system_monitor.update_system_readings()
        
        # Calculate response time
        response_time = end_time - start_time
        
        # Get current metrics
        system_metrics = self.system_monitor.get_system_metrics()
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        # Get averages
        avg_cpu = self.system_monitor.get_average_cpu_usage()
        avg_memory = self.system_monitor.get_average_memory_usage()
        
        performance_data = {
            'session_id': session_id,
            'response_time_seconds': round(response_time, 2),
            'prompt_length': prompt_length,
            'response_length': response_length,
            'timestamp': end_time,
            'system_metrics': {
                'current_memory_gb': system_metrics['memory_gb'],
                'current_memory_percent': system_metrics['memory_percent'],
                'average_memory_percent': avg_memory,
                'current_cpu_percent': system_metrics['cpu_percent'],
                'average_cpu_percent': avg_cpu
            },
            'gpu_metrics': gpu_info
        }
        
        return performance_data
    
    def print_enhanced_system_usage(self, performance_data: Dict):
        """Print enhanced system usage including GPU info and response time"""
        metrics = performance_data['system_metrics']
        gpu_info = performance_data['gpu_metrics']
        response_time = performance_data['response_time_seconds']
        
        print("=" * 70)
        print("🖥️  ENHANCED SYSTEM METRICS")
        print("=" * 70)
        
        # Response timing
        print(f"⏱️  Response Time: {response_time:.2f} seconds")
        print(f"📝 Prompt Length: {performance_data['prompt_length']} chars")
        print(f"💬 Response Length: {performance_data['response_length']} chars")
        print(f"🆔 Session ID: {performance_data['session_id']}")
        print("-" * 70)
        
        # CPU and Memory
        print(f"💾 Memory Usage: {metrics['current_memory_gb']:.2f} GB ({metrics['current_memory_percent']:.1f}%)")
        print(f"📊 Average Memory (last 10): {metrics['average_memory_percent']:.1f}%")
        print(f"⚡ CPU Usage: {metrics['current_cpu_percent']:.1f}%")
        print(f"📈 Average CPU (last 10): {metrics['average_cpu_percent']:.1f}%")
        print("-" * 70)
        
        # GPU Information
        print(f"🎮 CUDA Available: {'✅ Yes' if gpu_info['cuda_available'] else '❌ No'}")
        print(f"🔢 GPU Count: {gpu_info['gpu_count']}")
        print(f"🏷️  GPU Model: {gpu_info['gpu_model']}")
        print(f"🔧 Monitoring Method: {gpu_info['monitoring_method']}")
        
        if gpu_info['cuda_available'] and gpu_info['gpu_count'] > 0:
            print(f"🔥 GPU Memory: {gpu_info['gpu_memory_used_gb']:.2f} GB / {gpu_info['gpu_memory_total_gb']:.2f} GB ({gpu_info['gpu_memory_percent']:.1f}%)")
            print(f"⚙️  GPU Utilization: {gpu_info['gpu_utilization_percent']:.1f}%")
            if gpu_info['gpu_temperature'] > 0:
                print(f"🌡️  GPU Temperature: {gpu_info['gpu_temperature']}°C")
        else:
            print("🔥 GPU Memory: Not available")
            print("⚙️  GPU Utilization: Not available")
        
        print("=" * 70)

# Global performance tracker instance
_performance_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker

def initialize_gpu_monitoring():
    """Initialize GPU monitoring system"""
    global _performance_tracker
    print("🎮 Initializing GPU monitoring system...")
    
    try:
        _performance_tracker = PerformanceTracker()
        gpu_info = _performance_tracker.gpu_monitor.get_gpu_info()
        
        if gpu_info['cuda_available']:
            print(f"✅ CUDA detected: {gpu_info['gpu_model']} ({gpu_info['gpu_memory_total_gb']:.1f} GB)")
            print(f"🔧 Using monitoring method: {gpu_info['monitoring_method']}")
        else:
            print("❌ No CUDA-capable GPU detected")
        
        print("✅ GPU monitoring system initialized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize GPU monitoring: {e}")
        return False 