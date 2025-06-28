#!/usr/bin/env python3
"""
Test script for GPU monitoring functionality
Run this script to verify that GPU monitoring is working correctly
"""

import sys
import time
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_monitoring():
    """Test the GPU monitoring functionality"""
    print("🧪 Testing GPU Monitoring System")
    print("=" * 50)
    
    try:
        # Import the GPU monitoring module
        from utils.gpu_monitor import initialize_gpu_monitoring, get_performance_tracker
        
        # Initialize GPU monitoring
        print("1️⃣ Initializing GPU monitoring...")
        success = initialize_gpu_monitoring()
        print(f"   Status: {'✅ Success' if success else '⚠️ Limited functionality'}")
        
        # Get performance tracker
        print("\n2️⃣ Getting performance tracker...")
        tracker = get_performance_tracker()
        print("   ✅ Performance tracker obtained")
        
        # Test system metrics
        print("\n3️⃣ Testing system metrics...")
        system_metrics = tracker.system_monitor.get_system_metrics()
        print(f"   💾 Memory: {system_metrics['memory_gb']:.2f} GB ({system_metrics['memory_percent']:.1f}%)")
        print(f"   ⚡ CPU: {system_metrics['cpu_percent']:.1f}%")
        
        # Test GPU info
        print("\n4️⃣ Testing GPU information...")
        gpu_info = tracker.gpu_monitor.get_gpu_info()
        print(f"   🎮 CUDA Available: {gpu_info['cuda_available']}")
        print(f"   🔢 GPU Count: {gpu_info['gpu_count']}")
        print(f"   🏷️  GPU Model: {gpu_info['gpu_model']}")
        print(f"   🔧 Monitoring Method: {gpu_info['monitoring_method']}")
        
        if gpu_info['cuda_available'] and gpu_info['gpu_count'] > 0:
            print(f"   🔥 GPU Memory: {gpu_info['gpu_memory_used_gb']:.2f}/{gpu_info['gpu_memory_total_gb']:.2f} GB")
            print(f"   ⚙️  GPU Utilization: {gpu_info['gpu_utilization_percent']:.1f}%")
        
        # Test performance tracking
        print("\n5️⃣ Testing performance tracking...")
        start_time = time.time()
        time.sleep(0.5)  # Simulate some work
        end_time = time.time()
        
        performance_data = tracker.track_prompt_performance(
            start_time=start_time,
            end_time=end_time,
            session_id="test_session",
            prompt_length=100,
            response_length=500
        )
        
        print(f"   ⏱️  Response Time: {performance_data['response_time_seconds']:.2f} seconds")
        print(f"   📊 Tracking Data Keys: {list(performance_data.keys())}")
        
        # Test detailed GPU debug
        print("\n6️⃣ Testing detailed GPU debug...")
        debug_info = tracker.gpu_monitor.get_detailed_gpu_debug()
        print(f"   🔍 Available Libraries: {debug_info['libraries_available']}")
        print(f"   📝 Detailed Readings: {len(debug_info['detailed_readings'])} entries")
        
        print("\n" + "=" * 50)
        print("✅ All GPU monitoring tests completed successfully!")
        print("   You can now start the backend server to use GPU monitoring.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_monitoring()
    sys.exit(0 if success else 1) 