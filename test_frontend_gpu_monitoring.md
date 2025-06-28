# Testing Frontend GPU Monitoring

## Quick Test Steps

### 1. Start the Backend
```bash
cd backend
python app.py
```

You should see:
```
🎮 Initializing GPU monitoring system...
✅ CUDA detected: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)
✅ GPU monitoring system initialized
```

### 2. Start the Frontend
```bash
cd socrates
npm start
```

### 3. Test the Feature

1. **Login** to the application
2. **Send a simple test message** like "test" or "hello"
3. **Wait for the response** to be generated
4. **Look for the orange "🎮 System Info" button** at the bottom of the assistant's response
5. **Click the button** to open the performance metrics modal

### 4. Verify the Modal Content

The modal should display:
- ⏱️ Response timing (response time, prompt length, response length)
- 💾 System resources (memory and CPU usage)
- 🎮 GPU information (CUDA status, GPU model, memory usage, utilization)

### 5. Test Multiple Messages

Send several different messages to see how performance metrics vary:
- Short message: "hi"
- Medium message: "Explain CUDA programming"
- Long message: "Write a detailed CUDA kernel for matrix multiplication"

## Expected Behavior

### ✅ What Should Work
- System Info button appears on all new assistant messages
- Modal opens with comprehensive metrics
- Data matches what's shown in backend terminal
- Modal is responsive and closeable
- Button works from both message level and code blocks

### ❌ Troubleshooting

#### Button Not Appearing
1. Check backend console for GPU monitoring initialization
2. Verify performance metrics in API response:
   ```bash
   curl -X POST http://localhost:5001/api/chat \
   -H "Content-Type: application/json" \
   -d '{"message": "test", "session_id": "test_session"}'
   ```
3. Look for `performance_metrics` field in response

#### Modal Not Opening
1. Check browser console for JavaScript errors
2. Verify CSS files are loaded
3. Try hard refresh (Ctrl+F5)

#### Missing Performance Data
1. Ensure backend GPU monitoring is working:
   ```bash
   cd backend
   python test_gpu_monitoring.py
   ```
2. Check that GPU libraries are installed:
   ```bash
   pip install GPUtil torch
   ```

## Sample API Response

When working correctly, the `/api/chat` endpoint should return:

```json
{
  "response": "Hello! How can I help you with CUDA programming?",
  "session_id": "session_123",
  "is_follow_up": false,
  "context_used": false,
  "performance_metrics": {
    "response_time_seconds": 2.34,
    "prompt_length": 4,
    "response_length": 1023,
    "system_metrics": {
      "current_memory_gb": 8.45,
      "current_memory_percent": 52.3,
      "average_memory_percent": 48.7,
      "current_cpu_percent": 23.1,
      "average_cpu_percent": 19.8
    },
    "gpu_metrics": {
      "cuda_available": true,
      "gpu_count": 1,
      "gpu_model": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "monitoring_method": "gputil",
      "gpu_memory_used_gb": 3.25,
      "gpu_memory_total_gb": 8.0,
      "gpu_memory_percent": 40.6,
      "gpu_utilization_percent": 85.2,
      "gpu_temperature": 65
    }
  },
  "status": "success"
}
```

## Success Criteria

✅ System Info button appears on assistant messages
✅ Modal opens and displays formatted metrics
✅ Data is accurate and matches backend terminal output  
✅ Modal is responsive and user-friendly
✅ Button works from both message and code block levels
✅ Performance metrics persist across chat reloads
✅ Feature works on both desktop and mobile

The feature is working correctly when you can see the exact same GPU and system information in the frontend modal that appears in the backend terminal for each prompt. 