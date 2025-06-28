import numpy as np
import os
import sys

print("Python version:", sys.version)
print("NumPy version:", np.__version__)

# Check if numba is available
try:
    import numba
    print("Numba version:", numba.__version__)
except ImportError:
    print("Numba not installed")
    sys.exit(1)

# Check CUDA availability
try:
    from numba import cuda
    print("CUDA available:", cuda.is_available())
    if cuda.is_available():
        print("CUDA device count:", cuda.detect().count)
        print("CUDA devices:")
        for i in range(cuda.detect().count):
            device = cuda.get_device(i)
            print(f"  Device {i}: {device.name}")
    else:
        print("CUDA not available - checking why...")
        print("CUDA detect info:", cuda.detect())
except Exception as e:
    print(f"Error checking CUDA: {e}")

# Check environment variables
cuda_paths = [
    "CUDA_HOME",
    "CUDA_PATH", 
    "CUDA_ROOT",
    "PATH"
]

print("\nEnvironment variables:")
for var in cuda_paths:
    value = os.environ.get(var, "Not set")
    if var == "PATH":
        cuda_in_path = "cuda" in value.lower() or "nvidia" in value.lower()
        print(f"  {var}: {'Contains CUDA' if cuda_in_path else 'No CUDA found'}")
    else:
        print(f"  {var}: {value}")

# Try to find CUDA installation
possible_cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    r"C:\CUDA",
    r"C:\ProgramData\NVIDIA Corporation\CUDA",
]

print("\nChecking for CUDA installation:")
for path in possible_cuda_paths:
    if os.path.exists(path):
        print(f"  Found: {path}")
        try:
            subdirs = os.listdir(path)
            print(f"    Subdirectories: {subdirs}")
        except:
            print(f"    Cannot list contents")
    else:
        print(f"  Not found: {path}") 