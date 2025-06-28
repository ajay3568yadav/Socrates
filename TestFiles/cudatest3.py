import numpy as np
from numba import cuda
import numba
import sys

def main():
    print("Python version:", sys.version)
    print("Numba version:", numba.__version__)
    print("CUDA devices detected:", cuda.detect())
    
    # Try to get device info even if is_available() is False
    try:
        device = cuda.get_current_device()
        print(f"Current device: {device.name}")
        print(f"Compute capability: {device.compute_capability}")
        cuda_working = True
    except Exception as e:
        print(f"Error getting device: {e}")
        cuda_working = False
    
    if not cuda_working:
        print("\nCUDA setup issues detected. Trying alternative approach...")
        print("This might be a Windows-specific issue with Numba CUDA detection.")
        return

    # Test with a simple kernel
    @cuda.jit
    def vector_add(a, b, out):
        # Get the current thread index
        idx = cuda.grid(1)
        # Check bounds
        if idx < a.size:
            out[idx] = a[idx] + b[idx]

    n = 1_000_000
    a_host = np.random.rand(n).astype(np.float32)
    b_host = np.random.rand(n).astype(np.float32)
    out_host = np.empty_like(a_host)

    # Copy data to device
    a_dev = cuda.to_device(a_host)
    b_dev = cuda.to_device(b_host)
    out_dev = cuda.device_array_like(a_dev)

    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    print(f"Launching kernel with {blocks_per_grid} blocks, {threads_per_block} threads per block")

    # Launch kernel
    vector_add[blocks_per_grid, threads_per_block](a_dev, b_dev, out_dev)
    
    # Copy result back to host
    out_dev.copy_to_host(out_host)

    # Verify result
    expected = a_host + b_host
    np.testing.assert_allclose(out_host, expected, rtol=1e-5)
    print("Success! First 5 results:")
    print(f"GPU result: {out_host[:5]}")
    print(f"CPU result: {expected[:5]}")

if __name__ == "__main__":
    main() 