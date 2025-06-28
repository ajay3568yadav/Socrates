import numpy as np

# Try to import CUDA, fall back gracefully if not available
try:
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Numba CUDA not available. Using CPU implementation.")

# Define a simple CUDA kernel
if CUDA_AVAILABLE:
    @cuda.jit
    def add_matrices_kernel(mat1, mat2, result):
        row, col = cuda.grid(2)
        if row < result.shape[0] and col < result.shape[1]:
            result[row, col] = mat1[row, col] + mat2[row, col]

def cuda_kernel(matrix1, matrix2):
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    
    # Ensure input matrices are float32
    matrix1 = matrix1.astype(np.float32)
    matrix2 = matrix2.astype(np.float32)

    # Allocate result matrix
    result = np.empty_like(matrix1)

    # Allocate device memory
    d_matrix1 = cuda.to_device(matrix1)
    d_matrix2 = cuda.to_device(matrix2)
    d_result = cuda.device_array_like(matrix1)

    # Define grid/block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = (matrix1.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (matrix1.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    add_matrices_kernel[blocks_per_grid, threads_per_block](d_matrix1, d_matrix2, d_result)

    # Copy result back to host
    d_result.copy_to_host(result)
    return result

# Alternative CPU implementation for when CUDA is not available
def cpu_kernel(matrix1, matrix2):
    return matrix1 + matrix2

# Example usage
matrix1 = np.random.rand(1024, 1024)
matrix2 = np.random.rand(1024, 1024)

try:
    # Try CUDA first
    result = cuda_kernel(matrix1, matrix2)
    print("CUDA computation completed successfully.")
except Exception as e:
    print(f"CUDA not available: {e}")
    print("Falling back to CPU implementation...")
    result = cpu_kernel(matrix1, matrix2)
    print("CPU computation completed successfully.")

# Verify the result
expected = matrix1 + matrix2
print(f"Result is correct: {np.allclose(result, expected)}") 