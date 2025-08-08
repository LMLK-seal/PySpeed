# matrix_mult.py

import time
import numpy as np

def slow_matrix_multiply(A, B):
    """
    Performs matrix multiplication using a naive, pure Python triple-nested loop.

    This function is extremely slow for large matrices and is a perfect
    candidate for optimization by Numba, as it is a CPU-bound task with
    heavy array access and simple arithmetic inside tight loops.
    """
    # Get the dimensions of the matrices
    rows_A = A.shape[0]
    cols_A = A.shape[1]
    rows_B = B.shape[0]
    cols_B = B.shape[1]

    # The inner dimensions must match for multiplication to be possible
    if cols_A != rows_B:
        raise ValueError("Inner dimensions of matrices do not match!")

    # Create the result matrix, initialized with zeros
    C = np.zeros((rows_A, cols_B))

    # Perform the multiplication
    # For each row in the first matrix...
    for i in range(rows_A):
        # For each column in the second matrix...
        for j in range(cols_B):
            # For each element in the row/column pair...
            for k in range(cols_A):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

if __name__ == "__main__":
    # Define the size of the square matrices.
    # A size of 400-500 is large enough to take several seconds in pure Python.
    N = 400

    print(f"Creating two {N}x{N} random matrices...")
    # Create two random matrices with floating point numbers
    matrix_A = np.random.rand(N, N)
    matrix_B = np.random.rand(N, N)

    print("\n--- Running Naive Python Implementation ---")
    start_time_slow = time.perf_counter()
    result_slow = slow_matrix_multiply(matrix_A, matrix_B)
    end_time_slow = time.perf_counter()
    time_slow = end_time_slow - start_time_slow
    print(f"Result matrix shape: {result_slow.shape}")
    print(f"Time taken: {time_slow:.4f} seconds")

    print("\n--- Running Optimized NumPy Implementation (for reference) ---")
    start_time_fast = time.perf_counter()
    # Use NumPy's highly optimized dot product, which is implemented in C/Fortran
    result_fast = np.dot(matrix_A, matrix_B)
    end_time_fast = time.perf_counter()
    time_fast = end_time_fast - start_time_fast
    print(f"Result matrix shape: {result_fast.shape}")
    print(f"Time taken: {time_fast:.4f} seconds")

    # Verify that the results are the same (within a small tolerance for float errors)
    assert np.allclose(result_slow, result_fast)
    print("\nVerification successful: Naive and NumPy results are identical.")