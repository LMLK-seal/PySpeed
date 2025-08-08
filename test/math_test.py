# In a new file, e.g., math_test.py

import time

def calculate_pi(n_terms: int):
    """
    A numerically-intensive function that is a perfect
    candidate for Numba.
    """
    numerator = 4.0
    denominator = 1.0
    pi = 0.0
    # This loop with floating-point math is a strong signal.
    for _ in range(n_terms):
        pi += numerator / denominator
        denominator += 2.0
        pi -= numerator / denominator
        denominator += 2.0
    return pi

if __name__ == "__main__":
    start = time.time()
    # Use a large number to make the work significant
    result = calculate_pi(50_000_000) 
    end = time.time()
    print(f"Pi calculation result: {result}")
    print(f"Time taken: {end - start:.4f} seconds")