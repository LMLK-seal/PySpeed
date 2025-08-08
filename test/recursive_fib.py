# recursive_fib.py

import time
import functools

def fibonacci(n: int) -> int:
    """
    Calculates the nth Fibonacci number using a naive recursive approach.

    This function is extremely inefficient because it re-computes the same
    Fibonacci numbers over and over again. For example, fibonacci(5) calls
    fibonacci(4) and fibonacci(3). fibonacci(4) then calls fibonacci(3) again.

    This is a perfect candidate for memoization with @functools.lru_cache,
    as it's a pure function (always returns the same output for the same
    input) and has hashable integer arguments.
    """
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    # A value around 35-40 is high enough to take several seconds.
    # Warning: n=40 can take a very long time without optimization!
    number = 38

    print(f"Calculating Fibonacci number {number} using naive recursion...")
    
    start_time = time.perf_counter()
    result = fibonacci(number)
    end_time = time.perf_counter()
    
    time_taken = end_time - start_time
    
    print(f"Fibonacci({number}) = {result}")
    print(f"Time taken: {time_taken:.4f} seconds")
    
    # --- For reference, show the optimized version ---
    print("\n--- Running with @lru_cache (for reference) ---")
    
    @functools.lru_cache(maxsize=None)
    def fibonacci_cached(n: int) -> int:
        if n < 2:
            return n
        return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

    start_time_cached = time.perf_counter()
    result_cached = fibonacci_cached(number)
    end_time_cached = time.perf_counter()
    time_cached = end_time_cached - start_time_cached

    print(f"Fibonacci({number}) = {result_cached}")
    print(f"Time taken (cached): {time_cached:.8f} seconds (likely near zero)")