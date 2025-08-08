# time_series_sma.py

import time
import numpy as np
import pandas as pd

def calculate_sma_naive(data, window_size):
    """
    Calculates the Simple Moving Average (SMA) using a naive Python loop.

    For each point in the time series, it calculates the mean of the
    preceding 'window_size' points. This involves repeated array slicing
    and is very inefficient in pure Python, making it an excellent
    candidate for Numba's JIT compilation.
    """
    n = len(data)
    # The result will be shorter than the input array
    output_size = n - window_size + 1
    sma_values = np.zeros(output_size)

    # Loop through the data to calculate the moving average for each window
    for i in range(output_size):
        # Create a "window" of data by slicing the array
        window = data[i : i + window_size]
        # Calculate the mean of the window and store it
        sma_values[i] = np.mean(window)
    
    return sma_values

if __name__ == "__main__":
    # --- Setup ---
    # Create a realistic-looking time series (e.g., a sine wave with noise)
    num_points = 5_000_000
    window_size = 100

    print(f"Creating a test time series with {num_points:,} data points...")
    # Generate the time series data
    x = np.linspace(0, 200, num_points)
    # A base sine wave plus some random noise to simulate a stock price or sensor reading
    time_series_data = np.sin(x) + np.random.randn(num_points) * 0.1

    print(f"Calculating {window_size}-point SMA...")

    # --- Benchmark Naive Python Implementation ---
    print("\n--- Running Naive Python Implementation ---")
    start_time_naive = time.perf_counter()
    sma_naive = calculate_sma_naive(time_series_data, window_size)
    end_time_naive = time.perf_counter()
    time_naive = end_time_naive - start_time_naive
    print(f"Result shape: {sma_naive.shape}")
    print(f"Time taken: {time_naive:.4f} seconds")

    # --- Benchmark Optimized Pandas Implementation ---
    print("\n--- Running Optimized Pandas Implementation (for reference) ---")
    start_time_pandas = time.perf_counter()
    # Use pandas' highly optimized, C-backend rolling mean
    data_series = pd.Series(time_series_data)
    sma_pandas = data_series.rolling(window=window_size).mean().dropna().to_numpy()
    end_time_pandas = time.perf_counter()
    time_pandas = end_time_pandas - start_time_pandas
    print(f"Result shape: {sma_pandas.shape}")
    print(f"Time taken: {time_pandas:.4f} seconds")

    # Verify that the results are functionally identical
    assert np.allclose(sma_naive, sma_pandas)
    print("\nVerification successful: Naive and Pandas results are identical.")