# image_brighten.py

import time
import numpy as np

def brighten_image_loop(image_array, brightness_factor):
    """
    Brightens an image by adding a constant factor to each pixel using a loop.

    This is a classic element-wise operation that is very slow in pure Python
    but can be expressed as a single, highly efficient vectorized operation
    in NumPy.
    """
    # --- FIX: Convert to a larger integer type BEFORE the loop to prevent overflow ---
    # We work with a temporary array that can hold values larger than 255.
    work_array = image_array.astype(np.int16)
    
    # Loop over every pixel to apply the brightness
    for i in range(work_array.shape[0]):
        for j in range(work_array.shape[1]):
            # This is the slow, per-element operation
            work_array[i, j] += brightness_factor
            
    # Ensure pixel values stay within the valid 0-255 range and convert back to uint8
    return np.clip(work_array, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Create a large dummy image (e.g., 4K resolution)
    width, height = 3840, 2160
    
    print(f"Creating a {width}x{height} test image...")
    image = np.array([[(x + y) % 256 for x in range(width)] for y in range(height)], dtype=np.uint8)

    brightness = 50

    print("\n--- Running Naive Python Loop Implementation ---")
    start_time_loop = time.perf_counter()
    brightened_loop = brighten_image_loop(image, brightness)
    end_time_loop = time.perf_counter()
    time_loop = end_time_loop - start_time_loop
    print(f"Time taken: {time_loop:.4f} seconds")

    print("\n--- Running Optimized NumPy Vectorized Implementation ---")
    start_time_vec = time.perf_counter()
    # This is the highly efficient, one-line equivalent
    brightened_vec = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    end_time_vec = time.perf_counter()
    time_vec = end_time_vec - start_time_vec
    print(f"Time taken: {time_vec:.4f} seconds")

    # Verify the results are identical
    assert np.array_equal(brightened_loop, brightened_vec)
    print("\nVerification successful: Loop and Vectorized results are identical.")