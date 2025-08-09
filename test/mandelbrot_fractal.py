# mandelbrot_fractal.py

import time
import numpy as np
from PIL import Image

def generate_mandelbrot_naive(width, height, x_min, x_max, y_min, y_max, max_iter):
    """
    Generates a Mandelbrot set image using pure Python loops.

    This is an extremely CPU-intensive task. Each pixel is calculated
    independently, involving complex number arithmetic in a tight loop.
    It is a perfect candidate for both Numba JIT compilation and for
    parallelization across multiple CPU cores.
    """
    # Create an empty array to store the image data (iteration count for each pixel)
    mandelbrot_image = np.zeros((height, width))
    
    # Calculate the size of each step in the complex plane
    x_step = (x_max - x_min) / width
    y_step = (y_max - y_min) / height

    # Loop over every pixel
    for y in range(height):
        for x in range(width):
            # Map the pixel coordinates to the complex plane
            c = complex(x_min + x * x_step, y_min + y * y_step)
            z = 0
            iteration = 0
            
            # The core Mandelbrot calculation loop
            while abs(z) < 2 and iteration < max_iter:
                z = z*z + c
                iteration += 1
            
            # Store the number of iterations it took to escape
            mandelbrot_image[y, x] = iteration
            
    return mandelbrot_image

if __name__ == "__main__":
    # --- High-Stress Parameters ---
    # High resolution for a large workload
    IMG_WIDTH = 3840  # 4K Width
    IMG_HEIGHT = 2160 # 4K Height
    # High iteration count for deep calculation and rich colors
    MAX_ITERATIONS = 500

    print(f"Generating a {IMG_WIDTH}x{IMG_HEIGHT} Mandelbrot set with {MAX_ITERATIONS} iterations...")
    
    # Define the boundaries of the complex plane to render
    X_MIN, X_MAX = -2.0, 1.0
    Y_MIN, Y_MAX = -1.0, 1.0

    # --- Run the Naive, Slow Implementation ---
    start_time = time.perf_counter()
    mandelbrot_data = generate_mandelbrot_naive(IMG_WIDTH, IMG_HEIGHT, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITERATIONS)
    end_time = time.perf_counter()
    
    print(f"Time taken (pure Python): {end_time - start_time:.4f} seconds")

    # --- Colorize and Save the Image ---
    print("Colorizing and saving the fractal image...")
    # Normalize the iteration counts to be in the 0-1 range for coloring
    normalized_data = mandelbrot_data / MAX_ITERATIONS
    
    # Use a colormap to create a visually appealing image
    # This creates a smooth psychedelic gradient
    r = (0.5 + 0.5 * np.cos(3.0 + normalized_data * 15.0 + 0.2)).astype(np.float32)
    g = (0.5 + 0.5 * np.cos(3.0 + normalized_data * 15.0 + 0.4)).astype(np.float32)
    b = (0.5 + 0.5 * np.cos(3.0 + normalized_data * 15.0 + 0.6)).astype(np.float32)

    # Combine the channels and scale to 0-255
    colored_image_data = np.clip(np.dstack((r, g, b)) * 255, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(colored_image_data, 'RGB')
    image.save("mandelbrot_fractal.png")
    
    print("\nImage saved as 'mandelbrot_fractal.png'.")
    print("This script is a prime candidate for both Numba and Multiprocessing optimizations.")