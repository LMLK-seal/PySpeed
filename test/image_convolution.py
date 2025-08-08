# image_convolution.py

import time
import numpy as np
from PIL import Image

def apply_convolution(image_array, kernel):
    """
    Applies a convolution kernel to an image using pure Python loops.

    This function iterates over each pixel of the image and applies the
    convolutional kernel to its neighborhood. This is a classic image
    processing task that is computationally intensive and ideal for Numba.
    """
    # Get dimensions
    img_height, img_width = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # Create an output image array with the same dimensions, initialized to zeros
    output_array = np.zeros_like(image_array)

    # Calculate padding needed to handle borders
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Iterate over each pixel of the image (excluding the padded border)
    for y in range(pad_h, img_height - pad_h):
        for x in range(pad_w, img_width - pad_w):
            # Apply the kernel to the neighborhood of the current pixel
            pixel_sum = 0.0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    # Get the corresponding image pixel
                    img_pixel = image_array[y - pad_h + ky, x - pad_w + kx]
                    # Get the kernel value
                    kernel_val = kernel[ky, kx]
                    # Multiply and add to the sum
                    pixel_sum += img_pixel * kernel_val
            
            # Assign the convoluted value to the output image
            output_array[y, x] = pixel_sum

    return output_array

if __name__ == "__main__":
    # --- Setup ---
    # Create a dummy noisy image for the test
    image_size = 1000
    print(f"Creating a {image_size}x{image_size} test image...")
    # Base image with some patterns
    base_image = np.zeros((image_size, image_size), dtype=np.float32)
    base_image[200:800, 200:300] = 255  # A white bar
    base_image[200:300, 200:800] = 255  # Another white bar
    # Add random noise
    noise = np.random.randint(0, 100, size=(image_size, image_size))
    noisy_image_array = np.clip(base_image + noise, 0, 255)

    # Define a simple blurring (box blur) kernel
    blur_kernel = np.ones((7, 7), dtype=np.float32) / 49.0

    print("Applying convolution using pure Python...")
    start_time = time.perf_counter()
    
    # Run the convolution function
    convoluted_image_array = apply_convolution(noisy_image_array, blur_kernel)
    
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    
    print(f"Time taken: {time_taken:.4f} seconds")

    # --- Save the results for visual confirmation ---
    # Convert float arrays back to 8-bit integers for saving
    print("Saving original and convoluted images...")
    Image.fromarray(noisy_image_array.astype(np.uint8)).save("original_noisy_image.png")
    Image.fromarray(convoluted_image_array.astype(np.uint8)).save("convoluted_blurred_image.png")
    
    print("\nImages saved successfully.")
    print("Compare 'original_noisy_image.png' with 'convoluted_blurred_image.png' to see the result.")
