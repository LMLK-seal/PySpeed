# monte_carlo_pi.py

import time
import random

def estimate_pi_monte_carlo(n_points: int):
    """
    Estimates Pi using the Monte Carlo method.

    This function generates random points in a 1x1 square and checks
    how many fall within a circle of radius 1 inscribed within it.
    The ratio of points inside to the total points approximates Pi/4.

    This is a perfect candidate for Numba due to its tight loop with
    random number generation and floating-point arithmetic.
    """
    points_inside_circle = 0
    
    # The loop is the computational core of the function.
    for _ in range(n_points):
        # Generate a random point (x, y) in the square [-1, 1] x [-1, 1]
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # Calculate the distance from the origin
        distance_squared = x**2 + y**2
        
        # Check if the point is inside the unit circle
        if distance_squared <= 1:
            points_inside_circle += 1
            
    # The final calculation using the ratio
    pi_estimate = 4 * points_inside_circle / n_points
    return pi_estimate

if __name__ == "__main__":
    # Use a large number of points for an accurate estimate and significant work
    num_samples = 50_000_000
    
    print(f"Estimating Pi using the Monte Carlo method with {num_samples:,} samples...")
    
    start_time = time.time()
    
    # Run the estimation
    pi = estimate_pi_monte_carlo(num_samples)
    
    end_time = time.time()
    
    print(f"Estimated value of Pi: {pi}")
    print(f"Actual value of Pi:    3.1415926535...")
    print(f"Time taken: {end_time - start_time:.4f} seconds")