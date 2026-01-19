"""
Computation utilities - Low-level mathematical and image processing functions
Refactored from original computation.py to be more modular
"""

import numpy as np


class Computation:
    """
    Static utility class for mathematical computations.
    
    Organized into sections:
    - Image processing (filters, stretches)
    - Derivatives and edge detection
    - Geometric interpolations
    - Intersection calculations
    """

    # ==================== IMAGE PROCESSING ====================

    @staticmethod
    def median_filter(data, size=3):
        """Apply a median filter for noise reduction."""
        from scipy.ndimage import median_filter
        return median_filter(data, size=size)

    @staticmethod
    def gaussian_filter(data, sigma=1):
        """Apply a Gaussian filter for noise reduction."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(data, sigma=sigma)

    @staticmethod
    def linear_stretch(data):
        """Apply a linear stretch to normalize data range."""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max == data_min:
            return data
        stretched = (data - data_min) / (data_max - data_min)
        return stretched * (data_max - data_min) + data_min

    @staticmethod
    def asin_stretch(data):
        """Apply an arcsine stretch for better visualization."""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max == data_min:
            return data
        normalized = (data - data_min) / (data_max - data_min) * 2 - 1
        stretched = np.arcsin(normalized) / (np.pi / 2)
        return (stretched + 1) / 2 * (data_max - data_min) + data_min

    # ==================== DERIVATIVES ====================

    @staticmethod
    def mean_derivative_array(image_data):
        """
        Compute mean first derivative along X-axis across all columns.
        Used for detecting field-of-view bounds.
        
        Args:
            image_data (np.ndarray): 2D image array
        
        Returns:
            np.ndarray: Mean derivative along Y
        """
        derivatives = {x: np.diff(image_data[:, x])
                       for x in range(image_data.shape[1])}
        return np.mean(list(derivatives.values()), axis=0)

    @staticmethod
    def compute_first_derivative_at_positions(image_data, axis, positions):
        """
        Compute first derivative at specific positions with smoothing.
        
        Args:
            image_data (np.ndarray): 2D image array
            axis (int): 0 for X (horizontal), 1 for Y (vertical)
            positions (list): List of positions to compute derivatives
        
        Returns:
            dict: {position: derivative_array}
        """
        derivatives = {}

        if axis == 0:  # Derivative along X
            # Smooth over 5 rows (±2) and use gradient kernel
            kernel = np.array([2, 1, 0, -1, -2])
            for y in positions:
                # Average over 5 adjacent rows for robustness
                rows = [image_data[y + dy, :]
                        for dy in range(-2, 3) if 0 <= y + dy < image_data.shape[0]]
                if rows:
                    smoothed = np.mean(rows, axis=0)
                    derivatives[y] = np.convolve(smoothed, kernel, mode='same')

        elif axis == 1:  # Derivative along Y
            for x in positions:
                derivatives[x] = np.diff(image_data[:, x])

        return derivatives

    @staticmethod
    def filtre_de_sobel(derivatives):
        """
        Apply Sobel-like filter to enhance edges in derivative data.
        
        Args:
            derivatives (dict): Dictionary of derivative arrays
        
        Returns:
            dict: Filtered derivatives
        """
        sobel_kernel = np.array([1, 0, -1])
        return {key: np.convolve(deriv, sobel_kernel, mode='same')
                for key, deriv in derivatives.items()}

    @staticmethod
    def top_n_local_maxima(array, n):
        """
        Find indices of top N local maxima in an array.
        
        Args:
            array (np.ndarray): 1D array
            n (int): Number of maxima to find
        
        Returns:
            list: Indices of top N local maxima, sorted by value (descending)
        """
        # Find all local maxima
        maxima = [
            (i, array[i])
            for i in range(1, len(array) - 1)
            if array[i] > array[i-1] and array[i] > array[i+1]
        ]

        # Sort by value (descending) and return top N indices
        maxima.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in maxima[:n]]

    # ==================== GEOMETRIC INTERPOLATIONS ====================

    @staticmethod
    def parabolic_interpolation(p1, p2, p3):
        """
        Fit a parabola through three points.
        Solves y = ax² + bx + c for coefficients a, b, c.
        
        Args:
            p1, p2, p3 (tuple): Points as (x, y) tuples
        
        Returns:
            tuple: (a, b, c) coefficients
        """
        # Build system of equations: [x²  x  1] [a]   [y]
        #                            [x²  x  1] [b] = [y]
        #                            [x²  x  1] [c]   [y]
        A = np.array([
            [p1[0]**2, p1[0], 1],
            [p2[0]**2, p2[0], 1],
            [p3[0]**2, p3[0], 1]
        ])
        B = np.array([p1[1], p2[1], p3[1]])

        # Solve for coefficients
        coefficients = np.linalg.solve(A, B)
        return tuple(coefficients)

    @staticmethod
    def line_coefficients(p1, p2):
        """
        Compute line coefficients y = ax + b through two points.
        
        Args:
            p1, p2 (tuple): Points as (x, y) tuples
        
        Returns:
            tuple: (a, b) coefficients
        """
        # Handle vertical line
        if abs(p1[0] - p2[0]) < 1e-10:
            return (0, p1[1])

        # Compute slope and intercept
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0]

        return (a, b)

    # ==================== INTERSECTIONS ====================

    @staticmethod
    def find_intersection(parabola, line, near_point, display=False):
        """
        Find intersection between a parabola and a line.
        Uses discriminant of parabola to choose nearest solution.
        
        Parabola: y = a*x² + b*x + c
        Line: y = a_l*x + b_l (stored as (0, a_l, b_l) for consistency)
        
        Args:
            parabola (tuple): (a, b, c) coefficients
            line (tuple): (0, a_l, b_l) coefficients
            near_point (tuple): (x, y) to discriminate between two solutions
            display (bool): If True, plot the intersection (for debugging)
        
        Returns:
            tuple: (x, y) intersection point
        """
        a_p, b_p, c_p = parabola
        _, a_l, b_l = line

        # Solve: a_p*x² + b_p*x + c_p = a_l*x + b_l
        # Rearrange: a_p*x² + (b_p - a_l)*x + (c_p - b_l) = 0
        A = a_p + 1e-6  # Add small epsilon to avoid division by zero
        B = b_p - a_l
        C = c_p - b_l

        # Discriminant
        delta = B**2 - 4*A*C

        if delta < 0:
            # No real solution - shouldn't happen with proper geometry
            print(f"Warning: No intersection found (delta={delta})")
            return None

        # Two solutions from quadratic formula
        sqrt_delta = np.sqrt(delta)
        x1 = (-B + sqrt_delta) / (2*A)
        x2 = (-B - sqrt_delta) / (2*A)

        # Compute corresponding y values
        y1 = a_p * x1**2 + b_p * x1 + c_p
        y2 = a_p * x2**2 + b_p * x2 + c_p

        # Choose solution nearest to the reference point
        dist1 = np.hypot(x1 - near_point[0], y1 - near_point[1])
        dist2 = np.hypot(x2 - near_point[0], y2 - near_point[1])

        result = (x1, y1) if dist1 < dist2 else (x2, y2)

        # Validate solution (should satisfy both equations)
        y_parabola = a_p * result[0]**2 + b_p * result[0] + c_p
        y_line = a_l * result[0] + b_l
        error = abs(y_parabola - y_line)

        if error > 1.0:
            print(f"Warning: Intersection error = {error:.3f} pixels")

        # Optional visualization for debugging
        if display:
            Computation._plot_intersection(parabola, line, near_point, result)

        return result

    @staticmethod
    def _plot_intersection(parabola, line, near_point, result):
        """Helper method to visualize intersection (for debugging)."""
        import matplotlib.pyplot as plt

        a_p, b_p, c_p = parabola
        _, a_l, b_l = line

        # Generate X range around intersection
        x_min = min(result[0], near_point[0]) - 10
        x_max = max(result[0], near_point[0]) + 10
        x_vals = np.linspace(x_min, x_max, 400)

        # Compute Y values
        y_parabola = a_p * x_vals**2 + b_p * x_vals + c_p
        y_line = a_l * x_vals + b_l

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_parabola, 'b-', label='Parabola', linewidth=1)
        plt.plot(x_vals, y_line, 'r-', label='Line', linewidth=1)
        plt.scatter([result[0]], [result[1]], color='green',
                    s=50, label='Intersection', zorder=5)
        plt.scatter([near_point[0]], [near_point[1]],
                    color='orange', s=50, label='Near Point', zorder=5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Parabola-Line Intersection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
