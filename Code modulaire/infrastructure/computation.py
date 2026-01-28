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
    def median_filter(data, size=7):
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
            kernel = np.array([1, 0, -1])
            for y in positions:
                row = image_data[y, :]
                derivatives[y] = np.convolve(row, kernel, mode='same')

        elif axis == 1:  # Derivative along Y
            kernel = np.array([1, 0, -1])
            for x in positions:
                row = image_data[:, x]
                derivatives[x] = np.convolve(row, kernel, mode='same')

        return derivatives

    @staticmethod
    def compute_second_derivative_at_positions(image_data, axis, positions):
        """
        Compute second derivative at specific positions.

        Args:
            image_data (np.ndarray): 2D image array
            axis (int): 0 for X (horizontal), 1 for Y (vertical)
            positions (list): List of positions to compute derivatives

        Returns:
            dict: {position: second_derivative_array}
        """
        second_derivatives = {}

        if axis == 0:  # Second derivative along X
            kernel = np.array([1, -2, 1])
            for y in positions:
                row = image_data[y, :]
                second_derivatives[y] = np.convolve(row, kernel, mode='same')

        elif axis == 1:  # Second derivative along Y
            kernel = np.array([1, -2, 1])
            for x in positions:
                col = image_data[:, x]
                second_derivatives[x] = np.convolve(col, kernel, mode='same')

        return second_derivatives

    @staticmethod
    def compute_sobel_filter_at_positions(data, axis, positions):
        """Apply Sobel-like filter to calculate the second deriivative with a 30px ROI pooling at the specified positions."""
        sobel_results = {}

        if axis == 0:  # Sobel filter along X
            pooled = Computation.ROI_pooling(
                data, pool_size=30, axis=0, positions=positions)
            for y, line in zip(positions, pooled):
                filtered = Computation.Log(line)
                sobel_results[y] = filtered

        elif axis == 1:  # Sobel filter along Y
            pooled = Computation.ROI_pooling(
                data, pool_size=30, axis=1, positions=positions)
            for x, col in zip(positions, pooled):
                filtered = Computation.Log(col)
                sobel_results[x] = filtered

        return sobel_results

    @staticmethod
    def filtre_de_sobel(data):
        """
        Apply Sobel-like filter to enhance edges in derivative data.

        Args:
            data (np.ndarray): 1D array (line or column extracted from a matrix)

        Returns:
            np.ndarray: Filtered array
        """
        sobel_kernel = np.array([1, -2, -1])
        return np.convolve(data, sobel_kernel, mode='same')

    @staticmethod
    def Log(line, sigma=3):
        """
        Apply Laplacian of Gaussian (LoG) filter for edge detection.
        """
        from scipy.ndimage import gaussian_laplace
        return gaussian_laplace(line, sigma=sigma)

    @staticmethod
    def ROI_pooling(image_data, positions, axis=0, pool_size=30):
        """
        Apply ROI pooling by averaging over the perpendicular axis.
        Args :
            image_data (np.ndarray): 2D image array
            positions (list): List of positions to pool around
            axis (int): 0 for X (horizontal), 1 for Y (vertical)
            pool_size (int): Size of the pooling window
        Returns:
            list[np.ndarray]: List of pooled arrays (one per position)
            """
        pooled = []
        half_pool = pool_size // 2

        if axis == 0:  # Pooling along X (average rows around position)
            for y in positions:
                y_start = max(0, y - half_pool)
                y_end = min(image_data.shape[0], y + half_pool)
                pooled_row = np.median(image_data[y_start:y_end, :], axis=0)
                pooled.append(pooled_row)
        elif axis == 1:  # Pooling along Y (average columns around position)
            for x in positions:
                x_start = max(0, x - half_pool)
                x_end = min(image_data.shape[1], x + half_pool)
                pooled_col = np.median(image_data[:, x_start:x_end], axis=1)
                pooled.append(pooled_col)

        return pooled

    @staticmethod
    def top_n_local_maxima(array, n, radius=20):
        """
        Find indices of top N local maxima in an array.
        If radius is specified, returns only the strongest maxima within non-overlapping radius windows.

        Args:
            array (np.ndarray): 1D array
            n (int): Number of maxima to find
            radius (int, optional): Radius in pixels. If set, only the strongest maximum within each radius window is kept

        Returns:
            list: Indices of top N local maxima, sorted by value (descending)
        """
        # Find all local maxima
        maxima = [
            (i, array[i])
            for i in range(1, len(array) - 1)
            if array[i] > array[i-1] and array[i] > array[i+1]
        ]

        # Sort by value (descending)
        maxima.sort(key=lambda x: x[1], reverse=True)

        # If radius is specified, filter out maxima within radius of stronger ones
        if radius is not None:
            filtered_maxima = []
            excluded_ranges = []

            for idx, val in maxima:
                # Check if this maximum is within an excluded range
                is_excluded = any(abs(idx - excl_idx) <= radius for excl_idx, _ in excluded_ranges)

                if not is_excluded:
                    filtered_maxima.append((idx, val))
                    excluded_ranges.append((idx, val))

                if len(filtered_maxima) >= n:
                    break

            return [idx for idx, _ in filtered_maxima]
        else:
            return [idx for idx, _ in maxima[:n]]

    # ==================== GEOMETRIC INTERPOLATIONS ====================

    @staticmethod
    def parabolic_interpolation(p1, p2, p3, expected_sign=1):
        """
        Interpole une parabole avec validation du signe de 'a'.
        expected_sign: 1 pour une parabole convexe (U), -1 pour concave (pont).
        """
        points = np.array([p1, p2, p3])
        x = points[:, 0]
        y = points[:, 1]

        # Construction de la matrice de Vandermonde
        A = np.vander(x, 3)
        B = y

        try:
            # 1. Calcul des coefficients (a, b, c)
            coeffs = np.linalg.solve(A, B)
            a, _, _ = coeffs

            # 2. Vérification de la validité physique
            # Si 'a' est du mauvais signe ou presque nul, on rejette la parabole
            is_physically_wrong = np.sign(a) != np.sign(expected_sign)
            # Seuil à ajuster selon ton échelle
            is_nearly_linear = np.abs(a) < 1e-6

            if is_physically_wrong or is_nearly_linear:
                raise ValueError("Solution non physique ou trop linéaire")

            return tuple(coeffs)

        except (np.linalg.LinAlgError, ValueError):
            # 3. FALLBACK : Régression linéaire sur les 3 points
            # On ne peut plus faire d'interpolation exacte (3 points != 1 droite)
            # On utilise la pseudo-inverse pour trouver la droite la plus proche
            A_lin = np.vstack([x, np.ones(len(x))]).T
            m, q = np.linalg.lstsq(A_lin, y, rcond=None)[0]

            # On retourne (a=0, b=m, c=q) pour garder la signature (a, b, c)
            return (0.0, m, q)

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

        print(f"Finding intersection: delta={delta:.3f}")
        
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

        if error > 1:
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

    @staticmethod
    def stack_images(images):
        """
        Stack multiple images using median combination.

        Args:
            images (list): List of numpy arrays to stack

        Returns:
            np.ndarray: Median-stacked image
        """
        return np.median(np.array(images), axis=0)
