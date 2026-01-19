"""
Geometry Service - Handles all geometric transformations for MSDP channels
Extracted from channel.py and channel_normaliser.py
"""

import numpy as np
from infrastructure.computation import Computation

EPSILON = 1e-12


class GeometryService:
    """
    Service responsible for geometric operations on MSDP channels.
    Implements geometric transformations from ms2.f (srect, newgeom).
    
    This service handles:
    - Parabolic interpolation of channel edges
    - Computation of channel corners (ABCDEF points)
    - Normalization to rectangular solar coordinates
    """

    def __init__(self, config_manager):
        """
        Initialize geometry service with configuration.
        
        Args:
            config_manager: ConfigManager instance with geometry parameters
        """
        self.config = config_manager
        self.geometry_params = config_manager.get_geometry_params()

    def build_channel_edges(self, points):
        """
        Build parabolic and linear edges from detected points.
        Corresponds to Fortran construction of paraboles and droites.
        
        Args:
            points (dict): Dictionary with keys 'a', 'b', 'c', etc.
        
        Returns:
            tuple: (parabolas, lines) where:
                   parabolas = [(a,b,c) for left edge, (a,b,c) for right edge]
                   lines = [(0,a,b) for top edge, (0,a,b) for bottom edge]
        """
        try:
            # Left parabola through points a, b, c
            parabola_left = Computation.parabolic_interpolation(
                points['a'].xy(),
                points['b'].xy(),
                points['c'].xy()
            )

            # Right parabola through points d, e, f
            parabola_right = Computation.parabolic_interpolation(
                points['d'].xy(),
                points['e'].xy(),
                points['f'].xy()
            )

            # Top line through points l, n
            line_top = Computation.line_coefficients(
                points['l'].xy(),
                points['n'].xy()
            )

            # Bottom line through points k, m
            line_bottom = Computation.line_coefficients(
                points['k'].xy(),
                points['m'].xy()
            )

            parabolas = [parabola_left, parabola_right]
            # Lines stored as (0, a, b) for consistency with parabola format
            lines = [(0, line_top[0], line_top[1]),
                     (0, line_bottom[0], line_bottom[1])]

            return parabolas, lines

        except Exception as e:
            raise ValueError(f"Failed to build channel edges: {e}")

    def compute_channel_corners(self, parabolas, lines, near_points):
        """
        Compute the ABCDEF corner points from parabolas and lines.
        Corresponds to Fortran computation of points ABCDEF from intersections.
        
        Args:
            parabolas (list): [(a,b,c), (a,b,c)] for left and right edges
            lines (list): [(0,a,b), (0,a,b)] for top and bottom edges
            near_points (list): Approximate positions of corners for discrimination
        
        Returns:
            dict: Corner points {'A': (x,y), 'B': (x,y), ..., 'F': (x,y)}
        """
        # Extract parabolas and lines
        left_parabola = parabolas[0]
        right_parabola = parabolas[1]
        top_line = lines[0]
        bottom_line = lines[1]

        # Compute intersections (corresponds to Fortran ABCDEF calculation)
        # A = intersection(left_parabola, bottom_line) near point 'a'
        corner_A = Computation.find_intersection(
            left_parabola,
            bottom_line,
            near_points[2],  # Near point 'a' (bottom-left)
            display=False
        )

        # C = intersection(left_parabola, top_line) near point 'c'
        corner_C = Computation.find_intersection(
            left_parabola,
            top_line,
            near_points[0],  # Near point 'c' (top-left)
            display=False
        )

        # D = intersection(right_parabola, bottom_line) near point 'd'
        corner_D = Computation.find_intersection(
            right_parabola,
            bottom_line,
            near_points[3],  # Near point 'd' (bottom-right)
            display=False
        )

        # F = intersection(right_parabola, top_line) near point 'f'
        corner_F = Computation.find_intersection(
            right_parabola,
            top_line,
            near_points[1],  # Near point 'f' (top-right)
            display=False
        )

        # B and E are the middle points (directly from detected points)
        # They will be set by the caller from points['b'] and points['e']

        return {
            'A': corner_A,
            'C': corner_C,
            'D': corner_D,
            'F': corner_F
        }

    def compute_channel_size(self, corners):
        """
        Compute the output dimensions for a normalized solar channel.
        Corresponds to channel_size in channel_normaliser.py.
        
        Args:
            corners (list): [top-left, top-right, bottom-right, bottom-left]
        
        Returns:
            tuple: (height, width) in pixels
        """
        # Corners order: [haut-gauche, haut-droit, bas-droit, bas-gauche]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners

        # Width = average of top and bottom widths
        width_top = np.hypot(x1 - x0, y1 - y0)
        width_bottom = np.hypot(x2 - x3, y2 - y3)
        width = int(round((width_top + width_bottom) / 2))

        # Height = average of left and right heights
        height_left = np.hypot(x3 - x0, y3 - y0)
        height_right = np.hypot(x2 - x1, y2 - y1)
        height = int(round((height_left + height_right) / 2))

        return height, width

    def normalize_channel_to_rectangle(self, image_data, parabolas, output_shape):
        """
        Transform a parabolic channel shape into a rectangular solar coordinate system.
        Corresponds to extract_parabolic_shape_to_rect in channel_normaliser.py.
        
        This implements the geometric transformation from ms3.f (channels).
        
        Args:
            image_data (np.ndarray): Original CCD image
            parabolas (list): [left, right, top, bottom] edge equations
            output_shape (tuple): (height, width) of output rectangle
        
        Returns:
            np.ndarray: Normalized rectangular channel data
        """
        h_out, w_out = output_shape

        # Initialize output array with same dtype as input
        img_out = np.zeros(
            (h_out, w_out, *image_data.shape[2:]), dtype=image_data.dtype)

        # Extract parabola coefficients
        # Order: [left, right, top, bottom]
        (a_g, b_g, c_g) = parabolas[0]  # Left (gauche)
        (a_d, b_d, c_d) = parabolas[1]  # Right (droite)
        (a_h, b_h, c_h) = parabolas[2]  # Top (haut)
        (a_b, b_b, c_b) = parabolas[3]  # Bottom (bas)

        # For each pixel in the output rectangle
        for i in range(h_out):
            # Normalized vertical position (0 to 1)
            v = i / (h_out - 1) if h_out > 1 else 0

            for j in range(w_out):
                # Normalized horizontal position (0 to 1)
                u = j / (w_out - 1) if w_out > 1 else 0

                # Map to CCD coordinates using parabolic boundaries
                # Start with Y coordinate
                y = v * (image_data.shape[0] - 1)

                # Solve left and right parabolas to find X boundaries at this Y
                xs_left = self._solve_parabola_for_x(a_g, b_g, c_g, y)
                xs_right = self._solve_parabola_for_x(a_d, b_d, c_d, y)

                if len(xs_left) == 0 or len(xs_right) == 0:
                    continue  # No solution, skip this pixel

                # Choose the appropriate solution for each side
                x_left = min(xs_left) if a_g < 0 else max(xs_left)
                x_right = min(xs_right) if a_d < 0 else max(xs_right)

                # Interpolate X position
                x = x_left + u * (x_right - x_left)

                # Compute Y from top and bottom parabolas
                y_top = a_h * x**2 + b_h * x + c_h
                y_bottom = a_b * x**2 + b_b * x + c_b
                y = y_top + v * (y_bottom - y_top)

                # Bilinear interpolation from source image
                if 0 <= x < image_data.shape[1] and 0 <= y < image_data.shape[0]:
                    # Use OpenCV-style subpixel extraction
                    import cv2
                    img_out[i, j] = cv2.getRectSubPix(
                        image_data,
                        (1, 1),
                        (float(x), float(y))
                    )
                else:
                    img_out[i, j] = 0

        return img_out.squeeze()

    def _solve_parabola_for_x(self, a, b, c, y):
        """
        Solve y = a*x^2 + b*x + c for x.
        Returns the two possible x values.
        
        Args:
            a, b, c: Parabola coefficients
            y: Y value to solve for
        
        Returns:
            list: Possible x values (0, 1, or 2 solutions)
        """
        # Handle linear case
        if abs(a) < EPSILON:
            if abs(b) > EPSILON:
                return [(y - c) / b]
            else:
                return []

        # Quadratic formula: a*x^2 + b*x + (c-y) = 0
        discriminant = b**2 - 4*a*(c - y)

        if discriminant < 0:
            return []

        sqrt_disc = np.sqrt(discriminant)
        x1 = (-b - sqrt_disc) / (2*a)
        x2 = (-b + sqrt_disc) / (2*a)

        return [x1, x2]

    def compute_geometric_statistics(self, channels):
        """
        Compute geometric statistics for validation.
        Corresponds to Fortran distortion calculations and geo3.ps plots.
        
        Args:
            channels (list): List of Channel objects with computed corners
        
        Returns:
            dict: Statistics including mean channel width, spacing, distortion
        """
        # Mean channel width (Wij in CCD pixels)
        widths = []
        for channel in channels:
            if hasattr(channel, 'points_final') and 'A' in channel.points_final and 'D' in channel.points_final:
                A = channel.points_final['A']
                D = channel.points_final['D']
                width = np.sqrt((D.x - A.x)**2 + (D.y - A.y)**2)
                widths.append(width)

        Wij = np.mean(widths) if widths else None

        # Mean channel spacing (Tgij in CCD pixels)
        spacings = []
        for i in range(len(channels) - 1):
            if hasattr(channels[i], 'points_final') and 'A' in channels[i].points_final:
                if hasattr(channels[i+1], 'points_final') and 'A' in channels[i+1].points_final:
                    A1 = channels[i].points_final['A']
                    A2 = channels[i+1].points_final['A']
                    spacing = A2.x - A1.x
                    spacings.append(spacing)

        Tgij = np.mean(spacings) if spacings else None

        return {
            'Wij': Wij,
            'Tgij': Tgij,
            'num_channels': len(channels)
        }
