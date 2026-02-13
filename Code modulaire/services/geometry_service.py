"""
Geometry Service - Handles all geometric transformations for MSDP channels
Refactored to handle vertical parabolas (x as function of y) for channel edges
"""

import numpy as np
from infrastructure.computation import Computation

EPSILON = 1e-12


class GeometryService:
    """
    Service responsible for geometric operations on MSDP channels.
    """

    def __init__(self, config_manager):
        self.config = config_manager
        self.geometry_params = config_manager.get_geometry_params()

    def build_channel_edges(self, points):
        """
        Build parabolic and linear edges from detected points.
        
        CHANGES:
        - Left/Right edges are now computed as x = f(y) using parabolic_interpolation_vertical
        """
        try:
            # Left parabola (Vertical, x(y)) through points a, b, c
            parabola_left = Computation.parabolic_interpolation_vertical(
                points['a'].xy(),
                points['b'].xy(),
                points['c'].xy()
            )

            # Right parabola (Vertical, x(y)) through points d, e, f
            parabola_right = Computation.parabolic_interpolation_vertical(
                points['d'].xy(),
                points['e'].xy(),
                points['f'].xy()
            )

            # Top line through points l, n (Standard y = mx+p)
            line_top = Computation.line_coefficients(
                points['l'].xy(),
                points['n'].xy()
            )

            # Bottom line through points k, m (Standard y = mx+p)
            line_bottom = Computation.line_coefficients(
                points['k'].xy(),
                points['m'].xy()
            )

            parabolas = [parabola_left, parabola_right]
            # Lines stored as (0, a, b) for consistency
            lines = [(0, line_top[0], line_top[1]),
                     (0, line_bottom[0], line_bottom[1])]

            return parabolas, lines

        except Exception as e:
            raise ValueError(f"Failed to build channel edges: {e}")

    def compute_channel_corners(self, parabolas, lines, near_points):
        """
        Compute the ABCDEF corner points.
        Uses updated find_intersection which handles x(y) parabolas vs y(x) lines.
        """
        left_parabola = parabolas[0]   # x = ay^2 + by + c
        right_parabola = parabolas[1]  # x = ay^2 + by + c
        top_line = lines[0]            # y = mx + p
        bottom_line = lines[1]         # y = mx + p

        # A = intersection(left_parabola, bottom_line) near 'a'
        corner_A = Computation.find_intersection(
            left_parabola, bottom_line, near_points[2], display=False
        )

        # C = intersection(left_parabola, top_line) near 'c'
        corner_C = Computation.find_intersection(
            left_parabola, top_line, near_points[0], display=False
        )

        # D = intersection(right_parabola, bottom_line) near 'd'
        corner_D = Computation.find_intersection(
            right_parabola, bottom_line, near_points[3], display=False
        )

        # F = intersection(right_parabola, top_line) near 'f'
        corner_F = Computation.find_intersection(
            right_parabola, top_line, near_points[1], display=False
        )

        return {
            'A': corner_A, 'C': corner_C, 'D': corner_D, 'F': corner_F
        }

    def compute_channel_size(self, corners):
        """Compute the output dimensions for a normalized solar channel."""
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners

        width_top = np.hypot(x1 - x0, y1 - y0)
        width_bottom = np.hypot(x2 - x3, y2 - y3)
        width = int(round((width_top + width_bottom) / 2))

        height_left = np.hypot(x3 - x0, y3 - y0)
        height_right = np.hypot(x2 - x1, y2 - y1)
        height = int(round((height_left + height_right) / 2))

        return height, width

    def normalize_channel_to_rectangle(self, image_data, parabolas, output_shape):
        """
        Transform a vertical parabolic channel shape into a rectangular solar coordinate system.
        
        UPDATED LOGIC:
        - Assumes parabolas are x = ay^2 + by + c (left/right edges)
        - Assumes lines are y = ax + b (top/bottom edges)
        - Directly evaluates x positions from y, instead of solving quadratics.
        """
        h_out, w_out = output_shape
        img_out = np.zeros(
            (h_out, w_out, *image_data.shape[2:]), dtype=image_data.dtype)

        # Unpack geometric entities
        # Parabolas (vertical): x(y) = a*y^2 + b*y + c
        (a_g, b_g, c_g) = parabolas[0]  # Left
        (a_d, b_d, c_d) = parabolas[1]  # Right

        # Lines (horizontal-ish): y(x) = a*x + b
        (dummy1, m_h, k_h) = parabolas[2]  # Top
        (dummy2, m_b, k_b) = parabolas[3]  # Bottom

        # Pre-calculate normalized coordinates
        v_coords = np.linspace(0, 1, h_out)
        u_coords = np.linspace(0, 1, w_out)

        # Iterate over output grid
        for i in range(h_out):
            v = v_coords[i]

            # 1. First approximation of Y in CCD (linear map)
            # We map v=0 to roughly top line average, v=1 to bottom line average
            # Ideally we iterate to find exact Y, but for small tilts,
            # estimating Y based on image height is a starting point,
            # or better: interpolate between the Y-intercepts of lines.
            y_approx = k_h + v * (k_b - k_h)

            # 2. Refine Y/X mapping
            # Since the top/bottom lines are functions of X, and X is function of Y,
            # we can calculate exact x bounds for this specific y_approx row.

            # Calculate Left/Right X boundaries at this Y
            x_left = a_g * y_approx**2 + b_g * y_approx + c_g
            x_right = a_d * y_approx**2 + b_d * y_approx + c_d

            # Map the row of pixels
            for j in range(w_out):
                u = u_coords[j]

                # Interpolate X position between left and right parabolas
                x = x_left + u * (x_right - x_left)

                # 3. Re-calculate Y based on the top/bottom lines at this specific X
                # This corrects for the tilt of the top/bottom lines
                y_top_at_x = m_h * x + k_h
                y_bottom_at_x = m_b * x + k_b

                # The true Y for this (u,v) point is the interpolation between top/bottom lines
                y = y_top_at_x + v * (y_bottom_at_x - y_top_at_x)

                # Bilinear interpolation
                if 0 <= x < image_data.shape[1] - 1 and 0 <= y < image_data.shape[0] - 1:
                    # Optimized manual bilinear interp for speed or use cv2 if available
                    x0, y0 = int(x), int(y)
                    dx, dy = x - x0, y - y0

                    p00 = image_data[y0, x0]
                    p01 = image_data[y0, x0+1]
                    p10 = image_data[y0+1, x0]
                    p11 = image_data[y0+1, x0+1]

                    val = (p00 * (1-dx) * (1-dy) +
                           p01 * dx * (1-dy) +
                           p10 * (1-dx) * dy +
                           p11 * dx * dy)
                    img_out[i, j] = val
                else:
                    img_out[i, j] = 0

        return img_out.squeeze()

    def compute_geometric_statistics(self, channels):
        """Compute geometric statistics for validation."""
        widths = []
        for channel in channels:
            if hasattr(channel, 'points_final'):
                pf = channel.points_final
                if 'A' in pf and 'D' in pf:
                    # Width at bottom
                    width = np.hypot(pf['D'].x - pf['A'].x,
                                     pf['D'].y - pf['A'].y)
                    widths.append(width)

        Wij = np.mean(widths) if widths else 0.0

        spacings = []
        for i in range(len(channels) - 1):
            if hasattr(channels[i], 'points_final') and hasattr(channels[i+1], 'points_final'):
                p1 = channels[i].points_final
                p2 = channels[i+1].points_final
                if 'A' in p1 and 'A' in p2:
                    spacing = p2['A'].x - p1['A'].x
                    spacings.append(spacing)

        Tgij = np.mean(spacings) if spacings else 0.0

        return {'Wij': Wij, 'Tgij': Tgij, 'num_channels': len(channels)}
