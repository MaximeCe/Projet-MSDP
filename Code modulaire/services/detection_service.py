"""
Detection Service - Handles all edge detection logic for MSDP channels
Extracted from detector.py to separate concerns
"""

import numpy as np
from infrastructure.computation import Computation

class DetectionService:
    """
    Service responsible for detecting channel edges in flat field images.
    Implements the edge detection algorithms from ms2.f (newgeom).
    
    This service is stateless and operates purely on input data.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the detection service with configuration.
        
        Args:
            config_manager: ConfigManager instance containing detection parameters
        """
        self.config = config_manager
        self.detection_params = config_manager.get_detection_params()
    
    def detect_all_channel_edges(self, flat_image_data):
        """
        Main entry point: detect all edges for all channels in the flat field.
        
        Args:
            flat_image_data (np.ndarray): Flat field image data (after dark subtraction)
        
        Returns:
            dict: Dictionary with detected points for all channels
                  Format: {'as_': [...], 'bs': [...], ..., 'ks': [...], etc.}
        """
        # Step 1: Detect horizontal edges (a, b, c, d, e, f points)
        horizontal_edges = self._detect_horizontal_edges(flat_image_data)
        
        # Step 2: Detect vertical edges (k, l, m, n points)
        vertical_edges = self._detect_vertical_edges(
            flat_image_data, 
            horizontal_edges[1]  # Use b,e points to position vertical cuts
        )
        
        # Step 3: Combine into points dictionary
        points_dict = self._combine_edges_to_points(horizontal_edges, vertical_edges)
        
        return points_dict
    
    def _detect_horizontal_edges(self, image_data):
        """
        Detect horizontal edges (long edges parallel to X-axis).
        Corresponds to Fortran detection of a,b,c,d,e,f points.
        
        Args:
            image_data (np.ndarray): Flat field image
        
        Returns:
            dict: Detected horizontal edges with keys 'top', 'middle', 'bottom'
        """
        # Step 1: Compute mean derivative to find field-of-view bounds
        mean_derivative = Computation.mean_derivative_array(image_data)
        
        # Step 2: Detect top and bottom of field-of-view
        first_peak, last_peak = self._detect_field_bounds(mean_derivative)
        
        if first_peak is None or last_peak is None:
            raise ValueError("Failed to detect field-of-view bounds")
        
        # Step 3: Define 3 Y-positions for horizontal cuts (similar to Fortran ja(1), ja(2), ja(3))
        y_offset = self.detection_params.get('y_positions_offset', 60)
        y_positions = [
            first_peak + y_offset,           # Top third
            (first_peak + last_peak) // 2,   # Middle
            last_peak - y_offset             # Bottom third
        ]
        
        # Step 4: Compute derivatives at these positions
        derivatives = Computation.compute_first_derivative_at_positions(
            image_data, 
            axis=0,  # Along X
            positions=y_positions
        )
        
        # Step 5: Apply Sobel filter to enhance edges
        derivatives = Computation.filtre_de_sobel(derivatives)
        
        # Step 6: Detect edge positions (18 edges = 9 channels × 2 sides)
        detected_lines = self._detect_edge_positions_x(derivatives, y_positions)
        
        return detected_lines
    
    def _detect_vertical_edges(self, image_data, be_points):
        """
        Detect vertical edges (short edges parallel to Y-axis).
        Corresponds to Fortran detection of k,l,m,n points.
        
        Args:
            image_data (np.ndarray): Flat field image
            be_points (list): List of (x,y) tuples for b and e points
        
        Returns:
            list: Detected vertical edge points
        """
        # Step 1: Compute X positions for vertical cuts (between b and e points)
        x_positions = self._compute_vertical_cut_positions(be_points)
        
        # Step 2: Compute derivatives along Y at these X positions
        derivatives = Computation.compute_first_derivative_at_positions(
            image_data,
            axis=1,  # Along Y
            positions=x_positions
        )
        
        # Step 3: Detect top and bottom edge for each channel
        detected_edges = self._detect_edge_positions_y(derivatives, x_positions)
        
        return detected_edges
    
    def _detect_field_bounds(self, mean_derivative):
        """
        Detect the top and bottom boundaries of the field-of-view.
        Uses threshold on mean derivative similar to Fortran approach.
        
        Args:
            mean_derivative (np.ndarray): Mean derivative along X
        
        Returns:
            tuple: (first_peak, last_peak) indices
        """
        threshold_factor = self.detection_params.get('threshold_factor', 0.1)
        threshold = threshold_factor * np.max(np.abs(mean_derivative))
        
        # Find first position above threshold
        first_peak = next(
            (i for i, v in enumerate(mean_derivative) if abs(v) > threshold), 
            None
        )
        
        # Find last position above threshold
        last_peak = next(
            (i for i, v in reversed(list(enumerate(mean_derivative))) if abs(v) > threshold), 
            None
        )
        
        return first_peak, last_peak
    
    def _detect_edge_positions_x(self, derivatives, y_positions):
        """
        Detect edge positions from horizontal derivatives.
        Finds the N highest local maxima (N = number of channels × 2).
        
        Args:
            derivatives (dict): Derivatives at each Y position
            y_positions (list): Y positions where derivatives were computed
        
        Returns:
            list: Three lists of detected points (top, middle, bottom)
        """
        lines = []
        for y in y_positions:
            # Find 18 local maxima (9 channels × 2 edges)
            edges_x = Computation.top_n_local_maxima(
                -derivatives[y],  # Negative because we want minima of derivative
                18  # 9 channels × 2 edges
            )
            # Sort by X position and pair with Y
            line_points = sorted([(xi, y) for xi in edges_x], key=lambda p: p[0])
            lines.append(line_points)
        
        return lines
    
    def _compute_vertical_cut_positions(self, be_points):
        """
        Compute X positions for vertical cuts based on b,e points.
        Places cuts at 1/3 and 2/3 between each pair of b,e points.
        
        Args:
            be_points (list): Alternating b and e points
        
        Returns:
            list: X positions for vertical cuts
        """
        be_sorted = sorted(be_points, key=lambda p: p[0])
        x_positions = []
        
        # For each pair of (b, e) points
        for b, e in zip(be_sorted[::2], be_sorted[1::2]):
            x_b, x_e = b[0], e[0]
            # Two cuts at 1/3 and 2/3 of the interval
            x_positions.append(int(x_b + (x_e - x_b) / 3))
            x_positions.append(int(x_b + 2 * (x_e - x_b) / 3))
        
        return sorted(x_positions)
    
    def _detect_edge_positions_y(self, derivatives, x_positions):
        """
        Detect edge positions from vertical derivatives.
        Finds top and bottom edge for each X position.
        
        Args:
            derivatives (dict): Derivatives at each X position
            x_positions (list): X positions where derivatives were computed
        
        Returns:
            list: List of [(x, y_top), (x, y_bottom)] for each position
        """
        detected = []
        for x in x_positions:
            # Find 2 local maxima (top and bottom edge)
            ys = Computation.top_n_local_maxima(-derivatives[x], 2)
            # Sort by Y and pair with X
            points = sorted([(x, yi) for yi in ys], key=lambda p: (p[0], p[1]))
            detected.append(points)
        
        return detected
    
    def _combine_edges_to_points(self, horizontal_edges, vertical_edges):
        """
        Combine horizontal and vertical edge detections into named point groups.
        Separates alternating edges into left/right (a,b,c vs d,e,f) and top/bottom (k,l vs m,n).
        
        Args:
            horizontal_edges (list): Three lists of horizontal edge points
            vertical_edges (list): List of vertical edge point pairs
        
        Returns:
            dict: Dictionary with keys 'as_', 'bs', 'cs', etc.
        """
        # Separate horizontal edges into a,b,c (even indices) and d,e,f (odd indices)
        cs, fs, bs, es, as_, ds = [], [], [], [], [], []
        
        for i in range(len(horizontal_edges[0])):
            if i % 2 == 0:  # Left edges
                cs.append(horizontal_edges[0][i])  # Top
                bs.append(horizontal_edges[1][i])  # Middle
                as_.append(horizontal_edges[2][i])  # Bottom
            else:  # Right edges
                fs.append(horizontal_edges[0][i])
                es.append(horizontal_edges[1][i])
                ds.append(horizontal_edges[2][i])
        
        # Separate vertical edges into k,l (even indices) and m,n (odd indices)
        ls, ks, ns, ms = [], [], [], []
        
        for i in range(len(vertical_edges)):
            if i % 2 == 0:  # Left side
                ls.append(vertical_edges[i][0])  # Top
                ks.append(vertical_edges[i][1])  # Bottom
            else:  # Right side
                ns.append(vertical_edges[i][0])
                ms.append(vertical_edges[i][1])
        
        points_dict = {
            "as_": as_, "bs": bs, "cs": cs,
            "ds": ds, "es": es, "fs": fs,
            "ks": ks, "ls": ls, "ms": ms, "ns": ns
        }
        
        return points_dict