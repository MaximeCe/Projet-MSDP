import numpy as np
from tools.computation import Computation


class Detector:
    @staticmethod
    def top_and_bottom_detection(mean_derivatives) -> tuple[int | None, int | None]:
        """Detect the top and bottom lines of the channel"""
        threshold = 0.1 * np.max(np.abs(mean_derivatives))
        first_peak = next((i for i, v in enumerate(
            mean_derivatives) if abs(v) > threshold), None)
        last_peak = next((i for i, v in reversed(
            list(enumerate(mean_derivatives))) if abs(v) > threshold), None)
        return first_peak, last_peak

    @staticmethod
    def point_detection_x(derivatives, y_positions):
        """Detect a top point and a bottom point for each y_positions (2 points per channel for 3 y positions)"""
        lines = []
        for y in y_positions:
            edges_x = Computation.top_n_local_maxima(-derivatives[y], 18)
            lines.append(sorted([(xi, y)
                         for xi in edges_x], key=lambda p: p[0]))
        return lines

    @staticmethod
    def detect_edges_x(flat, display=False):
        """Detect three point on the left edge and on the right edge for each channel in the flat image

        Parameters
        ----------
        flat : Image
            Calibration Flat image
        display : bool, optional
            If True, display the detected points, by default False

        Returns
        -------
        list[list[tuple]]
            Points in the format (x,y) grouped by y coodinates; format of the output: (nbr_of_channels*2x2x2)
        """
        mean_der = Computation.mean_derivative(flat)
        # Display mean derivative for debugging

        
        first_peak, last_peak = Detector.top_and_bottom_detection(np.abs(mean_der))
        if first_peak is None or last_peak is None:
            return []
        y_positions = [first_peak + 20, (first_peak + last_peak) // 2, last_peak - 20]
        derivs = Computation.compute_first_derivative(flat, 0,y_positions)
        lines = Detector.point_detection_x(derivs, y_positions)
        
        if display == True:
            # Ad=ffichage des points (1px/point) de lines pour le debug
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Détection des points sur les dérivées en X")
            plt.imshow(flat.data, cmap='gray')
            plt.axis('off')
            for line in lines:
                xs, ys = zip(*line)
                plt.scatter(xs, ys, color='red', s=1)
            plt.show() 
        
        return lines

    @staticmethod
    def x_positions_computation(be_list):
        """Compute the x positions where the derivatives should be done"""
        be_list = sorted(be_list, key=lambda x: x[0])
        x_positions = []
        for b, e in zip(be_list[::2], be_list[1::2]):
            x_positions.append(int(b[0] + (e[0]-b[0]) / 3))
            x_positions.append(int(b[0] + 2*(e[0]-b[0]) / 3))
        return sorted(x_positions)

    @staticmethod
    def point_detection_y(derivatives, x_positions):
        """Detect a point on the bottom edge and top edge for each x_positions (2 x_position per channel)"""
        detected = []
        for x in x_positions:
            ys = Computation.top_n_local_maxima(-derivatives[x], 2)
            detected.append(sorted([(x, yi)
                            for yi in ys], key=lambda p: (p[0], p[1])))
        return detected

    @staticmethod
    def detect_edges_y(flat, be_list, display = False):
        """Detect a point on the bottom edge and top edge for each x_positions (2 x_position per channel eavenly distributed between a b and a e point)

        Parameters
        ----------
        flat : Image    
            
        be_list : list[tuple]
            list with b and e points alternating
        display : bool, optional
            If True, display the output points, by default False

        Returns
        -------
        list[list[tuple]]
            format : nbr of channel x [top,bottom] x (x,y) 
        """
        x_positions = Detector.x_positions_computation(be_list)
        derivs = Computation.compute_first_derivative(flat,1, x_positions)
        lines = Detector.point_detection_y(derivs, x_positions)
        
        if display == True:
            # Ad=ffichage des points (1px/point) de lines pour le debug
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Détection des points sur les dérivées en X")
            plt.imshow(flat.data, cmap='gray')
            plt.axis('off')
            for line in lines:
                xs, ys = zip(*line)
                plt.scatter(xs, ys, color='red', s=1)
            plt.show() 
            
        return lines
