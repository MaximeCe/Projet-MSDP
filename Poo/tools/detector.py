import numpy as np
from tools.computation import Computation


class Detector:
    @staticmethod
    def top_and_bottom_detection(mean_derivatives) -> tuple[int | None, int | None]:
        threshold = 0.1 * np.max(np.abs(mean_derivatives))
        first_peak = next((i for i, v in enumerate(
            mean_derivatives) if abs(v) > threshold), None)
        last_peak = next((i for i, v in reversed(
            list(enumerate(mean_derivatives))) if abs(v) > threshold), None)
        return first_peak, last_peak

    @staticmethod
    def point_detection_x(derivatives, y_positions):
        lines = []
        for y in y_positions:
            edges_x = Computation.top_n_local_maxima(-derivatives[y], 18)
            lines.append(sorted([(xi, y)
                         for xi in edges_x], key=lambda p: p[0]))
        return lines

    @staticmethod
    def detect_edges_x(flat, display=False):
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
        be_list = sorted(be_list, key=lambda x: x[0])
        x_positions = []
        for b, e in zip(be_list[::2], be_list[1::2]):
            x_positions.append(int(b[0] + (e[0]-b[0]) / 3))
            x_positions.append(int(b[0] + 2*(e[0]-b[0]) / 3))
        return sorted(x_positions)

    @staticmethod
    def point_detection_y(derivatives, x_positions):
        detected = []
        for x in x_positions:
            ys = Computation.top_n_local_maxima(-derivatives[x], 2)
            detected.append(sorted([(x, yi)
                            for yi in ys], key=lambda p: (p[0], p[1])))
        return detected

    @staticmethod
    def detect_edges_y(flat, be_list, display = False):
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
