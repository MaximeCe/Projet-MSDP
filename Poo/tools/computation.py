from cv2 import threshold
import numpy as np
from numpy.typing import NDArray
from typing import Any

class Computation:
    @staticmethod
    def mask(image, threshold, display=False):
        """Compute the masqued version of an image

        Parameters
        ----------
        image : NDArray
            The image that you want to mask
        threshold : float
            Threshold under which every pixel is set to 0
        display : bool, optional
            If True, display the masked image, by default False

        Returns
        -------
        NDArray
            The masqued image
        """
        masqued = image.data.copy()
        if threshold is None:
            threshold = 0.1 * np.max(masqued)
        
        masqued[masqued <= threshold] = 0
        # masqued[masqued != 0] = 1
        image.masqued = masqued
        # display du masque pour le debug
        if display == True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Masque de l'image")
            plt.imshow(image.masqued, cmap='gray')  # type: ignore
            plt.axis('off')  # type: ignore
            plt.show()
        return image.masqued 
    
    @staticmethod
    def filtre_de_sobel(derivative):
        sobel_kernel = np.array([1, 0, -1])
        return {x: np.convolve(derivative[x], sobel_kernel, mode='same') for x in derivative}
    
    
    @staticmethod
    def top_n_local_maxima(l, n):
        maxima = [(i, l[i]) for i in range(1, len(l)-1)
                  if l[i] > l[i-1] and l[i] > l[i+1]]
        maxima.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in maxima[:n]]


    @staticmethod
    def compute_first_derivative(image, axis, positions, filter=True, display=False, mask = False):
        if mask == True:    
            image = image.masqued
        else:
            image = image.data

        if axis == 'x' or axis == 0:
            derivatives = {y: np.diff(image[y, :]) for y in positions}
        elif axis == 'y' or axis == 1:
            derivatives = {x: np.diff(image[:, x]) for x in positions}
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        
        if filter == True:
            derivatives = Computation.filtre_de_sobel(derivatives)
        
        
        if display == True:
            import matplotlib.pyplot as plt
            for pos, deriv in derivatives.items():
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot derivative
                ax1.plot(deriv)
                ax1.set_title(f"First Derivative at position {pos}")
                ax1.set_xlabel("Pixel Index")
                ax1.set_ylabel("First Derivative Value")
                ax1.grid()
                
                # Plot image with line position
                ax2.imshow(image, cmap='gray')
                if axis == 'x' or axis == 0:
                    ax2.axhline(y=pos, color='r', linestyle='-')
                else:
                    ax2.axvline(x=pos, color='r', linestyle='-')
                ax2.set_title("Image with derivative position")
                
                plt.tight_layout()
                plt.show()
        
        return derivatives


    @staticmethod
    def compute_second_derivative(image, axis, positions, filter = False, display = True, mask= False):
        
        if mask == True:
            image = image.masqued
        else: 
            image = image.data
        
        if axis == 'x' or axis == 0:
            second_derivatives = {y: np.diff(np.diff(image[y, :])) for y in positions}
        elif axis == 'y' or axis == 1:
            second_derivatives = {x: np.diff(np.diff(image[:, x])) for x in positions}
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        
        if filter == True:
            second_derivatives = Computation.filtre_de_sobel(
                second_derivatives)
            
        if display == True:
            import matplotlib.pyplot as plt
            for pos, deriv in second_derivatives.items():
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Plot derivative
                ax1.plot(deriv)
                ax1.set_title(f"Second Derivative at position {pos}")
                ax1.set_xlabel("Pixel Index")
                ax1.set_ylabel("Second Derivative Value")
                ax1.grid()

                # Plot image with line position
                ax2.imshow(image, cmap='gray')
                if axis == 'x' or axis == 0:
                    ax2.axhline(y=pos, color='r', linestyle='-')
                else:
                    ax2.axvline(x=pos, color='r', linestyle='-')
                ax2.set_title("Image with derivative position")

                plt.tight_layout()
                plt.show()
        
        return second_derivatives     


    @staticmethod
    def mean_derivative(flat, display=False, mask= False):
        if mask == True:
            flat = flat.masqued
        else: 
            flat = flat.data
        
        derivatives = {x: np.diff(flat[:, x]) for x in range(flat.shape[1])}
        mean_derivatives = np.mean(list(derivatives.values()), axis=0)
        
        if display == True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Mean First Derivative along X-axis")
            plt.plot(mean_derivatives)
            plt.xlabel("Pixel Index")
            plt.ylabel("Mean First Derivative Value")
            plt.grid()
            plt.show()

        return mean_derivatives

    # ---- Interpolations & géométrie ----
    @staticmethod
    def parabolic_interpolation(p1, p2, p3):
        A = np.array([[p1[0]**2, p1[0], 1],
                      [p2[0]**2, p2[0], 1],
                      [p3[0]**2, p3[0], 1]])
        B = np.array([p1[1], p2[1], p3[1]])
        return tuple(np.linalg.solve(A, B))

    @staticmethod
    def line_coefficients(p1, p2):
        """Linear Interpolation"""
        if p1[0] == p2[0]:
            return 0, p1[1]
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0]
        return a, b

    # ---- Equations de parallélogramme ----
    @staticmethod
    def get_parallelogram_equations(parabolas, lines):
        """Useless function to clarify the code, parabolas and lines should be a class ??? """
        left_edge = parabolas[0]
        right_edge = parabolas[1]
        top_edge = lines[0]
        bottom_edge = lines[1]
        return left_edge, right_edge, top_edge, bottom_edge

    @staticmethod
    def find_intersection(parabola, line, near_point, display = False):
        """Compute the intersection between a line and a parabola, discrimine the 2 solutions thanks to a near point

        Parameters
        ----------
        parabola : tuple
            a,b,c coefficients of the parabola
        line : tuple
            0,a,b coefficients of the line
        near_point : tuple
            x,y coordinate of the nearest point to the desired intersection
        display : bool, optional
            If True, display the parabola, the line and the intersection point found, by default False

        Returns
        -------
        Tuple
            The intersection point between the line and the parabola under the (x,y) format

        Raises
        ------
        ValueError
            Triger if the output point doesn't solve the equation of the line and the equation of the parabola
        """
        a_p, b_p, c_p = parabola
        _, a_l, b_l = line

        A = a_p + 1e-6
        B = b_p - a_l
        C = c_p - b_l
        delta = B**2 - 4*A*C
        if delta < 0:
            return None

        x1 = (-B + np.sqrt(delta)) / (2*A)
        x2 = (-B - np.sqrt(delta)) / (2*A)
        y1 = a_p*x1**2 + b_p*x1 + c_p
        y2 = a_p*x2**2 + b_p*x2 + c_p

        d1 = np.hypot(x1-near_point[0], y1-near_point[1])
        d2 = np.hypot(x2-near_point[0], y2-near_point[1])
        
        result = (x1, y1) if d1 < d2 else (x2, y2)
        
        if a_p*result[0]**2 + b_p*result[0] + c_p - (a_l*result[0] + b_l) > 1:
            print(
                f"Distance between intersection and point : {a_p*result[0]**2 + b_p*result[0] + c_p - (a_l*result[0] + b_l)}")
            raise ValueError("Intersection point does not satisfy both equations")
        
        if display == True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Intersection Calculation")
            x_vals = np.linspace(
                min(result[0], near_point[0])-10, max(result[0], near_point[0])+10, 400)
            y_parabola = a_p*x_vals**2 + b_p*x_vals + c_p
            y_line = a_l*x_vals + b_l
            plt.plot(x_vals, y_parabola, label='Parabola', linewidth=0.5)
            plt.plot(x_vals, y_line, label='Line', linewidth=0.5)
            plt.scatter([result[0]], [result[1]], color='red', label='Intersection', s = 2)
            plt.scatter([near_point[0]], [near_point[1]], color='green', label='Near Point', s=2)
            plt.legend()
            plt.grid()
            plt.show()
        
        
        return result

    @staticmethod
    def find_quadrilateral_corners(parabolas, lines, near_points):
        left_edge, right_edge, top_edge, bottom_edge = Computation.get_parallelogram_equations(
            parabolas, lines)
        top_left = Computation.find_intersection(
            left_edge, top_edge, near_points[0])
        top_right = Computation.find_intersection(
            right_edge, top_edge, near_points[1])
        bottom_left = Computation.find_intersection(
            left_edge, bottom_edge, near_points[2])
        bottom_right = Computation.find_intersection(
            right_edge, bottom_edge, near_points[3])
        return [top_left, top_right, bottom_left, bottom_right]
