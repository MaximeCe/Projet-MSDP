import numpy as np


class Computation:
    # ---- Dérivées & maxima ----
    @staticmethod
    def top_n_local_maxima(l, n):
        maxima = [(i, l[i]) for i in range(1, len(l)-1)
                  if l[i] > l[i-1] and l[i] > l[i+1]]
        maxima.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in maxima[:n]]

    @staticmethod
    def compute_first_derivative_x(image, y_positions):
        image = image.data
        return {y: np.diff(image[y, :]) for y in y_positions}

    @staticmethod
    def compute_first_derivative_y(image, x_positions):
        image = image.data
        return {x: np.diff(image[:, x]) for x in x_positions}


    @staticmethod
    def mean_derivative(flat):
        flat = flat.data
        derivatives = {x: np.diff(flat[:, x]) for x in range(flat.shape[1])}
        return np.mean(list(derivatives.values()), axis=0)

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
        if p1[0] == p2[0]:
            return 0, p1[1]
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0]
        return a, b

    # ---- Equations de parallélogramme ----
    @staticmethod
    def get_parallelogram_equations(parabolas, lines):
        left_edge = parabolas[0]
        right_edge = parabolas[1]
        top_edge = lines[0]
        bottom_edge = lines[1]
        return left_edge, right_edge, top_edge, bottom_edge

    @staticmethod
    def find_intersection(parabola, line, near_point):
        a_p, b_p, c_p = parabola
        a_l, b_l, c_l = line

        A = a_p
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
        return (x1, y1) if d1 < d2 else (x2, y2)

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
