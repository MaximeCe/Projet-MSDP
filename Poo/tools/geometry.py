import numpy as np


def parabolic_interpolation(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([
        [x1 ** 2, x1, 1],
        [x2 ** 2, x2, 1],
        [x3 ** 2, x3, 1]
    ])
    b = np.array([y1, y2, y3])
    a, b, c = np.linalg.solve(A, b)
    return a, b, c


def line_coefficients(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        raise ValueError("Les points ont la même abscisse, pente infinie.")
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def compute_first_derivative(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def compute_second_derivative(p1, p2, p3):
    h = p2[0] - p1[0]
    f1 = compute_first_derivative(p1, p2)
    f2 = compute_first_derivative(p2, p3)
    return (f2 - f1) / h
