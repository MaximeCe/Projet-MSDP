import numpy as np

EPSILON = 1e-12

def channel_size(corners):
    """Compute the shape of a channel thanks to its corners

    Parameters
    ----------
    corners : list[tuple[int]]
        List of the corners, order : Top-Left; Top-Right; Bottom-Right; Bottom-Left

    Returns
    -------
    Tuple
        Hight and width of the channel
    """
    # Ordre: [haut-gauche, haut-droit, bas-droit, bas-gauche]
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
    # Largeur = moyenne des longueurs haut et bas
    width_top = np.hypot(x1 - x0, y1 - y0)
    width_bottom = np.hypot(x2 - x3, y2 - y3)
    width = int(round((width_top + width_bottom) / 2))
    # Hauteur = moyenne des longueurs gauche et droite
    height_left = np.hypot(x3 - x0, y3 - y0)
    height_right = np.hypot(x2 - x1, y2 - y1)
    height = int(round((height_left + height_right) / 2))
    h_out, w_out = height, width
    output_shape = h_out, w_out 
    print(f"Solar channels shape : {output_shape}")
    return output_shape


"""Résout y = a*x^2+b*x+c → retourne les deux solutions possibles en x"""
def solve_x_from_y(a, b, c, y):
    """Solve y = ax^2+bx+c

    Parameters
    ----------
    a : float
        a coefficient of a parabola
    b : float
        b coefficient of a parabola
    c : float
        c coefficient of a parabola
    y : float
        solve for y

    Returns
    -------
    list[float]
        The 2 solution of this equation
    """
    
    if abs(a) < EPSILON:  # Cas linéaire
        return [(y - c) / b] if abs(b) > EPSILON else []
    disc = b**2 - 4*a*(c - y)
    
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    return [(-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)]


def extract_parabolic_shape_to_rect(image, paraboles: list, output_shape: tuple, display = False):
    """Transform a parabolic shape into a rectangle

    Parameters
    ----------
    image : Image
        The image containing the parabolic shape (Channel)
    paraboles : list
        The 4 parabolas describing the edges of the parabolic shape
    output_shape : tuple
        Shape of the output image
    display : bool, optional
        If True, display the parabolas on the input image and the corresponding output image, by default False

    Returns
    -------
    ndarray
        The rectangle extracted from the image
    """
    import numpy as np
    import cv2

    h_out, w_out = output_shape
    img_out = np.zeros((h_out, w_out, *image.shape[2:]), dtype=image.dtype)
    (a_g, b_g, c_g), (a_d, b_d, c_d), (a_h, b_h, c_h), (a_b, b_b, c_b) = paraboles


    for i in range(h_out):
        v = i / (h_out - 1) if h_out > 1 else 0
        for j in range(w_out):
            u = j / (w_out - 1) if w_out > 1 else 0

            # On part d'une coordonnée y "image"
            y = v * (image.shape[0] - 1)

            # Bords gauche et droit : inverser les paraboles pour obtenir x
            xs_g = solve_x_from_y(a_g, b_g, c_g, y)
            xs_d = solve_x_from_y(a_d, b_d, c_d, y)

            if len(xs_g) == 0 or len(xs_d) == 0:
                print(f"Warning: No solution for y={y} on paraboles.")
                continue

            x_g = min(xs_g) if a_g < 0 else max(xs_g)  # côté gauche
            x_d = min(xs_d) if a_d < 0 else max(xs_d)  # côté droit
            x = x_g + u * (x_d - x_g)

            # Haut et bas : direct (y = f(x))
            y_h = a_h*x**2 + b_h*x + c_h
            y_b = a_b*x**2 + b_b*x + c_b
            y = y_h + v * (y_b - y_h)

            # Interpolation
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                img_out[i, j] = cv2.getRectSubPix(
                    image, (1, 1), (float(x), float(y)))
            else:
                img_out[i, j] = 0
            
        
    
    if display == True:
        # Affiche les paraboles sur l'image d'origine et l'image rectifiée
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.title("Image d'origine avec paraboles")
        plt.imshow(image, cmap='gray')
        x = np.arange(image.shape[1])
        y_g = a_g*x**2 + b_g*x + c_g
        y_d = a_d*x**2 + b_d*x + c_d
        y_h = a_h*x**2 + b_h*x + c_h
        y_b = a_b*x**2 + b_b*x + c_b
        plt.plot(x, y_g, 'r-', label = 'gauche', c='r', linewidth = 0.5)
        plt.plot(x, y_d, 'r-', label='droite', c='b', linewidth=0.5)
        plt.plot(x, y_h, 'r-', label='haut', c='g', linewidth=0.5)
        plt.plot(x, y_b, 'r-', label='bas', c='y', linewidth=0.5)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title("Image rectifiée")
        plt.imshow(img_out.squeeze(), cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return img_out.squeeze()



if __name__ == "__main__":
    pass