import numpy as np
import cv2


def extract_parabolic_shape_to_rect(image, paraboles, output_shape=None, corners=None):
    """
    Extrait la région délimitée par 4 paraboles et la remappe dans un rectangle.
    Si corners est fourni, la taille de sortie est automatiquement calculée.
    
    :param image: image source (numpy array)
    :param paraboles: liste de 4 tuples (a, b, c), ordre: [gauche, droite, haut, bas]
    :param output_shape: (h, w) taille du rectangle de sortie (optionnel si corners)
    :param corners: liste de 4 tuples (x, y) des coins (HG, HD, BD, BG) (optionnel)
    :return: image rectifiée (numpy array)
    """
    import numpy as np
    import cv2
    ## OUTSHAPE EST PREFERABLE POUR AVOIR TOUS LES CANAUX SOLAIRE DE MEME TAILLE
    if corners is not None:
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
    elif output_shape is not None:
        h_out, w_out = output_shape
    else:
        raise ValueError("Vous devez fournir output_shape ou corners.")

    img_out = np.zeros((h_out, w_out, *image.shape[2:]), dtype=image.dtype)
    a_g, b_g, c_g = paraboles[0]  # gauche
    a_d, b_d, c_d = paraboles[1]  # droite
    a_h, b_h, c_h = paraboles[2]  # haut
    a_b, b_b, c_b = paraboles[3]  # bas

    for i in range(h_out):
        v = i / (h_out - 1) if h_out > 1 else 0
        for j in range(w_out):
            u = j / (w_out - 1) if w_out > 1 else 0
            if corners is not None:
                # Homographie bilinéaire à partir des coins
                # (u,v) dans [0,1]x[0,1] → (x,y) dans l'image
                x = (1-u)*(1-v)*x0 + u*(1-v)*x1 + u*v*x2 + (1-u)*v*x3
                y = (1-u)*(1-v)*y0 + u*(1-v)*y1 + u*v*y2 + (1-u)*v*y3
            else:
                # Méthode paraboles
                y = v * (image.shape[0] - 1)
                x_g = a_g*y**2 + b_g*y + c_g
                x_d = a_d*y**2 + b_d*y + c_d
                x = x_g + u * (x_d - x_g)
                y_h = a_h*x**2 + b_h*x + c_h
                y_b = a_b*x**2 + b_b*x + c_b
                y = y_h + v * (y_b - y_h)
            # Interpolation bilinéaire
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                img_out[i, j] = cv2.getRectSubPix(
                    image, (1, 1), (float(x), float(y)))
            else:
                img_out[i, j] = 0
    return img_out.squeeze()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Création d'une image de test (gradient)
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (100, 60), (130, 130), 10, -1)

    # Paraboles fictives pour tester (bordure d'un rectangle)
    # gauche/droite : x = cte, haut/bas : y = cte
    paraboles = [
        (0.001, 1/3, 25),   # gauche: x = 50
        (0.001, 1/3, 125),  # droite: x = 250
        (0, 0, 50),   # haut: y = 50
        (0, 0, 150)   # bas: y = 150
    ]

    # Affichage des paraboles sur l'image d'origine
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image d'origine + paraboles")
    plt.imshow(img, cmap='gray')

    # Gauche et droite : x = a*y^2 + b*y + c
    y_vals = np.linspace(0, img.shape[0]-1, 300)
    for idx, (a, b, c) in enumerate(paraboles[:2]):
        x_vals = a*y_vals**2 + b*y_vals + c
        plt.plot(x_vals, y_vals, label=f'{"Gauche" if idx==0 else "Droite"}', color='red' if idx==0 else 'blue')

    # Haut et bas : y = a*x^2 + b*x + c
    x_vals = np.linspace(0, img.shape[1]-1, 300)
    for idx, (a, b, c) in enumerate(paraboles[2:]):
        y_vals_hb = a*x_vals**2 + b*x_vals + c
        plt.plot(x_vals, y_vals_hb, label=f'{"Haut" if idx==0 else "Bas"}', color='green' if idx==0 else 'orange')

    plt.legend()
    plt.axis('image')

    # Rectification
    rectified = extract_parabolic_shape_to_rect(
        img, paraboles, corners = [(44,50),(144,50),(197.5,150),(47.5,150)])

    plt.subplot(1, 2, 2)
    plt.title("Rectifiée")
    plt.imshow(rectified, cmap='gray')
    plt.axis('off')
    plt.show()