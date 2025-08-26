from matplotlib.widgets import Slider
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.path import Path


def load_fits(file_path):
    """Charge un fichier FITS si disponible, sinon retourne None."""
    try:
        with fits.open(file_path) as hdul:  # type: ignore
            print(f"✔️ Fichier chargé : {file_path}")
            return hdul[0].data.astype(np.float32)  # type: ignore
    except FileNotFoundError:
        print(f"⚠️ Fichier manquant : {file_path}")
        return None


def preprocess_fits(image_path, master_dark=None, master_flat=None, master_bias=None):
    """
    Applies preprocessing to a FITS image using Master Dark, Flat, and Bias frames.
    This function loads a FITS image and optionally applies corrections using 
    provided master calibration frames (dark, flat, and bias). It also displays 
    statistics (mean, median, and standard deviation) of the image before and 
    after the preprocessing.
    Args:
        image_path (str): Path to the FITS image to be processed.
        master_dark (str, optional): Path to the Master Dark FITS file. Defaults to None.
        master_flat (str, optional): Path to the Master Flat FITS file. Defaults to None.
        master_bias (str, optional): Path to the Master Bias FITS file. Defaults to None.
    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array, or None if the input 
        image could not be loaded.
    Notes:
        - If a Master Flat is provided, pixels with a value of 0 in the flat frame 
          are replaced with 1 to avoid division by zero.
        - The function prints messages indicating the success or failure of loading 
          each calibration frame.
        - The function assumes that the `load_fits` function is defined elsewhere 
          to handle the loading of FITS files.
    """
    """Applique le prétraitement à une image FITS en utilisant Master Dark, Flat et Bias."""
    # Chargement des fichiers
    image = load_fits(image_path)
    if image is None:
        print(
            f"❌ Impossible de traiter l'image : ⚠️ Fichier manquant : {image_path}")
        return

    dark = load_fits(master_dark) if master_dark else None
    if dark is not None:
        print(f"✔️ Master Dark chargé : {master_dark}")
    else:
        print("❌ Master Dark non chargé.")

    flat = load_fits(master_flat) if master_flat else None
    if flat is not None:
        print(f"✔️ Master Flat chargé : {master_flat}")
    else:
        print("❌ Master Flat non chargé.")

    bias = load_fits(master_bias) if master_bias else None
    if bias is not None:
        print(f"✔️ Master Bias chargé : {master_bias}")
    else:
        print("❌ Master Bias non chargé.")

    # Affichage des statistiques avant traitement
    print("\n📊 Statistiques avant traitement :")
    print(
        f"Moyenne : {np.mean(image):.2f}, Médiane : {np.median(image):.2f}, Écart-type : {np.std(image):.2f}")

    # Correction de l'image
    if bias is not None:
        image -= bias
    if dark is not None:
        image -= dark
    if flat is not None:
        flat[flat == 0] = 1  # Éviter la division par zéro
        image /= flat

    # Affichage des statistiques après traitement
    print("\n📊 Statistiques après traitement :")
    print(
        f"Moyenne : {np.mean(image):.2f}, Médiane : {np.median(image):.2f}, Écart-type : {np.std(image):.2f}")

    return image  # Retourne l'image traitée pour un éventuel enregistrement


def top_n_local_maxima(l, n):
    maxima = []

    # On parcourt la liste (hors bords)
    for i in range(1, len(l) - 1):
        if l[i] > l[i - 1] and l[i] > l[i + 1]:
            maxima.append((i, l[i]))

    # Tri par valeur décroissante
    maxima.sort(key=lambda x: x[1], reverse=True)

    # Retourne les n premiers indices seulement
    return [idx for idx, _ in maxima[:n]]


def compute_first_derivative_following_x(image, y_positions):
    """
    Compute the first derivative of specific rows in an image.
    This function calculates the first derivative along the horizontal axis
    for the specified rows in the input image. The results are stored in a
    dictionary where the keys are the row indices and the values are the
    computed derivatives.
    Args:
        image (numpy.ndarray): A 2D array representing the image.
        y_positions (list of int): A list of row indices for which the first
            derivative will be computed.
    Returns:
        dict: A dictionary where each key is a row index from `y_positions`,
        and the corresponding value is a 1D numpy array containing the first
        derivative of that row.
    Example:
        >>> import numpy as np
        >>> image = np.array([[1, 2, 4], [3, 6, 9], [2, 4, 6]])
        >>> y_positions = [0, 2]
        >>> compute_first_derivative(image, y_positions)
        Calcul des dérivées...
        Position des dérivées : [0, 2]
        Dérivées calculées.
        {0: array([1, 2]), 2: array([2, 2])}
    """
    """Calcule la dérivée seconde sur les lignes spécifiées."""
    print("Calcul des dérivées...")
    print("Position des dérivées :", y_positions)

    derivatives = {}
    for y in y_positions:
        row_values = image[y, :]  # Extraire la ligne horizontale à y
        first_derivative = np.diff(row_values)  # Dérivée première
        derivatives[y] = first_derivative
    print("Dérivées calculées.\n")

    return derivatives


def compute_first_derivative_following_y(image, x_positions):
    """
    Compute the first derivative along the y-axis for specified x positions in an image.
    This function calculates the first derivative of pixel intensity values along 
    the vertical axis (y-axis) for the specified x positions in the given image. 
    The derivatives are computed using the numpy `diff` function.
    Args:
        image (numpy.ndarray): A 2D array representing the image, where each element 
            corresponds to a pixel intensity value.
        x_positions (list of int): A list of x-coordinates (column indices) for which 
            the first derivative along the y-axis will be computed.
    Returns:
        dict: A dictionary where the keys are the x-coordinates from `x_positions`, 
            and the values are 1D numpy arrays containing the first derivative values 
            along the y-axis for the corresponding x-coordinate.
    Example:
        >>> import numpy as np
        >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> x_positions = [0, 2]
        >>> compute_first_derivative_y(image, x_positions)
        Calcul des dérivées...
        Position des dérivées : [0, 2]
        Dérivées calculées.
        {0: array([3, 3]), 2: array([3, 3])}
    """
    """Calcule la dérivée seconde sur les lignes spécifiées."""

    print("Calcul des dérivées...")
    print("Position des dérivées :", x_positions)

    derivatives = {}
    for x in x_positions:
        row_values = image[:, x]  # Extraire la ligne horizontale à x
        first_derivative = np.diff(row_values)  # Dérivée première
        derivatives[x] = first_derivative

    print("Dérivées calculées.\n")

    return derivatives


def mean_derivative(flat):
    # Initialisation des positions des lignes d'analyse
    # calcul de la dérivé moyenne dans la direction y
    print("Calcul des dérivées...")
    print("Position des dérivées : toutes")
    xmax = flat.shape[1]
    derivatives = {}
    for x in range(xmax):
        row_values = flat[:, x]  # Extraire la ligne horizontale à y
        first_derivative = np.diff(row_values)  # Dérivée première
        derivatives[x] = first_derivative

    print("Dérivées calculées.\n")

    # calcul de la moyenne des dérivées
    mean_derivatives = np.mean(list(derivatives.values()), axis=0)
    return mean_derivatives


def top_and_bottom_detection(mean_derivatives):
    # Détection du premier et dernier pic dans la moyenne des dérivées
    threshold = 0.1 * np.max(np.abs(mean_derivatives))
    first_peak = None
    for a in range(len(mean_derivatives)):
        if np.abs(mean_derivatives[a]) > threshold:
            first_peak = a
            break

    last_peak = None
    for b in range(len(mean_derivatives) - 1, -1, -1):
        if np.abs(mean_derivatives[b]) > threshold:
            last_peak = b
            break

    if first_peak is None or last_peak is None:
        raise ValueError(
            "Impossible de trouver les pics dans la dérivée moyenne.")
    return first_peak, last_peak


def point_detection_following_x(derivatives, y_positions):
    line1, line2, line3 = [], [], []

    for idx, (y, derivative) in enumerate(derivatives.items()):
        print("----------------------------------------------------------")
        print(f"Analyse de la ligne y={y}")
        print("Application du seuil")

        edges_x = top_n_local_maxima(np.abs(derivative), 18)
        filtered_points = [(xi, y) for xi in edges_x]

        if y == y_positions[0]:
            line1 = sorted(filtered_points, key=lambda point: point[0])
        elif y == y_positions[1]:
            line2 = sorted(filtered_points, key=lambda point: point[0])
        elif y == y_positions[2]:
            line3 = sorted(filtered_points, key=lambda point: point[0])

    return line1, line2, line3


def detect_edges_following_x(flat):
    """
    Processes a flat image, calculates the second derivative, and detects significant points.
    Parameters:
    -----------
    flat_path : str
        Path to the flat image file.
    dark_path : str
        Path to the dark image file.
    Returns:
    --------
    tuple of lists
        A tuple containing three lists (line1, line2, line3), each representing the detected points 
        for the corresponding analysis line. Each list contains up to 18 significant points.
    Notes:
    ------
    - The function preprocesses the flat image using the provided dark image.
    - It allows the user to interactively select three analysis lines using sliders.
    - For each selected line, the second derivative is computed, and significant points are detected 
      based on a threshold.
    - The detected points are filtered to retain the 18 most significant local maxima for each line.
    - The function displays plots for visualization of the flat image, sliders, and second derivatives.
    Sub-functions:
    --------------
    - update(val): Updates the positions of the analysis lines based on slider values.
    - update_plot(): Updates the plot to reflect the current positions of the analysis lines.
    """

    print("Début de la détection des bords...")
    flat = flat.data
    if flat is None:
        return [], [], []

    # calcul de la dérivé moyenne dans la direction y
    mean_derivatives = mean_derivative(flat)

    # Détection du premier et dernier pic dans la moyenne des dérivées
    first_peak, last_peak = top_and_bottom_detection(mean_derivatives)

    # Initialisation des positions des lignes d'analyse
    y_positions = [first_peak + 40, first_peak +
                   (last_peak - first_peak)//2, last_peak - 40]
    print("Lignes d'analyse retenues :", y_positions)

    # Calcul des dérivées après avoir choisi les lignes avec les sliders
    derivatives = compute_first_derivative_following_x(flat, y_positions)

    # Détection des points significatifs pour chaque ligne
    line1, line2, line3 = point_detection_following_x(derivatives, y_positions)

    # Retourne les points détectés pour chaque ligne len = 18
    return [line1, line2, line3]


def x_positions_computation(be_list):
    # trier par la première valeur du tuple
    be_list = sorted(be_list, key=lambda x: x[0])

    # Initialisation des positions des lignes d'analyse à partir de be_list
    # On suppose que be_list contient les indices de début et de fin des zones d'intérêt (alternativement)
    b_indices = [i for i in range(len(be_list)) if i % 2 == 0]
    e_indices = [i for i in range(len(be_list)) if i % 2 != 0]

    # Pour chaque paire (début, fin), on place deux lignes d'analyse à 1/3 et 2/3 de la distance
    x_positions = []
    for b_idx, e_idx in zip(b_indices, e_indices):
        start = be_list[b_idx]
        end = be_list[e_idx]
        x_positions.append(int(start[0] + (end[0] - start[0]) / 3))
        x_positions.append(int(start[0] + 2 * (end[0] - start[0]) / 3))
    x_positions = sorted(x_positions)

    print("Lignes d'analyse retenues :", x_positions)
    
    
    return x_positions


def point_detection_following_y(derivatives, x_positions):
    detected_points = []

    for idx, x in enumerate(x_positions):
        print("----------------------------------------------------------")
        print(f"Analyse de la ligne x={x}")
        print("Application du seuil")

        # Détection des pics locaux
        y = top_n_local_maxima(np.abs(derivatives[x]), 2)
        
        # ranger les points selon x puis y
        column = sorted([(x, yi)
                        for yi in y], key=lambda point: (point[0], point[1]))

        detected_points.append(column)

        print("Points détectés :", y)
        print("Nombre de points détectés :", len(y))
        print("fin de l'analyse de la ligne x=", x, '\n')
    return detected_points


def detect_edges_following_y(flat, be_list: list[tuple]):
    """
    Detects significant edge points along the y-axis after correcting the flat image.
    This function processes a flat image using a dark image for correction, calculates
    the second derivative along the y-axis, and detects significant edge points. It
    allows the user to interactively adjust analysis lines using sliders and returns
    the detected points for each line.
    Args:
        flat_path (str): Path to the flat image file.
        dark_path (str): Path to the dark image file.
    Returns:
        list[list[tuple[int, int]]]: A list of lists containing detected points for each
        analysis line. Each inner list contains up to two tuples representing the
        (x, y) coordinates of the detected points.
    Notes:
        - The function uses sliders to allow the user to adjust the positions of the
          analysis lines interactively.
        - The second derivative along the y-axis is computed for each analysis line,
          and significant points are detected based on a threshold.
        - Only the two most significant local maxima are retained for each line.
    Example:
        detected_points = detect_edges_y("path/to/flat_image.tif", "path/to/dark_image.tif")
    """
    """Corrige le flat, calcule la dérivée seconde sur y et détecte les points significatifs."""
    print("Début de la détection des bords...")
    n = 18
    flat = flat.data
    if flat is None:
        return [[] for _ in range(n)]

    # Initialisation des positions des lignes d'analyses
    x_positions = x_positions_computation(be_list)

    # Calcul des dérivées après avoir choisi les lignes avec les sliders
    derivatives = compute_first_derivative_following_y(flat, x_positions)

    # Détection des points significatifs pour chaque liqne
    detected_points = point_detection_following_y(derivatives, x_positions)

    # Retourne les points détectés pour chaque ligne shape = (18*2)
    print("Points détectés sur y :", detected_points)
    return detected_points


def parabolic_interpolation(point1, point2, point3):
    """
    Calcule les coefficients de l'équation d'une parabole passant par trois points donnés.
    Cette fonction utilise l'interpolation parabolique pour déterminer les coefficients
    a, b et c de l'équation de la forme f(x) = ax^2 + bx + c, où la parabole passe par
    les trois points spécifiés.
    Args:
        point1 (tuple): Un tuple (x1, y1) représentant les coordonnées du premier point.
        point2 (tuple): Un tuple (x2, y2) représentant les coordonnées du deuxième point.
        point3 (tuple): Un tuple (x3, y3) représentant les coordonnées du troisième point.
    Returns:
        tuple: Un tuple (a, b, c) contenant les coefficients de l'équation de la parabole.
    Raises:
        ValueError: Si l'un des points n'est pas un tuple de la forme (x, y).
    Exemple:
        >>> parabolic_interpolation((1, 2), (2, 3), (3, 5))
        (a, b, c)  # Coefficients de la parabole
    Note:
        Cette fonction suppose que les trois points fournis sont distincts et que
        leurs coordonnées x ne sont pas identiques, afin d'assurer une solution unique.
    """
    """Retourne les coefficients de l'équation d'une parabole passant par trois points."""
    if len(point1) != 2 or len(point2) != 2 or len(point3) != 2:
        raise ValueError(
            "Les points doivent être des tuples de la forme (x, y).")

    print("----------------------------------------------------------")
    print("Début de l'interpolation parabolique...")
    print("Points :", point1, point2, point3)
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Résoudre le système linéaire pour trouver les coefficients a, b et c
    A = np.array([
        [x1**2, x1, 1],
        [x2**2, x2, 1],
        [x3**2, x3, 1]
    ])
    B = np.array([y1, y2, y3])

    a, b, c = np.linalg.solve(A, B)
    print(
        "Équation de la parabole : f(x) = {:.2f}x^2 + {:.2f}x + {:.2f}".format(a, b, c))
    print("Interpolation parabolique terminée avec succès.\n")

    return (a, b, c)  # Retourne les coefficients (a, b, c) de la parabole


# def parabolic_interpolations(points):
    """Retourne les équations de paraboles pour 18 paires de points."""

    if len(points) != 3 or any(len(p) != 18 for p in points):
        raise ValueError(
            "La liste des points doit contenir exactement 3 sous-listes de 18 éléments chacune.")

    points1, points2, points3 = points

    equations = []
    for p1, p2, p3 in zip(points1, points2, points3):
        a, b, c = parabolic_interpolation(p1, p2, p3)
        equations.append((a, b, c))

    return equations


def line_coefficients(point1, point2):
    """
    Retourne les coefficients a et b de la droite passant par deux points.

    Args:
        point1 (tuple): Coordonnées du premier point (x1, y1).
        point2 (tuple): Coordonnées du deuxième point (x2, y2).

    Returns:
        tuple: Coefficients (a, b) de la droite y = ax + b.
    """
    print("----------------------------------------------------------")
    print("Début du calcul des coeficients de la droite...")
    print("Points :", point1, point2)
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError(
            "Les points doivent être des tuples de la forme (x, y).")

    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        a = 0
    else:
        # Calcul des coefficients a et b de la droite y = ax + b
        a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    print(
        "Équation de la droite : f(x) = {:.2f}x + {:.2f}".format(a, b))
    print("Droite déterminée avec succès.\n")

    return a, b


def get_parallelogram_equations(parabolas, lines) -> tuple[tuple, tuple, tuple, tuple]:
    """
    Retourne les équations des bords d'un parallélogramme pour un quadrilatère spécifié.

    Args:
        parabolas (list): Liste des coefficients des paraboles [(a1, b1, c1), (a2, b2, c2)].
        lines (list): Liste des coefficients des droites [(a1, b1), (a2, b2)].
        quadrilateral_index (int): Indice du quadrilatère voulu (de 1 à 9).

    Returns:
        tuple: 4 tuples correspondant aux équations des bords du parallélogramme.
    """
    print("----------------------------------------------------------")
    print(
        f"Début de la récupération des équations...")
    # Calculer l'indice de base pour le quadrilatère spécifié

    # Récupérer les équations des bords du parallélogramme
    left_edge = parabolas[0]  # Parabole gauche (indice pair)
    right_edge = parabolas[1]  # Parabole droite (indice impair)
    top_edge = lines[0]  # Droite haute (indice pair)
    bottom_edge = lines[1]  # Droite basse (indice impair)

    print("Équations des bords récupérées avec succès.")
    print("Parabole gauche :", left_edge)
    print("Parabole droite :", right_edge)
    print("Droite haute :", top_edge)
    print("Droite basse :", bottom_edge, '\n')

    return left_edge, right_edge, top_edge, bottom_edge


def find_intersection(parabola: tuple, line: tuple, near_point, point_name, quadrilateral_index):
    """
    Trouve l'intersection entre une parabole et une droite.

    Args:
        parabola (tuple): Coefficients de la parabole (a, b, c).
        line (tuple): Coefficients de la droite (a, b).
        near_point (tuple): Point proche pour déterminer l'intersection la plus pertinente.
        point_name (str): Nom du point (ex: "top_left").
        quadrilateral_index (int): Indice du quadrilatère en cours de traitement.

    Returns:
        tuple: Coordonnées (x, y) du point d'intersection.
    """

    print(f"----------------------------------------------------------")
    print(
        f"Début de la recherche de l'intersection pour le point '{point_name}' du quadrilatère {quadrilateral_index}...")

    a_p, b_p, c_p = parabola
    a_l, b_l ,c_l = line

    # Résoudre l'équation quadratique a_p*x^2 + (b_p - a_l)*x + (c_p - b_l) = 0
    A = a_p
    B = b_p - a_l
    C = c_p - b_l

    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        print(
            f"Pas d'intersection réelle pour le point '{point_name}' du quadrilatère {quadrilateral_index}. Delta < 0.")
        return None  # Pas d'intersection réelle

    x1 = (-B + np.sqrt(discriminant)) / (2*A)
    x2 = (-B - np.sqrt(discriminant)) / (2*A)

    # Calculer les coordonnées y correspondantes
    y1 = a_p*x1**2 + b_p*x1 + c_p
    y2 = a_p*x2**2 + b_p*x2 + c_p

    # Trouver le point le plus proche de near_point
    dist1 = np.sqrt((x1 - near_point[0])**2 + (y1 - near_point[1])**2)
    dist2 = np.sqrt((x2 - near_point[0])**2 + (y2 - near_point[1])**2)

    if dist1 < dist2:
        print(
            f"Point '{point_name}' retenu pour le quadrilatère {quadrilateral_index} : ({x1:.2f}, {y1:.2f})")
        return (x1, y1)
    else:
        print(
            f"Point '{point_name}' retenu pour le quadrilatère {quadrilateral_index} : ({x2:.2f}, {y2:.2f})")
        return (x2, y2)


def find_quadrilateral_corners(parabolas, lines, quadrilateral_index, near_points):
    """
    Trouve les 4 coins d'un quadrilatère spécifié par son indice.

    Args:
        parabolas (list): Liste des coefficients des paraboles [(a1, b1, c1), (a2, b2, c2)].
        lines (list): Liste des coefficients des droites [(a1, b1), (a2, b2, c2)].
        quadrilateral_index (int): Indice du quadrilatère voulu (de 1 à 9).
        near_points (list): Liste des points proches pour guider les intersections.

    Returns:
        list: Liste des coordonnées des 4 coins du quadrilatère [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """

    print(f"----------------------------------------------------------")
    print(
        f"Début de la détermination des coins pour le quadrilatère {quadrilateral_index}...")

    left_edge, right_edge, top_edge, bottom_edge = get_parallelogram_equations(
        parabolas, lines)

    top_left = find_intersection(
        left_edge, top_edge, near_points[0], "top_left", quadrilateral_index)
    top_right = find_intersection(
        right_edge, top_edge, near_points[1], "top_right", quadrilateral_index)
    bottom_left = find_intersection(
        left_edge, bottom_edge, near_points[2], "bottom_left", quadrilateral_index)
    bottom_right = find_intersection(
        right_edge, bottom_edge, near_points[3], "bottom_right", quadrilateral_index)

    print(f"Coins déterminés pour le quadrilatère {quadrilateral_index} :")
    print(f"  - top_left : {top_left}")
    print(f"  - top_right : {top_right}")
    print(f"  - bottom_left : {bottom_left}")
    print(f"  - bottom_right : {bottom_right}")

    return [top_left, top_right, bottom_left, bottom_right]


def crop(image, corners, quadrilateral_index):
    """
    Recadre une image en utilisant les 4 coins d'un quadrilatère et met à 0 les points en dehors du quadrilatère.

    Args:
        image (np.ndarray): Image sous forme de matrice NumPy.
        corners (list): Liste des coordonnées des 4 coins du quadrilatère [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

    Returns:
        np.ndarray: Matrice contenant uniquement la région délimitée par les 4 coins, avec les points en dehors mis à 0.
    """
    print("----------------------------------------------------------")
    print("Début du recadrage de l'image...")
    print("Coins reçus :", corners)

    if len(corners) != 4:
        raise ValueError(
            "La liste des coins doit contenir exactement 4 éléments.")

    # Réorganiser les coins dans le sens horaire pour garantir un quadrilatère correct
    corners = np.array(corners)
    center = np.mean(corners, axis=0)
    print("Centre du quadrilatère :", center)

    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    print("Coins triés dans le sens horaire :", sorted_corners)

    # Créer un masque pour le quadrilatère
    mask = np.zeros_like(image, dtype=bool)
    rr, cc = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    rr, cc = rr.flatten(), cc.flatten()

    # Utiliser une méthode pour vérifier si les points sont dans le quadrilatère
    polygon = Path(sorted_corners)
    points = np.vstack((rr, cc)).T
    inside = np.array(polygon.contains_points(points)).reshape(image.shape)
    print("Masque créé avec succès.")

    # Appliquer le masque
    cropped_image = np.zeros_like(image)
    cropped_image[inside] = image[inside]
    print("Recadrage terminé.\n")
    plt.imshow(cropped_image, cmap='gray')
    plt.title(f"Quadrilatère {quadrilateral_index}")
    plt.show()
    return cropped_image


def display_detected_points(flat, detected_points, detected_points_h):
    plt.figure(figsize=(10, 10))
    plt.imshow(flat, cmap='gray', extent=[0, flat.shape[1], flat.shape[0], 0])

    for points in [detected_points[0], detected_points[1], detected_points[2]]:
        for x, y in points:
            plt.plot(x, y, 'ro')

    for column_points in detected_points_h:
        for x, y in column_points:
            plt.plot(x, y, 'bo')

    plt.title("Points détectés par detect_edges et detect_edges_y")
    plt.xlim(0, flat.shape[1])
    plt.ylim(flat.shape[0], 0)
    # plt.show()


def display_parabolas_and_lines(flat, channel):
    edges = [edge for edge in channel.edges]
    parabolas = [edge.coefficients()
                        for edge in edges if "parabole" in edge.type]
    lines = [edge.coefficients()
                    for edge in edges if "parabole" not in edge.type]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(flat.data, cmap='gray')

    left_edge, right_edge, top_edge, bottom_edge = get_parallelogram_equations(
        parabolas, lines)

    x = np.arange(0, flat.shape[1])
    y_left = left_edge[0] * x ** 2 + left_edge[1] * x + left_edge[2]
    y_right = right_edge[0] * x ** 2 + right_edge[1] * x + right_edge[2]
    plt.plot(x, y_left, 'r', label="Parabole gauche")
    plt.plot(x, y_right, 'b', label="Parabole droite")

    y_top = top_edge[0] * x + top_edge[1]
    y_bottom = bottom_edge[0] * x + bottom_edge[1]
    plt.plot(x, y_top, 'g', label="Ligne haute")
    plt.plot(x, y_bottom, 'm', label="Ligne basse")

    plt.xlim(0, flat.shape[1])
    plt.ylim(flat.shape[0], 0)
    plt.title(f"Quadrilatère {channel.id}")
    plt.legend()
    plt.show()


def display_quadrilateral_corners(flat, corners, quadrilateral_index):
    plt.figure(figsize=(10, 10))
    plt.imshow(flat, cmap='gray', extent=[0, flat.shape[1], flat.shape[0], 0])
    for x, y in corners:
        plt.plot(x, y, 'ro')
    plt.title(f"Coins du quadrilatère {quadrilateral_index}")
    plt.xlim(0, flat.shape[1])
    plt.ylim(flat.shape[0], 0)
    # plt.show()


def display_start():
    print("__________________________________________________________")
    print("Exécution du programme de traitement d'images multicannaux")
    print("----------------------------------------------------------")
    print("Programme réalisé par :")
    print("CELLIER Maxime, POINTIER Paul, SEIXAS Mathilde, SAINT-PAUL Alexandre, VEILLAUD Baptiste, CANDUN Sam, LEVY Mathis, DA GRACA BAPTISTA Chloé, DUMUR Victor, FERNANDES DE SOUSA Valentin, LEKIC Anica de L'IPSA")
    print("En collaboration avec :")
    print("MEIN Pierre, SAYEDE Frédéric de l'Observatoire de Paris")
    print("----------------------------------------------------------\n")


def statistiques(cs, fs, bs, es, as_, ds, ls, ns, ks, ms, Cs, Fs, Ds, As):
    # Création des derniers points pour les calculs
    Bs = bs
    Es = es

    # Calcul des vecteurs DE, AB, DF, AC, CF, BE, AD
    DE = [np.array(d) - np.array(e) for d, e in zip(Ds, Es)]
    AB = [np.array(a) - np.array(b) for a, b in zip(As, Bs)]
    DF = [np.array(f) - np.array(d) for f, d in zip(Fs, Ds)]
    AC = [np.array(c) - np.array(a) for c, a in zip(Cs, As)]
    CF = [np.array(f) - np.array(c) for f, c in zip(Fs, Cs)]
    BE = [np.array(b) - np.array(e) for b, e in zip(Bs, Es)]
    AD = [np.array(a) - np.array(d) for a, d in zip(As, Ds)]

    # Vérification que tous les vecteurs sont des listes de tableaux NumPy
    vectors = [DE, AB, DF, AC, CF, BE, AD]
    labels = ["DE", "AB", "DF", "AC", "CF", "BE", "AD"]

    for i, vector in enumerate(vectors):
        if not all(isinstance(v, np.ndarray) and v.shape == (2,) for v in vector):
            print(f"Erreur dans le vecteur {labels[i]} : {vector}")
            # Remplacement par des vecteurs nuls
            vectors[i] = [np.array([0, 0])] * len(vector)

    # Plot des x, des y et du module de chaque vecteur dans un subplot
    fig, axs = plt.subplots(7, 3, figsize=(15, 15))
    fig.suptitle("Statistiques des vecteurs")
    i = 0
    axs[i, 0].set_title("Composante X")
    axs[i, 1].set_title("Composante Y")
    axs[i, 2].set_title("Module")

    for i, (vector, label) in enumerate(zip(vectors, labels)):
        # Composante X
        x_values = [v[0] for v in vector]
        min_x = min(x_values)
        normalized_x = [x - min_x for x in x_values]
        axs[i, 0].plot(normalized_x, marker='o', label=f"{label}")
        # Courbe de tendance pour X
        trend_x = np.polyfit(range(len(normalized_x)), normalized_x, 2)
        axs[i, 0].plot(range(len(normalized_x)), np.polyval(
            trend_x, range(len(normalized_x))), linestyle='--')

        # Composante Y
        y_values = [v[1] for v in vector]
        min_y = min(y_values)
        normalized_y = [y - min_y for y in y_values]
        axs[i, 1].plot(normalized_y, marker='o', label=f"{label}")
        # Courbe de tendance pour Y
        trend_y = np.polyfit(range(len(normalized_y)), normalized_y, 2)
        axs[i, 1].plot(range(len(normalized_y)), np.polyval(
            trend_y, range(len(normalized_y))), linestyle='--')

        # Module
        module = [np.sqrt(v[0]**2 + v[1]**2) for v in vector]
        min_module = min(module)
        normalized_module = [m - min_module for m in module]
        axs[i, 2].plot(normalized_module, marker='o', label=f"{label}")
        # Courbe de tendance pour le module
        trend_module = np.polyfit(
            range(len(normalized_module)), normalized_module, 2)
        axs[i, 2].plot(range(len(normalized_module)), np.polyval(
            trend_module, range(len(normalized_module))), linestyle='--')

        axs[i, 0].legend()
        axs[i, 1].legend()
        axs[i, 2].legend()

        # Fix y-axis limits for comparison
        axs[i, 0].set_ylim(0, 3)
        axs[i, 1].set_ylim(0, 3)
        axs[i, 2].set_ylim(0, 3)

    axs[i, 0].set_xlabel("Index")
    axs[i, 1].set_xlabel("Index")
    axs[i, 2].set_xlabel("Index")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.show()


def display_end():
    print("\n----------------------------------------------------------")
    print("Programme exécuté avec succès")
    print("Fin du programme")
    print("__________________________________________________________")




if __name__ == "__main__":
    # main()
    pass
