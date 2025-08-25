from matplotlib.widgets import Slider
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.path import Path


def load_fits(file_path):
    """Charge un fichier FITS si disponible, sinon retourne None."""
    try:
        with fits.open(file_path) as hdul:
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


def preprocess_flat(flat_path, dark_path):
    """
    Preprocesses a flat field image by subtracting a dark frame.
    This function loads a flat field image and a dark frame from the specified file paths.
    It then subtracts the dark frame from the flat field image to correct it. The function
    also prints statistical information (mean, median, and standard deviation) about the 
    flat field image before and after the correction.
    Args:
        flat_path (str): The file path to the flat field image in FITS format.
        dark_path (str): The file path to the dark frame in FITS format.
    Returns:
        numpy.ndarray or None: The corrected flat field image as a NumPy array if successful,
        or None if the flat field image or dark frame could not be loaded.
    """
    """Charge et corrige le flat en soustrayant le dark."""
    flat = load_fits(flat_path)
    dark = load_fits(dark_path)

    print("\n🔍 Prétraitement du flat en cours...")

    if flat is None:
        print("❌ Impossible de traiter le flat : fichier introuvable.")
        return None

    if dark is not None:
        # Affichage des statistiques avant traitement
        print("\n📊 Statistiques avant traitement :")
        print(
            f"Moyenne : {np.mean(flat):.2f}, Médiane : {np.median(flat):.2f}, Écart-type : {np.std(flat):.2f}")
        flat -= dark  # Correction du flat avec le dark
        # Affichage des statistiques après traitement
        print("\n📊 Statistiques après traitement :")
        print(
            f"Moyenne : {np.mean(flat):.2f}, Médiane : {np.median(flat):.2f}, Écart-type : {np.std(flat):.2f}")
    else:
        print("❌ Impossible de traiter le flat : dark manquant.")

    print("✔️ Prétraitement du flat terminé.\n")

    return flat


def compute_first_derivative(image, y_positions):
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


def compute_second_derivative(image, y_positions):
    """Calcule la dérivée seconde sur les lignes spécifiées."""
    derivatives = {}
    for y in y_positions:
        row_values = image[y, :]  # Extraire la ligne horizontale à y
        first_derivative = np.diff(row_values)  # Dérivée première
        second_derivative = np.diff(first_derivative)  # Dérivée seconde
        derivatives[y] = second_derivative

    return derivatives


def detect_edges(flat_path, dark_path):
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
    flat = preprocess_flat(flat_path, dark_path)
    if flat is None:
        return [], [], []

    ymax = flat.shape[0]
    n = 3  # Nombre de lignes d'analyse

    # Initialisation des positions des lignes d'analyse
    y_positions = [(i * ymax) // (n+1) for i in range(1, n+1)]

    # Fonction pour mettre à jour les lignes d'analyse
    def update(val):
        y_positions[0] = int(slider1.val)
        y_positions[1] = int(slider2.val)
        y_positions[2] = int(slider3.val)
        update_plot()

    # Fonction pour mettre à jour le graphique
    def update_plot():
        ax.clear()
        ax.imshow(flat, cmap='gray')
        for y in y_positions:
            ax.axhline(y, color='r')
        fig.canvas.draw_idle()

    # Création de la figure et des sliders
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, right=0.75, bottom=0.1, top=0.9)
    ax.imshow(flat, cmap='gray')
    for y in y_positions:
        ax.axhline(y, color='r')

    # Positionner les sliders sur le côté droit du graphique
    ax_slider1 = plt.axes((0.8, 0.7, 0.15, 0.03),
                          facecolor='lightgoldenrodyellow')
    ax_slider2 = plt.axes((0.8, 0.6, 0.15, 0.03),
                          facecolor='lightgoldenrodyellow')
    ax_slider3 = plt.axes((0.8, 0.5, 0.15, 0.03),
                          facecolor='lightgoldenrodyellow')

    slider1 = Slider(ax_slider1, 'Ligne 1', 0, ymax-1,
                     valinit=y_positions[0], valstep=1)
    slider2 = Slider(ax_slider2, 'Ligne 2', 0, ymax-1,
                     valinit=y_positions[1], valstep=1)
    slider3 = Slider(ax_slider3, 'Ligne 3', 0, ymax-1,
                     valinit=y_positions[2], valstep=1)

    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)

    plt.show()

    print("Lignes d'analyse retenues :", y_positions)

    # Calcul des dérivées après avoir choisi les lignes avec les sliders
    derivatives = compute_first_derivative(flat, y_positions)
    seuil = 0.1

    line1, line2, line3 = [], [], []

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()

    for idx, (y, second_derivative) in enumerate(derivatives.items()):
        print("----------------------------------------------------------")
        print(f"Analyse de la ligne y={y}")
        print("Application du seuil")
        tho = seuil * np.max(np.abs(second_derivative))  # Seuil à 10% du max
        x_coords = np.where(np.abs(second_derivative) > tho)[
            0]  # Détection des pics

        print("Filtrage des points")
        # Filtrer les points pour ne garder que les maximums locaux
        filtered_points = []
        if len(x_coords) > 0:
            max_point = x_coords[0]
            max_value = np.abs(second_derivative[max_point])
            for x in x_coords[1:]:
                if x <= max_point + 3:  # Vérifie si le point est à moins de 3 index
                    if np.abs(second_derivative[x]) > max_value:
                        max_point = x
                        max_value = np.abs(second_derivative[x])
                else:
                    filtered_points.append((max_point, y))
                    max_point = x
                    max_value = np.abs(second_derivative[x])
            filtered_points.append((max_point, y))

        print("Détection des 18 maximums locaux")
        # Garder les 18 plus grands maximums locaux sans modifier l'ordre des éléments de la liste
        filtered_points = sorted(filtered_points, key=lambda p: np.abs(
            second_derivative[p[0]]), reverse=True)[:18]
        # Trier par position x pour maintenir l'ordre original
        filtered_points.sort(key=lambda p: p[0])

        if y == y_positions[0]:
            line1 = filtered_points
        elif y == y_positions[1]:
            line2 = filtered_points
        elif y == y_positions[2]:
            line3 = filtered_points

        # Affichage des dérivées et du seuil
        axs[idx].plot(second_derivative, label=f"Dérivée (y={y})")
        axs[idx].axhline(tho, color='r', linestyle='--', label="Seuil +")
        axs[idx].axhline(-tho, color='r', linestyle='--', label="Seuil -")
        axs[idx].set_title(f"Dérivée seconde pour y={y}")
        axs[idx].legend()
        axs[idx].set_xlabel("x")
        axs[idx].set_ylabel("Dérivée")

        print("Points détectés :", filtered_points)
        print("Nombre de points détectés :", len(filtered_points))
        print("fin de l'analyse de la ligne y=", y, '\n')
    plt.tight_layout()
    plt.show()

    return line1, line2, line3  # Retourne les points détectés pour chaque ligne len = 18


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

    return a, b, c  # Retourne les coefficients (a, b, c) de la parabole


def parabolic_interpolations(points1, points2, points3):
    """Retourne les équations de paraboles pour 18 paires de points."""

    if len(points1) != 18 or len(points2) != 18:
        raise ValueError(
            "Les listes de points doivent contenir exactement 18 éléments chacune.")

    equations = []
    for p1, p2, p3 in zip(points1, points2, points3):
        a, b, c = parabolic_interpolation(p1, p2, p3)
        equations.append((a, b, c))

    return equations


def compute_first_derivative_y(image, x_positions):
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


def compute_second_derivative_y(image, x_positions):
    """Calcule la dérivée seconde sur les lignes spécifiées."""
    derivatives = {}
    for x in x_positions:
        row_values = image[:, x]  # Extraire la ligne horizontale à x
        first_derivative = np.diff(row_values)  # Dérivée première
        second_derivative = np.diff(first_derivative)  # Dérivée seconde
        derivatives[x] = second_derivative

    return derivatives


def detect_edges_y(flat_path, dark_path):
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
    flat = preprocess_flat(flat_path, dark_path)
    if flat is None:
        return [[] for _ in range(18)]

    xmax = flat.shape[1]
    n = 18  # Nombre de lignes d'analyse

    # Initialisation des positions des lignes d'analyse
    x_positions = [(i * xmax) // (n+1) for i in range(1, n+1)]

    # Fonction pour mettre à jour les lignes d'analyse
    def update(val):
        for i in range(n):
            x_positions[i] = int(sliders[i].val)
        update_plot()

    # Fonction pour mettre à jour le graphique
    def update_plot():
        ax.clear()
        ax.imshow(flat, cmap='gray')
        for x in x_positions:
            ax.axvline(x, color='r')
        fig.canvas.draw_idle()

    # Création de la figure et des sliders
    fig, ax = plt.subplots()
    # Ajuster pour laisser plus de place aux sliders
    plt.subplots_adjust(left=0.25, right=0.75, bottom=0.2, top=0.95)
    ax.imshow(flat, cmap='gray')
    for x in x_positions:
        ax.axvline(x, color='r')

    sliders = []
    for i in range(n):
        # Positionner les sliders plus haut
        ax_slider = plt.axes((0.8, 0.9 - i*0.05, 0.15, 0.03),
                             facecolor='lightgoldenrodyellow')
        slider = Slider(
            ax_slider, f'Ligne {i+1}', 0, xmax-1, valinit=x_positions[i], valstep=1)
        slider.on_changed(update)
        sliders.append(slider)

    plt.show()

    print("Lignes d'analyse retenues :", x_positions)

    # Calcul des dérivées après avoir choisi les lignes avec les sliders
    derivatives = compute_first_derivative_y(flat, x_positions)
    seuil = 0

    detected_points = [[] for _ in range(n)]

    for idx, x in enumerate(x_positions):
        print("----------------------------------------------------------")
        print(f"Analyse de la ligne x={x}")
        print("Application du seuil")

        second_derivative = derivatives[x]
        tho = seuil * np.max(np.abs(second_derivative))  # Seuil à 10% du max
        y_coords = np.where(np.abs(second_derivative) > tho)[
            0]  # Détection des pics

        print("Filtrage des points")
        # Filtrer les points pour ne garder que les maximums locaux
        filtered_points = []
        if len(y_coords) > 0:
            max_point = y_coords[0]
            max_value = np.abs(second_derivative[max_point])
            for y in y_coords[1:]:
                if y == max_point + 1:
                    if np.abs(second_derivative[y]) > max_value:
                        max_point = y
                        max_value = np.abs(second_derivative[y])
                else:
                    filtered_points.append((x, max_point))
                    max_point = y
                    max_value = np.abs(second_derivative[y])
            filtered_points.append((x, max_point))

        print("Détection des 2 maximums locaux")
        # Garder les 2 plus grands maximums locaux sans modifier l'ordre des éléments de la liste
        filtered_points = sorted(filtered_points, key=lambda p: np.abs(
            second_derivative[p[1]]), reverse=True)[:2]
        # Trier par position y pour maintenir l'ordre original
        filtered_points.sort(key=lambda p: p[1])

        detected_points[idx] = filtered_points

        print("Points détectés :", filtered_points)
        print("Nombre de points détectés :", len(filtered_points))
        print("fin de l'analyse de la ligne x=", x, '\n')

    # Retourne les points détectés pour chaque ligne shape = (18*2)
    return detected_points


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


def compute_lines_coefficients(detected_columns):
    """Calculate the coefficients (a, b) of lines formed by pairs of columns 
    from a list of detected points.
    This function takes a list of 18 sublists, where each sublist contains 
    points representing detected columns. It computes the coefficients of 
    the lines formed by pairs of columns (e.g., columns 1&2, 3&4, ..., 15&16). 
    The coefficients are calculated for both the top and bottom points of 
    each pair of columns.
    Args:
        detected_columns (list): A list of 18 sublists, where each sublist 
        contains two points representing a detected column.
    Returns:
        list: A list of tuples representing the coefficients (a, b) of the 
        lines for each pair of columns. Each pair contributes two tuples 
        (one for the top points and one for the bottom points).
    Raises:
        ValueError: If there is an error during the calculation of the 
        coefficients for any pair of columns.
    Notes:
        - If the input list contains fewer than 18 columns, an error message 
          is printed, and the function returns None.
        - If a problem is encountered during the calculation of coefficients, 
          the function appends default coefficients (0, 0) for the problematic 
          pair and raises a ValueError."""
    """
    Prend une liste de 18 listes de points et calcule les coefficients (a, b)
    des droites formées par les colonnes 1&2, 3&4, ..., 15&16.
    """
    if len(detected_columns) < 18:
        print("Erreur : la liste doit contenir 18 colonnes de points.")
        return None

    lines_coefficients = []

    for i in range(0, 18, 2):  # Boucle sur les paires (0&1, 2&3, ..., 14&15)
        coeffs_h = line_coefficients(
            detected_columns[i][0], detected_columns[i+1][0])
        coeffs_b = line_coefficients(
            detected_columns[i][1], detected_columns[i+1][1])
        if coeffs_h:
            lines_coefficients.append(coeffs_h)
            lines_coefficients.append(coeffs_b)
        else:
            # si un problème est rencontré déclencher une exception
            print("coeffs_h", coeffs_h)
            print("coeffs_b", coeffs_b)
            lines_coefficients.append((0, 0))
            lines_coefficients.append((0, 0))
            raise ValueError(
                "Erreur lors du calcul des coefficients de la droite.")

    return lines_coefficients


def get_parallelogram_equations(parabolas, lines, quadrilateral_index):
    """
    Retourne les équations des bords d'un parallélogramme pour un quadrilatère spécifié.

    Args:
        parabolas (list): Liste des coefficients des paraboles [(a, b, c), ...].
        lines (list): Liste des coefficients des droites [(a, b), ...].
        quadrilateral_index (int): Indice du quadrilatère voulu (de 1 à 9).

    Returns:
        tuple: 4 tuples correspondant aux équations des bords du parallélogramme.
    """
    if len(parabolas) != 18 or len(lines) != 18:
        raise ValueError(
            "Les listes de coefficients doivent contenir exactement 18 éléments chacune.")

    if quadrilateral_index < 1 or quadrilateral_index > 9:
        raise ValueError(
            "L'indice du quadrilatère doit être compris entre 1 et 9.")

    print("----------------------------------------------------------")
    print(
        f"Début de la récupération des équations pour le quadrilatère {quadrilateral_index}...")
    # Calculer l'indice de base pour le quadrilatère spécifié
    base_index = (quadrilateral_index - 1) * 2

    # Récupérer les équations des bords du parallélogramme
    left_edge = parabolas[base_index]  # Parabole gauche (indice pair)
    right_edge = parabolas[base_index + 1]  # Parabole droite (indice impair)
    top_edge = lines[base_index]  # Droite haute (indice pair)
    bottom_edge = lines[base_index + 1]  # Droite basse (indice impair)

    print("Équations des bords récupérées avec succès.")
    print("Parabole gauche :", left_edge)
    print("Parabole droite :", right_edge)
    print("Droite haute :", top_edge)
    print("Droite basse :", bottom_edge, '\n')

    return left_edge, right_edge, top_edge, bottom_edge


def find_intersection(parabola, line, near_point, point_name, quadrilateral_index):
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
    a_l, b_l = line

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
        parabolas (list): Liste des coefficients des paraboles [(a, b, c), ...].
        lines (list): Liste des coefficients des droites [(a, b), ...].
        quadrilateral_index (int): Indice du quadrilatère voulu (de 1 à 9).
        near_points (list): Liste des points proches pour guider les intersections.

    Returns:
        list: Liste des coordonnées des 4 coins du quadrilatère [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """

    print(f"----------------------------------------------------------")
    print(
        f"Début de la détermination des coins pour le quadrilatère {quadrilateral_index}...")

    left_edge, right_edge, top_edge, bottom_edge = get_parallelogram_equations(
        parabolas, lines, quadrilateral_index)

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


def crop(image, corners):
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

    return cropped_image


def display_detected_points(flat, detected_points_1, detected_points_2, detected_points_3, detected_points_h):
    plt.figure(figsize=(10, 10))
    plt.imshow(flat, cmap='gray', extent=[0, flat.shape[1], flat.shape[0], 0])

    for points in [detected_points_1, detected_points_2, detected_points_3]:
        for x, y in points:
            plt.plot(x, y, 'ro')

    for column_points in detected_points_h:
        for x, y in column_points:
            plt.plot(x, y, 'bo')

    plt.title("Points détectés par detect_edges et detect_edges_y")
    plt.xlim(0, flat.shape[1])
    plt.ylim(flat.shape[0], 0)
    plt.show()


def display_parabolas_and_lines(flat, parabolas, lines):
    for i in range(9):
        plt.figure(figsize=(8, 8))
        plt.imshow(flat, cmap='gray', extent=[
                   0, flat.shape[1], flat.shape[0], 0])

        left_edge, right_edge, top_edge, bottom_edge = get_parallelogram_equations(
            parabolas, lines, i + 1)

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
        plt.title(f"Quadrilatère {i + 1}")
        plt.legend()
        plt.show()


def process_quadrilaterals(flat, parabolas, lines, detected_points_h, image_path):
    transformed_points_h = []
    for i in range(0, len(detected_points_h), 2):
        combined_points = detected_points_h[i] + detected_points_h[i + 1]
        transformed_points_h.append(combined_points)

    for quadrilateral_index in range(1, 10):
        near_points = transformed_points_h[quadrilateral_index - 1]
        corners = find_quadrilateral_corners(
            parabolas, lines, quadrilateral_index, near_points)
        if None in corners:
            print("Erreur lors de la détermination des coins du quadrilatère.")
            return

        display_quadrilateral_corners(flat, corners, quadrilateral_index)

        image = load_fits(image_path)
        if image is None:
            print("Erreur lors du chargement de l'image.")
            return

        cropped_image = crop(image, corners)
        display_cropped_image(cropped_image, quadrilateral_index)


def display_quadrilateral_corners(flat, corners, quadrilateral_index):
    plt.figure(figsize=(10, 10))
    plt.imshow(flat, cmap='gray', extent=[0, flat.shape[1], flat.shape[0], 0])
    for x, y in corners:
        plt.plot(x, y, 'ro')
    plt.title(f"Coins du quadrilatère {quadrilateral_index}")
    plt.xlim(0, flat.shape[1])
    plt.ylim(flat.shape[0], 0)
    plt.show()


def display_cropped_image(cropped_image, quadrilateral_index):
    plt.imshow(cropped_image, cmap='gray')
    plt.title(f"Quadrilatère {quadrilateral_index}")
    plt.show()


def display_start():
    print("__________________________________________________________")
    print("Exécution du programme de traitement d'images multicannaux")
    print("----------------------------------------------------------")
    print("Programme réalisé par :")
    print("CELLIER Maxime, POINTIER Paul, SEIXAS Mathilde, SAINT-PAUL Alexandre, VEILLAUD Baptiste, CANDUN Sam, LEVY Mathis, DA GRACA BAPTISTA Chloé, DUMUR Victor, FERNANDES DE SOUSA Valentin, LEKIC Anica de L'IPSA")
    print("En collaboration avec :")
    print("MEIN Pierre, SAYEDE Frédéric de l'Observatoire de Paris")
    print("----------------------------------------------------------\n")


def display_end():
    print("\n----------------------------------------------------------")
    print("Programme exécuté avec succès")
    print("Fin du programme")
    print("__________________________________________________________")


def main():
    # Affiche les informations de démarrage du programme
    display_start()

    # Chemins des fichiers nécessaires
    flat_path = "flat.fits"  # Chemin du fichier flat
    dark_path = "dark.fits"  # Chemin du fichier dark
    image_path = "image.fits"  # Chemin de l'image à traiter

    # Prétraitement du flat avec le dark
    flat = preprocess_flat(flat_path, dark_path)
    if flat is None:
        print("Erreur lors du chargement du flat.")
        return

    # Détection des bords horizontaux sur le flat
    detected_points_1, detected_points_2, detected_points_3 = detect_edges(
        flat_path, dark_path)
    if not detected_points_1 or not detected_points_2 or not detected_points_3:
        print("Erreur lors de la détection des bords.")
        return

    # Détection des bords verticaux sur le flat
    detected_points_h = detect_edges_y(flat_path, dark_path)
    if not detected_points_h:
        print("Erreur lors de la détection des bords.")
        return

    # Affichage des points détectés sur l'image flat
    display_detected_points(
        flat, detected_points_1, detected_points_2, detected_points_3, detected_points_h)

    # Calcul des équations des paraboles pour les bords horizontaux
    parabolas = parabolic_interpolations(
        detected_points_1, detected_points_2, detected_points_3)
    # Calcul des coefficients des droites pour les bords verticaux
    lines = compute_lines_coefficients(detected_points_h)
    if not parabolas or not lines:
        print("Erreur lors du calcul des coefficients.")
        return

    # Affichage des paraboles et des droites détectées sur l'image flat
    display_parabolas_and_lines(flat, parabolas, lines)

    # Traitement des quadrilatères détectés dans l'image
    process_quadrilaterals(flat, parabolas, lines,
                           detected_points_h, image_path)

    # Affiche les informations de fin du programme
    display_end()


if __name__ == "__main__":
    main()











