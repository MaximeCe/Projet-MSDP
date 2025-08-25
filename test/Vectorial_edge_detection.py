from matplotlib.widgets import Slider
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.path import Path


def load_fits(file_path):
    try:
        with fits.open(file_path) as hdul:
            print(f"✔️ Fichier chargé : {file_path}")
            return hdul[0].data.astype(np.float32)
    except FileNotFoundError:
        print(f"⚠️ Fichier manquant : {file_path}")
        return None


def preprocess_flat(flat_path, dark_path):
    flat = load_fits(flat_path)
    dark = load_fits(dark_path)

    print("\n🔍 Prétraitement du flat en cours...")

    if flat is None:
        print("❌ Impossible de traiter le flat : fichier introuvable.")
        return None

    if dark is not None:
        print("\n📊 Statistiques avant traitement :")
        print(
            f"Moyenne : {np.mean(flat):.2f}, Médiane : {np.median(flat):.2f}, Écart-type : {np.std(flat):.2f}")
        flat -= dark
        print("\n📊 Statistiques après traitement :")
        print(
            f"Moyenne : {np.mean(flat):.2f}, Médiane : {np.median(flat):.2f}, Écart-type : {np.std(flat):.2f}")
    else:
        print("❌ Impossible de traiter le flat : dark manquant.")

    print("✔️ Prétraitement du flat terminé.\n")
    return flat


def find_top_local_extrema(signal, n, sign="both"):
    candidates = []
    for i in range(1, len(signal) - 1):
        if sign == "pos" and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            candidates.append((i, abs(signal[i])))
        elif sign == "neg" and signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            candidates.append((i, abs(signal[i])))
        elif sign == "both" and (
            (signal[i] > signal[i - 1] and signal[i] > signal[i + 1]) or
            (signal[i] < signal[i - 1] and signal[i] < signal[i + 1])
        ):
            candidates.append((i, abs(signal[i])))

    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    best_indices = [idx for idx, _ in sorted_candidates[:n]]

    while len(best_indices) < n:
        best_indices.append(best_indices[-1] if best_indices else 0)

    return best_indices


def detect_points_ABCDEF_vector(image, n, line_offset=15):
    rows, cols = image.shape
    mid_y = rows // 2
    upper_y = max(0, mid_y - line_offset)

    dx_mid = np.gradient(image[mid_y, :])
    dx_upper = np.gradient(image[upper_y, :])

    B_indices = find_top_local_extrema(dx_mid, n=n, sign="pos")
    E_indices = find_top_local_extrema(dx_mid, n=n, sign="neg")
    upper_indices = find_top_local_extrema(dx_upper, n=n, sign="pos")

    B_points = [(x, mid_y) for x in B_indices]
    E_points = [(x, mid_y) for x in E_indices]
    upper_points = [(x, upper_y) for x in upper_indices]

    vectors = [np.array([x1 - x0, y1 - y0])
               for (x0, y0), (x1, y1) in zip(upper_points, B_points)]
    u = np.mean(vectors, axis=0) if vectors else np.array([0.0, 1.0])
    u = u / \
        np.linalg.norm(u) if np.linalg.norm(u) > 0 else np.array([0.0, 1.0])

    dy, dx = np.gradient(image)
    v = np.array([1, 1]) / np.sqrt(2)
    w = np.array([-1, 1]) / np.sqrt(2)

    def extract_directional_candidates(proj_map_1, proj_map_2, n, epsilon=1e-2):
        candidates = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                val_1 = proj_map_1[y, x]
                val_2 = proj_map_2[y, x]
                if val_1 > proj_map_1[y, x - 1] and val_1 > proj_map_1[y, x + 1]:
                    if abs(val_2) < epsilon:
                        grad_norm = np.sqrt(dx[y, x]**2 + dy[y, x]**2)
                        candidates.append((x, y, val_1, val_2, grad_norm))
        sorted_pts = sorted(candidates, key=lambda x: x[4], reverse=True)
        coords = [(x, y, vval, wval)
                  for (x, y, vval, wval, _) in sorted_pts[:n]]
        while len(coords) < n:
            coords.append(coords[-1] if coords else (0, 0, 0.0, 0.0))
        return coords

    proj_v = dx * v[0] + dy * v[1]
    proj_w = dx * w[0] + dy * w[1]

    A_points = extract_directional_candidates(proj_v, proj_w, n)
    C_points = extract_directional_candidates(-proj_w, proj_v, n)
    D_points = extract_directional_candidates(-proj_v, proj_w, n)
    F_points = extract_directional_candidates(proj_w, proj_v, n)

    return {
        "A": A_points,
        "B": B_points,
        "C": C_points,
        "D": D_points,
        "E": E_points,
        "F": F_points,
        "u": u,
        "v": v,
        "w": w,
        "dx": dx,
        "dy": dy
    }


"""----------------------MAIN---------------------"""
flat_path = "flat.fits"
dark_path = "dark.fits"
image_path = "image.fits"

image = preprocess_flat(flat_path, dark_path)

n = 9
results = detect_points_ABCDEF_vector(image, n)
points_dict = {k: v for k, v in results.items() if k in "ABCDEF"}

print("\nVecteur u (orientation moyenne) :", results["u"])
print("Vecteur v (1,1)/√2 :", results["v"])
print("Vecteur w (-1,1)/√2 :", results["w"])



# Recalculer dx, dy pour affichage
_, dx = np.gradient(image)
dy, _ = np.gradient(image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")

plt.subplot(1, 3, 2)
plt.imshow(dx * results["v"][0] + dy * results["v"][1], cmap='bwr')
plt.title("Projection sur v")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(dx * results["w"][0] + dy * results["w"][1], cmap='bwr')
plt.title("Projection sur w")
plt.colorbar()

plt.tight_layout()
plt.show()
