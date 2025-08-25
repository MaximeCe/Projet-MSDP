import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from skimage import measure, filters, exposure
import cv2


class CanalDetector:
    def __init__(self, image_path: str, dark_path: str):
        self.image_path = image_path
        self.dark_path = dark_path
        self.original = None
        self.corrected = None
        self.binary = None
        self.canaux = []

    def charger_images(self):
        image_data = fits.getdata(self.image_path)
        dark_data = fits.getdata(self.dark_path)
        self.original = image_data
        self.corrected = image_data - dark_data
        self.corrected = exposure.rescale_intensity(
            self.corrected, in_range='image', out_range=(0, 1))
        print("Images chargées, dark soustrait et intensité étirée.")

    def binariser(self):
        seuil = filters.threshold_otsu(self.corrected)
        self.binary = self.corrected > seuil * 0.3
        print(f"Seuil utilisé (abaissé) : {seuil * 0.3:.2f}")

    def detecter_canaux(self):
        contours = measure.find_contours(self.binary.astype(float), 0.8)
        print(f"{len(contours)} contours trouvés.")
        grands_contours = [c for c in contours if len(c) > 100]
        grands_contours.sort(key=lambda c: np.mean(
            c[:, 1]))  # tri de gauche à droite
        for i, c in enumerate(grands_contours, 1):
            self.canaux.append(Canal(c, nom=str(i)))
        print(f"{len(self.canaux)} canaux retenus après filtrage et tri.")

    def afficher_image_binaire(self):
        plt.imshow(self.binary, cmap='gray')
        plt.title("Image binaire")
        plt.show()

    def afficher_canaux(self):
        plt.imshow(self.corrected, cmap='gray')
        for canal in self.canaux:
            c = canal.contour
            plt.plot(c[:, 1], c[:, 0], label=f"Canal {canal.nom}")
        plt.title("Contours des canaux")
        plt.legend()
        plt.show()


class Canal:
    def __init__(self, contour, nom):
        self.contour = contour
        self.nom = nom
        self.bords = {}
        self.ajuster_bords()

    def ajuster_bords(self):
        x = self.contour[:, 1]
        y = self.contour[:, 0]
        hauteur = np.max(y) - np.min(y)

        # Coupe 2% pour détection des paraboles (gauche/droite uniquement)
        y_min = np.min(y) + hauteur * 0.02
        y_max = np.max(y) - hauteur * 0.02
        masque_y = (y >= y_min) & (y <= y_max)
        contour_filtré = self.contour[masque_y]

        # Isolation des deux segments : médiane de x par ligne
        x_unique = np.unique(contour_filtré[:, 0].astype(int))
        gauche, droite = [], []

        for x_val in x_unique:
            ligne_points = contour_filtré[(
                contour_filtré[:, 0].astype(int) == x_val)]
            if len(ligne_points) >= 2:
                sorted_ligne = ligne_points[np.argsort(ligne_points[:, 1])]
                gauche.append(sorted_ligne[0])
                droite.append(sorted_ligne[-1])

        gauche = np.array(gauche)
        droite = np.array(droite)

        haut = self.contour[self.contour[:, 0]
                            < np.min(self.contour[:, 0]) + 5]
        bas = self.contour[self.contour[:, 0] > np.max(self.contour[:, 0]) - 5]

        self.bords['gauche'] = np.polyfit(gauche[:, 0], gauche[:, 1], 2)
        self.bords['droite'] = np.polyfit(droite[:, 0], droite[:, 1], 2)
        self.bords['haut'] = np.polyfit(haut[:, 1], haut[:, 0], 1)
        self.bords['bas'] = np.polyfit(bas[:, 1], bas[:, 0], 1)

    def afficher_bords(self, image):
        x_vals = np.linspace(0, image.shape[0] - 1, 500)
        plt.imshow(image, cmap='gray', alpha=0.6)

        for nom, coeffs in self.bords.items():
            if len(coeffs) == 3:
                y_vals = coeffs[0] * x_vals**2 + coeffs[1] * x_vals + coeffs[2]
                plt.plot(y_vals, x_vals, label=nom)
            else:
                y_vals = coeffs[0] * x_vals + coeffs[1]
                plt.plot(x_vals, y_vals, label=nom)

        plt.title(f"Ajustement des bords - Canal {self.nom}")
        plt.legend()
        plt.show()


class CalibrateurLambda:
    

    def fit_raie(self, image, canaux: list[Canal], n, d_lambda=0.01, delta_lambda=0.1):
        
        # Détermination des points de calibration
        # Rectangle de largeur l=1 et longueur L=1, i points en largeur, j en longueur
        
        centre_raie = []
        x_total = []
        y_total = []
        decalage = 0
        
        # récupération des intensitées sur lesquels on fera l'interpolation
        points1 = []
        points2 = []
        for canal in canaux :
            points1.append(canal.bords['haut'])
            points2.append(canal.bords['bas'])
        
        points1 = np.array(points1)
        points2 = np.array(points2)
        ensembles = []
        ys = []
        for i in range(1, n+1):
            y = i / (n+1)
            ys.append(y)
            inter = (1-y)*points1 + y*points2
            ensembles.append(inter)
        
        intensites = ensembles
        
        for y0 in ys:
            # formation de la gaussienne à partir des canaux
            intensites = [canal[:,y0] for canal in intensites]
            for c in intensites :
                # Inverser l'axe x si besoin (on suppose que tous sont à inverser)
                c = c[::-1]
                x = np.arange(len(c)) + decalage
                x_total.append(x)
                y_total.append(c)
                # On suppose un chevauchement de 10% entre chaque segment
                decalage += int(len(c) * 0.9)
            
            lambda_initiale = np.concatenate(x_total)
            intensites = np.concatenate(y_total)
            
            # Pour les x dupliqués (chevauchement), on fait la moyenne des y
            x_unique, indices = np.unique(intensites, return_inverse=True)
            y_fusionné = np.zeros_like(x_unique, dtype=float)
            compte = np.zeros_like(x_unique, dtype=int)
            for idx, y in zip(indices, lambda_initiale):
                y_fusionné[idx] += y
                compte[idx] += 1
            intensites /= compte

            # Normalisation de x pour le fit
            lambda_initiale = (x_unique - x_unique.min()) / \
                (x_unique.max() - x_unique.min())

            # affichage de lambda_initiale et intensites
            plt.plot(lambda_initiale, intensites, 'o', label=f"y0={y0:.2f}")
            plt.title("Intensités par rapport à lambda initiale")
            plt.show()

            # fit de la gaussienne 
            try:
                # Ajustement de la gaussienne
                def gaussienne(x, a, x0, sigma, c):
                    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + c
                popt, _ = curve_fit(gaussienne, lambda_initiale, intensites, p0=[1, 0, 1, 0])
                
                a, x0, sigma, c = popt
                print(f"Paramètres ajustés : a={a}, x0={x0}, sigma={sigma}, c={c}")
                
                # Calcul du centre de la raie à partir de la gaussienne ajustée
                centre = x0  # x0 est le centre de la gaussienne ajustée (minimum de la raie)
                centre_raie.append(centre)

            except RuntimeError as e:
                print(f"Erreur lors de l'ajustement : {e}")
        
        
        
        return None


class ReconstructeurImage:
    def reconstruire_monochromatique(self, canaux):
        print("Reconstruction de l'image finale...")
        result = np.zeros((100, 100))
        plt.imshow(result, cmap='gray')
        plt.title("Image monochromatique finale (placeholder)")
        plt.show()


# Utilisation
image_path = "image.fits"
dark_path = "dark.fits"

detecteur = CanalDetector(image_path, dark_path)
detecteur.charger_images()
detecteur.binariser()
detecteur.afficher_image_binaire()
detecteur.detecter_canaux()
detecteur.afficher_canaux()

for canal in detecteur.canaux:
    canal.ajuster_bords()
    canal.afficher_bords(detecteur.corrected)

calibrateur = CalibrateurLambda()
centre_raie = calibrateur.fit_raie(detecteur.corrected, detecteur.canaux, 10)

