import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path


class Display:
    """Not used, should replace the diplay = True arguments ???"""
    @staticmethod
    def crop(image, corners, quadrilateral_index):
        corners = np.array(corners)
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1]-center[1], corners[:, 0]-center[0])
        sorted_corners = corners[np.argsort(angles)]

        rr, cc = np.meshgrid(
            np.arange(image.shape[1]), np.arange(image.shape[0]))
        polygon = Path(sorted_corners)
        inside = np.array(polygon.contains_points(
            np.vstack((rr.flatten(), cc.flatten())).T)).reshape(image.shape)
        cropped = np.zeros_like(image)
        cropped[inside] = image[inside]
        plt.imshow(cropped, cmap='gray')
        plt.title(f"Quadrilatère {quadrilateral_index}")
        plt.show()
        return cropped

    @staticmethod
    def display_detected_points(flat, points_x, points_y):
        plt.figure(figsize=(10, 10))
        plt.imshow(flat, cmap='gray', extent=[
                   0, flat.shape[1], flat.shape[0], 0])
        for line in points_x:
            for x, y in line:
                plt.plot(x, y, 'ro')
        for col in points_y:
            for x, y in col:
                plt.plot(x, y, 'bo')
        plt.title("Points détectés")
        plt.show()

    @staticmethod
    def display_parabolas_and_lines(flat, parabolas, lines):
        flat = flat.data
        plt.imshow(flat, cmap='gray')
        x = np.arange(flat.shape[1])
        y_left = parabolas[0][0]*x**2 + parabolas[0][1]*x + parabolas[0][2]
        y_right = parabolas[1][0]*x**2 + parabolas[1][1]*x + parabolas[1][2]
        plt.plot(x, y_left, 'r', label="Parabole gauche")
        plt.plot(x, y_right, 'b', label="Parabole droite")
        y_top = lines[0][0]*x + lines[0][1]
        y_bottom = lines[1][0]*x + lines[1][1]
        plt.plot(x, y_top, 'g', label="Ligne haute")
        plt.plot(x, y_bottom, 'm', label="Ligne basse")
        plt.legend()
        plt.show()

    @staticmethod
    def display_quadrilateral_corners(flat, corners, index):
        plt.imshow(flat, cmap='gray', extent=[
                   0, flat.shape[1], flat.shape[0], 0])
        for x, y in corners:
            plt.plot(x, y, 'ro')
        plt.title(f"Coins quadrilatère {index}")
        plt.show()

    @staticmethod
    def statistiques(DE, AB, DF, AC, CF, BE, AD):
        # idem code original, mais rangé ici
        pass  # <-- à réintégrer si tu veux garder les tracés vectoriels

    @staticmethod
    def display_start():
        print("__________________________________________________________")
        print("Exécution du programme de traitement d'images multicannaux")

    @staticmethod
    def display_end():
        print("Programme exécuté avec succès ✔️")
