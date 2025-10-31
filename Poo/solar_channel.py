import numpy as np
from channel import Channel
from tools.channel_normaliser import channel_size, extract_parabolic_shape_to_rect


class SolarChannel(Channel):
    def __init__(self, id, image, index, points, paraboles, output_shape):
        """
        Channel in solar coordinates

        :param id: identifiant unique du canal (1 à 9)
        :param image: instance de la classe Image
        :param index: position du canal dans la liste
        :param data_points: dictionnaire des points détectés
        :param paraboles: liste de 4 tuples (a, b, c), ordre: [gauche, droite, haut, bas]
        :param output_shape: (h, w) taille du rectangle de sortie (optionnel si corners)
        :param corners: liste de 4 tuples (x, y) des coins (optionnel)
        """
        # super().__init__(id, image, index, points)
        # Extrait la région normalisée du canal solaire
        
        self.data = extract_parabolic_shape_to_rect(
            image.data, paraboles, output_shape=output_shape
        )
        self.paraboles = paraboles
        self.resolution = self.data.shape
        self.idx = index
        self.indice = index - 1
        self.lambda_list = [None for _ in range(self.resolution[1])]
    




