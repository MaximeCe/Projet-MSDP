from tools.io import Io
from flat import Flat

class Image:
    def __init__(self, image_path, dark_path, nombre_canaux=9) -> None:
        """Not used. Should be a static class for flat, light and maybe dark ???"""
        self.image_path= image_path
        self.dark_path = dark_path
        self.nombre_canaux = nombre_canaux

        # Création et traitement de l'image
        self.resolution = None
        self.data = None
        self.shape = self.data.shape if self.data is not None else (0, 0)
        image = Io.load_fits(self.image_path)
        dark = Io.load_fits(self.dark_path)
        
        self.data = image - dark        