import numpy as np
from astropy.io import fits


class Io:
    @staticmethod
    def load_fits(file_path: str):
        """Charge un fichier FITS si disponible, sinon retourne None."""
        try:
            with fits.open(file_path) as hdul:
                print(f"✔️ Fichier chargé : {file_path}")
                return hdul[0].data.astype(np.float32) # type: ignore
        except FileNotFoundError:
            print(f"⚠️ Fichier manquant : {file_path}")
            return None

    @staticmethod
    def preprocess_fits(image_path, master_dark=None, master_flat=None, master_bias=None):
        """Prétraite une image FITS avec Dark, Flat, Bias (optionnels)."""
        image = Io.load_fits(image_path)
        if image is None:
            return None

        dark = Io.load_fits(master_dark) if master_dark else None
        flat = Io.load_fits(master_flat) if master_flat else None
        bias = Io.load_fits(master_bias) if master_bias else None

        if bias is not None:
            image -= bias
        if dark is not None:
            image -= dark
        if flat is not None:
            flat[flat == 0] = 1
            image /= flat

        return image
