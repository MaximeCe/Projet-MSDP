from astropy.io import fits
import numpy as np

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

    print("\\n🔍 Prétraitement du flat en cours...")

    if flat is None:
        print("❌ Impossible de traiter le flat : fichier introuvable.")
        return None

    if dark is not None:
        print("\\n📊 Statistiques avant traitement :")
        print(f"Moyenne : {np.mean(flat):.2f}, Médiane : {np.median(flat):.2f}, Écart-type : {np.std(flat):.2f}")
        flat -= dark
        print("\\n📊 Statistiques après traitement :")
        print(f"Moyenne : {np.mean(flat):.2f}, Médiane : {np.median(flat):.2f}, Écart-type : {np.std(flat):.2f}")
    else:
        print("❌ Impossible de traiter le flat : dark manquant.")

    print("✔️ Prétraitement du flat terminé.\\n")
    return flat
