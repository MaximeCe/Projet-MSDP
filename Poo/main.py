from matplotlib import pyplot as plt
from flat import Flat
from light import Lights
import numpy as np
import os

def main():
    
    def stack_images(images):
        """Stack images together"""
        return np.median(np.array(images), axis=0)
    
    def stack_folder(folder_name):
        """stack all the fit images in a folder and save the master fit in the master file"""
        import os
        from tools.io import Io
        if not os.path.exists('master'):
            os.makedirs('master')
        image_files = [f for f in os.listdir(folder_name) if f.endswith('.fit') or f.endswith('.fits')]
        images = [Io.load_fits(os.path.join(folder_name, f)) for f in image_files]
        stacked_image = stack_images(images)
        Io.save_fits(stacked_image, f'master/{folder_name}.fits')
        print(f"Master stacked image saved to 'master/{folder_name}.fits'")
    
    # stack_folder("Flats")
    # stack_folder("Darks")
    
    images_path = [f for f in os.listdir(
        "Lights") if f.endswith('.fit') or f.endswith('.fits')]
    images_path = [os.path.join("Lights", f) for f in images_path]
    
    
    
    lights = Lights(images_path, flat_path="master/Flats.fits", dark_path="master/Darks.fits", nombre_canaux=9)
    
    # # Chargement de l'image FITS avec soustraction du dark
    # image = Flat(
    #     flat_path="image.fits",
    #     dark_path="dark.fits"
    # )

    # # Récupère les valeurs min et max globales pour l'échelle de gris
    # all_data = [sc.data for sc in image.solar_channels]
    # vmin = min(data.min() for data in all_data)
    # vmax = max(data.max() for data in all_data)

    # # Affiche tous les canaux solaires sur le même graphique
    # fig, axes = plt.subplots(1, len(image.solar_channels), figsize=(15, 5))
    # if len(image.solar_channels) == 1:
    #     axes = [axes]
    # for ax, solar_channel in zip(axes, image.solar_channels):
    #     im = ax.imshow(solar_channel.data, cmap='gray',# type: ignore
    #                    vmin=vmin, vmax=vmax)  
    #     ax.set_title(f"Solar Channel {solar_channel.id}")  # type: ignore
    #     ax.axis('off')#type: ignore
    # plt.tight_layout()
    # plt.show()
    # print(f"Ts (en pixels solaire) = {image.Ts}")


if __name__ == "__main__":
    main()
