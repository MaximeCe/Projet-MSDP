from matplotlib import pyplot as plt
from flat import Flat
from light import Lights

def main():
    
    lights = Lights(["image.fit"], flat_path="flat.fits", dark_path="dark.fits", nombre_canaux=9)
    
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
