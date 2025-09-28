from matplotlib import pyplot as plt
from image import Image
from tools.display import Display


def main():
    # Chargement de l'image FITS avec soustraction du dark
    image = Image(
        image_path="image.fits",
        master_dark="dark.fits"
    )

    # Récupère les valeurs min et max globales pour l'échelle de gris
    all_data = [sc.data for sc in image.solar_channels]
    vmin = min(data.min() for data in all_data)
    vmax = max(data.max() for data in all_data)

    # Affiche tous les canaux solaires sur le même graphique
    fig, axes = plt.subplots(1, len(image.solar_channels), figsize=(15, 5))
    if len(image.solar_channels) == 1:
        axes = [axes]
    for ax, solar_channel in zip(axes, image.solar_channels):
        im = ax.imshow(solar_channel.data, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"Solar Channel {solar_channel.id}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Ts (en pixels solaire) = {image.Ts}")


if __name__ == "__main__":
    main()
