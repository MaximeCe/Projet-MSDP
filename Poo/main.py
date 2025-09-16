from matplotlib import pyplot as plt
from image import Image
from tools.display import Display


def main():
    # Chargement de l'image FITS avec soustraction du dark
    image = Image(
        image_path="image.fits",
        master_dark="dark.fits"
    )

    # Affichage des data de chaque canal solaire
    for i, solar_channel in enumerate(image.solar_channels):
        plt.figure()
        plt.imshow(solar_channel.data, cmap='gray')
        plt.title(f"Solar Channel {solar_channel.id}")
        plt.axis('off')
    plt.show()
    print(f"Ts (en pixels solaire) = {image.Ts}")


if __name__ == "__main__":
    main()
