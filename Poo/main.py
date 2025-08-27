from matplotlib import lines
from image import Image
from tools.display import Display


def main():
    print("Chargement de l'image et initialisation des canaux...\n")


    image = Image(
        image_path="image.fits",
        master_dark="dark.fits"
    )

    print(image)
    # image.afficher()

    print("\nPremier canal :")
    first_channel = image.channels[0]
    print(first_channel)
    first_channel.compute_corners()

    print("\nPoints minuscule du premier canal (a–f, k–n) :")

    points = first_channel.points.values()
    for point in points:
        print(f"  - {point.nom} : ({point.x:.1f}, {point.y:.1f})")
    
    image.afficher(points)
    
    
    print("\nPoints majuscule du premier canal (A–F) :")
    points = first_channel.points_final.values()

    # Ranger les points dans l'ordre alphabétique
    points = sorted(points, key=lambda p: p.nom)
    for point in points:
        print(f"  - {point.nom} : ({point.x:.1f}, {point.y:.1f})")
    image.afficher(points)
    
    # afficher les bords du canal
    parabolas = [edge.coefficients() for edge in first_channel.edges if "parabole" in edge.type]
    lines = [edge.coefficients() for edge in first_channel.edges if "parabole" not in edge.type]
    Display.display_parabolas_and_lines(image, parabolas, lines) ## TODO: Debug ici


if __name__ == "__main__":
    main()
 