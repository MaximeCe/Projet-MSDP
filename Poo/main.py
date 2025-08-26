from image import Image
import tools


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
    
    # addicher les bords du canal
    tools.display_parabolas_and_lines(image, first_channel)



if __name__ == "__main__":
    main()
