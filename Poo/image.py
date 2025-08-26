from tools import detect_edges_following_x, detect_edges_following_y, load_fits
from channel import Channel  # à créer ensuite


class Image:
    def __init__(self, image_path, master_dark=None, nombre_canaux=9):
        self.image_path = image_path
        self.master_dark = master_dark
        self.nombre_canaux = nombre_canaux

        self.resolution = None
        self.data = None
        self.load_and_process_image()
        self.shape = self.data.shape if self.data is not None else (0, 0)
        self.channels = []

        self.create_channels()



    def load_and_process_image(self):
        """Charge l’image FITS et applique uniquement le dark."""
        image = load_fits(self.image_path)
        if image is None:
            raise ValueError(
                f"❌ Impossible de charger l'image : {self.image_path}")

        if self.master_dark:
            dark = load_fits(self.master_dark)
            if dark is not None:
                image -= dark
            else:
                print(f"⚠️ Dark introuvable : {self.master_dark}")

        self.data = image
        self.resolution = f"{self.data.shape[1]}x{self.data.shape[0]}"


    def create_channels(self):
        """Crée les canaux après détection des points initiaux."""
        print("🔍 Détection des points pour les canaux...")

        # Appliquer le dark au flat et détecter les points
        flat_path = "flat.fits"
        dark_path = self.master_dark

        # Détection horizontale (3 lignes → 18 points pour chaque)
        detected = detect_edges_following_x(self)
        if not all(detected):
            raise ValueError("❌ Erreur : points horizontaux non détectés.")

        # Séparer les points en 6 listes de 9
        cs, fs, bs, es, as_, ds = [], [], [], [], [], []
        for i in range(0, len(detected[0])):
            if i % 2 == 0:
                cs.append(detected[0][i])
                bs.append(detected[1][i])
                as_.append(detected[2][i])
            else:
                fs.append(detected[0][i])
                es.append(detected[1][i])
                ds.append(detected[2][i])

        # Détection verticale (18 colonnes → 2 points chacune)
        detected_h = detect_edges_following_y(
            self, be_list=detected[1])
        if not detected_h:
            raise ValueError("❌ Erreur : points verticaux non détectés.")

        ls, ks, ns, ms = [], [], [], []
        for i in range(len(detected_h)):
            if i % 2 == 0:
                ls.append(detected_h[i][0])
                ks.append(detected_h[i][1])
            else:
                ns.append(detected_h[i][0])
                ms.append(detected_h[i][1])

        # Regrouper les points dans un dict
        data_dict = {
            "as_": as_, "bs": bs, "cs": cs, "ds": ds, "es": es, "fs": fs,
            "ks": ks, "ls": ls, "ms": ms, "ns": ns
        }
        print(data_dict)
        # Créer les canaux à partir des données
        for i in range(self.nombre_canaux):
            canal = Channel(id=i + 1, image=self, index=i, data=data_dict)
            self.channels.append(canal)


    def __str__(self):
        return f"Image(resolution={self.resolution}, canaux={len(self.channels)})"

    def afficher(self, points=None):
        import matplotlib.pyplot as plt
        if self.data is not None:
            plt.imshow(self.data, cmap='gray')
            plt.title("Image traitée")
            plt.axis("off")

            # Afficher les points s'ils sont fournis
            if points:
                for point in points:
                    plt.plot(point.x, point.y, 'ro')  # 'ro' pour des points rouges

            plt.show()
        else:
            raise ValueError("Les données de l'image sont vides. Impossible d'afficher l'image.")
