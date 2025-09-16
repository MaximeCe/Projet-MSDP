from solar_channel import SolarChannel
from channel import Channel  
from tools.detector import Detector
from tools.io import Io
import numpy as np
from tools.channel_normaliser import channel_size
import matplotlib.pyplot as plt

class Image:
    def __init__(self, image_path, master_dark=None, nombre_canaux=9):
        self.image_path = image_path
        self.master_dark = master_dark
        self.nombre_canaux = nombre_canaux

        # Création et traitement de l'image
        self.resolution = None
        self.data = None
        self.load_and_process_image()
        self.shape = self.data.shape if self.data is not None else (0, 0)

        # Création des canaux
        self.channels = []
        self.create_channels()
        

        # # affichage des points détectés pour vérification
        # for canal in self.channels:
        #     self.afficher(points=[p for p in canal.points.values() if p])

        # Création des canaux solaires
        self.solar_channels = []
        self.create_solar_channels()
        
        # Paramètre de calibration en lambda
        self.t1_mm = 2.5
        self.t2_mm = 9.0
        self.Wij = np.mean(
            [np.sqrt((canal.points_final["D"].x-canal.points_final["A"].x)**2+(canal.points_final["D"].y-canal.points_final["A"].y)**2) for canal in self.channels])
        self.Tgij = np.mean(
            [self.channels[i+1].points_final["A"].x-self.channels[i].points_final["A"].x for i in range(len(self.channels)-1) if self.channels[i].points_final])
        self.W = np.mean([canal.resolution[0] for canal in self.solar_channels if canal.resolution])
        self.Ts = self.Tgij*self.W*self.t1_mm/(self.t2_mm*self.Wij)


    def load_and_process_image(self):
        """Charge l’image FITS et applique uniquement le dark."""
        image = Io.load_fits(self.image_path)
        if image is None:
            raise ValueError(
                f"❌ Impossible de charger l'image : {self.image_path}")

        if self.master_dark:
            dark = Io.load_fits(self.master_dark)
            if dark is not None:
                image -= dark
            else:
                print(f"⚠️ Dark introuvable : {self.master_dark}")

        self.data = image
        self.resolution = f"{self.data.shape[1]}x{self.data.shape[0]}"



    def create_points_dict(self):
        # Détection horizontale (3 lignes → 18 points pour chaque)
        detected = Detector.detect_edges_x(self)
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
        detected_h = Detector.detect_edges_y(
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
        points_dict = {
            "as_": as_, "bs": bs, "cs": cs, "ds": ds, "es": es, "fs": fs,
            "ks": ks, "ls": ls, "ms": ms, "ns": ns
        }
        return points_dict
    
    def create_channels(self):
        points_dict = self.create_points_dict()
        # Créer les canaux à partir des données
        for i in range(self.nombre_canaux):
            canal = Channel(id=i + 1, image=self, index=i, points=points_dict)
            self.channels.append(canal)


    def create_solar_channels(self):
        """Crée les canaux solaires normalisés à partir des canaux détectés."""
        print("🔄 Création des canaux solaires normalisés...")
        
        points_dict = self.create_points_dict()
        
        canal = self.channels[self.nombre_canaux//2] # utilliser le canal centrale pour définir les coins
        if hasattr(canal, "points_final") and canal.points_final:
        # Ordre: [haut-gauche, haut-droit, bas-droit, bas-gauche]
            pf = canal.points_final
            if all(k in pf for k in ["C", "F", "D", "A"]):
                corners = [
                    (pf["C"].x, pf["C"].y),
                    (pf["F"].x, pf["F"].y),
                    (pf["D"].x, pf["D"].y),
                    (pf["A"].x, pf["A"].y),
                ]

        # Afficher les coins sur l'image pour vérification
        # self.afficher(points=[pf["C"], pf["F"], pf["D"], pf["A"]])
        
        
        print(corners)
        output_shape = channel_size(corners)
        # output_shape = (800,100)
        
        
        for i, canal in enumerate(self.channels):
            
            # Récupérer les paraboles (gauche, droite, haut, bas) pour chaque canal
            paraboles = [edge.coefficients()
                                           for edge in canal.edges if "parabole" in edge.type]
            droites = [edge.coefficients()
                                         for edge in canal.edges if "parabole" not in edge.type]
            # Ordre attendu : [gauche, droite, haut, bas]
            if len(paraboles) == 2 and len(droites) == 2:
                # On suppose l'ordre des edges : [parabole_gauche, parabole_droite, droite_haut, droite_bas]
                paraboles_ordre = paraboles + droites
                # Affichage de canal.points pour vérification
                # print(f"Canal {canal.id} points: { canal.points}")
                
                # # Affichage des paraboles et des droites sur l'image pour vérification
                # x = np.linspace(0, self.shape[0], 500)
                # for i in range(4):
                #     y = paraboles_ordre[i][0]*x**2 + paraboles_ordre[i][1]*x + paraboles_ordre[i][2]
                #     plt.plot(x, y)
            
                # plt.imshow(self.data, cmap='gray')
                # plt.legend(['gauche', 'droite', 'haut', 'bas'])
                # plt.show()
                
                solar_channel = SolarChannel(
                    id=canal.id,
                    image=self,
                    index=i,
                    points=points_dict,
                    paraboles=paraboles_ordre,
                    output_shape=output_shape,
                    
                )
                self.solar_channels.append(solar_channel)
            else:
                print(f"⚠️ Canal {canal.id}: paraboles/droites manquantes, canal solaire non créé.")

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

