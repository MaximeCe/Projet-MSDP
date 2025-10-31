from tools.computation import Computation
from tools.detector import Detector
from tools.io import Io
from tools.channel_normaliser import channel_size, extract_parabolic_shape_to_rect
from solar_channel import SolarChannel
from channel import Channel  
import numpy as np
import matplotlib.pyplot as plt


class Flat():
    def __init__(self, flat_path, dark_path, nombre_canaux=9):
        self.flat_path = flat_path
        self.master_dark = dark_path
        self.nombre_canaux = nombre_canaux

        # Création et traitement de l'image
        self.resolution = None
        self.data = None
        dark_data = Io.load_fits(dark_path)
        # Seuil fixé pour que 98% des pixels du FITS soient inférieurs
        # self.load_and_process_image()
        self.data = Io.load_fits(self.flat_path)- dark_data
        self.threshold = 0.1* np.max(self.data)
        self.masqued=Computation.mask(self, self.threshold)
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
        
        # Largeur moyenne des cannaux (px)
        self.Wij = np.mean(
            [np.sqrt((canal.points_final["D"].x-canal.points_final["A"].x)**2+(canal.points_final["D"].y-canal.points_final["A"].y)**2) for canal in self.channels])
        
        # Distance entre deux cannaux successifs (px)
        self.Tgij = np.mean(
            [self.channels[i+1].points_final["A"].x-self.channels[i].points_final["A"].x for i in range(len(self.channels)-1) if self.channels[i].points_final])
        
        # Largeur des cannaux en pixels solaires 
        self.W = self.solar_channels[0].resolution[1]
        
        # Translation spectral entre deux cannaux successifs (px solaires)
        self.Ts = self.Tgij*self.W*self.t1_mm/(self.t2_mm*self.Wij)
        
        # Calibration photométrique des cannaux 
        self.photometric_calibration()
        
        # Calibration spectrométrique des cannaux
        self.spectrometric_calibration()


    def load_and_process_image(self):
        """Charge l’image FITS et applique uniquement le dark."""
        image = Io.load_fits(self.flat_path)
        if image is None:
            raise ValueError(
                f"❌ Impossible de charger l'image : {self.flat_path}")

        if self.master_dark:
            dark = Io.load_fits(self.master_dark)
            if dark is not None:
                image -= dark
            else:
                print(f"⚠️ Dark introuvable : {self.master_dark}")

        self.data = image
        self.resolution = f"{self.data.shape[1]}x{self.data.shape[0]}"


    def create_points_dict(self):
        """Generate a dictionnary with key = 'name' and value = list[point] each point being on one edge of a channel following the figure 3 in the pdf file"""
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
        self.output_shape = channel_size(corners)
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
                    output_shape=self.output_shape,
                    
                )
                self.solar_channels.append(solar_channel)
            else:
                print(f"⚠️ Canal {canal.id}: paraboles/droites manquantes, canal solaire non créé.")


    def photometric_calibration(self):
        """Photometric calibration of each channels. lambda = wavelength; isolambda = same wavelength column in a channel in solar coordinate.
        Computation => Isolambda in channel n / isolambda in channel n + 1  = calibration Ratio 
        channel *= calibration Ratio"""
        xmax = self.solar_channels[0].resolution[1]
        begining = int(0.1*xmax)
        end = int(xmax-0.1*xmax)
        Ts = round(self.Ts)
        
        self.photometric_ratios = {}
        
        for idx in range(len(self.solar_channels[:-1])):
            map = self.solar_channels[idx].data
            map1 = self.solar_channels[idx + 1].data
            
            # take all isolambda column on channel n and n+1 exept for the edges (5%)
            isolambda_n = map[:,begining:end-Ts ] 
            isolambda_n1 = map1[:,begining+Ts:end]
            
            # compute the mean of each column
            mean_n = np.mean(isolambda_n, axis=0)
            mean_n1 = np.mean(isolambda_n1, axis=0)
            # print(f"Canal means, n={mean_n.mean()}, n+1={mean_n1.mean()}")
            
            # compute the ratio
            ratio =  np.mean([mean_n[k]/mean_n1[k] for k in range(len(mean_n1))])
            self.photometric_ratios[idx] = ratio

            # apply the ratio to channel n+1
            self.solar_channels[idx + 1].data *= ratio 


    def spectrometric_calibration(self, display = True):
        """Compute the calibration in wavelength. Detect the most centered minimum as the raie of interest (Ha) and 
        propagate the wavelength thanks to Ts inside the channel and ad 0.3A to an isolambda column to change channel"""
        # initialisations
        xmax = self.solar_channels[0].resolution[1]
        begining = int(0*xmax)
        end = int(xmax-0*xmax)
        
        # columns intensities for each channel
        mean_columns_intensities = [[np.mean(canal.data[:,begining:end], axis = 0)] for canal in self.solar_channels]
        
        # replace mean columns intensities by an approximate parabolic fit and save the original in plot
        plots = []
        for i in range(len(mean_columns_intensities)):
            plots.append(mean_columns_intensities[i])
            mean_columns_intensities[i] = np.polyval(np.polyfit(np.arange(len(mean_columns_intensities[i][0])), mean_columns_intensities[i][0], 2), np.arange(len(mean_columns_intensities[i][0])))
        # len(mean_columns_intensities[i]) = len(plots[i])
        
        # idx of the minimum intensity for each channel + begining offset
        idx_min_intensities = [begining+np.argmin(intensities) for intensities in mean_columns_intensities]
        min_intensities = [np.min(intensities) for intensities in mean_columns_intensities]
        
        # find the argument of the list closest to half of xmax
        ha_channel = np.argmin([abs(idx_min_intensities[idx]-xmax/2)
                                        * min_intensities[idx] for idx in range(len(idx_min_intensities))])
        ha_idx = idx_min_intensities[ha_channel]
        print("Ha_channel", ha_channel)
        self.solar_channels[ha_channel].lambda_list[ha_idx] = 6562.8  # H-alpha
        print(f"Canal Ha (n°{ha_channel+1}) : min intensité à l'index {ha_idx} (milieu={xmax/2})")
        """ Test de la méthode empirique de calcul de Ts et k (échec)
        # Afficher les intensités de chaque canaux cote à cote pour vérification
        for i, intensities in enumerate(mean_columns_intensities):
            plt.plot(intensities, label=f"Canal {i+1} (min at {idx_min_intensities[i]})")
            plt.scatter(
                np.arange(len(plots[i][0])), plots[i])

            plt.legend()
            plt.show()
            
        
        # k computation
        k_theorical = 0.3/self.Ts
        Ts_theorical = self.Ts
        
        # get the min intensity index for the channel before and after ha_channel
        ks = []
        for i in range(1, 1+self.nombre_canaux//2): 
            for idx in [ha_channel-i, ha_channel+i]:
                # calcul du premier coefficient de la parabole correspondant à l'index
                a = Computation.parabolic_interpolation(
                    (0, mean_columns_intensities[idx][0]), 
                    (len(mean_columns_intensities[idx])//2, mean_columns_intensities[idx][len(mean_columns_intensities[idx])//2]), 
                    (len(mean_columns_intensities[idx]), mean_columns_intensities[idx][len(mean_columns_intensities[idx])-1]))[0]
                
                if begining>idx_min_intensities[idx]>end or a <0:
                    print(f" Pas de minimum dans le canal {idx} sortie de la boucle")
                    break
                
                delta_idx = idx_min_intensities[idx] - ha_idx
                print("delta_idx", delta_idx)
                Ts_mesured = abs(delta_idx)
                print("Ts_mesured", Ts_mesured)
                k_mesured = 0.3/Ts_mesured
                ks.append(k_mesured)

        for k in ks :
            if abs(k- k_theorical)/k_theorical > 0.1:
                print(f"⚠️ Attention : k mesuré={k:.6f} diffère de k théorique={k_theorical:.6f} de plus de 10%")
        
        k = np.mean(ks)
        print(f"k final = {k:.4f} (théorique={k_theorical:.4f})")
        """
        # k computation
        self.k = 0.3/self.Ts
        
        # propagation de k dans la liste de lambda pour chaque canal
        for idx,canal in enumerate( self.solar_channels):
            canal.lambda_list[ha_idx]= 6562.8+0.3*(idx-ha_channel)
            for i in range(0, xmax):
                canal.lambda_list[i] = 6562.8+0.3*(idx-ha_channel)+(ha_idx-i)*self.k

        # mean_complete_columns_intensities=[np.mean(canal.data, axis=0)
        #  for canal in self.solar_channels]
        # plot en x les lambdas et en y les intensités moyennes sur un même graphique
        if display == True:
            for idx, canal in enumerate(self.solar_channels):
                plt.plot(canal.lambda_list[begining:end],
                        plots[idx][0], label=f"Canal {idx+1}")
                
            plt.show()
    
    def apply_flat_correction(self, light):
        """Appllique la correction géométrique puis photométrique et la calibration spectrométrique à une image light."""
        # Appliquer la correction géométrique
        light_solar_channels = []
        for solar_channel in self.solar_channels:
            light_solar_channel = extract_parabolic_shape_to_rect(light, solar_channel.paraboles, self.output_shape)
            light_solar_channels.append(light_solar_channel)
        
        
        # Appliquer la calibration photométrique
        for idx in range(len(self.solar_channels[:-1])):
            ratio = self.photometric_ratios.get(idx, 1.0)
            light_solar_channels[idx + 1] *= ratio
        
        light_lambda_lists = []
        # Appliquer la calibration spectrométrique
        for idx, canal in enumerate(light_solar_channels):
            light_lambda_list = self.solar_channels[idx].lambda_list
            light_lambda_lists.append(light_lambda_list)
        
        return light_solar_channels, light_lambda_lists
        
        
        
        
        
        
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

