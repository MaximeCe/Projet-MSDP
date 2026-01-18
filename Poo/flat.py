from tools.computation import Computation
from tools.detector import Detector
from tools.io import Io
from tools.channel_normaliser import channel_size, extract_parabolic_shape_to_rect
from solar_channel import SolarChannel
from channel import Channel
from config_manager import ConfigManager
import numpy as np
import matplotlib.pyplot as plt


class Flat():
    def __init__(self, flat_path, dark_path, nombre_canaux=9, config_path="config.yml"):
        self.flat_path = flat_path
        self.master_dark = dark_path
        self.nombre_canaux = nombre_canaux

        # Initialize configuration manager
        self.config = ConfigManager(config_path)

        # Load or set basic parameters
        self._load_basic_params()

        # Image processing
        self.resolution = None
        self.data = None
        dark_data = Io.load_fits(dark_path)
        self.data = Io.load_fits(self.flat_path) - dark_data
        
        # apply median filter
        self.data = Computation.median_filter(self.data, size=3)
        # asin stretch
        self.data = Computation.linear_stretch(self.data)
        
        

        # ms.par: threshold_factor - image masking threshold
        # threshold_factor = self.config.get(
        #     'detection', 'threshold_factor', 0.1)
        # self.threshold = threshold_factor * np.max(self.data)
        # self.masqued = Computation.mask(self, self.threshold)
        # self.shape = self.data.shape if self.data is not None else (0, 0)

        # Create channels
        self.channels = []
        self.create_channels()

        # Create solar channels
        self.solar_channels = []
        self.create_solar_channels()

        # Calibration parameters
        self._setup_calibration_params()

        # Computed geometric values
        self._compute_or_load_geometric_values()

        # Photometric and spectrometric calibration
        self.photometric_calibration()
        self.spectrometric_calibration()

    def _load_basic_params(self):
        """Load or initialize basic parameters from config"""
        # ms.par: nm - number of channels
        nm = self.config.get('geometry', 'nm', 9)
        if nm != self.nombre_canaux:
            self.config.set('geometry', 'nm', self.nombre_canaux)

    def _setup_calibration_params(self):
        """Setup calibration parameters from config or defaults"""
        # ms.par: t1_mm, t2_mm - slicer translations (see msdp.par step 3a)
        self.t1_mm = self.config.get('calibration', 't1_mm', 2.5)
        self.t2_mm = self.config.get('calibration', 't2_mm', 9.0)

    def _compute_or_load_geometric_values(self):
        """Compute or load geometric values (Wij, Tgij, W, Ts)"""

        # ms.par: Wij - mean channel width in CCD pixels (computed from geometry)
        if self.config.should_compute('Wij'):
            self.Wij = np.mean([
                np.sqrt((canal.points_final["D"].x - canal.points_final["A"].x)**2 +
                        (canal.points_final["D"].y - canal.points_final["A"].y)**2)
                for canal in self.channels
            ])
            self.config.update_computed_values(Wij=float(self.Wij))
        else:
            self.Wij = self.config.get_computed_value('Wij')

        # ms.par: Tgij - distance between successive channels in CCD pixels
        if self.config.should_compute('Tgij'):
            self.Tgij = np.mean([
                self.channels[i+1].points_final["A"].x -
                self.channels[i].points_final["A"].x
                for i in range(len(self.channels)-1)
                if self.channels[i].points_final
            ])
            self.config.update_computed_values(Tgij=float(self.Tgij))
        else:
            self.Tgij = self.config.get_computed_value('Tgij')

        # ms.par: W - channel width in solar pixels (from output_shape)
        if self.config.should_compute('W'):
            self.W = self.solar_channels[0].resolution[1]
            self.config.update_computed_values(W=int(self.W))
        else:
            self.W = self.config.get_computed_value('W')

        # ms.par: Ts - spectral translation between channels (solar pixels)
        # Formula from MSDP documentation: Ts = Tgij * W * t1 / (t2 * Wij)
        if self.config.should_compute('Ts'):
            self.Ts = self.Tgij * self.W * self.t1_mm / (self.t2_mm * self.Wij)
            self.config.update_computed_values(Ts=float(self.Ts))
        else:
            self.Ts = self.config.get_computed_value('Ts')

    def create_points_dict(self):
        """Generate a dictionary with key = 'name' and value = list[point]"""
        # Detection using parameters from config
        detected = Detector.detect_edges_x(self)
        if not all(detected):
            raise ValueError("❌ Erreur : points horizontaux non détectés.")

        # Separate points into 6 lists of 9
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

        # Vertical detection
        detected_h = Detector.detect_edges_y(self, be_list=detected[1])
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

        points_dict = {
            "as_": as_, "bs": bs, "cs": cs, "ds": ds, "es": es, "fs": fs,
            "ks": ks, "ls": ls, "ms": ms, "ns": ns
        }
        return points_dict

    def create_channels(self):
        points_dict = self.create_points_dict()
        for i in range(self.nombre_canaux):
            canal = Channel(id=i + 1, image=self, index=i, points=points_dict)
            self.channels.append(canal)

    def create_solar_channels(self, display=True):
        """Create normalized solar channels from detected channels"""
        print("🔄 Création des canaux solaires normalisés...")

        points_dict = self.create_points_dict()

        # Use central channel to define corners
        canal = self.channels[self.nombre_canaux//2]
        if hasattr(canal, "points_final") and canal.points_final:
            pf = canal.points_final
            if all(k in pf for k in ["C", "F", "D", "A"]):
                corners = [
                    (pf["C"].x, pf["C"].y),
                    (pf["F"].x, pf["F"].y),
                    (pf["D"].x, pf["D"].y),
                    (pf["A"].x, pf["A"].y),
                ]

        if display:
            self.afficher(points=[pf["C"], pf["F"], pf["D"], pf["A"]])
            print(corners)

        # ms.par: output_shape - computed from corners (height, width in solar pixels)
        if self.config.should_compute('output_shape'):
            self.output_shape = channel_size(corners)
            self.config.update_computed_values(
                output_shape=list(self.output_shape)
            )
        else:
            output_shape_list = self.config.get_computed_value('output_shape')
            self.output_shape = tuple(
                output_shape_list) if output_shape_list else channel_size(corners)

        for i, canal in enumerate(self.channels):
            paraboles = [edge.coefficients()
                         for edge in canal.edges if "parabole" in edge.type]
            droites = [edge.coefficients()
                       for edge in canal.edges if "parabole" not in edge.type]

            if len(paraboles) == 2 and len(droites) == 2:
                paraboles_ordre = paraboles + droites

                if display:
                    print(f"Canal {canal.id} points: {canal.points}")
                    x = np.linspace(0, self.shape[0], 500)
                    for i in range(4):
                        y = paraboles_ordre[i][0]*x**2 + \
                            paraboles_ordre[i][1]*x + paraboles_ordre[i][2]
                        plt.plot(x, y)
                    plt.imshow(self.data, cmap='gray')
                    plt.legend(['gauche', 'droite', 'haut', 'bas'])
                    plt.show()

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
                print(f"⚠️ Canal {canal.id}: paraboles/droites manquantes")

    def photometric_calibration(self):
        """Photometric calibration of channels"""
        # ms.par: channel_offset - avoid edge effects (5%)
        channel_offset = self.config.get('detection', 'channel_offset', 0.05)

        xmax = self.solar_channels[0].resolution[1]
        begining = int(channel_offset * xmax)
        end = int(xmax - channel_offset * xmax)
        Ts = round(self.Ts)

        self.photometric_ratios = {}

        for idx in range(len(self.solar_channels[:-1])):
            map = self.solar_channels[idx].data
            map1 = self.solar_channels[idx + 1].data

            isolambda_n = map[:, begining:end-Ts]
            isolambda_n1 = map1[:, begining+Ts:end]

            mean_n = np.mean(isolambda_n, axis=0)
            mean_n1 = np.mean(isolambda_n1, axis=0)

            ratio = np.mean([mean_n[k]/mean_n1[k]
                            for k in range(len(mean_n1))])
            self.photometric_ratios[idx] = ratio
            self.solar_channels[idx + 1].data *= ratio

        # Save photometric ratios to config
        ratios_dict = {str(k): float(v)
                       for k, v in self.photometric_ratios.items()}
        self.config.update_computed_values(photometric_ratios=ratios_dict)

    def spectrometric_calibration(self, display=True):
        """Compute wavelength calibration"""
        # ms.par: targeted_lambda - H-alpha line wavelength
        targeted_lambda = self.config.get(
            'calibration', 'targeted_lambda', 6562.8)
        # ms.par: lambda_offset - wavelength difference between channels
        lambda_offset = self.config.get('calibration', 'lambda_offset', 0.3)
        # ms.par: channel_offset
        channel_offset = self.config.get('detection', 'channel_offset', 0.05)

        xmax = self.solar_channels[0].resolution[1]
        begining = int(channel_offset * xmax)
        end = int(xmax - channel_offset * xmax)

        mean_columns_intensities = [
            [np.mean(canal.data[:, begining:end], axis=0)]
            for canal in self.solar_channels
        ]

        plots = []
        for i in range(len(mean_columns_intensities)):
            plots.append(mean_columns_intensities[i])
            mean_columns_intensities[i] = np.polyval(
                np.polyfit(np.arange(len(mean_columns_intensities[i][0])),
                           mean_columns_intensities[i][0], 2),
                np.arange(len(mean_columns_intensities[i][0]))
            )

        idx_min_intensities = [
            begining + np.argmin(intensities) for intensities in mean_columns_intensities]
        min_intensities = [np.min(intensities)
                           for intensities in mean_columns_intensities]

        ha_channel = np.argmin([
            abs(idx_min_intensities[idx] - xmax/2) * min_intensities[idx]
            for idx in range(len(idx_min_intensities))
        ])
        ha_idx = idx_min_intensities[ha_channel]
        print(f"Ha_channel {ha_channel}")
        self.solar_channels[ha_channel].lambda_list[ha_idx] = targeted_lambda
        print(
            f"Canal Ha (n°{ha_channel+1}) : min intensité à l'index {ha_idx} (milieu={xmax/2})")

        # ms.par: k - wavelength calibration coefficient (Angstroms/pixel)
        if self.config.should_compute('k'):
            self.k = lambda_offset / self.Ts
            self.config.update_computed_values(k=float(self.k))
        else:
            self.k = self.config.get_computed_value('k')

        # Propagate wavelength calibration
        for idx, canal in enumerate(self.solar_channels):
            canal.lambda_list[ha_idx] = targeted_lambda + \
                lambda_offset * (idx - ha_channel)
            for i in range(0, xmax):
                canal.lambda_list[i] = targeted_lambda + \
                    lambda_offset * (idx - ha_channel) + (ha_idx - i) * self.k

        if display:
            for idx, canal in enumerate(self.solar_channels):
                plt.plot(canal.lambda_list[begining:end],
                         plots[idx][0], label=f"Canal {idx+1}")
            plt.show()

    def apply_flat_correction(self, light):
        """Apply geometric, photometric and spectrometric corrections to light image"""
        light_solar_channels = []
        for solar_channel in self.solar_channels:
            light_solar_channel = extract_parabolic_shape_to_rect(
                light, solar_channel.paraboles, self.output_shape
            )
            light_solar_channels.append(light_solar_channel)

        # Apply photometric calibration
        for idx in range(len(self.solar_channels[:-1])):
            ratio = self.photometric_ratios.get(idx, 1.0)
            light_solar_channels[idx + 1] *= ratio

        # Apply spectrometric calibration
        light_lambda_lists = []
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
            if points:
                for point in points:
                    plt.plot(point.x, point.y, 'ro')
            plt.show()
        else:
            raise ValueError("Les données de l'image sont vides.")
