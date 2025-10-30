from tools.io import Io
from flat import Flat

class Lights():
    def __init__(self, lights_path: list[str], flat_path, dark_path, nombre_canaux=9) -> None:
        self.lights_path = lights_path
        self.flat_path = flat_path
        self.dark_path = dark_path
        self.nombre_canaux = nombre_canaux

        # Création et traitement de l'image
        self.resolution = None
        self.data = None
        self.shape = self.data.shape if self.data is not None else (0, 0)
        self.lights = [Io.load_fits(light_path) for light_path in self.lights_path]
        dark = Io.load_fits(self.dark_path)
        
        self.datas = self.lights - dark
        self.flat = Flat(flat_path=self.flat_path, dark_path=self.dark_path, nombre_canaux=self.nombre_canaux)
        
        self.light_solar_channels = []
        self.light_lambda_lists = []
        for light in self.lights:
            light_solar_channels, light_lambda_lists = self.flat.apply_flat_correction(
                light)
            self.light_solar_channels.append(light_solar_channels)
            self.light_lambda_lists.append(light_lambda_lists)
        
            
        
    
        