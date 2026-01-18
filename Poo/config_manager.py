"""
Configuration Manager for MSDP Data Processing
Maps parameters from ms.par to Python configuration
"""

import yaml
import os
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path="config.yml"):
        self.config_path = config_path
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self):
        """Load existing config or create default structure"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"✔️ Configuration chargée depuis {self.config_path}")
                return config
        else:
            config = self._create_default_config()
            self._save_config(config)
            print(f"✔️ Configuration par défaut créée : {self.config_path}")
            return config
    
    def _create_default_config(self):
        """Create default configuration structure matching ms.par"""
        return {
            'geometry': {
                # ===== GEOMETRY PARAMETERS (from ms.par) =====
                'li': None,           # ms.par: li - field stop length (arcsec/1000)
                'lj': None,           # ms.par: lj - field stop width (arcsec/1000)
                'nm': 9,              # ms.par: nm - number of channels
                'nbcln': 1024,        # ms.par: nbcln - final number of X-CCD pixels
                'nblgn': 1536,        # ms.par: nblgn - final number of Y-CCD pixels
                'interc': 15,         # ms.par: interc - approx distance between channels (CCD pixels)
                'milsec': 500,        # ms.par: milsec - output pixel (unit arcsec/1000)
                'i1': 1,              # ms.par: i1 - first useful pixel in i-direction
                'i2m': 0,             # ms.par: i2m - last useful pixel offset
                'j1': 1,              # ms.par: j1 - first useful pixel in j-direction
                'j2m': 0,             # ms.par: j2m - last useful pixel offset
                'milangi': -40,       # ms.par: milangi - angle between channel edge and CCD (rad/1000)
                'milangj': None,      # ms.par: milangj (computed from geometry)
                'lip': 40,            # ms.par: lip - curvature determination percentage
                'jeps': 20,           # ms.par: jeps - edge detection tolerance (pixels)
                'intvi': 60,          # ms.par: intvi - integration interval for i-edges
                'intvj': 30,          # ms.par: intvj - integration interval for j-edges
                'leps': 50,           # ms.par: leps - gradient max search interval
                'distor': 1,          # ms.par: distor - 1=curvature taken into account
                'ipermu': 1,          # ms.par: ipermu - permutation of CCD X and Y
            },
            
            'detection': {
                # ===== EDGE DETECTION PARAMETERS =====
                'threshold_factor': 0.1,        # Python: THRESHOLD_FACTOR - image masking threshold
                'si': 15,                       # ms.par: si - intensity threshold for edge detection vs X
                'sgi': 10,                      # ms.par: sgi - gradient threshold vs X
                'sj': 15,                       # ms.par: sj - intensity threshold vs Y
                'sgj': 10,                      # ms.par: sgj - gradient threshold vs Y
                'mingrad': 8,                   # ms.par: mingrad - minimum intensity gradient
                'interp': 1,                    # ms.par: interp - parabolic interpolation flag
                'y_positions_offset': 60,       # Python: Y_POSITIONS_OFFSET
                'channel_offset': 0.05,         # Python: CHANNEL_OFFSET (5% edge avoidance)
            },
            
            'calibration': {
                # ===== SPECTRAL CALIBRATION =====
                'lbda': 6563,         # ms.par: lbda - line wavelength (Angstroms)
                'dlbd': 300,          # ms.par: dlbd - wavelength distance between channels (mA)
                'targeted_lambda': 6562.8,  # Python: TARGETED_LAMBDA (H-alpha)
                'lambda_offset': 0.3,       # Python: LAMBDA_OFFSET (Angstroms)
                'mupris': 9000,       # ms.par: mupris - translation between output channels (microns)
                'mustep': 2500,       # ms.par: mustep - distance between successive slits (microns)
                't1_mm': 2.5,         # Python: slicer input translation (mm)
                't2_mm': 9.0,         # Python: slicer output translation (mm)
                'idc': 1,             # ms.par: idc - dark current subtraction flag
            },
            
            'computed_values': {
                # ===== VALUES COMPUTED DURING PROCESSING =====
                'Wij': None,          # Python: mean channel width in CCD pixels
                'Tgij': None,         # Python: distance between successive channels (CCD pixels)
                'W': None,            # Python: channel width in solar pixels
                'Ts': None,           # Python: spectral translation between channels (solar pixels)
                'k': None,            # Python: wavelength calibration coefficient (A/pixel)
                'output_shape': None, # Python: (height, width) of solar channels
                'photometric_ratios': {},  # Python: calibration ratios between channels
            },
            
            'processing': {
                # ===== PROCESSING FLAGS (from ms.par) =====
                'ixy': 1,             # ms.par: ixy - compute average dark/flat/field-stop
                'igeom': 1,           # ms.par: igeom - geometry
                'iflat': 0,           # ms.par: iflat - calibration
                'ibmc': 0,            # ms.par: ibmc - elementary calibrated c-files
                'icmd': 0,            # ms.par: icmd - d-files (spectroheliograms)
                'ides': 0,            # ms.par: ides - plots
                'iquick': 0,          # ms.par: iquick - q-files for scanned targets
                'igrayq': 0,          # ms.par: igrayq - plots of q-files
                'igeo': 1,            # ms.par: igeo - plot geo.ps
                'iflat1': 0,          # ms.par: iflat1 - plot flat1.ps
                'iflat2': 0,          # ms.par: iflat2 - plot flat2.ps
                'ical': 0,            # ms.par: ical - plot cal.ps
            },
            
            'files': {
                # ===== FILE PATHS =====
                'flat_path': None,
                'dark_path': None,
                'lights_path': [],
            }
        }
    
    def _save_config(self, config=None):
        """Save configuration to YAML file"""
        if config is None:
            config = self.config
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, section, key, default=None):
        """Get a configuration value"""
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section, key, value):
        """Set a configuration value and save"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self._save_config()
    
    def update_computed_values(self, **kwargs):
        """Update computed values section"""
        for key, value in kwargs.items():
            self.config['computed_values'][key] = value
        self._save_config()
    
    def get_geometry_params(self):
        """Get all geometry parameters as dict"""
        return self.config['geometry']
    
    def get_detection_params(self):
        """Get all detection parameters as dict"""
        return self.config['detection']
    
    def get_calibration_params(self):
        """Get all calibration parameters as dict"""
        return self.config['calibration']
    
    def get_computed_value(self, key):
        """Get a computed value, return None if not yet computed"""
        return self.config['computed_values'].get(key)
    
    def should_compute(self, key):
        """Check if a value needs to be computed"""
        return self.config['computed_values'].get(key) is None