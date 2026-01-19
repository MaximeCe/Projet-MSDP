# domain/flat.py
"""
Flat - Represents a flat field calibration image
Pure data class - all processing delegated to services
"""


class Flat:
    """
    Represents a flat field calibration image with its processed channels.
    
    This is now a pure data container. All processing is performed by:
    - DetectionService: edge detection
    - GeometryService: geometric transformations
    - CalibrationService: photometric and spectrometric calibration
    """

    def __init__(self, flat_path, dark_path):
        """
        Initialize a flat field object.
        
        Args:
            flat_path (str): Path to flat field FITS file
            dark_path (str): Path to dark frame FITS file
        """
        self.flat_path = flat_path
        self.dark_path = dark_path

        # Raw image data (after dark subtraction)
        self.data = None
        self.shape = None

        # Detected channels in CCD coordinates
        self.channels = []

        # Normalized channels in solar coordinates
        self.solar_channels = []

        # Calibration data
        self.photometric_ratios = {}
        self.spectral_calibration = {}

        # Computed geometric parameters
        self.Wij = None  # Mean channel width (CCD pixels)
        self.Tgij = None  # Mean channel spacing (CCD pixels)
        self.W = None  # Channel width (solar pixels)
        self.Ts = None  # Spectral translation (solar pixels)
        self.k = None  # Wavelength per pixel (Angstroms/pixel)

    def set_data(self, data):
        """Set the preprocessed flat field data."""
        self.data = data
        self.shape = data.shape

    def set_channels(self, channels):
        """Set the detected channels."""
        self.channels = channels

    def set_solar_channels(self, solar_channels):
        """Set the normalized solar channels."""
        self.solar_channels = solar_channels

    def set_geometric_params(self, Wij, Tgij, W, Ts):
        """Set computed geometric parameters."""
        self.Wij = Wij
        self.Tgij = Tgij
        self.W = W
        self.Ts = Ts

    def set_calibration_data(self, photometric_ratios, spectral_calibration):
        """Set calibration data."""
        self.photometric_ratios = photometric_ratios
        self.spectral_calibration = spectral_calibration
        self.k = spectral_calibration.get('k')

    def __str__(self):
        return f"Flat(shape={self.shape}, channels={len(self.channels)})"

    def __repr__(self):
        return f"Flat(flat_path='{self.flat_path}', channels={len(self.channels)})"
