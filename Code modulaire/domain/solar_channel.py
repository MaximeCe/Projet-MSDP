# domain/solar_channel.py
"""
SolarChannel - Represents a channel in normalized solar coordinates
Pure data class - processing delegated to GeometryService
"""


class SolarChannel:
    """
    Represents a spectral channel in normalized rectangular solar coordinates.
    Contains the rectified image data and wavelength calibration.
    
    All geometric transformations are performed by GeometryService.
    All calibrations are performed by CalibrationService.
    """

    def __init__(self, channel_id, index, shape):
        """
        Initialize a solar channel.
        
        Args:
            channel_id (int): Channel identifier (1 to nm)
            index (int): Zero-based index in channel list
            shape (tuple): (height, width) of the normalized channel
        """
        self.id = channel_id
        self.index = index
        self.resolution = shape

        # Normalized rectangular data
        self.data = None

        # Wavelength calibration (one wavelength per X pixel)
        self.lambda_list = [None] * shape[1]

        # Edge equations in CCD coordinates (for reference)
        self.parabolas = None

    def set_data(self, data):
        """Set the normalized image data."""
        self.data = data

    def set_parabolas(self, parabolas):
        """Store the parabolic edge equations for reference."""
        self.parabolas = parabolas

    def get_wavelength_range(self):
        """Get the wavelength range covered by this channel."""
        valid_lambdas = [l for l in self.lambda_list if l is not None]
        if valid_lambdas:
            return min(valid_lambdas), max(valid_lambdas)
        return None, None

    def __str__(self):
        return f"SolarChannel {self.id} (shape={self.resolution})"

    def __repr__(self):
        return f"SolarChannel(id={self.id}, resolution={self.resolution})"
