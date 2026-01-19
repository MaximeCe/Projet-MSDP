"""
Calibration Service - Handles photometric and spectrometric calibration
Extracted from flat.py to separate calibration logic
"""

import numpy as np


class CalibrationService:
    """
    Service responsible for photometric and spectrometric calibration of MSDP channels.
    Implements calibration methods from ms3.f (calib, profmean).
    
    This service handles:
    - Photometric calibration (intensity normalization between channels)
    - Spectrometric calibration (wavelength assignment)
    """

    def __init__(self, config_manager):
        """
        Initialize calibration service with configuration.
        
        Args:
            config_manager: ConfigManager instance with calibration parameters
        """
        self.config = config_manager
        self.calib_params = config_manager.get_calibration_params()
        self.detection_params = config_manager.get_detection_params()

    def compute_photometric_calibration(self, solar_channels, Ts):
        """
        Compute photometric calibration ratios between channels.
        Implements the intensity normalization from MSDP-methods Step 3b.
        
        The method compares intensities at the same wavelength in adjacent channels
        and computes correction ratios to normalize them.
        
        Args:
            solar_channels (list): List of SolarChannel objects with .data arrays
            Ts (float): Spectral translation between channels (in pixels)
        
        Returns:
            dict: Photometric ratios {channel_index: ratio_to_apply}
        """
        channel_offset = self.detection_params.get('channel_offset', 0.05)

        xmax = solar_channels[0].resolution[1]
        beginning = int(channel_offset * xmax)
        end = int(xmax - channel_offset * xmax)
        Ts_int = round(Ts)

        photometric_ratios = {}

        # For each pair of adjacent channels
        for idx in range(len(solar_channels) - 1):
            channel_n = solar_channels[idx].data
            channel_n1 = solar_channels[idx + 1].data

            # Extract iso-wavelength columns (shifted by Ts)
            # Column i in channel n corresponds to column i+Ts in channel n+1
            isolambda_n = channel_n[:, beginning:end - Ts_int]
            isolambda_n1 = channel_n1[:, beginning + Ts_int:end]

            # Compute mean intensity for each wavelength
            mean_n = np.mean(isolambda_n, axis=0)
            mean_n1 = np.mean(isolambda_n1, axis=0)

            # Compute ratio (avoiding division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = mean_n / mean_n1
                ratios = ratios[np.isfinite(ratios)]

            # Average ratio across wavelengths
            if len(ratios) > 0:
                ratio = np.mean(ratios)
            else:
                ratio = 1.0

            photometric_ratios[idx] = ratio

        return photometric_ratios

    def apply_photometric_calibration(self, solar_channels, photometric_ratios):
        """
        Apply photometric calibration ratios to solar channels.
        Modifies the data in place.
        
        Args:
            solar_channels (list): List of SolarChannel objects
            photometric_ratios (dict): Ratios from compute_photometric_calibration
        """
        for idx in range(len(solar_channels) - 1):
            ratio = photometric_ratios.get(idx, 1.0)
            # Apply ratio to next channel
            solar_channels[idx + 1].data *= ratio

    def compute_spectrometric_calibration(self, solar_channels, Ts):
        """
        Compute wavelength calibration for all channels.
        Implements wavelength assignment from MSDP-methods Step 3a.
        
        This method:
        1. Detects the H-alpha line center in the central channel
        2. Computes wavelength calibration coefficient k
        3. Assigns wavelength to each pixel in each channel
        
        Args:
            solar_channels (list): List of SolarChannel objects
            Ts (float): Spectral translation between channels (in pixels)
        
        Returns:
            dict: Calibration data including k, ha_channel, ha_idx
        """
        # Get calibration parameters
        targeted_lambda = self.calib_params.get('targeted_lambda', 6562.8)
        lambda_offset = self.calib_params.get('lambda_offset', 0.3)
        channel_offset = self.detection_params.get('channel_offset', 0.05)

        xmax = solar_channels[0].resolution[1]
        beginning = int(channel_offset * xmax)
        end = int(xmax - channel_offset * xmax)

        # Step 1: Compute mean column intensities for each channel
        mean_columns = []
        for channel in solar_channels:
            mean_col = np.mean(channel.data[:, beginning:end], axis=0)
            mean_columns.append(mean_col)

        # Step 2: Fit parabolas to smooth the profiles
        fitted_profiles = []
        for mean_col in mean_columns:
            x_vals = np.arange(len(mean_col))
            coeffs = np.polyfit(x_vals, mean_col, 2)
            fitted = np.polyval(coeffs, x_vals)
            fitted_profiles.append(fitted)

        # Step 3: Find line center (minimum intensity) in each channel
        idx_min_list = [beginning + np.argmin(profile)
                        for profile in fitted_profiles]
        min_intensities = [np.min(profile) for profile in fitted_profiles]

        # Step 4: Identify the channel containing H-alpha
        # Choose channel with minimum closest to center and deepest line
        ha_channel = np.argmin([
            abs(idx_min_list[i] - xmax/2) * min_intensities[i]
            for i in range(len(idx_min_list))
        ])
        ha_idx = idx_min_list[ha_channel]

        # Step 5: Compute wavelength calibration coefficient k
        # k = wavelength change per pixel = lambda_offset / Ts
        k = lambda_offset / Ts

        # Step 6: Assign wavelengths to all channels
        for idx, channel in enumerate(solar_channels):
            # Central wavelength for this channel
            channel_lambda = targeted_lambda + \
                lambda_offset * (idx - ha_channel)

            # Wavelength for each pixel
            channel.lambda_list = [
                channel_lambda + (ha_idx - i) * k
                for i in range(xmax)
            ]

        return {
            'k': k,
            'ha_channel': ha_channel,
            'ha_idx': ha_idx,
            'targeted_lambda': targeted_lambda,
            'fitted_profiles': fitted_profiles,
            'original_profiles': mean_columns
        }

    def get_wavelength_at_position(self, solar_channel, x_pixel):
        """
        Get the wavelength at a specific pixel position in a channel.
        
        Args:
            solar_channel: SolarChannel object with lambda_list
            x_pixel (int): Pixel index
        
        Returns:
            float: Wavelength in Angstroms
        """
        if 0 <= x_pixel < len(solar_channel.lambda_list):
            return solar_channel.lambda_list[x_pixel]
        return None

    def compute_calibration_statistics(self, calibration_data):
        """
        Compute statistics about the calibration for validation.
        
        Args:
            calibration_data (dict): Data from compute_spectrometric_calibration
        
        Returns:
            dict: Statistics including wavelength range, resolution, etc.
        """
        k = calibration_data['k']
        ha_channel = calibration_data['ha_channel']
        targeted_lambda = calibration_data['targeted_lambda']

        return {
            'spectral_resolution': k,  # Angstroms per pixel
            'ha_channel_index': ha_channel,
            'central_wavelength': targeted_lambda,
            'calibration_method': 'single_method'  # Could be extended to multiple methods
        }
