"""
Visualizer - Handles all visualization and plotting for MSDP processing
Separated from processing logic for cleaner architecture
"""

import matplotlib.pyplot as plt
import numpy as np


class MSDPVisualizer:
    """
    Handles all visualization for MSDP data processing.
    Separated from processing logic to maintain single responsibility.
    
    Methods are organized by processing stage:
    - Detection visualization
    - Geometry visualization
    - Calibration visualization
    - Channel visualization
    """

    def __init__(self, config_manager=None):
        """
        Initialize visualizer.
        
        Args:
            config_manager: Optional ConfigManager for plot parameters
        """
        self.config = config_manager

    # ==================== DETECTION VISUALIZATION ====================

    def plot_detected_points(self, image_data, points_dict, title="Detected Channel Points"):
        """
        Display detected points overlaid on the image.
        
        Args:
            image_data (np.ndarray): Flat field image
            points_dict (dict): Dictionary with point lists (as_, bs, etc.)
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.imshow(image_data, cmap='gray', aspect='auto')
        ax.set_title(title)

        # Plot each group of points with different colors
        colors = {
            'as_': 'red', 'bs': 'blue', 'cs': 'green',
            'ds': 'red', 'es': 'blue', 'fs': 'green',
            'ks': 'yellow', 'ls': 'cyan', 'ms': 'yellow', 'ns': 'cyan'
        }

        for name, points in points_dict.items():
            if points:
                xs, ys = zip(*points)
                color = colors.get(name, 'white')
                ax.scatter(xs, ys, c=color, s=10, alpha=0.7, label=name)

        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_channel_with_points(self, image_data, channel, title=None):
        """
        Display a single channel with its detected points and edges.
        
        Args:
            image_data (np.ndarray): Image data
            channel: Channel object with points and edges
            title (str): Optional title
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.imshow(image_data, cmap='gray', aspect='auto')
        ax.set_title(title or f"Channel {channel.id}")

        # Plot detected points
        for name, point in channel.points.items():
            ax.scatter(point.x, point.y, c='red', s=20)
            ax.text(point.x + 10, point.y, name, color='yellow', fontsize=8)

        # Plot final corner points if available
        if channel.points_final:
            for name, point in channel.points_final.items():
                ax.scatter(point.x, point.y, c='blue', s=30, marker='s')
                ax.text(point.x + 10, point.y, name,
                        color='cyan', fontsize=10, weight='bold')

        # Plot edges if available
        if channel.edges:
            x_range = np.arange(image_data.shape[1])
            for edge in channel.edges:
                a, b, c = edge.coefficients()
                y = a * x_range**2 + b * x_range + c
                # Only plot where y is within image bounds
                mask = (y >= 0) & (y < image_data.shape[0])
                color = 'red' if 'parabola' in edge.type else 'yellow'
                ax.plot(x_range[mask], y[mask], color=color,
                        linewidth=0.5, alpha=0.7)

        ax.axis('off')
        plt.tight_layout()
        plt.show()

    # ==================== GEOMETRY VISUALIZATION ====================

    def plot_all_channels_geometry(self, image_data, channels, save_path=None):
        """
        Display all channels with their geometry overlaid.
        Similar to Fortran geo1.ps and geo2.ps plots.
        
        Args:
            image_data (np.ndarray): Flat field image
            channels (list): List of Channel objects
            save_path (str): Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        ax.imshow(image_data, cmap='gray', aspect='auto')
        ax.set_title("All Channels - Geometry")

        # Plot each channel's edges
        x_range = np.arange(image_data.shape[1])
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(channels)))

        for idx, channel in enumerate(channels):
            color = colors[idx]

            # Plot edges
            for edge in channel.edges:
                a, b, c = edge.coefficients()
                y = a * x_range**2 + b * x_range + c
                mask = (y >= 0) & (y < image_data.shape[0])
                linestyle = '-' if 'parabola' in edge.type else '--'
                ax.plot(x_range[mask], y[mask], color=color,
                        linewidth=1, linestyle=linestyle, alpha=0.7)

            # Plot channel number at center
            if channel.points_final and 'A' in channel.points_final:
                center_x = (
                    channel.points_final['A'].x + channel.points_final['D'].x) / 2
                center_y = image_data.shape[0] / 2
                ax.text(center_x, center_y, str(channel.id),
                        color=color, fontsize=12, weight='bold',
                        ha='center', va='center')

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    # ==================== CALIBRATION VISUALIZATION ====================

    def plot_spectral_profiles(self, solar_channels, calibration_data, save_path=None):
        """
        Display spectral profiles for all channels.
        Similar to Fortran cal.ps plots.
        
        Args:
            solar_channels (list): List of SolarChannel objects
            calibration_data (dict): Data from CalibrationService
            save_path (str): Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Get profile data
        original_profiles = calibration_data.get('original_profiles', [])
        fitted_profiles = calibration_data.get('fitted_profiles', [])
        ha_channel = calibration_data.get(
            'ha_channel', len(solar_channels) // 2)

        # Plot 1: Original vs fitted profiles
        ax1.set_title("Mean Spectral Profiles (Original and Fitted)")
        ax1.set_xlabel("Pixel Index")
        ax1.set_ylabel("Mean Intensity")

        for idx, (orig, fitted) in enumerate(zip(original_profiles, fitted_profiles)):
            x = np.arange(len(orig))
            color = 'red' if idx == ha_channel else 'blue'
            alpha = 1.0 if idx == ha_channel else 0.3

            ax1.plot(x, orig, 'o', markersize=2, alpha=alpha/2, color=color)
            ax1.plot(x, fitted, '-', linewidth=1.5, alpha=alpha,
                     color=color, label=f'Channel {idx+1}' if idx == ha_channel else '')

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Wavelength-calibrated profiles
        ax2.set_title("Wavelength-Calibrated Spectral Profiles")
        ax2.set_xlabel("Wavelength (Å)")
        ax2.set_ylabel("Mean Intensity")

        for idx, channel in enumerate(solar_channels):
            if channel.lambda_list and any(l is not None for l in channel.lambda_list):
                # Get valid wavelength range
                lambda_vals = [l for l in channel.lambda_list if l is not None]
                if len(lambda_vals) == len(original_profiles[idx]):
                    color = 'red' if idx == ha_channel else 'blue'
                    alpha = 1.0 if idx == ha_channel else 0.3

                    ax2.plot(lambda_vals, original_profiles[idx], '-',
                             linewidth=1.5, alpha=alpha, color=color,
                             label=f'Channel {idx+1}' if idx == ha_channel else '')

        ax2.axvline(x=calibration_data.get('targeted_lambda', 6562.8),
                    color='green', linestyle='--', label='H-alpha center')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def plot_photometric_ratios(self, photometric_ratios, save_path=None):
        """
        Display photometric calibration ratios between channels.
        
        Args:
            photometric_ratios (dict): Ratios from CalibrationService
            save_path (str): Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if photometric_ratios:
            indices = sorted(photometric_ratios.keys())
            ratios = [photometric_ratios[i] for i in indices]

            ax.bar(indices, ratios, color='steelblue', alpha=0.7)
            ax.axhline(y=1.0, color='red', linestyle='--', label='Unity')
            ax.set_xlabel("Channel Index")
            ax.set_ylabel("Photometric Ratio")
            ax.set_title("Photometric Calibration Ratios")
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    # ==================== SOLAR CHANNEL VISUALIZATION ====================

    def plot_solar_channels(self, solar_channels, vmin=None, vmax=None, save_path=None):
        """
        Display all solar channels side by side.
        
        Args:
            solar_channels (list): List of SolarChannel objects
            vmin, vmax (float): Intensity range for display
            save_path (str): Optional path to save figure
        """
        n_channels = len(solar_channels)
        fig, axes = plt.subplots(1, n_channels, figsize=(3*n_channels, 5))

        if n_channels == 1:
            axes = [axes]

        # Determine global intensity range if not provided
        if vmin is None or vmax is None:
            all_data = [
                ch.data for ch in solar_channels if ch.data is not None]
            if all_data:
                vmin = min(np.min(data) for data in all_data)
                vmax = max(np.max(data) for data in all_data)

        # Plot each channel
        for idx, (ax, channel) in enumerate(zip(axes, solar_channels)):
            if channel.data is not None:
                im = ax.imshow(channel.data, cmap='gray',
                               vmin=vmin, vmax=vmax, aspect='auto')
                ax.set_title(f"Channel {channel.id}")
                ax.axis('off')

        # Add colorbar
        fig.colorbar(im, ax=axes, orientation='horizontal',
                     fraction=0.05, pad=0.05)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    # ==================== COMPARISON PLOTS ====================

    def plot_before_after(self, before_data, after_data, title="Before/After Comparison"):
        """
        Display before and after comparison of processing steps.
        
        Args:
            before_data (np.ndarray): Original data
            after_data (np.ndarray): Processed data
            title (str): Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.imshow(before_data, cmap='gray', aspect='auto')
        ax1.set_title("Before")
        ax1.axis('off')

        ax2.imshow(after_data, cmap='gray', aspect='auto')
        ax2.set_title("After")
        ax2.axis('off')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
