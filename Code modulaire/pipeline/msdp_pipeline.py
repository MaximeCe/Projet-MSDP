"""
MSDP Pipeline - Orchestrates the complete processing workflow
Coordinates all services to process MSDP data from raw FITS to calibrated channels
"""

import numpy as np
from domain.flat import Flat
from domain.channel import Channel
from domain.solar_channel import SolarChannel
from domain.point import Point
from domain.edge import Edge
from services.detection_service import DetectionService
from services.geometry_service import GeometryService
from services.calibration_service import CalibrationService
from infrastructure.io import Io
from infrastructure.computation import Computation
from visualization.visualizer import MSDPVisualizer


class MSDPPipeline:
    """
    Orchestrates the complete MSDP data processing workflow.
    
    This pipeline implements the processing steps from MSDP-methods-2024-02.pdf:
    - Step 1: Average dark/flat fields
    - Step 2: Geometry detection and normalization
    - Step 3: Photometric and spectrometric calibration
    
    The pipeline separates concerns:
    - Domain objects (Flat, Channel) contain only data
    - Services (Detection, Geometry, Calibration) perform processing
    - Visualizer handles all plotting
    - ConfigManager manages parameters
    """

    def __init__(self, config_manager, visualizer=None):
        """
        Initialize the pipeline with configuration and services.
        
        Args:
            config_manager: ConfigManager instance
            visualizer: Optional MSDPVisualizer for plotting
        """
        self.config = config_manager

        # Initialize services
        self.detection_service = DetectionService(config_manager)
        self.geometry_service = GeometryService(config_manager)
        self.calibration_service = CalibrationService(config_manager)

        # Initialize visualizer
        self.visualizer = visualizer or MSDPVisualizer(config_manager)

        # Calibration parameters
        self.t1_mm = config_manager.get('calibration', 't1_mm', 2.5)
        self.t2_mm = config_manager.get('calibration', 't2_mm', 9.0)

    def process_flat_field(self, flat_path, dark_path, num_channels=9, visualize=False):
        """
        Complete pipeline to process a flat field image.
        
        This is the main entry point that orchestrates all processing steps.
        
        Args:
            flat_path (str): Path to flat field FITS file
            dark_path (str): Path to dark frame FITS file
            num_channels (int): Number of spectral channels (default: 9)
            visualize (bool): If True, display intermediate results
        
        Returns:
            Flat: Processed Flat object with all calibrations
        """
        print("=" * 60)
        print("MSDP Pipeline - Processing Flat Field")
        print("=" * 60)

        # Step 1: Load and preprocess image
        flat = self._load_and_preprocess(flat_path, dark_path, visualize)

        # Step 2: Detect channel geometry
        self._detect_channels(flat, num_channels, visualize)

        # Step 3: Normalize to solar coordinates
        self._normalize_channels(flat, visualize)

        # Step 4: Compute geometric parameters
        self._compute_geometric_parameters(flat)

        # Step 5: Photometric calibration
        self._apply_photometric_calibration(flat, visualize)

        # Step 6: Spectrometric calibration
        self._apply_spectrometric_calibration(flat, visualize)

        # Step 7: Save computed values to config
        self._save_to_config(flat)

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print(f"Processed {len(flat.solar_channels)} channels")
        print(f"Wavelength calibration: k = {flat.k:.6f} Å/pixel")
        print("=" * 60)

        return flat

    def _load_and_preprocess(self, flat_path, dark_path, visualize):
        """Step 1: Load FITS files and apply preprocessing."""
        print("\n[1/7] Loading and preprocessing image...")

        # Create Flat domain object
        flat = Flat(flat_path, dark_path)

        # Load FITS data
        dark_data = Io.load_fits(dark_path)
        flat_data = Io.load_fits(flat_path)

        if dark_data is None or flat_data is None:
            raise ValueError("Failed to load FITS files")

        # Subtract dark
        preprocessed = flat_data - dark_data

        # Apply filters and stretches
        preprocessed = Computation.median_filter(preprocessed, size=3)
        preprocessed = Computation.linear_stretch(preprocessed)

        flat.set_data(preprocessed)

        print(f"   Image shape: {flat.shape}")
        print(
            f"   Intensity range: [{np.min(preprocessed):.1f}, {np.max(preprocessed):.1f}]")

        if visualize:
            self.visualizer.plot_before_after(flat_data, preprocessed,
                                              title="Preprocessing: Dark Subtraction and Filtering")

        return flat

    def _detect_channels(self, flat, num_channels, visualize):
        """Step 2: Detect channel edges and compute geometry."""
        print("\n[2/7] Detecting channel edges...")

        # Detect all channel edges using DetectionService
        points_dict = self.detection_service.detect_all_channel_edges(
            flat.data)

        print(f"   Detected points for {num_channels} channels")

        if visualize:
            self.visualizer.plot_detected_points(flat.data, points_dict,
                                                 title="Detected Channel Points")

        # Create Channel objects and assign detected points
        channels = []
        for i in range(num_channels):
            channel = Channel(channel_id=i + 1, index=i)

            # Extract points for this channel from points_dict
            channel_points = {}
            for name in ['a', 'b', 'c', 'd', 'e', 'f', 'k', 'l', 'm', 'n']:
                point_list = points_dict.get(
                    f"{name}s") or points_dict.get(f"{name}s_")
                if point_list and len(point_list) > i:
                    x, y = point_list[i]
                    channel_points[name] = Point(name, x, y, channel)

            channel.set_points(channel_points)

            # Build edges using GeometryService
            parabolas, lines = self.geometry_service.build_channel_edges(
                channel_points)

            # Create Edge objects
            edges = [
                Edge("parabola_left", *parabolas[0], channel),
                Edge("parabola_right", *parabolas[1], channel),
                Edge("line_top", *lines[0], channel),
                Edge("line_bottom", *lines[1], channel)
            ]
            channel.set_edges(edges)

            # Compute final corner points (ABCDEF)
            near_points = [
                channel_points['c'].xy(),
                channel_points['f'].xy(),
                channel_points['a'].xy(),
                channel_points['d'].xy()
            ]
            corners = self.geometry_service.compute_channel_corners(
                parabolas, lines, near_points
            )

            # Add B and E from detected points
            final_points = {
                'A': Point('A', *corners['A'], channel),
                'C': Point('C', *corners['C'], channel),
                'D': Point('D', *corners['D'], channel),
                'F': Point('F', *corners['F'], channel),
                'B': Point('B', channel_points['b'].x, channel_points['b'].y, channel),
                'E': Point('E', channel_points['e'].x, channel_points['e'].y, channel)
            }
            channel.set_final_points(final_points)

            channels.append(channel)

        flat.set_channels(channels)

        if visualize and channels:
            self.visualizer.plot_all_channels_geometry(flat.data, channels)

    def _normalize_channels(self, flat, visualize):
        """Step 3: Normalize channels to rectangular solar coordinates."""
        print("\n[3/7] Normalizing channels to solar coordinates...")

        # Determine output shape from central channel
        central_channel = flat.channels[len(flat.channels) // 2]
        pf = central_channel.points_final

        # Check if output_shape is already computed
        output_shape_list = self.config.get_computed_value('output_shape')
        if output_shape_list:
            output_shape = tuple(output_shape_list)
        else:
            corners = [
                (pf['C'].x, pf['C'].y),
                (pf['F'].x, pf['F'].y),
                (pf['D'].x, pf['D'].y),
                (pf['A'].x, pf['A'].y)
            ]
            output_shape = self.geometry_service.compute_channel_size(corners)
            self.config.update_computed_values(output_shape=list(output_shape))

        print(f"   Output shape: {output_shape}")

        # Normalize each channel
        solar_channels = []
        for channel in flat.channels:
            # Get parabolic edge equations
            parabolas = [edge.coefficients() for edge in channel.edges]

            # Create SolarChannel
            solar_channel = SolarChannel(
                channel.id, channel.index, output_shape)
            solar_channel.set_parabolas(parabolas)

            # Normalize using GeometryService
            normalized_data = self.geometry_service.normalize_channel_to_rectangle(
                flat.data,
                parabolas,
                output_shape
            )
            solar_channel.set_data(normalized_data)

            solar_channels.append(solar_channel)

        flat.set_solar_channels(solar_channels)

        if visualize and solar_channels:
            self.visualizer.plot_solar_channels(solar_channels)

    def _compute_geometric_parameters(self, flat):
        """Step 4: Compute geometric parameters (Wij, Tgij, W, Ts)."""
        print("\n[4/7] Computing geometric parameters...")

        # Wij, Tgij: From CCD geometry
        if self.config.should_compute('Wij') or self.config.should_compute('Tgij'):
            stats = self.geometry_service.compute_geometric_statistics(
                flat.channels)
            Wij = stats['Wij']
            Tgij = stats['Tgij']
            self.config.update_computed_values(
                Wij=float(Wij), Tgij=float(Tgij))
        else:
            Wij = self.config.get_computed_value('Wij')
            Tgij = self.config.get_computed_value('Tgij')

        # W: From solar channel resolution
        if self.config.should_compute('W'):
            W = flat.solar_channels[0].resolution[1]
            self.config.update_computed_values(W=int(W))
        else:
            W = self.config.get_computed_value('W')

        # Ts: Spectral translation (formula from MSDP-methods)
        if self.config.should_compute('Ts'):
            Ts = Tgij * W * self.t1_mm / (self.t2_mm * Wij)
            self.config.update_computed_values(Ts=float(Ts))
        else:
            Ts = self.config.get_computed_value('Ts')

        flat.set_geometric_params(Wij, Tgij, W, Ts)

        print(f"   Wij (channel width CCD): {Wij:.2f} pixels")
        print(f"   Tgij (channel spacing CCD): {Tgij:.2f} pixels")
        print(f"   W (channel width solar): {W} pixels")
        print(f"   Ts (spectral translation): {Ts:.2f} pixels")

    def _apply_photometric_calibration(self, flat, visualize):
        """Step 5: Photometric calibration between channels."""
        print("\n[5/7] Applying photometric calibration...")

        # Compute ratios
        photometric_ratios = self.calibration_service.compute_photometric_calibration(
            flat.solar_channels,
            flat.Ts
        )

        # Apply ratios
        self.calibration_service.apply_photometric_calibration(
            flat.solar_channels,
            photometric_ratios
        )

        print(f"   Computed {len(photometric_ratios)} inter-channel ratios")

        if visualize:
            self.visualizer.plot_photometric_ratios(photometric_ratios)

        # Store ratios
        flat.photometric_ratios = photometric_ratios

    def _apply_spectrometric_calibration(self, flat, visualize):
        """Step 6: Spectrometric (wavelength) calibration."""
        print("\n[6/7] Applying spectrometric calibration...")

        # Compute wavelength calibration
        spectral_calibration = self.calibration_service.compute_spectrometric_calibration(
            flat.solar_channels,
            flat.Ts
        )

        ha_channel = spectral_calibration['ha_channel']
        k = spectral_calibration['k']

        print(f"   H-alpha found in channel {ha_channel + 1}")
        print(f"   Wavelength resolution: k = {k:.6f} Å/pixel")

        # Store calibration data
        flat.set_calibration_data(
            flat.photometric_ratios, spectral_calibration)

        if visualize:
            self.visualizer.plot_spectral_profiles(
                flat.solar_channels, spectral_calibration)

    def _save_to_config(self, flat):
        """Step 7: Save computed values to configuration."""
        print("\n[7/7] Saving computed values to configuration...")

        # Save photometric ratios
        ratios_dict = {str(k): float(v)
                       for k, v in flat.photometric_ratios.items()}
        self.config.update_computed_values(photometric_ratios=ratios_dict)

        # Save k if not already saved
        if self.config.should_compute('k'):
            self.config.update_computed_values(k=float(flat.k))

        print("   Configuration updated")

    def apply_flat_to_light(self, flat, light_path, visualize=False):
        """
        Apply flat field corrections to a light frame.
        
        Args:
            flat (Flat): Processed flat field object
            light_path (str): Path to light frame FITS file
            visualize (bool): If True, display results
        
        Returns:
            list: List of corrected SolarChannel objects
        """
        print("\nApplying flat corrections to light frame...")

        # Load light frame
        light_data = Io.load_fits(light_path)
        dark_data = Io.load_fits(flat.dark_path)

        # Preprocess
        light_preprocessed = light_data - dark_data
        light_preprocessed = Computation.median_filter(
            light_preprocessed, size=3)
        light_preprocessed = Computation.linear_stretch(light_preprocessed)

        # Normalize each channel using same geometry as flat
        light_channels = []
        for solar_channel in flat.solar_channels:
            # Create new SolarChannel for light
            light_solar_channel = SolarChannel(
                solar_channel.id,
                solar_channel.index,
                solar_channel.resolution
            )

            # Normalize using same parabolas
            normalized = self.geometry_service.normalize_channel_to_rectangle(
                light_preprocessed,
                solar_channel.parabolas,
                solar_channel.resolution
            )
            light_solar_channel.set_data(normalized)
            light_solar_channel.set_parabolas(solar_channel.parabolas)

            # Copy wavelength calibration
            light_solar_channel.lambda_list = solar_channel.lambda_list.copy()

            light_channels.append(light_solar_channel)

        # Apply photometric calibration
        self.calibration_service.apply_photometric_calibration(
            light_channels,
            flat.photometric_ratios
        )

        if visualize:
            self.visualizer.plot_solar_channels(light_channels)

        return light_channels
