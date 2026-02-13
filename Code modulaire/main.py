"""
Main entry point for MSDP data processing
Uses the new modular architecture with Pipeline orchestration
"""

import os
import numpy as np
from infrastructure.config_manager import ConfigManager
from infrastructure.io import Io
from pipeline.msdp_pipeline import MSDPPipeline
from visualization.visualizer import MSDPVisualizer
from services.detection_service import DetectionService
from services.calibration_service import CalibrationService


def main():
    """
    Main processing workflow using the modular pipeline.
    
    This demonstrates the new architecture:
    1. ConfigManager handles all parameters
    2. Pipeline orchestrates services
    3. Services perform specialized tasks
    4. Domain objects contain only data
    5. Visualizer handles all plotting
    """

    print("=" * 70)
    print("MSDP Data Processing - Modular Architecture")
    print("=" * 70)

    # ===== STEP 1: CONFIGURATION =====
    print("\n[Configuration]")
    config_path = "config.yml"
    config = ConfigManager(config_path)
    print(f"Configuration loaded from: {config_path}")

    # ===== STEP 2: PREPARE MASTER CALIBRATION FRAMES =====
    print("\n[Master Calibration Frames]")

    # Check if masters already exist
    dark_path = 'Code modulaire/data/dark.fits'
    flat_path = 'Code modulaire/data/flat.fits'

    # ===== STEP 3: INITIALIZE PIPELINE =====
    print("\n[Pipeline Initialization]")

    # Create visualizer (optional - set to None to disable all plots)
    visualizer = MSDPVisualizer(config)

    # Create pipeline with all services
    pipeline = MSDPPipeline(config, visualizer)
    print("Pipeline initialized with:")
    print("   - DetectionService")
    print("   - GeometryService")
    print("   - CalibrationService")
    print("   - MSDPVisualizer")

    # ===== STEP 4: PROCESS FLAT FIELD =====
    print("\n[Flat Field Processing]")

    # Process flat with full pipeline
    # Set visualize=True to see intermediate plots
    # Set visualize=False for batch processing
    flat = pipeline.process_flat_field(
        flat_path=flat_path,
        dark_path=dark_path,
        num_channels=9,
        visualize=True  # Set to True to see plots
    )

    # ===== STEP 5: DISPLAY RESULTS =====
    print("\n[Calibration Results]")
    print(f"Geometric parameters:")
    print(f"   Wij  = {flat.Wij:.2f} pixels (channel width in CCD)")
    print(f"   Tgij = {flat.Tgij:.2f} pixels (channel spacing in CCD)")
    print(f"   W    = {flat.W} pixels (channel width in solar coords)")
    print(f"   Ts   = {flat.Ts:.2f} pixels (spectral translation)")
    print(f"   k    = {flat.k:.6f} Å/pixel (wavelength calibration)")

    print(f"\nPhotometric ratios:")
    for idx, ratio in flat.photometric_ratios.items():
        print(f"   Channel {idx} → {idx+1}: {ratio:.4f}")

    # Display solar channels
    print("\n[Visualization]")
    visualizer.plot_solar_channels(flat.solar_channels)

    # ===== STEP 6: PROCESS LIGHT FRAMES (OPTIONAL) =====
    print("\n[Light Frame Processing]")

    # Find all light frames
    lights_dir = "Code modulaire/data/lights"
    if os.path.exists(lights_dir):
        light_files = [f for f in os.listdir(lights_dir)
                       if f.endswith('.fit') or f.endswith('.fits')]

        if light_files:
            print(f"Found {len(light_files)} light frames")

            # Process first light frame as example
            example_light = os.path.join(lights_dir, light_files[0])
            print(f"Processing example: {example_light}")

            light_channels = pipeline.apply_flat_to_light(
                flat,
                example_light,
                visualize=True
            )

            print(f"✓ Processed {len(light_channels)} channels")
        else:
            print(f"No light frames found in {lights_dir}")
    else:
        print(
            f"Directory '{lights_dir}' not found - skipping light processing")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"\nProcessed data stored in Flat object:")
    print(f"   - {len(flat.channels)} channels in CCD coordinates")
    print(f"   - {len(flat.solar_channels)} channels in solar coordinates")
    print(
        f"   - Photometric calibration: {len(flat.photometric_ratios)} ratios")
    print(f"   - Wavelength calibration: k = {flat.k:.6f} Å/pixel")
    print(f"\nConfiguration saved to: {config_path}")
    print("\nTo access processed data:")
    print("   flat.solar_channels[i].data          # Channel image data")
    print("   flat.solar_channels[i].lambda_list   # Wavelength calibration")
    print()


def demonstrate_modular_architecture():
    """
    Demonstrate the benefits of the new modular architecture.
    Shows how to use individual services independently.
    """
    print("\n" + "=" * 70)
    print("Architecture Demonstration - Using Services Independently")
    print("=" * 70)

    # Example 1: Use only DetectionService
    print("\nExample 1: Edge Detection Only")
    config = ConfigManager()
    detection_service = DetectionService(config)

    # Load an image
    flat_data = Io.load_fits('master/Flats.fits')
    dark_data = Io.load_fits('master/Darks.fits')
    image = flat_data - dark_data

    # Detect edges
    points = detection_service.detect_all_channel_edges(image)
    print(f"Detected {len(points['as_'])} channels")

    # Example 2: Use only CalibrationService
    print("\nExample 2: Calibration Only (with existing SolarChannels)")
    calibration_service = CalibrationService(config)
    # (Would need pre-processed solar channels here)

    print("\nThese examples show how services can be used independently")
    print("for testing, debugging, or custom workflows.")


if __name__ == "__main__":
    # Run main processing
    main()

    # Optionally demonstrate architecture
    demonstrate_modular_architecture()
