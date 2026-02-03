"""
Example: Complete PSD Calculation Workflow
===========================================

This script demonstrates the full pipeline for computing PSD from a 3D micro-CT scan.

Workflow:
    1. Load segmented volume (or create synthetic test data)
    2. Configure pipeline parameters
    3. Compute PSD with checkpointing
    4. Export results
    5. Visualize PSD curves
"""

import numpy as np
from pathlib import Path

# Import PSD pipeline
from pores_analysis import (
    compute_psd,
    psd_to_dataframe,
    save_psd_dataframe,
    plot_psd
)


def example_synthetic_data():
    """
    Example 1: Synthetic test volume
    
    Creates a volume with multiple spherical pores of varying sizes.
    This demonstrates the pipeline without requiring actual μCT data.
    """
    print("=" * 70)
    print("EXAMPLE 1: Synthetic Test Volume")
    print("=" * 70)
    
    # Generate synthetic volume (128³ with random spheres)
    size = 128
    volume = np.zeros((size, size, size), dtype=bool)
    
    # Add several spheres of different sizes
    spheres = [
        ((40, 64, 64), 20),   # Large sphere
        ((88, 64, 64), 12),   # Medium sphere
        ((64, 40, 40), 8),    # Small sphere
        ((64, 88, 88), 15),   # Another medium sphere
    ]
    
    for center, radius in spheres:
        cz, cy, cx = center
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt(
                        (z - cz)**2 + 
                        (y - cy)**2 + 
                        (x - cx)**2
                    )
                    if dist <= radius:
                        volume[z, y, x] = True
    
    print(f"Volume shape: {volume.shape}")
    print(f"Porosity: {volume.mean():.4f}")
    print(f"Expected pore diameters: 40, 24, 16, 30 voxels")
    
    # Compute PSD (CPU mode for compatibility)
    print("\nComputing PSD...")
    psd = compute_psd(
        volume,
        voxel_spacing=(1.0, 1.0, 1.0),  # Isotropic voxels
        use_gpu=False,                  # CPU mode
        use_chunking=False              # No chunking for small volume
    )
    
    # Convert to DataFrame
    df = psd_to_dataframe(psd)
    
    # Display results
    print("\nPSD Results:")
    print(df.head(10))
    print(f"\nTotal bins: {len(df)}")
    print(f"Reliable bins: {df['is_reliable'].sum()}/{len(df)}")
    
    # Save results
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    save_psd_dataframe(df, output_dir / "synthetic_psd.csv")
    print(f"\nResults saved to {output_dir}/synthetic_psd.csv")
    
    # Plot (requires matplotlib)
    try:
        plot_psd(df, save_path=output_dir / "synthetic_psd_plot.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")


def example_real_data_with_chunking():
    """
    Example 2: Large volume with block processing and checkpointing
    
    Demonstrates the recommended workflow for processing large μCT scans
    in Google Colab with GPU acceleration and checkpoint resilience.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Large Volume with Chunking & Checkpointing")
    print("=" * 70)
    print("\n⚠️  This example shows the recommended configuration.")
    print("    Uncomment and adapt for your actual data.\n")
    
    # Example configuration (commented out - adapt for your data)
    code_example = '''
# Load your segmented volume
# volume = np.load('/content/drive/MyDrive/scans/scan_001_segmented.npy')
# 
# # Or load from TIFF stack
# import tifffile
# volume = tifffile.imread('/path/to/stack/*.tif') > threshold
# volume = volume.astype(bool)

# Configure pipeline for large volumes
psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),  # Anisotropic spacing (Z, Y, X) in μm
    use_gpu=True,                   # GPU acceleration via CuPy
    use_chunking=True,              # Enable block processing
    chunk_size=(128, 128, 128),     # Adjust based on GPU memory
    halo_width=50,                  # >= max expected pore diameter
    checkpoint_dir='/content/drive/MyDrive/psd_checkpoints',
    run_id='scan_001_psd',
    resume=True                     # Resume from checkpoint if interrupted
)

# Export results
df = psd_to_dataframe(psd)
save_psd_dataframe(
    df,
    '/content/drive/MyDrive/results/scan_001_psd.csv',
    metadata={
        'scan_id': 'scan_001',
        'sample_name': 'soil_core_A1',
        'acquisition_date': '2026-02-03',
        'notes': 'Agricultural soil, 0-5cm depth'
    }
)

# Filter reliable data
reliable_df = df[df['is_reliable']]
print(f"Reliable data: {len(reliable_df)}/{len(df)} bins")

# Plot
plot_psd(df, save_path='/content/drive/MyDrive/results/scan_001_plot.png')
'''
    
    print(code_example)


def example_anisotropic_voxels():
    """
    Example 3: Anisotropic voxel spacing
    
    Demonstrates handling of non-cubic voxels (common in μCT).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Anisotropic Voxel Spacing")
    print("=" * 70)
    
    # Create synthetic volume
    size = 64
    volume = np.zeros((size, size, size), dtype=bool)
    
    # Single sphere
    center = (32, 32, 32)
    radius = 15
    
    for z in range(size):
        for y in range(size):
            for x in range(size):
                dist = np.sqrt(
                    (z - center[0])**2 + 
                    (y - center[1])**2 + 
                    (x - center[2])**2
                )
                if dist <= radius:
                    volume[z, y, x] = True
    
    # Anisotropic voxel spacing (Z spacing 2× larger than X/Y)
    voxel_spacing = (2.0, 1.0, 1.0)  # μm
    
    print(f"Volume shape: {volume.shape}")
    print(f"Voxel spacing: {voxel_spacing} μm")
    print(f"Physical dimensions: {size*2:.1f} × {size*1:.1f} × {size*1:.1f} μm³")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=voxel_spacing,
        use_gpu=False,
        use_chunking=False
    )
    
    df = psd_to_dataframe(psd)
    
    print("\nPeak diameter:")
    peak_idx = df['Volume_Count'].argmax()
    print(f"  Voxel units: {df.loc[peak_idx, 'Diameter_px']:.2f} voxels")
    print(f"  Physical units: {df.loc[peak_idx, 'Diameter_um']:.2f} μm")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PSD CALCULATOR - EXAMPLE WORKFLOWS")
    print("=" * 70)
    
    # Example 1: Synthetic data (executable)
    example_synthetic_data()
    
    # Example 2: Large volume workflow (template)
    example_real_data_with_chunking()
    
    # Example 3: Anisotropic voxels
    example_anisotropic_voxels()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Adapt Example 2 for your actual data")
    print("  2. Run validation tests: python test_psd_synthetic.py")
    print("  3. Review output DataFrames for reliability flags")
    print("\nDocumentation: See pores_analysis/README.md")


if __name__ == "__main__":
    main()
