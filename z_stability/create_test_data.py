#!/usr/bin/env python3
"""Create synthetic test data for the pipeline."""
import numpy as np
from pathlib import Path
from PIL import Image

def create_synthetic_scan(output_folder, z_slices=20, height=100, width=100):
    """Create a synthetic scan with masks."""
    print(f"Creating synthetic scan in {output_folder}...")
    
    mask_folder = Path(output_folder) / "masks"
    mask_folder.mkdir(parents=True, exist_ok=True)
    
    for z in range(z_slices):
        # Create a mask with three classes (0, 1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Background (class 0) - default
        
        # Stable target region (class 1)
        mask[20:40, 20:40] = 1
        
        # Another stable region (class 2)
        mask[60:80, 60:80] = 2
        
        # Unstable target region - flickers
        if z % 2 == 0:
            mask[20:30, 60:70] = 1
        
        # Small components (will be filtered by correction)
        if z in [5, 6, 7]:
            mask[50:52, 50:52] = 1
        
        # Growing component
        size = min(10 + z, 30)
        mask[70:70+size, 20:20+size] = 1
        
        # Save slice
        img = Image.fromarray(mask)
        img.save(mask_folder / f"slice_{z:04d}.png")
    
    print(f"  Created {z_slices} mask slices")
    return str(output_folder)

if __name__ == "__main__":
    scan_path = create_synthetic_scan("data/test_scan", z_slices=20, height=100, width=100)
    print(f"Synthetic scan created at: {scan_path}")
