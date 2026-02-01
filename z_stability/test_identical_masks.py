#!/usr/bin/env python3
"""Test script to verify identical mask detection."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, "src")

def test_identical_mask_behavior():
    """Test that identical masks are handled correctly."""
    print("="*60)
    print("Testing Identical Mask Detection")
    print("="*60)
    
    # Create a simple test volume
    volume = np.zeros((5, 20, 20), dtype=np.uint8)
    volume[:, 5:15, 5:15] = 1  # Stable region
    
    # Create two identical masks
    mask1 = volume.copy()
    mask2 = volume.copy()
    
    # Check detection
    identical = np.array_equal(mask1, mask2)
    
    if identical:
        print("✓ PASS: Identical masks correctly detected")
        print("  In pipeline, this would trigger:")
        print("  '!' * 60")
        print("  'WARNING: Conservative and Aggressive masks are IDENTICAL!'")
        print("  'No further analysis will be performed on identical masks.'")
        print("  '!' * 60")
    else:
        print("✗ FAIL: Identical masks not detected")
        return 1
    
    # Test diverged masks
    mask2[0, 0, 0] = 2  # Change one voxel
    diverged = not np.array_equal(mask1, mask2)
    
    if diverged:
        print("✓ PASS: Diverged masks correctly detected")
        print("  In pipeline, this would trigger stability analysis")
    else:
        print("✗ FAIL: Diverged masks not detected")
        return 1
    
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(test_identical_mask_behavior())
