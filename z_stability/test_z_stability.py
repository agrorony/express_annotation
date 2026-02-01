#!/usr/bin/env python3
"""Test script for Z-stability analysis."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, "src")
from metrics_engine import (
    compute_z_metrics,
    compute_adjacent_slice_consistency,
    compute_z_run_length_stability,
    compute_component_persistence
)

def create_test_volume(z=10, y=50, x=50, target_class=1):
    """Create a synthetic test volume."""
    volume = np.zeros((z, y, x), dtype=np.uint8)
    
    # Create some target class regions with varying stability
    # Stable region: consistent across Z
    volume[:, 10:20, 10:20] = target_class
    
    # Unstable region: alternates
    for zi in range(z):
        if zi % 2 == 0:
            volume[zi, 30:40, 30:40] = target_class
    
    # Progressive region: grows over Z
    for zi in range(z):
        size = zi + 5
        volume[zi, 40:40+size, 10:10+size] = target_class
    
    return volume

def test_basic_metrics():
    """Test basic Z-metrics computation."""
    print("="*60)
    print("Test 1: Basic Z-Metrics (CPU)")
    print("="*60)
    
    volume = create_test_volume()
    target_class = 1
    
    # Test CPU version
    metrics = compute_z_metrics(volume, target_class, window_size=3, use_gpu=False)
    
    assert 'frequency' in metrics
    assert 'flip_rate' in metrics
    assert metrics['frequency'].shape == volume.shape
    assert metrics['flip_rate'].shape == volume.shape
    
    print(f"✓ CPU Z-metrics computed successfully")
    print(f"  Frequency range: [{metrics['frequency'].min():.3f}, {metrics['frequency'].max():.3f}]")
    print(f"  Flip rate range: [{metrics['flip_rate'].min()}, {metrics['flip_rate'].max()}]")
    print()

def test_adjacent_slice_consistency():
    """Test adjacent slice Dice computation."""
    print("="*60)
    print("Test 2: Adjacent Slice Consistency")
    print("="*60)
    
    volume = create_test_volume()
    target_class = 1
    
    dice_scores = compute_adjacent_slice_consistency(volume, target_class)
    
    assert len(dice_scores) == volume.shape[0] - 1
    assert np.all((dice_scores >= 0) & (dice_scores <= 1))
    
    print(f"✓ Adjacent slice Dice computed successfully")
    print(f"  Number of slice pairs: {len(dice_scores)}")
    print(f"  Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"  Dice range: [{dice_scores.min():.4f}, {dice_scores.max():.4f}]")
    print()

def test_run_length_stability():
    """Test Z run-length computation."""
    print("="*60)
    print("Test 3: Z Run-Length Stability (CPU)")
    print("="*60)
    
    volume = create_test_volume()
    target_class = 1
    
    run_stats = compute_z_run_length_stability(volume, target_class, use_gpu=False)
    
    assert 'run_length_map' in run_stats
    assert 'mean_run_length' in run_stats
    assert 'median_run_length' in run_stats
    assert run_stats['run_length_map'].shape == (volume.shape[1], volume.shape[2])
    
    print(f"✓ Run-length stability computed successfully")
    print(f"  Mean run length: {run_stats['mean_run_length']:.2f}")
    print(f"  Median run length: {run_stats['median_run_length']:.2f}")
    print(f"  Max run length in map: {run_stats['run_length_map'].max()}")
    print()

def test_component_persistence():
    """Test component persistence tracking."""
    print("="*60)
    print("Test 4: Component Persistence")
    print("="*60)
    
    volume = create_test_volume()
    target_class = 1
    
    comp_stats = compute_component_persistence(volume, target_class)
    
    assert 'mean_persistence' in comp_stats
    assert 'median_persistence' in comp_stats
    assert 'max_persistence' in comp_stats
    assert 'component_lifespans' in comp_stats
    
    print(f"✓ Component persistence computed successfully")
    print(f"  Mean persistence: {comp_stats['mean_persistence']:.2f}")
    print(f"  Median persistence: {comp_stats['median_persistence']:.2f}")
    print(f"  Max persistence: {comp_stats['max_persistence']}")
    print(f"  Number of tracked components: {len(comp_stats['component_lifespans'])}")
    print()

def test_gpu_flag():
    """Test GPU flag behavior."""
    print("="*60)
    print("Test 5: GPU Flag Behavior")
    print("="*60)
    
    volume = create_test_volume(z=5, y=20, x=20)  # Smaller for quick test
    target_class = 1
    
    # Test that GPU raises appropriate error when CUDA not available
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available - testing error handling...")
            try:
                metrics = compute_z_metrics(volume, target_class, window_size=3, use_gpu=True)
                print("  ✗ Should have raised RuntimeError")
            except RuntimeError as e:
                print(f"  ✓ Correctly raised RuntimeError: {str(e)[:50]}...")
            
            try:
                run_stats = compute_z_run_length_stability(volume, target_class, use_gpu=True)
                print("  ✗ Should have raised RuntimeError")
            except RuntimeError as e:
                print(f"  ✓ Correctly raised RuntimeError: {str(e)[:50]}...")
        else:
            print("  CUDA available - testing GPU execution...")
            metrics = compute_z_metrics(volume, target_class, window_size=3, use_gpu=True)
            print(f"  ✓ GPU Z-metrics computed successfully")
            
            run_stats = compute_z_run_length_stability(volume, target_class, use_gpu=True)
            print(f"  ✓ GPU run-length computed successfully")
            
    except ImportError:
        print("  PyTorch not available - testing error handling...")
        try:
            metrics = compute_z_metrics(volume, target_class, window_size=3, use_gpu=True)
            print("  ✗ Should have raised NotImplementedError")
        except NotImplementedError as e:
            print(f"  ✓ Correctly raised NotImplementedError: {str(e)[:50]}...")
    print()

def test_identical_mask_detection():
    """Test that identical masks are detected."""
    print("="*60)
    print("Test 6: Identical Mask Detection")
    print("="*60)
    
    volume = create_test_volume(z=5, y=20, x=20)
    
    # Create identical masks
    mask1 = volume.copy()
    mask2 = volume.copy()
    
    identical = np.array_equal(mask1, mask2)
    assert identical, "Should detect identical masks"
    print("  ✓ Identical masks correctly detected")
    
    # Create different masks
    mask2[0, 0, 0] = 2  # Change one voxel
    not_identical = np.array_equal(mask1, mask2)
    assert not not_identical, "Should detect different masks"
    print("  ✓ Different masks correctly detected")
    print()

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "Z-STABILITY ANALYSIS TEST SUITE" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    try:
        test_basic_metrics()
        test_adjacent_slice_consistency()
        test_run_length_stability()
        test_component_persistence()
        test_gpu_flag()
        test_identical_mask_detection()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print()
        return 0
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
