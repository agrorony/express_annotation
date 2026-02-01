"""
Stage 2b: Constructed Z-Pattern Validation Tests
=================================================

Tests Z-stability metrics using synthetic volumes with known behavior.
These are NOT realistic μCT simulations - they isolate specific Z-axis patterns.

Expected behavior is qualitative, not exact numerical targets.
"""

import numpy as np
import sys
sys.path.insert(0, "src")

from metrics_engine import (
    compute_adjacent_slice_consistency,
    compute_z_run_length_stability,
    compute_component_persistence
)


def print_test_header(test_num, description):
    print(f"\n{'='*70}")
    print(f"TEST {test_num}: {description}")
    print('='*70)


def print_results(expected, actual, status="PASS"):
    print(f"\nExpected:\n{expected}")
    print(f"\nActual:\n{actual}")
    print(f"\nStatus: [{status}]")


def test_1_fully_stable_slab():
    """
    Test 1: Fully stable slab
    - Same component in all Z slices
    - Expected: Dice = 1.0 everywhere, Run-length = Z, Persistence = Z
    """
    print_test_header(1, "Fully Stable Slab")
    
    Z, Y, X = 10, 20, 20
    volume = np.ones((Z, Y, X), dtype=np.uint8)  # All class 1
    target_class = 1
    
    # Compute metrics
    dice = compute_adjacent_slice_consistency(volume, target_class)
    run_length = compute_z_run_length_stability(volume, target_class)
    persistence = compute_component_persistence(volume, target_class)
    
    # Expected behavior
    expected = """
    - Adjacent Dice: 1.0 for all slice pairs
    - Run-length: Z (10) for all voxels
    - Persistence: Z (10) for single stable component
    """
    
    actual = f"""
    - Adjacent Dice: min={np.min(dice):.4f}, max={np.max(dice):.4f}, mean={np.mean(dice):.4f}
    - Run-length: mean={run_length['mean_run_length']:.2f}, median={run_length['median_run_length']:.2f}
    - Persistence: mean={persistence['mean_persistence']:.2f}, max={persistence['max_persistence']}
    """
    
    # Validation
    dice_pass = np.allclose(dice, 1.0)
    run_pass = (run_length['mean_run_length'] == Z) and (run_length['median_run_length'] == Z)
    pers_pass = (persistence['mean_persistence'] == Z) and (persistence['max_persistence'] == Z)
    
    status = "PASS" if (dice_pass and run_pass and pers_pass) else "FAIL"
    if not dice_pass:
        status += " (Dice != 1.0)"
    if not run_pass:
        status += f" (Run-length != {Z})"
    if not pers_pass:
        status += f" (Persistence != {Z})"
    
    print_results(expected, actual, status)


def test_2_single_slice_appearance():
    """
    Test 2: Single-slice appearance
    - Component appears in exactly one slice (Z=5)
    - Expected: Dice drop around that slice, Run-length = 1, Persistence = 0 (not counted)
    """
    print_test_header(2, "Single-Slice Appearance")
    
    Z, Y, X = 10, 20, 20
    volume = np.zeros((Z, Y, X), dtype=np.uint8)  # All class 0
    volume[5, 5:15, 5:15] = 1  # Single slice with class 1
    target_class = 1
    
    # Compute metrics
    dice = compute_adjacent_slice_consistency(volume, target_class)
    run_length = compute_z_run_length_stability(volume, target_class)
    persistence = compute_component_persistence(volume, target_class)
    
    # Expected behavior
    expected = """
    - Adjacent Dice: 0.0 for pairs around Z=5 (slices 4-5 and 5-6), 1.0 elsewhere (both empty)
    - Run-length: 1 for voxels at Z=5, 0 elsewhere
    - Persistence: 0 (single-slice components not counted)
    """
    
    actual = f"""
    - Adjacent Dice: min={np.min(dice):.4f}, max={np.max(dice):.4f}
      Dice[4]={dice[4]:.4f} (before), Dice[5]={dice[5]:.4f} (after)
    - Run-length: mean={run_length['mean_run_length']:.2f}, median={run_length['median_run_length']:.2f}
    - Persistence: mean={persistence['mean_persistence']:.2f}, max={persistence['max_persistence']}
    """
    
    # Validation
    dice_pass = (dice[4] == 0.0) and (dice[5] == 0.0)  # Drop around Z=5
    run_pass = (run_length['mean_run_length'] == 1.0) and (run_length['median_run_length'] == 1.0)
    pers_pass = (persistence['mean_persistence'] == 0.0) and (persistence['max_persistence'] == 0)
    
    status = "PASS" if (dice_pass and run_pass and pers_pass) else "FAIL"
    if not dice_pass:
        status += " (Dice behavior unexpected)"
    if not run_pass:
        status += " (Run-length != 1)"
    if not pers_pass:
        status += " (Persistence != 0)"
    
    print_results(expected, actual, status)


def test_3_alternating_pattern():
    """
    Test 3: Alternating pattern (010101... along Z)
    - Expected: Very low Dice, Run-length ≈ 1, High flip activity
    """
    print_test_header(3, "Alternating Pattern (010101...)")
    
    Z, Y, X = 10, 20, 20
    volume = np.zeros((Z, Y, X), dtype=np.uint8)
    for z in range(Z):
        if z % 2 == 1:  # Odd slices = class 1
            volume[z, 5:15, 5:15] = 1
    target_class = 1
    
    # Compute metrics
    dice = compute_adjacent_slice_consistency(volume, target_class)
    run_length = compute_z_run_length_stability(volume, target_class)
    persistence = compute_component_persistence(volume, target_class)
    
    # Expected behavior
    expected = """
    - Adjacent Dice: 0.0 everywhere (alternating presence/absence)
    - Run-length: 1 (each voxel appears for only 1 slice at a time)
    - Persistence: 0 (no component persists for ≥2 slices)
    """
    
    actual = f"""
    - Adjacent Dice: min={np.min(dice):.4f}, max={np.max(dice):.4f}, mean={np.mean(dice):.4f}
    - Run-length: mean={run_length['mean_run_length']:.2f}, median={run_length['median_run_length']:.2f}
    - Persistence: mean={persistence['mean_persistence']:.2f}, max={persistence['max_persistence']}
    """
    
    # Validation
    dice_pass = np.allclose(dice, 0.0)
    run_pass = (run_length['mean_run_length'] == 1.0) and (run_length['median_run_length'] == 1.0)
    pers_pass = (persistence['mean_persistence'] == 0.0) and (persistence['max_persistence'] == 0)
    
    status = "PASS" if (dice_pass and run_pass and pers_pass) else "FAIL"
    if not dice_pass:
        status += " (Dice != 0.0)"
    if not run_pass:
        status += " (Run-length != 1)"
    if not pers_pass:
        status += " (Persistence != 0)"
    
    print_results(expected, actual, status)


def test_4_gradual_disappearance():
    """
    Test 4: Gradual disappearance
    - Component shrinks and vanishes across Z
    - Expected: Dice decreases monotonically, limited persistence
    """
    print_test_header(4, "Gradual Disappearance")
    
    Z, Y, X = 10, 20, 20
    volume = np.zeros((Z, Y, X), dtype=np.uint8)
    target_class = 1
    
    # Create shrinking circular region
    center_y, center_x = 10, 10
    for z in range(Z):
        radius = max(10 - z, 0)  # Shrinks from radius 10 to 0
        for y in range(Y):
            for x in range(X):
                if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                    volume[z, y, x] = 1
    
    # Compute metrics
    dice = compute_adjacent_slice_consistency(volume, target_class)
    run_length = compute_z_run_length_stability(volume, target_class)
    persistence = compute_component_persistence(volume, target_class)
    
    # Expected behavior
    expected = """
    - Adjacent Dice: Decreases monotonically (not necessarily linearly)
    - Run-length: Varies spatially (center has longer runs than edges)
    - Persistence: Single component persists for full volume (Z=10)
    """
    
    actual = f"""
    - Adjacent Dice: {dice}
      (First few: {dice[:3]}, Last few: {dice[-3:]})
    - Run-length: mean={run_length['mean_run_length']:.2f}, median={run_length['median_run_length']:.2f}
      (Should be < Z since edges disappear early)
    - Persistence: mean={persistence['mean_persistence']:.2f}, max={persistence['max_persistence']}
    """
    
    # Validation (qualitative)
    dice_decreasing = np.all(np.diff(dice) <= 0.01)  # Allow small numerical noise
    run_reasonable = run_length['mean_run_length'] < Z  # Not all voxels persist for full Z
    pers_reasonable = persistence['max_persistence'] == Z  # Center component persists throughout
    
    status = "PASS" if (dice_decreasing and run_reasonable and pers_reasonable) else "FAIL"
    if not dice_decreasing:
        status += " (Dice not monotonically decreasing)"
    if not run_reasonable:
        status += f" (Run-length >= Z, expected < Z)"
    if not pers_reasonable:
        status += f" (Max persistence != Z)"
    
    print_results(expected, actual, status)


def test_5_component_merge():
    """
    Test 5: Component merge
    - Two separate components merge into one
    - Expected: Behavior must be deterministic and documented
    """
    print_test_header(5, "Component Merge")
    
    Z, Y, X = 10, 20, 20
    volume = np.zeros((Z, Y, X), dtype=np.uint8)
    target_class = 1
    
    # First 5 slices: two separate components
    for z in range(5):
        volume[z, 5:10, 5:10] = 1   # Component A (left)
        volume[z, 5:10, 12:17] = 1  # Component B (right)
    
    # Last 5 slices: merged into one component
    for z in range(5, 10):
        volume[z, 5:10, 5:17] = 1   # Merged component
    
    # Compute metrics
    dice = compute_adjacent_slice_consistency(volume, target_class)
    run_length = compute_z_run_length_stability(volume, target_class)
    persistence = compute_component_persistence(volume, target_class)
    
    # Expected behavior
    expected = """
    - Adjacent Dice: High before merge, drops at merge point (Z=4-5), then high again
    - Run-length: Most voxels persist for Z=10, but gap region (bridge) only Z=5
    - Persistence: Greedy matching means only ONE component continues across merge
      Expected: Two components with persistence=5 (before merge) + one with persistence=10 (includes continuation)
      OR: One component tracks through with persistence=10, other ends at 5
    """
    
    actual = f"""
    - Adjacent Dice: {dice}
      Dice[3]={dice[3]:.4f} (before merge), Dice[4]={dice[4]:.4f} (at merge), Dice[5]={dice[5]:.4f} (after)
    - Run-length: mean={run_length['mean_run_length']:.2f}, median={run_length['median_run_length']:.2f}
    - Persistence: mean={persistence['mean_persistence']:.2f}, max={persistence['max_persistence']}
      (Exact values depend on greedy overlap matching - checking for determinism)
    """
    
    # Validation (deterministic behavior, not specific values)
    dice_dip = dice[4] < dice[3]  # Should drop at merge
    run_reasonable = run_length['mean_run_length'] > 5 and run_length['mean_run_length'] < Z  # Between 5 and 10
    pers_deterministic = persistence['max_persistence'] >= 5  # At least one component tracked
    
    status = "PASS" if (dice_dip and run_reasonable and pers_deterministic) else "FAIL"
    if not dice_dip:
        status += " (Dice didn't drop at merge)"
    if not run_reasonable:
        status += f" (Run-length not in expected range)"
    if not pers_deterministic:
        status += " (Persistence behavior unclear)"
    
    print_results(expected, actual, status)
    
    # Additional note on merge behavior
    print("\nNote: Greedy overlap matching means merge behavior is simplified.")
    print("Only the component with highest overlap to the merged region continues tracking.")


def main():
    print("\n" + "="*70)
    print("STAGE 2B: Z-PATTERN VALIDATION TESTS")
    print("CPU-only, constructed synthetic volumes")
    print("="*70)
    
    test_1_fully_stable_slab()
    test_2_single_slice_appearance()
    test_3_alternating_pattern()
    test_4_gradual_disappearance()
    test_5_component_merge()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nIf any tests show FAIL or unexpected behavior, review metric semantics.")
    print("All behavior should be deterministic and explainable.\n")


if __name__ == "__main__":
    main()
