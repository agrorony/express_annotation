import yaml
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "src")

from io_utils import load_image_stack, load_mask_stack, save_mask, save_diagnostic_map, get_scan_name
from metrics_engine import (compute_z_metrics, compute_2d_metrics_stack, 
                             compute_adjacent_slice_consistency, compute_z_run_length_stability,
                             compute_component_persistence)
from correction_logic import apply_conservative_correction, apply_aggressive_correction

def check_masks_identical(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Check if two masks are identical."""
    return np.array_equal(mask1, mask2)

def print_stability_summary(mask: np.ndarray, target_class: int, mask_type: str, use_gpu: bool = False):
    """Print concise stability summary for a mask."""
    print(f"\n{'='*60}")
    print(f"Z-Stability Analysis for {mask_type.upper()}")
    print(f"{'='*60}")
    
    # Adjacent slice consistency
    dice_scores = compute_adjacent_slice_consistency(mask, target_class)
    mean_dice = np.mean(dice_scores)
    print(f"Adjacent Slice Consistency (Dice): mean={mean_dice:.4f}, min={np.min(dice_scores):.4f}, max={np.max(dice_scores):.4f}")
    
    # Run-length stability
    run_stats = compute_z_run_length_stability(mask, target_class, use_gpu=use_gpu)
    print(f"Z Run-Length Stability: mean={run_stats['mean_run_length']:.2f}, median={run_stats['median_run_length']:.2f}")
    
    # Component persistence
    comp_stats = compute_component_persistence(mask, target_class)
    print(f"Component Persistence: mean={comp_stats['mean_persistence']:.2f}, median={comp_stats['median_persistence']:.2f}, max={comp_stats['max_persistence']}")
    print(f"{'='*60}\n")
    
    return {
        'mean_dice': mean_dice,
        'dice_scores': dice_scores,
        'run_length_stats': run_stats,
        'component_stats': comp_stats
    }

def save_stability_metrics(metrics: dict, output_path: Path, mask_type: str):
    """Save detailed stability metrics to files."""
    metrics_dir = output_path / f"stability_metrics_{mask_type}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Dice scores
    np.save(metrics_dir / "adjacent_slice_dice.npy", metrics['dice_scores'])
    np.savetxt(metrics_dir / "adjacent_slice_dice.csv", metrics['dice_scores'], delimiter=',', fmt='%.6f')
    
    # Save run-length map
    np.save(metrics_dir / "run_length_map.npy", metrics['run_length_stats']['run_length_map'])
    
    print(f"Saved stability metrics to {metrics_dir}")

def main():
    print("Starting Pipeline...")
    with open("configs/pipeline.yaml", 'r') as f: config = yaml.safe_load(f)
    
    scan_folder = config['data']['scan_folder']
    use_gpu = config['compute']['use_gpu']
    target_class = config['target_class']
    
    print(f"Processing: {scan_folder}")
    print(f"GPU Acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
    
    # Load
    mask_volume = load_mask_stack(str(Path(scan_folder)/"masks"))
    
    # Metrics (with GPU flag propagated)
    print("\nComputing Z-axis metrics...")
    m_short = compute_z_metrics(mask_volume, target_class, config['windows']['short'], use_gpu=use_gpu)
    m_long = compute_z_metrics(mask_volume, target_class, config['windows']['long'], use_gpu=use_gpu)
    print("Computing 2D spatial metrics...")
    m_2d = compute_2d_metrics_stack(mask_volume, target_class, use_gpu=use_gpu)
    
    # Correction
    out_path = Path(config['output']['output_folder']) / get_scan_name(scan_folder)
    
    print("\nApplying corrections...")
    mask_cons, conf_cons = apply_conservative_correction(mask_volume, m_short, m_long, m_2d, config['thresholds']['conservative'])
    save_mask(mask_cons, str(out_path / "mask_conservative"))
    save_diagnostic_map(conf_cons, str(out_path / "conf_conservative"), colormap='hot')
    
    mask_agg, conf_agg = apply_aggressive_correction(mask_volume, m_short, m_long, m_2d, config['thresholds']['aggressive'])
    save_mask(mask_agg, str(out_path / "mask_aggressive"))
    
    # Check if masks are identical
    masks_identical = check_masks_identical(mask_cons, mask_agg)
    if masks_identical:
        print("\n" + "!"*60)
        print("WARNING: Conservative and Aggressive masks are IDENTICAL!")
        print("No further analysis will be performed on identical masks.")
        print("!"*60 + "\n")
    else:
        print("\nMasks diverged - performing Z-stability analysis...")
        
        # Compute and report stability metrics for both masks
        cons_metrics = print_stability_summary(mask_cons, target_class, "conservative", use_gpu=use_gpu)
        agg_metrics = print_stability_summary(mask_agg, target_class, "aggressive", use_gpu=use_gpu)
        
        # Save detailed metrics if requested
        if config['output'].get('save_diagnostic_maps', False):
            save_stability_metrics(cons_metrics, out_path, "conservative")
            save_stability_metrics(agg_metrics, out_path, "aggressive")
    
    print("Done!")

if __name__ == "__main__":
    main()
