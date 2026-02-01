import yaml
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "src")

from io_utils import load_image_stack, load_mask_stack, save_mask, save_diagnostic_map, get_scan_name
from metrics_engine import compute_z_metrics, compute_2d_metrics_stack
from correction_logic import apply_conservative_correction, apply_aggressive_correction

def main():
    print("Starting Pipeline...")
    with open("configs/pipeline.yaml", 'r') as f: config = yaml.safe_load(f)
    
    scan_folder = config['data']['scan_folder']
    print(f"Processing: {scan_folder}")
    
    # Load
    mask_volume = load_mask_stack(str(Path(scan_folder)/"masks"))
    
    # Metrics
    m_short = compute_z_metrics(mask_volume, config['target_class'], config['windows']['short'])
    m_long = compute_z_metrics(mask_volume, config['target_class'], config['windows']['long'])
    m_2d = compute_2d_metrics_stack(mask_volume, config['target_class'])
    
    # Correction
    out_path = Path(config['output']['output_folder']) / get_scan_name(scan_folder)
    
    mask_cons, conf_cons = apply_conservative_correction(mask_volume, m_short, m_long, m_2d, config['thresholds']['conservative'])
    mask_agg, conf_agg = apply_aggressive_correction(mask_volume, m_short, m_long, m_2d, config['thresholds']['aggressive'])
    
    # Validation: ensure masks are different
    assert not np.array_equal(mask_cons, mask_agg), "ERROR: Conservative and aggressive masks are identical!"
    
    # Log statistics
    print(f"\n=== MASK STATISTICS ===")
    print(f"Conservative: unique={np.unique(mask_cons)}, sum={np.sum(mask_cons)}, voxels={mask_cons.size}")
    print(f"Aggressive:   unique={np.unique(mask_agg)}, sum={np.sum(mask_agg)}, voxels={mask_agg.size}")
    diff_count = np.sum(mask_cons != mask_agg)
    print(f"Difference:   {diff_count} voxels differ ({100*diff_count/mask_cons.size:.2f}%)")
    
    # Save
    save_mask(mask_cons, str(out_path / "mask_conservative"))
    save_diagnostic_map(conf_cons, str(out_path / "conf_conservative"), colormap='hot')
    save_mask(mask_agg, str(out_path / "mask_aggressive"))
    save_diagnostic_map(conf_agg, str(out_path / "conf_aggressive"), colormap='hot')
    
    print("\nDone!")

if __name__ == "__main__":
    main()
