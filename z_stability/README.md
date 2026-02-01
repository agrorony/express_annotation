# Z-Stability Analysis Tool

A research-oriented tool for analyzing Z-axis stability in 3D segmentation masks.

## Features

### 1. Z-Axis Stability Metrics
- **Frequency analysis**: Per-voxel class frequency within sliding Z-windows
- **Flip rate**: Detection of class transitions along Z-axis
- **Class fractions**: Distribution of competing classes

### 2. Advanced Stability Analysis
- **Adjacent-slice consistency**: Dice coefficient between consecutive slices
- **Z run-length stability**: Longest continuous runs per voxel
- **Component persistence**: Tracking of connected components across slices

### 3. GPU Acceleration
- Optional GPU acceleration using PyTorch
- Implemented for Z-axis voxel metrics and run-length operations
- Automatic error handling when GPU unavailable
- Connected-component analysis remains on CPU for optimal performance

### 4. Mask Correction Strategies
- **Conservative correction**: Requires all instability indicators
- **Aggressive correction**: Triggers on any instability indicator
- **Identical mask detection**: Warns when strategies produce identical results

## Installation

```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install torch
```

## Usage

### Basic Usage

```bash
cd z_stability
python run_pipeline.py
```

### Configuration

Edit `configs/pipeline.yaml`:

```yaml
data:
  scan_folder: "data/test_scan"
  
windows:
  short: 3    # Short-window size for local analysis
  long: 7     # Long-window size for broader context
  
target_class: 1  # Segmentation class to analyze

compute:
  use_gpu: false  # Enable GPU acceleration (requires PyTorch + CUDA)

thresholds:
  conservative:
    min_short_frequency: 0.4
    max_flip_rate: 3
    min_long_frequency: 0.2
    min_component_size: 5
  aggressive:
    min_short_frequency: 0.6
    max_flip_rate: 2
    min_long_frequency: 0.3
    min_component_size: 10

output:
  save_diagnostic_maps: true
  output_folder: "outputs"
```

## Output

### Corrected Masks
- `mask_conservative/`: Conservative correction results
- `mask_aggressive/`: Aggressive correction results
- `conf_conservative/`: Confidence maps for corrections

### Stability Metrics
When masks diverge, detailed stability metrics are generated:

- `stability_metrics_conservative/`
  - `adjacent_slice_dice.csv`: Dice scores for each slice pair
  - `adjacent_slice_dice.npy`: NumPy array of Dice scores
  - `run_length_map.npy`: Maximum run length per voxel

- `stability_metrics_aggressive/`
  - (Same structure as conservative)

### Console Output

```
============================================================
Z-Stability Analysis for CONSERVATIVE
============================================================
Adjacent Slice Consistency (Dice): mean=0.9155, min=0.8921, max=0.9378
Z Run-Length Stability: mean=12.18, median=13.00
Component Persistence: mean=4.00, median=1.00, max=20
============================================================
```

## Testing

Run the test suite:

```bash
# Test all metrics
python test_z_stability.py

# Test GPU error handling
python test_gpu_handling.py

# Test identical mask detection
python test_identical_masks.py

# Create synthetic test data
python create_test_data.py
```

## API Reference

### Core Functions

#### `compute_z_metrics(volume, target_class, window_size, use_gpu=False)`
Compute Z-axis stability metrics.

**Parameters:**
- `volume`: 3D numpy array (Z, Y, X)
- `target_class`: Target segmentation class
- `window_size`: Sliding window size for local analysis
- `use_gpu`: Enable GPU acceleration (requires PyTorch + CUDA)

**Returns:**
Dictionary with:
- `frequency`: Per-voxel frequency of target class
- `flip_rate`: Number of class transitions
- `class_0_fraction`, `class_2_fraction`: Competing class fractions
- `effective_window_size`: Actual window size used (varies at boundaries)

#### `compute_adjacent_slice_consistency(mask, target_class)`
Compute Dice coefficient between adjacent slices.

**Returns:**
1D array of Dice scores (length Z-1)

#### `compute_z_run_length_stability(volume, target_class, use_gpu=False)`
Compute longest continuous run per voxel along Z.

**Returns:**
Dictionary with:
- `run_length_map`: 2D array (Y, X) of maximum run lengths
- `mean_run_length`: Mean of non-zero runs
- `median_run_length`: Median of non-zero runs

#### `compute_component_persistence(mask, target_class)`
Track connected components slice-to-slice.

**Returns:**
Dictionary with:
- `mean_persistence`: Average component lifespan in slices
- `median_persistence`: Median component lifespan
- `max_persistence`: Maximum component lifespan
- `component_lifespans`: List of all component lifespans

## Design Decisions

### GPU Implementation
- Only Z-axis voxel operations are GPU-accelerated
- Connected-component analysis stays on CPU (SciPy is already optimized)
- Explicit tensor movement prevents memory leaks
- Clear error messages when GPU unavailable

### Identical Mask Detection
- Checks if conservative and aggressive masks are identical
- Prevents wasted computation on meaningless comparisons
- Warns user that correction thresholds may need adjustment

### Component Persistence
- Uses greedy overlap matching (not over-engineered tracking)
- Sufficient for research-grade analysis
- Fast and interpretable results

## Limitations

- No visualization framework (research tool, not production)
- 2D metrics do not have GPU acceleration (connected components stay on CPU)
- Component tracking uses greedy matching (not optimal for complex scenarios)
- Requires consistent slice ordering

## Contributing

This is a research-oriented tool. When contributing:
- Keep changes localized to z_stability folder
- Do not touch preprocessing code
- Maintain existing test infrastructure
- Document new metrics clearly

## License

See repository root for license information.
