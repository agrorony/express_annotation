import numpy as np
from scipy.ndimage import label, convolve
from typing import Dict, Optional, Tuple
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def compute_z_metrics(volume: np.ndarray, target_class: int, window_size: int, use_gpu: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute Z-axis stability metrics including frequency, flip rate, class fractions,
    and optionally using GPU acceleration.
    
    Args:
        volume: 3D volume (Z, Y, X)
        target_class: Target class to analyze
        window_size: Window size for local Z analysis
        use_gpu: If True, use GPU acceleration (requires PyTorch)
    
    Returns:
        Dictionary with metric arrays
    """
    if use_gpu:
        if not TORCH_AVAILABLE:
            raise NotImplementedError("GPU acceleration requires PyTorch. Install with: pip install torch")
        return _compute_z_metrics_gpu(volume, target_class, window_size)
    else:
        return _compute_z_metrics_cpu(volume, target_class, window_size)

def _compute_z_metrics_cpu(volume: np.ndarray, target_class: int, window_size: int) -> Dict[str, np.ndarray]:
    """CPU implementation of Z-axis metrics."""
    Z, Y, X = volume.shape
    half_window = window_size // 2
    frequency = np.zeros((Z, Y, X), dtype=np.float32)
    flip_rate = np.zeros((Z, Y, X), dtype=np.int32)
    class_0_fraction = np.zeros((Z, Y, X), dtype=np.float32)
    class_2_fraction = np.zeros((Z, Y, X), dtype=np.float32)
    effective_window_size = np.zeros((Z, Y, X), dtype=np.int32)

    for z in range(Z):
        z_start = max(0, z - half_window)
        z_end = min(Z, z + half_window + 1)
        window = volume[z_start:z_end, :, :]
        window_len = z_end - z_start
        effective_window_size[z, :, :] = window_len
        frequency[z] = np.sum(window == target_class, axis=0).astype(np.float32) / window_len
        class_0_fraction[z] = np.sum(window == 0, axis=0).astype(np.float32) / window_len
        class_2_fraction[z] = np.sum(window == 2, axis=0).astype(np.float32) / window_len
        if window_len > 1:
            transitions = np.diff(window, axis=0) != 0
            flip_rate[z] = np.sum(transitions, axis=0).astype(np.int32)
            
    return {'frequency': frequency, 'flip_rate': flip_rate, 'class_0_fraction': class_0_fraction, 
            'class_2_fraction': class_2_fraction, 'effective_window_size': effective_window_size}

def _compute_z_metrics_gpu(volume: np.ndarray, target_class: int, window_size: int) -> Dict[str, np.ndarray]:
    """GPU implementation of Z-axis metrics using PyTorch."""
    Z, Y, X = volume.shape
    half_window = window_size // 2
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("GPU requested but CUDA is not available")
    
    volume_gpu = torch.from_numpy(volume).to(device)
    
    # Prepare outputs
    frequency = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    flip_rate = torch.zeros((Z, Y, X), dtype=torch.int32, device=device)
    class_0_fraction = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    class_2_fraction = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    effective_window_size = torch.zeros((Z, Y, X), dtype=torch.int32, device=device)
    
    for z in range(Z):
        z_start = max(0, z - half_window)
        z_end = min(Z, z + half_window + 1)
        window = volume_gpu[z_start:z_end, :, :]
        window_len = z_end - z_start
        effective_window_size[z, :, :] = window_len
        frequency[z] = (window == target_class).sum(dim=0).float() / window_len
        class_0_fraction[z] = (window == 0).sum(dim=0).float() / window_len
        class_2_fraction[z] = (window == 2).sum(dim=0).float() / window_len
        if window_len > 1:
            transitions = torch.diff(window, dim=0) != 0
            flip_rate[z] = transitions.sum(dim=0).int()
    
    # Move back to CPU as NumPy
    return {
        'frequency': frequency.cpu().numpy(),
        'flip_rate': flip_rate.cpu().numpy(),
        'class_0_fraction': class_0_fraction.cpu().numpy(),
        'class_2_fraction': class_2_fraction.cpu().numpy(),
        'effective_window_size': effective_window_size.cpu().numpy()
    }

def compute_2d_metrics(slice_mask: np.ndarray, target_class: int, use_gpu: bool = False) -> Dict[str, np.ndarray]:
    if use_gpu:
        raise NotImplementedError("GPU acceleration for 2D metrics not implemented - keep on CPU")
    Y, X = slice_mask.shape
    class_0_neighbors = np.zeros((Y, X), dtype=np.float32)
    class_2_neighbors = np.zeros((Y, X), dtype=np.float32)
    component_size = np.zeros((Y, X), dtype=np.int32)
    
    target_mask = (slice_mask == target_class)
    if not np.any(target_mask):
        return {'class_0_neighbors': class_0_neighbors, 'class_2_neighbors': class_2_neighbors, 'component_size': component_size}

    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(target_mask, structure=structure)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        component_size = component_sizes[labeled]

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    total_neighbors = convolve(np.ones_like(slice_mask, dtype=np.float32), kernel, mode='constant', cval=0)
    
    class_0_neighbor_counts = convolve((slice_mask == 0).astype(np.float32), kernel, mode='constant', cval=0)
    class_2_neighbor_counts = convolve((slice_mask == 2).astype(np.float32), kernel, mode='constant', cval=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        c0 = np.nan_to_num(class_0_neighbor_counts / total_neighbors)
        c2 = np.nan_to_num(class_2_neighbor_counts / total_neighbors)

    class_0_neighbors = np.where(target_mask, c0, 0.0)
    class_2_neighbors = np.where(target_mask, c2, 0.0)
    
    return {'class_0_neighbors': class_0_neighbors, 'class_2_neighbors': class_2_neighbors, 'component_size': component_size}

def compute_2d_metrics_stack(volume: np.ndarray, target_class: int, use_gpu: bool = False) -> Dict[str, np.ndarray]:
    Z = volume.shape[0]
    first = compute_2d_metrics(volume[0], target_class, use_gpu)
    metrics_3d = {k: np.zeros((Z,) + volume.shape[1:], dtype=v.dtype) for k, v in first.items()}
    print(f"Computing 2D metrics for {Z} slices...")
    for z in range(Z):
        res = compute_2d_metrics(volume[z], target_class, use_gpu)
        for k, v in res.items(): metrics_3d[k][z] = v
        if (z+1) % 50 == 0: print(f" Processed {z+1}/{Z}")
    return metrics_3d

def compute_adjacent_slice_consistency(mask: np.ndarray, target_class: int) -> np.ndarray:
    """
    Compute Dice coefficient between adjacent slices for the target class.
    
    Args:
        mask: 3D mask volume (Z, Y, X)
        target_class: Target class to analyze
    
    Returns:
        1D array of Dice scores for each adjacent slice pair (length Z-1)
    """
    Z = mask.shape[0]
    dice_scores = np.zeros(Z - 1, dtype=np.float32)
    
    for z in range(Z - 1):
        slice_a = (mask[z] == target_class)
        slice_b = (mask[z + 1] == target_class)
        
        intersection = np.sum(slice_a & slice_b)
        union = np.sum(slice_a) + np.sum(slice_b)
        
        if union > 0:
            dice_scores[z] = 2.0 * intersection / union
        else:
            dice_scores[z] = 1.0  # Both empty, perfect agreement
    
    return dice_scores

def compute_z_run_length_stability(volume: np.ndarray, target_class: int, use_gpu: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute Z-axis run-length stability: longest continuous run per voxel.
    
    Args:
        volume: 3D volume (Z, Y, X)
        target_class: Target class to analyze
        use_gpu: If True, use GPU acceleration
    
    Returns:
        Dictionary with 'run_length_map' (Y, X) and statistics
    """
    if use_gpu:
        if not TORCH_AVAILABLE:
            raise NotImplementedError("GPU acceleration requires PyTorch")
        return _compute_z_run_length_gpu(volume, target_class)
    else:
        return _compute_z_run_length_cpu(volume, target_class)

def _compute_z_run_length_cpu(volume: np.ndarray, target_class: int) -> Dict[str, np.ndarray]:
    """CPU implementation of run-length computation."""
    Z, Y, X = volume.shape
    max_run_length = np.zeros((Y, X), dtype=np.int32)
    
    target_volume = (volume == target_class).astype(np.int32)
    
    for y in range(Y):
        for x in range(X):
            z_line = target_volume[:, y, x]
            current_run = 0
            max_run = 0
            for val in z_line:
                if val == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_run_length[y, x] = max_run
    
    runs = max_run_length[max_run_length > 0]
    mean_run = np.mean(runs) if len(runs) > 0 else 0.0
    median_run = np.median(runs) if len(runs) > 0 else 0.0
    
    return {
        'run_length_map': max_run_length,
        'mean_run_length': mean_run,
        'median_run_length': median_run
    }

def _compute_z_run_length_gpu(volume: np.ndarray, target_class: int) -> Dict[str, np.ndarray]:
    """GPU implementation of run-length computation using PyTorch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("GPU requested but CUDA is not available")
    
    Z, Y, X = volume.shape
    volume_gpu = torch.from_numpy(volume).to(device)
    target_volume = (volume_gpu == target_class).int()
    
    max_run_length = torch.zeros((Y, X), dtype=torch.int32, device=device)
    
    # Process each XY column along Z
    for y in range(Y):
        for x in range(X):
            z_line = target_volume[:, y, x]
            # Compute run lengths using diff
            padded = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), z_line, 
                               torch.zeros(1, dtype=torch.int32, device=device)])
            changes = torch.diff(padded)
            starts = torch.where(changes == 1)[0]
            ends = torch.where(changes == -1)[0]
            
            if len(starts) > 0:
                run_lengths = ends - starts
                max_run_length[y, x] = torch.max(run_lengths).item()
    
    max_run_np = max_run_length.cpu().numpy()
    runs = max_run_np[max_run_np > 0]
    mean_run = np.mean(runs) if len(runs) > 0 else 0.0
    median_run = np.median(runs) if len(runs) > 0 else 0.0
    
    return {
        'run_length_map': max_run_np,
        'mean_run_length': mean_run,
        'median_run_length': median_run
    }

def compute_component_persistence(mask: np.ndarray, target_class: int) -> Dict[str, any]:
    """
    Track connected components slice-to-slice and measure persistence.
    
    Args:
        mask: 3D mask volume (Z, Y, X)
        target_class: Target class to analyze
    
    Returns:
        Dictionary with persistence statistics
    """
    Z = mask.shape[0]
    persistence_counts = []
    
    structure = np.ones((3, 3), dtype=int)
    
    # Track components through slices
    prev_labeled = None
    prev_num = 0
    component_lifespans = {}  # component_id -> lifespan
    next_component_id = 0
    
    for z in range(Z):
        slice_mask = (mask[z] == target_class)
        if not np.any(slice_mask):
            # Reset tracking if empty slice
            prev_labeled = None
            prev_num = 0
            continue
        
        curr_labeled, curr_num = label(slice_mask, structure=structure)
        
        if prev_labeled is not None:
            # Match components between slices by overlap
            for curr_id in range(1, curr_num + 1):
                curr_component = (curr_labeled == curr_id)
                best_overlap = 0
                best_prev_id = None
                
                for prev_id in range(1, prev_num + 1):
                    prev_component = (prev_labeled == prev_id)
                    overlap = np.sum(curr_component & prev_component)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_prev_id = prev_id
                
                # Track persistence
                if best_overlap > 0:
                    # Component continues from previous slice
                    if best_prev_id in component_lifespans:
                        component_lifespans[best_prev_id] += 1
                else:
                    # New component
                    component_lifespans[next_component_id] = 1
                    next_component_id += 1
        else:
            # First non-empty slice
            for curr_id in range(1, curr_num + 1):
                component_lifespans[next_component_id] = 1
                next_component_id += 1
        
        prev_labeled = curr_labeled
        prev_num = curr_num
    
    # Compute statistics
    if component_lifespans:
        lifespans = list(component_lifespans.values())
        mean_persistence = np.mean(lifespans)
        median_persistence = np.median(lifespans)
        max_persistence = np.max(lifespans)
    else:
        mean_persistence = 0.0
        median_persistence = 0.0
        max_persistence = 0
    
    return {
        'mean_persistence': mean_persistence,
        'median_persistence': median_persistence,
        'max_persistence': max_persistence,
        'component_lifespans': list(component_lifespans.values())
    }
