import numpy as np
from scipy.ndimage import label, convolve
from typing import Dict

def compute_z_metrics(volume: np.ndarray, target_class: int, window_size: int, use_gpu: bool = False) -> Dict[str, np.ndarray]:
    if use_gpu: raise NotImplementedError("GPU acceleration will be added in Phase 2")
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

def compute_2d_metrics(slice_mask: np.ndarray, target_class: int, use_gpu: bool = False) -> Dict[str, np.ndarray]:
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
