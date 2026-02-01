import numpy as np
from typing import Dict, Tuple

def apply_conservative_correction(original_mask, metrics_short, metrics_long, metrics_2d, thresholds, target_class=1):
    corrected = original_mask.copy()
    target_voxels = (original_mask == target_class)
    
    unstable_short = metrics_short['frequency'] < thresholds['min_short_frequency']
    high_flip = metrics_short['flip_rate'] > thresholds['max_flip_rate']
    unstable_long = metrics_long['frequency'] < thresholds['min_long_frequency']
    small_comp = metrics_2d['component_size'] < thresholds['min_component_size']
    
    # Conservative: Require ALL bad signs
    noise_mask = target_voxels & unstable_short & high_flip & unstable_long & small_comp
    
    relabel_pore = noise_mask & (metrics_short['class_0_fraction'] > metrics_short['class_2_fraction'])
    relabel_solid = noise_mask & (metrics_short['class_0_fraction'] <= metrics_short['class_2_fraction'])
    
    corrected[relabel_pore] = 0
    corrected[relabel_solid] = 2
    
    # Simple confidence
    conf = np.zeros_like(original_mask, dtype=np.float32)
    if np.any(noise_mask):
         c1 = np.clip(1 - metrics_short['frequency']/thresholds['min_short_frequency'],0,1)
         c2 = np.clip((metrics_short['flip_rate']-thresholds['max_flip_rate'])/thresholds['max_flip_rate'],0,1)
         conf[noise_mask] = (c1+c2)[noise_mask] / 2
         
    print(f"Conservative: Corrected {np.sum(noise_mask)} voxels")
    return corrected, conf

def apply_aggressive_correction(original_mask, metrics_short, metrics_long, metrics_2d, thresholds, target_class=1):
    corrected = original_mask.copy()
    target_voxels = (original_mask == target_class)
    
    unstable_short = metrics_short['frequency'] < thresholds['min_short_frequency']
    high_flip = metrics_short['flip_rate'] > thresholds['max_flip_rate']
    unstable_long = metrics_long['frequency'] < thresholds['min_long_frequency']
    small_comp = metrics_2d['component_size'] < thresholds['min_component_size']
    
    # Aggressive: Require ANY bad sign
    noise_mask = target_voxels & (unstable_short | high_flip | unstable_long | small_comp)
    
    relabel_pore = noise_mask & (metrics_short['class_0_fraction'] > metrics_short['class_2_fraction'])
    relabel_solid = noise_mask & (metrics_short['class_0_fraction'] <= metrics_short['class_2_fraction'])
    
    corrected[relabel_pore] = 0
    corrected[relabel_solid] = 2
    
    conf = np.zeros_like(original_mask, dtype=np.float32)
    print(f"Aggressive: Corrected {np.sum(noise_mask)} voxels")
    return corrected, conf
