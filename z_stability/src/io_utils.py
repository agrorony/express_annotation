import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional

def load_image_stack(folder_path: str) -> np.ndarray:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    image_files = sorted(folder.glob("*.png"))
    if not image_files:
        raise ValueError(f"No PNG files found in {folder_path}")
    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape[:2]
    n_slices = len(image_files)
    volume = np.zeros((n_slices, height, width), dtype=np.uint8)
    for i, img_path in enumerate(image_files):
        img = np.array(Image.open(img_path))
        if img.ndim == 3: img = img[:, :, 0]
        volume[i] = img
    print(f"Loaded {n_slices} slices from {folder_path} -> shape {volume.shape}")
    return volume

def load_mask_stack(folder_path: str) -> np.ndarray:
    return load_image_stack(folder_path)

def save_mask(array: np.ndarray, output_path: str) -> None:
    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    n_slices = array.shape[0]
    for i in range(n_slices):
        slice_img = Image.fromarray(array[i].astype(np.uint8))
        slice_img.save(output_folder / f"slice_{i:04d}.png")
    print(f"Saved {n_slices} slices to {output_path}")

def save_diagnostic_map(array: np.ndarray, output_path: str, colormap: str = "viridis", vmin=None, vmax=None) -> None:
    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(colormap)
    if vmin is None: vmin = np.nanmin(array)
    if vmax is None: vmax = np.nanmax(array)
    n_slices = array.shape[0]
    for i in range(n_slices):
        slice_data = array[i]
        if vmax > vmin: normalized = (slice_data - vmin) / (vmax - vmin)
        else: normalized = np.zeros_like(slice_data)
        normalized = np.clip(normalized, 0, 1)
        colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(output_folder / f"diagnostic_{i:04d}.png")
    print(f"Saved {n_slices} diagnostic slices to {output_path}")

def get_scan_name(scan_folder_path: str) -> str:
    return Path(scan_folder_path).name
