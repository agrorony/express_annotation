from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image


# =========================
# SETTINGS (CHANGE THESE)
# =========================

INPUT_DIR = r"C:\Users\first\Documents\soilNWater\year3\img processing\express_annotations\dataset\01_raw"


OUTPUT_DIR = r"C:\Users\first\Documents\soilNWater\year3\img processing\express_annotations\dataset\01_standartizied"


SUFFIX = "_norm8"


TO8_METHOD = "percentile"
P_LOW = 0.5
P_HIGH = 99.5


def to_uint8(
    img: np.ndarray,
    method: str = "percentile",
    p_low: float = 0.5,
    p_high: float = 99.5
) -> np.ndarray:
    """
    Convert image/stack to uint8 [0..255].
    method:
      - "percentile": robust scaling using percentiles
      - "minmax": linear scaling using min/max
      - "divide256": uint16 -> uint8 by /256 (fast, often worse)
    """
    if img.dtype == np.uint8:
        return img

    x = img.astype(np.float32)

    if method == "divide256":
        x = np.clip(x / 256.0, 0, 255)
        return x.astype(np.uint8)

    if method == "minmax":
        mn = float(np.min(x))
        mx = float(np.max(x))
    elif method == "percentile":
        mn = float(np.percentile(x, p_low))
        mx = float(np.percentile(x, p_high))
    else:
        raise ValueError("Unknown method. Use: 'percentile', 'minmax', 'divide256'.")

    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - mn) * (255.0 / (mx - mn))
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def detect_mode_above_100(img: np.ndarray, low: int = 100, high: int = 254) -> int:
    """
    img: TIFF ndarray (2D или 3D stack).
    Возвращает mode значений в диапазоне [low, high] для 8-bit изображения.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image (0-255). Got dtype={img.dtype}")

    flat = img.ravel()
    mask = (flat >= low) & (flat <= high)
    if not np.any(mask):
        return 0

    counts = np.bincount(flat[mask], minlength=256)
    mode = int(np.argmax(counts[low:high + 1]) + low)
    return mode


def normalize_stack_to_mode_200(img: np.ndarray, mode: int, target: float = 200.0) -> np.ndarray:
    """
    Multiply by factor=target/mode (как ImageJ Multiply...) и clip к 0..255.
    """
    if mode <= 0:
        return img

    factor = target / float(mode)
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def process_folder(
    input_dir: str,
    output_dir: str,
    suffix: str = "_norm8",
    to8_method: str = "percentile",
    p_low: float = 0.5,
    p_high: float = 99.5
) -> None:
    in_path = Path(input_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {in_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tif_files = sorted([p for p in in_path.iterdir()
                        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])

    if not tif_files:
        print(f"No TIFF files found in: {in_path}")
        return

    print(f"Input folder : {in_path}")
    print(f"Output folder: {out_path}")
    print(f"Found {len(tif_files)} TIFF file(s).")
    print("-" * 60)

    for p in tif_files:
        img = tiff.imread(p)  # 2D или 3D stack
        print(f"{p.name}: dtype={img.dtype}, shape={img.shape}")

        # Convert to 8-bit if needed
        img8 = to_uint8(img, method=to8_method, p_low=p_low, p_high=p_high)
        print(f" -> uint8: dtype={img8.dtype}, min={img8.min()}, max={img8.max()}")

        mode = detect_mode_above_100(img8, low=100, high=254)
        print(f" -> mode(100..254)={mode}")

        norm = normalize_stack_to_mode_200(img8, mode, target=200.0)

        out_name = f"{p.stem}{suffix}.png"
        out_file = out_path / out_name

        # Save as PNG
        if norm.ndim == 3:
            # If stack (3D), save only first slice
            Image.fromarray(norm[0]).save(out_file)
        else:
            # 2D image
            Image.fromarray(norm).save(out_file)

        print(f" -> saved: {out_file.name}")
        print("-" * 60)

    print("Done.")


if __name__ == "__main__":
    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        suffix=SUFFIX,
        to8_method=TO8_METHOD,
        p_low=P_LOW,
        p_high=P_HIGH
    )
