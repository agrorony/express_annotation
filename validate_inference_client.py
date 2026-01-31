import base64
import os
import sys

import requests

# This script is a minimal validation client for the SAM inference server. No napari involved.

BASE_URL = "https://monogenetic-nonmalarial-nia.ngrok-free.dev"
IMAGE_PATH = r"C:\Users\ronys\soil_and_water\research_exercise\express_annotation\test_images\roi_0000_slice0120_norm8_clahe61_med5.png"
BOX = [50, 50, 300, 300]


def main() -> int:
    if not os.path.isfile(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        return 1

    with open(IMAGE_PATH, "rb") as image_file:
        image_bytes = image_file.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image": image_b64, "box": BOX}

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1

    print(f"HTTP status code: {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return 1

    try:
        response_data = response.json()
    except ValueError:
        print("Response is not valid JSON.")
        return 1

    if "mask" not in response_data:
        print("Response missing 'mask'.")
        return 1

    mask_b64 = response_data["mask"]
    try:
        mask_bytes = base64.b64decode(mask_b64)
    except (ValueError, TypeError) as exc:
        print(f"Failed to decode mask: {exc}")
        return 1

    with open("mask.png", "wb") as mask_file:
        mask_file.write(mask_bytes)

    mask_metadata = response_data.get("mask_metadata", {})
    mask_shape = mask_metadata.get("shape")
    mask_dtype = mask_metadata.get("dtype")

    print(f"mask shape: {mask_shape}")
    print(f"mask dtype: {mask_dtype}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
