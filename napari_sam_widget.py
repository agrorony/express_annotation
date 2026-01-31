import base64
import io

import numpy as np
import requests
import imageio.v3 as iio
from magicgui import magicgui
import napari
from napari.layers import Image


def _encode_png_base64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    iio.imwrite(buffer, image, extension=".png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_png_base64(encoded_png: str) -> np.ndarray:
    png_bytes = base64.b64decode(encoded_png)
    return iio.imread(io.BytesIO(png_bytes), extension=".png")


@magicgui(
    call_button="Run SAM Inference",
    server_url={"label": "Server Base URL"},
    viewer={"visible": False},
)
def sam_inference_widget(viewer: napari.Viewer, server_url: str = "") -> None:
    layer = viewer.layers.selection.active
    if layer is None:
        raise AssertionError("No active layer selected.")
    if not isinstance(layer, Image):
        raise AssertionError("Active layer is not an image layer.")

    image = np.asarray(layer.data)
    if image.ndim != 2:
        raise AssertionError("Active image layer must be a 2D array.")

    encoded_image = _encode_png_base64(image)
    payload = {
        "image": encoded_image,
        "box": [50, 50, 300, 300],
    }

    url = server_url.rstrip("/") + "/predict"
    response = requests.post(url, json=payload, timeout=30)
    print(f"HTTP status code: {response.status_code}")

    response.raise_for_status()
    data = response.json()
    mask = _decode_png_base64(data["mask"])
    if mask.shape != image.shape:
        raise AssertionError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )

    print(f"mask shape: {mask.shape}")
    print(f"mask dtype: {mask.dtype}")

    viewer.add_labels(mask, name="SAM Mask")


def make_sam_dock_widget(viewer: napari.Viewer):
    widget = sam_inference_widget
    widget.viewer.value = viewer
    return widget


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(make_sam_dock_widget(viewer), area="right")
    napari.run()
