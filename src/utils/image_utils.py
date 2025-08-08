import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to RGB format.

    Args:
        path (str):
            Path to the image file on disk.

    Returns:
        np.ndarray:
            Image as a NumPy array in RGB color space with shape (H, W, 3)
            and dtype `uint8`.

    Raises:
        FileNotFoundError:
            If the file at `path` does not exist or cannot be read as an image.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str):
    """
    Save an RGB image to disk in its original quality.

    Args:
        image (np.ndarray):
            Image as a NumPy array in RGB color space with shape (H, W, 3)
            and dtype `uint8`.
        path (str):
            Destination path (including filename and extension) where the image will be saved.

    Raises:
        IOError:
            If the image cannot be written to the specified path.
        cv2.error:
            If the input array is not a valid 3-channel image.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr)
    if not ok:
        raise IOError(f"Failed to save image: {path}")
