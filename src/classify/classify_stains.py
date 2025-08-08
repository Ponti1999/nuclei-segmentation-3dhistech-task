from typing import Dict, Tuple
import numpy as np
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from tqdm import tqdm


def _to_rgb_float01(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 RGB format with values in [0, 1].

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Converted image as float32 with normalized values.
    """
    if image.dtype not in (np.float32, np.float64):
        return image.astype(np.float32) / 255.0
    return image


def _dab_h_from_rgb01(image_rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RGB image to DAB and Hematoxylin channels.

    Args:
        image_rgb01 (np.ndarray): RGB image with values in [0, 1].

    Returns:
        tuple: Two np.ndarray values:
            - DAB channel (float32, rescaled to [0, 1]).
            - Hematoxylin channel (float32, rescaled to [0, 1]).
    """
    hed = rgb2hed(image_rgb01)
    dab = rescale_intensity(hed[:, :, 2], out_range=(0, 1)).astype(np.float32)
    he = rescale_intensity(hed[:, :, 0], out_range=(0, 1)).astype(np.float32)
    return dab, he


def _build_color_map(class_colors: Dict[str, list]) -> Dict[str, Tuple[int, int, int]]:
    """Build a color map from given class colors with defaults.

    Args:
        class_colors (dict): Mapping of class names to RGB lists.

    Returns:
        dict: Mapping of class names to RGB tuples (B, G, R).
    """
    defaults = {
        "red": (0, 0, 255),
        "orange": (0, 165, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }
    out = {}
    for k, dv in defaults.items():
        v = class_colors.get(k, list(dv))
        out[k] = tuple(int(c) for c in v)
    return out


def _per_label_means(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute the mean value for each label.

    Args:
        values (np.ndarray): Value array (same shape as labels).
        labels (np.ndarray): Label array of integer IDs.

    Returns:
        np.ndarray: Mean value for each label ID.
    """
    lab = labels.ravel()
    val = values.ravel()
    max_id = int(lab.max())
    if max_id <= 0:
        return np.zeros(1, dtype=np.float32)
    sums = np.bincount(lab, weights=val, minlength=max_id + 1)
    cnts = np.bincount(lab, minlength=max_id + 1)
    cnts[cnts == 0] = 1
    return sums / cnts


def classify_nuclei(
    image: np.ndarray,
    label_map: np.ndarray,
    thresholds: Dict[str, float],
    class_colors: Dict[str, list],
    auto_calibrate: bool = True,
    percentiles: Dict[str, int] | None = None,
) -> Dict[int, tuple]:
    """Classify nuclei by DAB intensity.

    Computes mean DAB intensity per nucleus, assigns a color-coded class
    (red, orange, yellow, blue) based on thresholds, and optionally
    auto-calibrates thresholds using percentiles.

    Args:
        image (np.ndarray): Input RGB image.
        label_map (np.ndarray): Segmentation label map.
        thresholds (dict): Fixed DAB mean thresholds for each class.
        class_colors (dict): Mapping of class names to RGB lists.
        auto_calibrate (bool, optional): Whether to auto-calculate thresholds.
        percentiles (dict, optional): Percentile values for auto-calibration.

    Returns:
        dict: Mapping from nucleus ID to RGB color tuple.
    """
    rgb01 = _to_rgb_float01(image)
    dab, _ = _dab_h_from_rgb01(rgb01)

    labels = label_map.astype(np.int32)
    max_id = int(labels.max())
    if max_id <= 0:
        return {}

    means = _per_label_means(dab, labels)
    nuc_vals = means[1:]

    if auto_calibrate:
        if not percentiles:
            percentiles = {"yellow": 40, "orange": 65, "red": 85}
        t_yellow = float(np.percentile(nuc_vals, percentiles["yellow"]))
        t_orange = float(np.percentile(nuc_vals, percentiles["orange"]))
        t_red = float(np.percentile(nuc_vals, percentiles["red"]))
        print(
            f"[CLASSIFY] Auto thresholds (DAB mean): yellow>{t_yellow:.3f}  orange>{t_orange:.3f}  red>{t_red:.3f}"
        )
    else:
        t_red = float(thresholds.get("red", 0.75))
        t_orange = float(thresholds.get("orange", 0.50))
        t_yellow = float(thresholds.get("yellow", 0.25))
        print(
            f"[CLASSIFY] Fixed thresholds (DAB mean): yellow>{t_yellow:.3f}  orange>{t_orange:.3f}  red>{t_red:.3f}"
        )

    cmap = _build_color_map(class_colors)

    out: Dict[int, tuple] = {}
    counts = {"red": 0, "orange": 0, "yellow": 0, "blue": 0}

    chunk = 10000
    ids = np.arange(1, max_id + 1, dtype=np.int32)
    with tqdm(total=ids.size, desc="Classifying nuclei", unit="cell") as pbar:
        for start in range(0, ids.size, chunk):
            end = min(start + chunk, ids.size)
            cids = ids[start:end]
            m = means[cids]

            red_mask = m > t_red
            orange_mask = (m > t_orange) & ~red_mask
            yellow_mask = (m > t_yellow) & ~(red_mask | orange_mask)
            blue_mask = ~(red_mask | orange_mask | yellow_mask)

            for cid in cids[red_mask]:
                out[int(cid)] = cmap["red"]
                counts["red"] += 1
            for cid in cids[orange_mask]:
                out[int(cid)] = cmap["orange"]
                counts["orange"] += 1
            for cid in cids[yellow_mask]:
                out[int(cid)] = cmap["yellow"]
                counts["yellow"] += 1
            for cid in cids[blue_mask]:
                out[int(cid)] = cmap["blue"]
                counts["blue"] += 1

            pbar.update(cids.size)

    total = sum(counts.values())
    if total > 0:
        fr = {k: f"{v} ({100*v/total:.1f}%)" for k, v in counts.items()}
        print(
            f"[CLASSIFY] Class counts: red={fr['red']}, orange={fr['orange']}, yellow={fr['yellow']}, blue={fr['blue']}"
        )

    return out
