from collections import defaultdict
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def draw_outlines(
    image: np.ndarray,
    label_map: np.ndarray,
    classifications: Dict[int, Tuple[int, int, int]],
    thickness: int,
) -> np.ndarray:
    """
    Draw colored outlines for nuclei in an image, grouped by color.

    Args:
        image (np.ndarray): Input image in BGR format on which outlines will be drawn.
        label_map (np.ndarray): 2D array of integer nucleus IDs.
        classifications (Dict[int, Tuple[int, int, int]]): Mapping of nucleus ID to RGB color tuple.
        thickness (int): Thickness of the outline.

    Returns:
        np.ndarray: Image with outlines drawn.
    """
    outlined = image.copy()
    groups: Dict[Tuple[int, int, int], list] = defaultdict(list)

    uids = np.unique(label_map)
    uids = uids[uids != 0]

    for cid in uids:
        rgb_color = classifications.get(int(cid), (0, 255, 0))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        groups[bgr_color].append(int(cid))

    print(
        f"[INFO] Drawing outlines for {len(uids)} nuclei in {len(groups)} color groups (thickness={thickness})"
    )

    for color, ids in tqdm(groups.items(), desc="Outlining groups", unit="group"):
        mask = np.isin(label_map, ids).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(outlined, contours, -1, color, thickness)

    return outlined
