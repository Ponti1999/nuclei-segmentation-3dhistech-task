import math
import time
import gc
from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm
import torch
from cellpose import models


def _try_eval(
    model: models.CellposeModel, img: np.ndarray, batch_size: int, diameter: int
):
    """Evaluate a single image tile with a Cellpose model.

    Runs Cellpose inference on the given image tile and normalizes the output
    format to always return `(masks, flows, styles, diams)`.

    Args:
        model (models.CellposeModel): The Cellpose model to use.
        img (np.ndarray): Image tile for processing.
        batch_size (int): Number of images to process per batch.
        diameter (int): Estimated object diameter in pixels.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Segmentation masks.
            - np.ndarray: Flow fields from Cellpose.
            - np.ndarray: Style vectors.
            - np.ndarray or None: Estimated diameters, if available.
    """
    out = model.eval(
        img, channels=[0, 0], diameter=diameter, resample=True, batch_size=batch_size
    )
    if isinstance(out, (list, tuple)):
        if len(out) == 4:
            masks, flows, styles, diams = out
        else:
            masks, flows, styles = out
            diams = None
    else:
        masks = out.get("masks")
        flows = out.get("flows")
        styles = out.get("styles")
        diams = out.get("diams")
    return masks, flows, styles, diams


def segment_nuclei(
    image: np.ndarray,
    try_patch_sizes: Iterable[int],
    overlap: float,
    batch_size: int,
    diameter: int,
) -> np.ndarray:
    """Perform tiled Cellpose segmentation with memory-safe patching.

    Splits the image into overlapping tiles to prevent GPU/CPU memory issues
    during segmentation. Tests multiple patch sizes until one fits into
    memory, then processes all tiles with progress tracking.

    Args:
        image (np.ndarray): Input image for segmentation.
        try_patch_sizes (Iterable[int]): Patch sizes to try, from largest to smallest.
        overlap (float): Fraction of tile overlap between adjacent patches (0.0–1.0).
        batch_size (int): Number of tiles processed per batch.
        diameter (int): Estimated object diameter in pixels.

    Returns:
        np.ndarray: A 2D array of segmentation mask IDs matching the image size.

    Raises:
        RuntimeError: If no patch size fits into available GPU/CPU memory.
    """
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)

    h, w = image.shape[:2]
    masks_full = np.zeros((h, w), dtype=np.int32)
    next_id = 1

    chosen_ps = None
    last_err = None
    test_y0, test_x0 = 0, 0
    for ps in try_patch_sizes:
        y1 = min(test_y0 + ps, h)
        x1 = min(test_x0 + ps, w)
        tile = image[test_y0:y1, test_x0:x1]
        try:
            _ = _try_eval(model, tile, batch_size=batch_size, diameter=diameter)
            chosen_ps = ps
            break
        except RuntimeError as e:
            last_err = e
            torch.cuda.empty_cache()
            gc.collect()
    if chosen_ps is None:
        raise RuntimeError(
            f"Could not find a tile size that fits GPU/CPU. Last error: {last_err}"
        )

    ps = chosen_ps
    step = int(ps * (1.0 - overlap))
    ny = math.ceil((h - ps) / step) + 1 if h > ps else 1
    nx = math.ceil((w - ps) / step) + 1 if w > ps else 1
    total_tiles = ny * nx
    print(
        f"[Cellpose] Using patch_size={ps}, overlap={overlap}, tiles={ny}x{nx}={total_tiles}, "
        f"batch_size={batch_size}, diameter={diameter}, gpu={use_gpu}"
    )

    t0 = time.time()
    pbar = tqdm(total=total_tiles, unit="tile")
    for iy in range(ny):
        y0 = min(iy * step, h - ps) if h > ps else 0
        y1 = min(y0 + ps, h)
        for ix in range(nx):
            x0 = min(ix * step, w - ps) if w > ps else 0
            x1 = min(x0 + ps, w)
            tile_img = image[y0:y1, x0:x1]

            for _attempt in range(2):
                try:
                    tile_masks, _, _, _ = _try_eval(
                        model, tile_img, batch_size=batch_size, diameter=diameter
                    )
                    break
                except RuntimeError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if batch_size > 1:
                        batch_size = 1
                        continue
                    raise

            if tile_masks is not None:
                tids = np.unique(tile_masks)
                tids = tids[tids != 0]
                for tid in tids:
                    masks_full[y0:y1, x0:x1][tile_masks == tid] = next_id
                    next_id += 1

            del tile_img, tile_masks
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)

    pbar.close()
    mins = (time.time() - t0) / 60.0
    print(f"[Cellpose] Done in {mins:.1f} min. Cells: {next_id - 1}")
    return masks_full
