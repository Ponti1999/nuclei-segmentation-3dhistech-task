import sys
import os
import json
import numpy as np
from utils.image_utils import load_image, save_image
from segment.segment_nuclei import segment_nuclei
from classify.classify_stains import classify_nuclei
from visualize.draw_outlines import draw_outlines


def load_config() -> dict:
    """Load the configuration from config.json.

    Returns:
        dict: The loaded configuration data.
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Missing config.json at: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config.json: {e}")
        sys.exit(1)


def main():
    """Run the main processing pipeline."""
    if len(sys.argv) != 3:
        print("Usage: python -u src/main.py <input_image> <output_image>")
        sys.exit(1)

    image_path, output_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(image_path):
        print(f"[ERROR] Input not found: {image_path}")
        sys.exit(1)

    config = load_config()

    cache_dir = config["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    lm_path = os.path.join(cache_dir, config["paths"]["label_map_file"])
    cls_path = os.path.join(cache_dir, config["paths"]["classifications_file"])

    print(f"[INFO] Loading image: {image_path}")
    image = load_image(image_path)

    if os.path.exists(lm_path):
        print("[INFO] Loading cached label_map...")
        label_map = np.load(lm_path)
    else:
        print("[INFO] Segmenting nuclei...")
        label_map = segment_nuclei(
            image,
            try_patch_sizes=tuple(config["segmentation"]["try_patch_sizes"]),
            overlap=config["segmentation"]["overlap"],
            batch_size=config["segmentation"]["batch_size"],
            diameter=config["segmentation"]["diameter"],
        )
        np.save(lm_path, label_map)

    if os.path.exists(cls_path):
        print("[INFO] Loading cached classifications...")
        classifications = np.load(cls_path, allow_pickle=True).item()
    else:
        print("[INFO] Classifying nuclei...")
        classifications = classify_nuclei(
            image,
            label_map,
            thresholds=config["classification"]["thresholds"],
            class_colors=config["classification"]["class_colors"],
        )
        np.save(cls_path, classifications, allow_pickle=True)

    print("[INFO] Drawing outlines...")
    outlined = draw_outlines(
        image,
        label_map,
        classifications,
        thickness=config["drawing"]["outline_thickness"],
    )

    save_image(outlined, output_path)
    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()
