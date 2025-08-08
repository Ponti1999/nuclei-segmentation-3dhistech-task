# Cell Classifier

This repository contains the implementation of an interview task for **3DHistech Kft.**.
The goal was to develop a pipeline for nuclei segmentation and classification using [Cellpose](https://www.cellpose.org/) and stain intensity analysis.
The solution was designed to handle large microscopy images efficiently using tiled segmentation with GPU acceleration where available.


---

## System Requirements

This project was developed and tested using:

- **Python:** 3.10
- **CUDA:** 11.8

Python 3.10 was chosen because it is fully compatible with the available PyTorch build for CUDA 11.8 at the time of development.
CUDA 11.8 support ensures GPU acceleration for the segmentation step when running Cellpose, significantly improving performance on large images compared to CPU-only processing.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ponti1999/nuclei-segmentation-3dhistech-task.git
   cd nuclei-segmentation-3dhistech-task
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv310
   .\.venv310\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Folder Structure

```
cell_classifier/
│
├── img/
│   ├── input/       # Place input images here
│   ├── output/      # Processed segmentation and classification results
│   └── ref/         # Reference or ground truth images
│
├── src/             # Source code
│   ├── segment/     # Nuclei segmentation logic
│   ├── classify/    # Stain intensity classification
│   ├── visualize/   # Drawing and visualization functions
│   └── utils/       # Utility modules
│
├── cache/           # Automatically generated intermediate files
├── tests/           # Reserved for future tests
├── requirements.txt
├── config.json      # Configuration file controlling processing parameters
└── README.md
```

---

## Configuration

The `config.json` file allows customization of segmentation, classification, and output parameters without modifying the source code.

Example:
```json
{
    "paths": {
        "cache_dir": "cache",
        "label_map_file": "label_map.npy",
        "classifications_file": "classifications.npy"
    },
    "segmentation": {
        "try_patch_sizes": [1024, 768, 512],
        "overlap": 0.1,
        "batch_size": 4,
        "diameter": 30
    },
    "classification": {
        "thresholds": {
            "red": 0.75,
            "orange": 0.50,
            "yellow": 0.25
        },
        "class_colors": {
            "red": [0, 0, 255],
            "orange": [0, 165, 255],
            "yellow": [0, 255, 255],
            "blue": [255, 0, 0]
        }
    },
    "drawing": {
        "outline_thickness": 2
    }
}
```

**Key sections:**
- **paths**
  - `cache_dir`: Directory to store intermediate files.
  - `label_map_file`: Filename for saved segmentation label maps.
  - `classifications_file`: Filename for saved classification results.
- **segmentation**
  - `try_patch_sizes`: List of patch sizes (in pixels) to attempt for tiled segmentation, in descending order.
  - `overlap`: Fractional overlap between adjacent tiles (0–1).
  - `batch_size`: Number of tiles processed per batch.
  - `diameter`: Estimated nucleus diameter in pixels.
- **classification**
  - `thresholds`: Fixed thresholds for DAB mean intensity classification.
  - `class_colors`: RGB color codes for each classification group.
- **drawing**
  - `outline_thickness`: Pixel thickness for classification outlines.

---

## Usage

Basic command:
```bash
python -u src/main.py <input_image> <output_image>
```

Example:
```bash
python -u src/main.py img/input/sample.jpg img/output/result.jpg
```

- `<input_image>`: Path to the source image (recommended: in `img/input/`).
- `<output_image>`: Path to save the processed output (recommended: in `img/output/`).

The pipeline will:
1. Load the image.
2. Check the cache for existing segmentation and classification results.
3. If not found, perform segmentation and classification.
4. Draw classification outlines.
5. Save the final processed image.

---

## How It Works

1. **Segmentation**
   - Uses Cellpose for nuclei segmentation.
   - Operates in a tiled mode to manage memory constraints.
   - Automatically selects the largest tile size that fits into GPU/CPU memory.

2. **Classification**
   - Converts the image to Hematoxylin-Eosin-DAB (HED) space.
   - Calculates mean DAB intensity for each nucleus.
   - Assigns a classification color based on fixed or auto-calculated thresholds.

3. **Visualization**
   - Draws color-coded outlines for each classified nucleus.
   - Saves results in the specified output path.

---

## Performance

Processing time depends on:
- Image size and resolution.
- GPU availability.
- Chosen patch size and overlap.

---

## Notes

- The `cache/` folder stores intermediate results to avoid redundant processing.
- Delete cached files if you want to reprocess an image from scratch.
- Ensure input images are in a supported format (e.g., `.jpg`, `.png`).

---

## License

This project is released under the MIT License.
