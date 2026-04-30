# Cross-modal Building Image Alignment

Given an architectural sketch and an aerial/satellite photo of the same building, this tool uses SAM3 to segment the building footprint from the aerial image, then warps the sketch onto it via affine transform.

**`building_matcher5_2.py` is the final version.** Earlier scripts (`v1`–`v5`) are kept for reference only.

| Script | Purpose |
|--------|---------|
| `building_matcher4_2.py` | Single-pair, interactive — tune and inspect results |
| `building_matcher5_2.py` | **Final.** Batch, fully automated |
| `sam3_explore.py` | Debug tool — visualize what SAM3 detects for a given set of text prompts on a single image, before running the full pipeline |

---

## How it works

1. **Sketch feature extraction** — load sketch → grayscale → binarize (threshold 200) → morphological close → find largest contour → compute `minAreaRect`
2. **SAM3 detection** — text-prompt segmentation with positive prompts; per-mask keep largest connected component; area filter; per-prompt dedup (keep largest area mask); negative prompt filter; IoU dedup
3. **Auto-merge** — check bbox gap between all candidate pairs; if any gap is smaller than 10% of the image diagonal, merge all triggered candidates automatically
4. **Affine alignment** — fit a similarity transform (scale + rotation + translation) between sketch `minAreaRect` corners and aerial mask corners; try all 4 corner orderings; keep the best IoU; composite sketch as transparent overlay onto aerial

---

## Requirements

### Hardware
- NVIDIA GPU with ≥ 6 GB VRAM recommended (CPU works but is very slow)

### Python environment

```bash
conda create -n sam3 python=3.11 -y
conda activate sam3

# PyTorch with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install ultralytics opencv-python numpy matplotlib
```

### Model weights

```bash
hf download 1038lab/sam3 sam3.pt --local-dir checkpoints
```

Expected layout:
```
SAMLearning/
└── checkpoints/
    └── sam3.pt        (~3.5 GB)
```

---

## `building_matcher4_2.py` — Interactive single-pair mode

Best for exploring a new image pair or tuning parameters. Shows a side-by-side detection preview, prints all candidates with confidence scores, then asks which masks to use.

### Configuration

Edit the path variables near the top:

```python
SKETCH_PATH = "sketch_2.jpg"
AERIAL_PATH = "aerial_2.jpg"
```

### Running

```bash
python building_matcher4_2.py
```

### What happens

1. SAM3 detects candidates and displays them in a numbered overlay
2. If **auto-merge fires** (bbox gap < threshold): regions are merged silently and alignment proceeds
3. If **only one candidate**: selected automatically, no prompt
4. If **multiple candidates, no auto-merge**: prints a table and asks:
   ```
   Enter mask indices to use (space- or comma-separated; multiple = merge; Enter = use #0):
   > 0 1
   ```
   To skip the prompt entirely, set `MANUAL_SEL_INDICES` in the config:
   ```python
   MANUAL_SEL_INDICES = [0, 1]   # always merge #0 and #1, no input needed
   ```

### Outputs (`output/`)

| File | Description |
|------|-------------|
| `alignment_result.png` | Sketch warped onto aerial image |
| `final_comparison.png` | 4-panel: sketch / SAM3 mask / isolated mask / alignment zoom |
| `detection_preview.png` | Numbered candidate masks (saved when multiple candidates exist) |

---

## `building_matcher5_2.py` — Batch automated mode (final)

Processes multiple sketch/aerial pairs in one run. No interactive input at any point. Model is loaded once and reused across all pairs. Each pair is wrapped in error handling so a failure on one pair does not stop the others.

### Configuration

Edit `SKETCH_AERIAL_PAIRS`:

```python
SKETCH_AERIAL_PAIRS = [
    ("sketch_2.jpg",  "aerial_2.jpg"),
    ("sketch_3.png",  "aerial_3.png"),
    ("sketch.png",    "aerial.png"),
]
```

All paths are relative to the script's directory.

### Running

```bash
python building_matcher5_2.py
```

### What happens

The detection and merge logic is identical to v4.2. The only difference is candidate handling when auto-merge does not trigger:

- **1 candidate** → selected automatically
- **Multiple candidates, no auto-merge** → all are merged automatically; `detection_preview_N.png` is saved for inspection

End-of-run summary:

```
============================================================
  DONE  3 OK  /  0 failed
  Output → .../output/pair_v5_2
============================================================
```

### Outputs (`output/pair_v5_2/`)

Files are numbered by pair index:

| File | Description |
|------|-------------|
| `alignment_result_1.png` | Sketch warped onto aerial (pair 1) |
| `final_comparison_1.png` | 4-panel figure (pair 1) |
| `detection_preview_1.png` | Candidate masks — only saved when auto-merge does not trigger |

---

## `sam3_explore.py` — SAM3 detection debugger

Use this before running the alignment pipeline to check whether SAM3 can correctly detect the building in a given aerial image. Adjust prompts here first, then copy working settings into `building_matcher4_2.py` or `building_matcher5_2.py`.

### Configuration

Edit the variables at the top of the file:

```python
IMAGE_PATH = "aerial_2.jpg"

PROMPTS = [
    "building",
    "outdoor building seating area",
]

NEGATIVE_PROMPTS = [
    "car",
    "tree",
    "road",
]
```

### Running

```bash
python sam3_explore.py
```

Saves a side-by-side visualization to `output/sam3_explore.png` showing the original image alongside all detected masks, each numbered and color-coded with a confidence score in the legend.

---

## Tuning parameters

Both scripts share the same parameter names. All are in the `★` section near the top of each file.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAM3_PROMPTS` | `["building", "outdoor building seating area"]` | Text queries sent to SAM3 |
| `SAM3_CONF` | `0.10` | Minimum detection confidence (keep low; matches `sam3_explore.py`) |
| `SAM3_NEGATIVE_PROMPTS` | `["car", "tree", "road", "swimming pool"]` | Masks with IoU > `NEG_IOU_THRESH` against any of these are discarded |
| `NEG_IOU_THRESH` | `0.3` | IoU threshold for negative filtering |
| `IOU_DEDUP` | `0.5` | IoU threshold for deduplicating overlapping positive masks |
| `MIN_MASK_FRACTION` | `0.01` | Minimum mask area as a fraction of total image pixels |
| `MAX_MASK_FRACTION` | `0.70` | Maximum mask area as a fraction of total image pixels |
| `AUTO_MERGE_PROXIMITY` | `0.1` | Bbox gap threshold as a fraction of image diagonal; `0` = disabled |
| `ZOOM_PAD_RATIO` | `0.20` | Padding around the target region in the alignment zoom panel |
| `CLAHE_ENABLED` | `False` | Contrast enhancement before SAM3 inference; keep `False` to match `sam3_explore.py` |
