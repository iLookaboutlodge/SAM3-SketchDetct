# Cross-modal Building Image Alignment

Given an architectural sketch and an aerial/satellite photo of the same building, this tool uses SAM3 to segment the building footprint from the aerial image, then warps the sketch onto it via affine transform.

**`building_matcher6.py` is the current version.** Earlier scripts are kept for reference only.

| Script | Purpose |
|--------|---------|
| `building_matcher4_2.py` | Single-pair, interactive — tune and inspect results |
| `building_matcher5_2.py` | Batch, fully automated (v5 baseline) |
| `building_matcher6.py` | **Current.** Batch + confidence scoring + discrepancy detection |
| `sam3_explore.py` | Debug tool — visualize what SAM3 detects for a given set of text prompts on a single image, before running the full pipeline |

---

## How it works

1. **Module A — Sketch feature extraction** — load sketch → grayscale → binarize (threshold 200) → morphological close → find largest contour → compute `minAreaRect`
2. **Module B — SAM3 detection** — text-prompt segmentation with positive prompts; per-mask keep largest connected component; area filter; per-prompt dedup (keep largest area mask); negative prompt filter; IoU dedup
3. **Auto-merge / target selection** — proximity clustering via union-find on pairwise bbox gap; if all masks form one cluster the merged mask is used directly; if multiple clusters the best-fitting one is selected and the rest flagged as unmatched buildings
4. **Module D — Affine alignment** — fit a similarity transform (scale + rotation + translation) between sketch `minAreaRect` corners and aerial mask corners; try all 4 corner orderings; keep the best IoU; composite sketch as transparent overlay onto aerial
5. **Module E — Confidence scoring** *(v6 new)* — compute a composite alignment score from IoU, HD95, Chamfer distance, PoLiS, and extra-area coverage; each metric is converted to a [0, 1] similarity via exponential decay and combined with configurable weights
6. **Module F — Discrepancy detection** *(v6 new)* — identify extra protrusions in the aerial footprint not covered by the sketch, sketch regions missing from the aerial, and fully unmatched buildings; filter out slivers by minimum area and compactness (isoperimetric quotient)

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
pip install ultralytics opencv-python numpy matplotlib scipy

# Optional — enables PoLiS metric in Module E
pip install shapely
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

## `building_matcher6.py` — Batch mode with confidence scoring and discrepancy detection (current)

Extends v5.2 with two new pipeline stages. Processes multiple sketch/aerial pairs in one run with no interactive input.

### Configuration

Edit `SKETCH_AERIAL_PAIRS`:

```python
SKETCH_AERIAL_PAIRS = [
    ("sketch_2.jpg",  "aerial_2.jpg"),
    ("sketch_3.png",  "aerial_3.png"),
    ("sketch.png",    "aerial.png"),
]
```

### Running

```bash
python building_matcher6.py
```

### Outputs (`output/pair_v6/`)

| File | Description |
|------|-------------|
| `alignment_result_N.png` | Sketch warped onto aerial (pair N) |
| `discrepancy_map_N.png` | Aerial with color-coded diff overlay + legend |
| `final_comparison_N.png` | **6-panel figure**: ① sketch ② SAM3 mask ③ isolated mask ④ alignment zoom ⑤ discrepancy map ⑥ confidence bar chart |
| `detection_preview_N.png` | Candidate masks — saved when multiple candidates exist |

### Discrepancy map color coding

| Color | Meaning |
|-------|---------|
| Green | Matching area (sketch ∩ aerial) |
| Cyan | Extra protrusion in aerial not covered by sketch |
| Red | Sketch area missing from aerial |
| Orange | Unmatched building(s) detected in aerial |

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

## `building_matcher5_2.py` — Batch automated mode (v5 baseline)

Same pipeline as v4.2 but headless. Use v6 for new work; v5.2 is kept as a simpler reference without Module E/F.

### Running

```bash
python building_matcher5_2.py
```

### Outputs (`output/pair_v5_2/`)

| File | Description |
|------|-------------|
| `alignment_result_N.png` | Sketch warped onto aerial (pair N) |
| `final_comparison_N.png` | 4-panel figure (pair N) |
| `detection_preview_N.png` | Candidate masks — only saved when auto-merge does not trigger |

---

## `sam3_explore.py` — SAM3 detection debugger

Use this before running the alignment pipeline to check whether SAM3 can correctly detect the building in a given aerial image. Adjust prompts here first, then copy working settings into the matcher scripts.

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

All parameters are in the `★` section near the top of each script.

### Detection & alignment (all versions)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAM3_PROMPTS` | `["building", "outdoor building seating area"]` | Text queries sent to SAM3 |
| `SAM3_CONF` | `0.10` | Minimum detection confidence (keep low; matches `sam3_explore.py`) |
| `SAM3_NEGATIVE_PROMPTS` | `["car", "tree", "road", "swimming pool"]` | Masks with IoU > `NEG_IOU_THRESH` against any of these are discarded |
| `NEG_IOU_THRESH` | `0.3` | IoU threshold for negative filtering |
| `IOU_DEDUP` | `0.5` | IoU threshold for deduplicating overlapping positive masks |
| `MIN_MASK_FRACTION` | `0.01` | Minimum mask area as a fraction of total image pixels |
| `MAX_MASK_FRACTION` | `0.70` | Maximum mask area as a fraction of total image pixels |
| `ZOOM_PAD_RATIO` | `0.20` | Padding around the target region in the alignment zoom panel |
| `CLAHE_ENABLED` | `False` | Contrast enhancement before SAM3 inference; keep `False` to match `sam3_explore.py` |

### Multi-building handling (v6)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MULTI_BLDG_PROXIMITY` | `0.10` | Bbox gap threshold (fraction of image diagonal) for clustering masks into one building |
| `MULTI_BLDG_IOU_THRESH` | `0.35` | Merged IoU below this triggers per-cluster fallback |
| `MULTI_BLDG_MISSING_THRESH` | `0.10` | Missing area > this fraction of merged building area → re-evaluate clusters individually |

### Module E — Confidence scoring weights (v6)

Weights must sum to 1.0. Distance metrics use exponential decay: `sim = exp(−distance / (CONF_DECAY_FRAC × diagonal))`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONF_WEIGHT_IOU` | `0.30` | Alignment IoU |
| `CONF_WEIGHT_HD95` | `0.20` | 95th-percentile Hausdorff boundary similarity |
| `CONF_WEIGHT_CHAMFER` | `0.15` | Average Chamfer boundary similarity |
| `CONF_WEIGHT_POLIS` | `0.10` | PoLiS polygon-line similarity (requires `shapely`) |
| `CONF_WEIGHT_EXTRA` | `0.25` | Extra-area penalty: uncovered aerial area reduces score |
| `CONF_DECAY_FRAC` | `0.05` | Distance at which similarity ≈ 0.37, as a fraction of image diagonal |

### Module F — Discrepancy filtering (v6)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DISCREP_MIN_AREA_FRAC` | `0.02` | Ignore discrepancy regions smaller than 2% of the largest building area |
| `DISCREP_MIN_COMPACTNESS` | `0.05` | Isoperimetric quotient threshold (4πA/P²); regions below this are treated as slivers and discarded |
