# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Cross-modal Building Image Alignment System: given an architectural sketch and an aerial/satellite image of the same building, segment the building footprint from the aerial image and warp the sketch onto it via affine transform.

## Active scripts

```bash
# v4.2 — SAM3 explore-style detection, interactive single-pair mode
python building_matcher4_2.py

# v5.2 — batch version of v4.2, fully automated, no interactive input (FINAL)
python building_matcher5_2.py

# SAM3 exploration — visualize what text prompts detect on a single image
python sam3_explore.py
```

Older scripts (`building_matcher.py` through `building_matcher5.py`) are kept for reference only. No build step. Conda environment with PyTorch + CUDA recommended.

## Dependencies

- `ultralytics` (SAM3): `pip install ultralytics`
- `pip install opencv-python numpy matplotlib torch torchvision`
- V1/V2/V3 additionally require the `sam2` package (not needed for active scripts)

## Model weights

```bash
hf download 1038lab/sam3 sam3.pt --local-dir checkpoints
```

Place `sam3.pt` (~3.5 GB) in `checkpoints/`. All paths use `Path(__file__).parent` so no hardcoding is needed.

## Architecture (v4.2 / v5.2)

Both active scripts share the same four-stage pipeline inside `BuildingMatcher`:

- **Module A** (`extract_sketch_features`): load sketch → grayscale → binarize (threshold 200) → morphological close → largest contour → Hu moments + `minAreaRect`
- **Module B** (`detect_masks`): SAM3 text-prompt inference → per-mask keep largest connected component → area filter → per-prompt dedup (keep largest area) → negative prompt filter → IoU dedup (conf-priority)
- **Auto-merge**: check bbox gap between all candidate pairs; if any gap < `AUTO_MERGE_PROXIMITY × diagonal`, merge all triggered candidates
- **Module D** (`_compute_affine` / `align_and_overlay`): fit similarity transform between sketch `minAreaRect` corners and aerial mask corners; try all 4 orderings; pick best IoU; composite sketch as transparent RGBA over aerial

Key difference between v4.2 and v5.2: v4.2 shows a detection preview and asks the user to pick masks; v5.2 is headless, merges all candidates automatically, and loops over `SKETCH_AERIAL_PAIRS`.

## Tuning parameters

All parameters are in the `★` section at the top of each script:

- `SAM3_PROMPTS` — text queries sent to SAM3
- `SAM3_CONF` — minimum detection confidence (keep at `0.10` to match `sam3_explore.py`)
- `SAM3_NEGATIVE_PROMPTS` — masks overlapping these are discarded
- `NEG_IOU_THRESH` — IoU threshold for negative filtering
- `AUTO_MERGE_PROXIMITY` — bbox gap threshold as fraction of image diagonal; `0` = disabled
- `MIN_MASK_FRACTION` / `MAX_MASK_FRACTION` — area bounds relative to image size
- `CLAHE_ENABLED` — keep `False` to match `sam3_explore.py` behavior

## Outputs

- `output/` (v4.2): `alignment_result.png`, `final_comparison.png`, `detection_preview.png`
- `output/pair_v5_2/` (v5.2): `alignment_result_N.png`, `final_comparison_N.png`, `detection_preview_N.png`
