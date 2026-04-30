"""
Cross-modal Building Image Alignment System — v4
Architecture: SAM3 text-prompt → Hu-moment filter → affine align → transparent overlay

V4 upgrade over V3:
  - SAM2 AutomaticMaskGenerator → SAM3 concept-based segmentation (ultralytics)
  - Natural-language prompt targets the building directly — no brute-force full scan
  - No sam2 package required; ultralytics auto-downloads SAM3 weights on first run
  - Shape matching retained as secondary confidence filter on returned masks
"""

import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Image discovery helper
# ============================================================
def _find_image(base_dir: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        p = base_dir / (stem + ext)
        if p.exists():
            return p
    return base_dir / (stem + ".png")


# ============================================================
# Paths & Global Config
# ============================================================
BASE_DIR   = Path(r"D:\2026 myplan\SAMLearning")
OUTPUT_DIR = BASE_DIR / "output"

MAX_IMAGE_DIM = 1024

# SKETCH_PATH: Path | None = "sketch_3.png"
# AERIAL_PATH: Path | None = "aerial_3.png"
# SKETCH_PATH: Path | None = "sketch_2.jpg"
# AERIAL_PATH: Path | None = "aerial_2.jpg"
SKETCH_PATH: Path | None = "sketch.png"
AERIAL_PATH: Path | None = "aerial.png"

# ★ SAM3 — tuning
SAM3_MODEL  = BASE_DIR / "checkpoints" / "sam3.pt"
SAM3_PROMPTS = [
    "building",
    "outdoor building seating area"
]
SAM3_IMGSZ  = 640            # inference resolution; reduce to 512 if VRAM is tight
SAM3_CONF   = 0.1       # confidence threshold; keep low to match sam3_explore
SAM3_NEGATIVE_PROMPTS = [  # negative filter: positive masks with IoU > NEG_IOU_THRESH are discarded
    "car",
    "tree",
    "road",
    "garden",       # semantically overlaps with outdoor seating; may false-drop positive candidates
    "swimming pool",
]
NEG_IOU_THRESH = 0.3       # discard positive mask if IoU with any negative exceeds this

# ★ Area filter
MIN_MASK_FRACTION = 0.01
MAX_MASK_FRACTION = 0.70

# ★ Overlap dedup: drop smaller mask if this fraction of it falls inside the larger
# 0.5 = smaller mask more than half covered → redundant; 0 = disabled
OVERLAP_CONTAIN_THRESH = 0.5

# ★ Adjacent merge: two masks within N px generate a merged candidate added to the pool
# both merged and original candidates enter shape matching; Hu moments pick the best
# larger value = merge more distant masks; 0 = disabled
MERGE_ADJACENT_PX = 30

# ★ Top-K combo merge: after shape ranking, incrementally stack top-1..K into combined candidates
# Hu moments select the combination most similar to the sketch
# handles detections that belong to the same building but are farther than MERGE_ADJACENT_PX
# 0 or 1 = disabled
TOP_K_COMBO_MERGE = 2

# ★ Candidate selection mode
#   None      — interactive each run (export folder opens; enter rank in terminal)
#   []        — fully automatic, use Hu-moment rank 0, no prompt
#   [0, 2, 5] — merge these ranks directly, no prompt (for batch/automation)
MANUAL_MERGE_RANKS = None

# ★ Bbox gap auto-merge: auto-merge candidates whose bbox gap < this fraction of image diagonal
# 0.1 = 10%; set to 0 to disable
AUTO_MERGE_PROXIMITY = 0.1

# ★ Largest-first auto-select: if second-largest area < largest × this ratio, pick largest
# 0.7 = second < 70% of largest → clearly dominant → auto-select; 0 = disabled
AUTO_LARGEST_RATIO = 0.7

# ★ Contour smoothing & visualization
APPROX_EPSILON = 0.001  # SAM3 is precise; 0 = keep original contour exactly
ZOOM_PAD_RATIO = 0.20

# ★ Candidate export (debug)
EXPORT_CANDIDATES = True
EXPORT_TOP_N      = 8

# ★ CLAHE
CLAHE_ENABLED    = False   # tree shade = low local contrast; CLAHE helps here
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)


# ============================================================
# BuildingMatcher v4
# ============================================================
class BuildingMatcher:
    """
    V4 pipeline:
      Module A — Sketch feature extraction          (identical to V3)
      Module B — SAM3 text-prompt segmentation      (replaces SAM2 AutoMask)
      Module D — Affine alignment & transparent overlay (identical to V3)
    """

    def __init__(self, model_name: str = SAM3_MODEL):
        self.device = self._select_device()
        self._load_model(model_name)

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    def _select_device(self) -> str:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1024 ** 3
            print(f"[GPU] {props.name}  VRAM: {vram:.1f} GB")
            if vram < 4:
                print("[WARN] Low VRAM — consider reducing SAM3_IMGSZ to 512.")
            return "cuda"
        print("[WARN] CUDA unavailable — running on CPU (slow).")
        return "cpu"

    def _load_model(self, model_name):
        from ultralytics.models.sam import SAM3SemanticPredictor
        print(f"[INFO] Loading SAM3 model: {model_name} ...")
        overrides = dict(
            conf=SAM3_CONF,
            task="segment",
            mode="predict",
            model=str(model_name),
            imgsz=SAM3_IMGSZ,
            device=self.device,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        print("[INFO] SAM3 ready.")

    # ----------------------------------------------------------
    # Module A — Sketch feature extraction (unchanged from V3)
    # ----------------------------------------------------------
    def extract_sketch_features(self, sketch_path: Path) -> dict:
        img = cv2.imread(str(sketch_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load sketch: {sketch_path}")

        if img.ndim == 2:
            gray = img
        elif img.shape[2] == 4:
            alpha = img[:, :, 3]
            gray  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            if np.mean(gray[alpha < 10]) < 128:
                gray = cv2.bitwise_not(gray)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in sketch — check that lines are visible.")

        main_cnt = max(contours, key=cv2.contourArea)
        area     = cv2.contourArea(main_cnt)
        M        = cv2.moments(main_cnt)
        if M["m00"] == 0:
            raise ValueError("Sketch contour has zero area.")

        cx        = M["m10"] / M["m00"]
        cy        = M["m01"] / M["m00"]
        mu20      = M["mu20"] / M["m00"]
        mu02      = M["mu02"] / M["m00"]
        mu11      = M["mu11"] / M["m00"]
        angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        print(f"[Sketch] centroid=({cx:.1f},{cy:.1f})  "
              f"angle={np.degrees(angle_rad):.1f}°  area={area:.0f} px²")

        return {
            "contour":   main_cnt,
            "binary":    binary,
            "centroid":  (cx, cy),
            "angle_deg": np.degrees(angle_rad),
            "hu":        cv2.HuMoments(M).flatten(),
            "min_rect":  cv2.minAreaRect(main_cnt),
            "area":      area,
        }

    # ----------------------------------------------------------
    # Contour helpers
    # ----------------------------------------------------------
    @staticmethod
    def _unified_contour(mask_bool: np.ndarray, close_px: int = 0):
        """
        Extract unified contour from a merged mask.
        When close_px > 0, apply morphological close to fill gaps between sub-regions,
        ensuring a single continuous outer contour rather than multiple independent lines.
        Returns the largest outer contour; the mask itself is not modified.
        """
        seg = mask_bool.astype(np.uint8) * 255
        if close_px > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
            )
            seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kern)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cnts, key=cv2.contourArea) if cnts else None

    # ----------------------------------------------------------
    # Image enhancement (unchanged from V3)
    # ----------------------------------------------------------
    @staticmethod
    def _apply_clahe(img_bgr: np.ndarray,
                     clip_limit: float = 3.0,
                     tile_grid: tuple  = (8, 8)) -> np.ndarray:
        lab          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b      = cv2.split(lab)
        clahe        = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        lab_enhanced = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ----------------------------------------------------------
    # Module B — SAM3 text-prompt segmentation
    # ----------------------------------------------------------
    def segment_aerial(self, aerial_path: Path, sketch_features: dict):
        """
        SAM3 receives a concept prompt and returns only semantically relevant masks.
        Shape matching is kept as a secondary ranking step in case multiple masks
        are returned for the same concept.
        """
        img_bgr = cv2.imread(str(aerial_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load aerial image: {aerial_path}")

        h, w  = img_bgr.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_IMAGE_DIM:
            scale   = MAX_IMAGE_DIM / max(h, w)
            new_w   = int(w * scale)
            new_h   = int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w    = new_h, new_w
            print(f"[WARN] Aerial resized to {new_w}×{new_h} (scale={scale:.3f})")

        if CLAHE_ENABLED:
            img_bgr = self._apply_clahe(img_bgr, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)
            print(f"[INFO] CLAHE applied (clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID})")
        else:
            print("[INFO] CLAHE skipped")

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        total_px = h * w

        print(f"[INFO] Running SAM3 with prompts: {SAM3_PROMPTS} ...")
        # set_image expects BGR numpy array (same as cv2.imread output)
        self.predictor.set_image(img_bgr)
        results = self.predictor(text=SAM3_PROMPTS)

        # --- Extract masks from ultralytics Results ---
        raw_total = sum(len(r.masks.data) for r in results if r.masks is not None)
        print(f"[DEBUG] SAM3 raw output: {raw_total} masks before any filter")

        mask_list = []
        for r in results:
            if r.masks is None:
                continue

            masks_np = r.masks.data.cpu().numpy()      # [N, H, W]  float 0~1

            # confidence scores: use box conf if present, else 1.0 per mask
            if r.boxes is not None and len(r.boxes.conf) == len(masks_np):
                confs = r.boxes.conf.cpu().numpy()
            else:
                confs = np.ones(len(masks_np), dtype=np.float32)

            for mask_arr, conf in zip(masks_np, confs):
                # ultralytics masks may be at inference resolution; resize to match img_bgr
                if mask_arr.shape != (h, w):
                    mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_LINEAR)

                mask_bool = mask_arr > 0.5
                area      = int(mask_bool.sum())
                frac      = area / total_px

                if not (MIN_MASK_FRACTION <= frac <= MAX_MASK_FRACTION):
                    print(f"  [skip] mask fraction={frac:.3f} out of range "
                          f"({MIN_MASK_FRACTION}~{MAX_MASK_FRACTION})")
                    continue

                seg  = mask_bool.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea)
                mask_list.append({
                    "mask":      mask_bool,
                    "contour":   cnt,
                    "area":      area,
                    "iou_score": float(conf),
                })

        print(f"[INFO] SAM3 returned {len(mask_list)} masks after area filter")

        if not mask_list:
            raise RuntimeError(
                f"SAM3 found no masks for prompts {SAM3_PROMPTS} that pass the area filter "
                f"({MIN_MASK_FRACTION}–{MAX_MASK_FRACTION}).\n"
                "Suggestions:\n"
                "  1. Try a different prompt (e.g. 'building', 'rooftop', 'structure')\n"
                "  2. Widen MIN/MAX_MASK_FRACTION\n"
                "  3. Try a larger model: sam3_b.pt or sam3_l.pt"
            )

        # --- Negative prompt filter ---
        if SAM3_NEGATIVE_PROMPTS:
            self.predictor.set_image(img_bgr)
            neg_masks = []
            for r in self.predictor(text=SAM3_NEGATIVE_PROMPTS):
                if r.masks is None:
                    continue
                for mask_arr in r.masks.data.cpu().numpy():
                    if mask_arr.shape != (h, w):
                        mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_LINEAR)
                    m = mask_arr > 0.5
                    if m.any():
                        neg_masks.append(m.ravel())

            if neg_masks:
                kept_list = []
                for mc in mask_list:
                    flat = mc["mask"].ravel()
                    max_iou = max(
                        np.logical_and(flat, nm).sum() /
                        (np.logical_or(flat, nm).sum() + 1e-9)
                        for nm in neg_masks
                    )
                    if max_iou > NEG_IOU_THRESH:
                        print(f"  [neg-drop] area={mc['area']}  max_neg_iou={max_iou:.3f}")
                    else:
                        kept_list.append(mc)
                print(f"[INFO] Negative filter: removed {len(mask_list) - len(kept_list)} mask(s) "
                      f"(prompts={SAM3_NEGATIVE_PROMPTS}, iou_thresh={NEG_IOU_THRESH})")
                mask_list = kept_list

        # --- Overlap dedup: drop smaller mask if covered > OVERLAP_CONTAIN_THRESH ---
        if OVERLAP_CONTAIN_THRESH > 0:
            mask_list.sort(key=lambda x: x["area"], reverse=True)  # 大的优先
            deduped = []
            for mc in mask_list:
                contained = False
                for kept in deduped:
                    inter = int(np.logical_and(mc["mask"], kept["mask"]).sum())
                    ratio = inter / mc["area"]
                    if ratio > OVERLAP_CONTAIN_THRESH:
                        print(f"  [overlap-drop] area={mc['area']}  contain_ratio={ratio:.3f}")
                        contained = True
                        break
                if not contained:
                    deduped.append(mc)
            removed = len(mask_list) - len(deduped)
            if removed:
                print(f"[INFO] Overlap dedup removed {removed} contained mask(s)")
            mask_list = deduped

        # save original masks for candidate selection; auto-merge appends composites but user picks from originals
        original_masks = list(mask_list)

        # --- Adjacent mask merge: add composite candidates; keep originals ---
        if MERGE_ADJACENT_PX > 0 and len(mask_list) > 1:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (MERGE_ADJACENT_PX * 2 + 1,) * 2
            )
            # union-find: group adjacent masks into clusters, supports n-way merging naturally
            n       = len(mask_list)
            parent  = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                parent[find(x)] = find(y)

            for i in range(n):
                dil_i = cv2.dilate(mask_list[i]["mask"].astype(np.uint8), kern).astype(bool)
                for j in range(i + 1, n):
                    if np.logical_and(dil_i, mask_list[j]["mask"]).any():
                        union(i, j)

            # each multi-member cluster generates one merged candidate
            from collections import defaultdict
            groups = defaultdict(list)
            for i in range(n):
                groups[find(i)].append(i)

            merged_candidates = []
            for members in groups.values():
                if len(members) < 2:
                    continue
                combined = mask_list[members[0]]["mask"].copy()
                for idx in members[1:]:
                    combined = np.logical_or(combined, mask_list[idx]["mask"])
                area = int(combined.sum())
                frac = area / total_px
                if not (MIN_MASK_FRACTION <= frac <= MAX_MASK_FRACTION):
                    continue
                seg  = combined.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                merged_candidates.append({
                    "mask":      combined,
                    "contour":   max(cnts, key=cv2.contourArea),
                    "area":      area,
                    "iou_score": min(mask_list[m]["iou_score"] for m in members),
                })
                print(f"  [merge] cluster{members} → area={area}px²")

            if merged_candidates:
                print(f"[INFO] Generated {len(merged_candidates)} merged candidate(s)")
                mask_list.extend(merged_candidates)
            else:
                print(f"[WARN] No adjacent clusters found within {MERGE_ADJACENT_PX}px — "
                      f"try increasing MERGE_ADJACENT_PX")

        # --- Hu-moment shape matching — initial ranking ---
        sketch_cnt = sketch_features["contour"]
        for mc in mask_list:
            mc["shape_score"] = cv2.matchShapes(
                sketch_cnt, mc["contour"], cv2.CONTOURS_MATCH_I1, 0
            )
        mask_list.sort(key=lambda x: x["shape_score"])

        # --- Top-K combo merge: stack top-1..K into extra candidates ---
        if TOP_K_COMBO_MERGE >= 2 and len(mask_list) >= 2:
            top_k = mask_list[:min(TOP_K_COMBO_MERGE, len(mask_list))]
            combo_candidates = []
            combined = top_k[0]["mask"].copy()
            for k_idx, mc in enumerate(top_k[1:], start=2):
                combined = np.logical_or(combined, mc["mask"])
                area = int(combined.sum())
                frac = area / total_px
                if not (MIN_MASK_FRACTION <= frac <= MAX_MASK_FRACTION):
                    print(f"  [top-{k_idx} combo] skip: frac={frac:.3f} out of range")
                    continue
                seg  = combined.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                new_mc = {
                    "mask":      combined.copy(),
                    "contour":   max(cnts, key=cv2.contourArea),
                    "area":      area,
                    "iou_score": min(m["iou_score"] for m in top_k[:k_idx]),
                }
                new_mc["shape_score"] = cv2.matchShapes(
                    sketch_cnt, new_mc["contour"], cv2.CONTOURS_MATCH_I1, 0
                )
                combo_candidates.append(new_mc)
                print(f"  [top-{k_idx} combo] area={area}px²  shape={new_mc['shape_score']:.4f}")

            if combo_candidates:
                mask_list.extend(combo_candidates)
                mask_list.sort(key=lambda x: x["shape_score"])

        # --- Flood fill: from best candidate, iteratively absorb all spatially adjacent masks ---
        if MERGE_ADJACENT_PX > 0 and len(mask_list) >= 2:
            kern_exp = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (MERGE_ADJACENT_PX * 2 + 1,) * 2
            )
            flood_mask    = mask_list[0]["mask"].copy()
            remaining_idx = list(range(1, len(mask_list)))
            absorbed_idx  = [0]
            changed = True
            while changed:
                changed = False
                dil = cv2.dilate(flood_mask.astype(np.uint8), kern_exp).astype(bool)
                still_left = []
                for idx in remaining_idx:
                    if np.logical_and(dil, mask_list[idx]["mask"]).any():
                        flood_mask = np.logical_or(flood_mask, mask_list[idx]["mask"])
                        absorbed_idx.append(idx)
                        changed = True
                    else:
                        still_left.append(idx)
                remaining_idx = still_left

            if len(absorbed_idx) > 1:
                area = int(flood_mask.sum())
                frac = area / total_px
                if MIN_MASK_FRACTION <= frac <= MAX_MASK_FRACTION:
                    seg  = flood_mask.astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        flood_mc = {
                            "mask":      flood_mask,
                            "contour":   max(cnts, key=cv2.contourArea),
                            "area":      area,
                            "iou_score": min(mask_list[k]["iou_score"] for k in absorbed_idx),
                        }
                        flood_mc["shape_score"] = cv2.matchShapes(
                            sketch_cnt, flood_mc["contour"], cv2.CONTOURS_MATCH_I1, 0
                        )
                        mask_list.append(flood_mc)
                        mask_list.sort(key=lambda x: x["shape_score"])
                        print(f"  [flood] best+{len(absorbed_idx)-1} adjacent → "
                              f"area={area}px²  shape={flood_mc['shape_score']:.4f}")

        top3 = [f"{m['shape_score']:.4f}" for m in mask_list[:3]]
        print(f"[INFO] Top-3 shape scores: {top3}")

        # --- Debug export ---
        orig_sorted = sorted(original_masks, key=lambda x: x["shape_score"])

        if EXPORT_CANDIDATES:
            crop_dir = OUTPUT_DIR / "ai_test_crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            for old in crop_dir.glob("candidate_*.png"):
                old.unlink()

            for rank, mc in enumerate(orig_sorted[:EXPORT_TOP_N]):
                x, y, bw, bh = cv2.boundingRect(mc["contour"])
                pad = int(max(bw, bh) * ZOOM_PAD_RATIO)
                x1  = max(0, x - pad)
                y1  = max(0, y - pad)
                x2  = min(img_bgr.shape[1], x + bw + pad)
                y2  = min(img_bgr.shape[0], y + bh + pad)
                crop = img_bgr[y1:y2, x1:x2].copy()

                shifted = mc["contour"] - np.array([x1, y1])
                cv2.drawContours(crop, [shifted], -1, (0, 140, 255), 2)

                label = (f"#{rank}  shape={mc['shape_score']:.4f}"
                         f"  area={mc['area']}  conf={mc['iou_score']:.3f}")
                cv2.putText(crop, label, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(crop, label, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.imwrite(str(crop_dir / f"candidate_{rank:02d}_shape{mc['shape_score']:.4f}.png"), crop)

            print(f"[Save] {len(orig_sorted[:EXPORT_TOP_N])} candidate crops → {crop_dir}")

        # --- Candidate selection: interactive / config / auto ---
        print("\n" + "─" * 62)
        print(f"  {'#':<5} {'shape_score':<14} {'area(px²)':<12} conf  (original masks)")
        print("─" * 62)
        for i, mc in enumerate(orig_sorted):
            print(f"  [{i}]   {mc['shape_score']:<14.4f} {mc['area']:<12} {mc['iou_score']:.3f}")
        print("─" * 62)

        # --- Bbox gap auto-merge ---
        auto_sel = None
        if AUTO_MERGE_PROXIMITY > 0 and len(orig_sorted) >= 2:
            diag = float(np.sqrt(img_bgr.shape[1] ** 2 + img_bgr.shape[0] ** 2))
            thresh = AUTO_MERGE_PROXIMITY * diag

            def _bbox_gap(mc_a, mc_b):
                ax, ay, aw, ah = cv2.boundingRect(mc_a["contour"])
                bx, by, bw, bh = cv2.boundingRect(mc_b["contour"])
                gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
                return float(np.sqrt(gap_x ** 2 + gap_y ** 2))

            print(f"[AUTO] image diagonal={diag:.0f}px  merge threshold={thresh:.1f}px")
            print(f"  {'pair':<10} {'bbox_gap(px)':<16} < threshold?")
            found = []
            for i in range(len(orig_sorted)):
                for j in range(i + 1, len(orig_sorted)):
                    gap = _bbox_gap(orig_sorted[i], orig_sorted[j])
                    hit = gap < thresh
                    print(f"  #{i} vs #{j}    {gap:<16.1f} {'<-- auto-merge' if hit else ''}")
                    if hit:
                        found.append((gap, i, j))

            if found:
                found.sort()
                _, i0, j0 = found[0]
                auto_sel = [i0, j0]
                print(f"[AUTO] bbox gap {found[0][0]:.1f}px < threshold {thresh:.1f}px "
                      f"→ auto-merging candidates #{i0} + #{j0}")

        if auto_sel is not None:
            sel = auto_sel
        elif AUTO_LARGEST_RATIO > 0 and len(orig_sorted) >= 2:
            by_area = sorted(range(len(orig_sorted)),
                             key=lambda i: orig_sorted[i]["area"], reverse=True)
            largest_area = orig_sorted[by_area[0]]["area"]
            second_area  = orig_sorted[by_area[1]]["area"]
            ratio = second_area / largest_area
            print(f"[AUTO] largest={largest_area}px²  second={second_area}px²  "
                  f"ratio={ratio:.2f}  threshold={AUTO_LARGEST_RATIO}")
            if ratio < AUTO_LARGEST_RATIO:
                sel = [by_area[0]]
                print(f"[AUTO] second < {AUTO_LARGEST_RATIO*100:.0f}% of largest → auto-select #{by_area[0]}")
            else:
                print(f"[AUTO] candidates are similar in size (ratio={ratio:.2f}), falling back to manual selection")
                sel = None  # 走下面的 MANUAL_MERGE_RANKS / 交互流程
        elif AUTO_LARGEST_RATIO > 0 and len(orig_sorted) == 1:
            sel = [0]
            print(f"[AUTO] only one candidate, auto-select #0")
        else:
            sel = None

        if sel is not None:
            pass
        elif MANUAL_MERGE_RANKS is None:
            # interactive mode: open export folder and wait for user input
            import os
            if EXPORT_CANDIDATES:
                try:
                    os.startfile(str(OUTPUT_DIR / "ai_test_crops"))
                except Exception:
                    pass
            raw = input(
                "Enter mask indices to merge (space- or comma-separated); single = select only; Enter = use #0:\n> "
            ).strip()
            if raw:
                try:
                    sel = [int(x) for x in raw.replace(",", " ").split()]
                except ValueError:
                    print("[WARN] Could not parse input, falling back to #0")
                    sel = [0]
            else:
                sel = [0]
        elif MANUAL_MERGE_RANKS:
            sel = list(MANUAL_MERGE_RANKS)
        else:
            sel = [0]

        # clamp indices and pick masks from orig_sorted
        sel = [s for s in sel if s < len(orig_sorted)]
        if not sel:
            sel = [0]

        if len(sel) == 1:
            best = orig_sorted[sel[0]]
            print(f"[INFO] Selected mask #{sel[0]}  shape={best['shape_score']:.4f}  "
                  f"area={best['area']} px²  conf={best['iou_score']:.4f}")
        else:
            combined = orig_sorted[sel[0]]["mask"].copy()
            for s in sel[1:]:
                combined = np.logical_or(combined, orig_sorted[s]["mask"])
            area = int(combined.sum())
            # morphological close fills gaps between sub-regions → single unified contour
            cnt = self._unified_contour(combined, close_px=MERGE_ADJACENT_PX)
            if cnt is None:
                cnt = orig_sorted[sel[0]]["contour"]
            best = {
                "mask":      combined,
                "contour":   cnt,
                "area":      area,
                "iou_score": min(orig_sorted[s]["iou_score"] for s in sel),
                "shape_score": cv2.matchShapes(sketch_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0),
            }
            print(f"[INFO] Merged masks {sel} → area={area}px²  shape={best['shape_score']:.4f}")

        perimeter       = cv2.arcLength(best["contour"], True)
        best["contour"] = cv2.approxPolyDP(best["contour"], APPROX_EPSILON * perimeter, True)
        print(f"[INFO] Contour smoothed → {len(best['contour'])} vertices")

        return img_bgr, img_rgb, [best], scale

    # ----------------------------------------------------------
    # Module D — Geometric alignment & overlay (unchanged from V3)
    # ----------------------------------------------------------
    def _compute_affine(self, sketch_features: dict, target_contour, img_shape) -> np.ndarray:
        sk_rect  = sketch_features["min_rect"]
        tgt_rect = cv2.minAreaRect(target_contour)
        tgt_ctr  = tgt_rect[0]

        sk_box  = cv2.boxPoints(sk_rect).astype(np.float32)
        tgt_box = cv2.boxPoints(tgt_rect).astype(np.float32)
        sk_cnt  = sketch_features["contour"]

        # pre-draw target mask for IoU scoring
        tgt_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(tgt_mask, [target_contour], -1, 255, -1)

        best_M, best_iou = None, -1.0
        for offset in range(4):
            src  = np.roll(sk_box, offset, axis=0)
            M, _ = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2), tgt_box.reshape(-1, 1, 2), method=cv2.LMEDS
            )
            if M is None:
                continue

            # transform real sketch contour; pixel-level IoU disambiguates 0°/180° (bbox alone cannot)
            transformed_cnt = cv2.transform(sk_cnt.reshape(-1, 1, 2), M).astype(np.int32)
            sk_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(sk_mask, [transformed_cnt], -1, 255, -1)

            intersection = cv2.bitwise_and(tgt_mask, sk_mask)
            union        = cv2.bitwise_or(tgt_mask, sk_mask)
            iou = cv2.countNonZero(intersection) / (cv2.countNonZero(union) + 1e-5)

            if iou > best_iou:
                best_iou, best_M = iou, M

        if best_M is None:
            raise RuntimeError("estimateAffinePartial2D failed for all 4 corner orderings.")

        scale = float(np.sqrt(best_M[0, 0] ** 2 + best_M[0, 1] ** 2))
        angle = float(np.degrees(np.arctan2(best_M[1, 0], best_M[0, 0])))
        print(f"[Align] scale={scale:.3f}  angle={angle:.1f}°  "
              f"center=({tgt_ctr[0]:.1f},{tgt_ctr[1]:.1f})  IoU={best_iou:.4f}")
        if not (0.05 < scale < 20.0):
            print(f"[WARN] Scale={scale:.3f} looks unreasonable — verify building is in aerial image.")
        return best_M

    @staticmethod
    def make_transparent(sketch_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
        rgba = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255 - gray
        return rgba

    def align_and_overlay(self, sketch_path, aerial_bgr, sketch_features, target_mask, output_dir):
        h, w = aerial_bgr.shape[:2]
        M    = self._compute_affine(sketch_features, target_mask["contour"], (h, w))

        sketch_rgba = self.make_transparent(cv2.imread(str(sketch_path)))
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "sketch_transparent.png"), sketch_rgba)

        warped_rgba = cv2.warpAffine(sketch_rgba, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255, 0))

        alpha_f = warped_rgba[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255.0
        result  = np.clip(
            aerial_bgr.astype(np.float32) * (1 - alpha_f) +
            warped_rgba[:, :, :3].astype(np.float32) * alpha_f,
            0, 255,
        ).astype(np.uint8)

        out_path = output_dir / "alignment_result.png"
        cv2.imwrite(str(out_path), result)
        print(f"[Save] Alignment result → {out_path}")
        return result

    # ----------------------------------------------------------
    # Visualization — 4-panel figure (unchanged from V3)
    # ----------------------------------------------------------
    def visualize_result(self, sketch_path, aerial_rgb, aerial_bgr, mask, result_img, output_dir):
        overlay = np.zeros_like(aerial_rgb)
        overlay[mask["mask"]] = [255, 165, 0]
        vis_bgr = cv2.cvtColor(
            cv2.addWeighted(aerial_rgb, 0.6, overlay, 0.4, 0), cv2.COLOR_RGB2BGR
        )
        cv2.drawContours(vis_bgr, [mask["contour"]], -1, (0, 140, 255), 2)
        cv2.imwrite(str(output_dir / "masks_preview.png"), vis_bgr)

        mask_iso = np.full_like(aerial_bgr, 240)
        mask_iso[mask["mask"]] = aerial_bgr[mask["mask"]]
        cv2.drawContours(mask_iso, [mask["contour"]], -1, (0, 140, 255), 2)

        x, y, bw, bh = cv2.boundingRect(mask["contour"])
        pad = int(max(bw, bh) * ZOOM_PAD_RATIO)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(aerial_bgr.shape[1], x + bw + pad)
        y2  = min(aerial_bgr.shape[0], y + bh + pad)
        zoom = result_img[y1:y2, x1:x2]

        sketch_bgr = cv2.imread(str(sketch_path))
        fig, axes  = plt.subplots(2, 2, figsize=(18, 14))
        panels = [
            (sketch_bgr, "① Original Sketch"),
            (vis_bgr,    f"② SAM3 Mask  (shape={mask['shape_score']:.4f}  conf={mask['iou_score']:.3f})"),
            (mask_iso,   "③ Target Mask (isolated)"),
            (zoom,       "④ Alignment Zoom"),
        ]
        for ax, (img, title) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.suptitle(f"Cross-modal Building Alignment v4 — SAM3 prompts: {SAM3_PROMPTS}",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        fig_path = output_dir / "final_comparison.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"[Save] Final comparison → {fig_path}")
        plt.show()

    # ----------------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------------
    def run(self, sketch_path: Path, aerial_path: Path, output_dir: Path):
        print("\n" + "=" * 60)
        print("  MODULE A — Sketch Feature Extraction")
        print("=" * 60)
        sketch_features = self.extract_sketch_features(sketch_path)

        print("\n" + "=" * 60)
        print(f"  MODULE B — SAM3 Segmentation  prompts={SAM3_PROMPTS}")
        print("=" * 60)
        aerial_bgr, aerial_rgb, mask_list, scale = self.segment_aerial(aerial_path, sketch_features)
        target_mask = mask_list[0]

        print("\n" + "=" * 60)
        print("  MODULE D — Geometric Alignment & Overlay")
        print("=" * 60)
        result_img = self.align_and_overlay(
            sketch_path, aerial_bgr, sketch_features, target_mask, output_dir
        )

        print("\n" + "=" * 60)
        print("  VISUALIZATION")
        print("=" * 60)
        self.visualize_result(
            sketch_path, aerial_rgb, aerial_bgr, target_mask, result_img, output_dir
        )

        return result_img


# ============================================================
# Entry point
# ============================================================
def _check_prerequisites(sketch_path: Path, aerial_path: Path):
    missing = [str(p) for p in [sketch_path, aerial_path] if not p.exists()]
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print(f"  • {m}")
        sys.exit(1)


def _resolve_path(p, base: Path) -> Path:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else base / p


def main():
    sketch_path = _resolve_path(SKETCH_PATH, BASE_DIR) or _find_image(BASE_DIR, "sketch")
    aerial_path = _resolve_path(AERIAL_PATH, BASE_DIR) or _find_image(BASE_DIR, "aerial")
    _check_prerequisites(sketch_path, aerial_path)

    print(f"[INFO] Sketch : {sketch_path}")
    print(f"[INFO] Aerial : {aerial_path}")

    matcher = BuildingMatcher(model_name=SAM3_MODEL)
    matcher.run(sketch_path=sketch_path, aerial_path=aerial_path, output_dir=OUTPUT_DIR)
    print("\n[DONE] Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
