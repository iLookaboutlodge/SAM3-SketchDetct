"""
Cross-modal Building Image Alignment System — v5.2
Batch version of v4.2: no interactive input, headless, model loaded once.

Output: output/pair_v5_2/
  alignment_result_1.png, final_comparison_1.png  (per pair)
"""

import sys
import matplotlib
matplotlib.use("Agg")   # headless — must be before any other matplotlib import

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
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
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "pair_v5_2"

MAX_IMAGE_DIM = 1024

# ★ Batch sketch / aerial pairs — add or remove as needed
SKETCH_AERIAL_PAIRS = [
    ("sketch_2.jpg",  "aerial_2.jpg"),
    ("sketch_3.png",  "aerial_3.png"),
    ("sketch.png",    "aerial.png"),
]

# ★ SAM3 — tuning (matches sam3_explore settings)
SAM3_MODEL  = BASE_DIR / "checkpoints" / "sam3.pt"
SAM3_PROMPTS = [
    "building",
    "outdoor building seating area",
]
SAM3_IMGSZ  = 640
SAM3_CONF   = 0.10

SAM3_NEGATIVE_PROMPTS = [
    "car",
    "tree",
    "road",
    "swimming pool",
]
NEG_IOU_THRESH = 0.3
IOU_DEDUP      = 0.5

# ★ Area filter
MIN_MASK_FRACTION = 0.01
MAX_MASK_FRACTION = 0.70

# ★ Bbox gap auto-merge: merge all candidates if any bbox gap < image diagonal × this ratio
# 0.1 = 10%; set to 0 to disable
AUTO_MERGE_PROXIMITY = 0.1

# ★ Contour smoothing & visualization
APPROX_EPSILON = 0.001
ZOOM_PAD_RATIO = 0.20

# ★ CLAHE
CLAHE_ENABLED    = False
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)

COLORS = [
    (255,  60,  60), (60, 180, 255), ( 60, 220,  60), (255, 180,  30),
    (180,  60, 255), (255, 100, 180), ( 60, 220, 200), (200, 200,  60),
]


# ============================================================
# BuildingMatcher v5.2
# ============================================================
class BuildingMatcher:

    def __init__(self, model_name=SAM3_MODEL):
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
            return "cuda"
        print("[WARN] CUDA unavailable — running on CPU (slow).")
        return "cpu"

    def _load_model(self, model_name):
        from ultralytics.models.sam import SAM3SemanticPredictor
        print(f"[INFO] Loading SAM3 model: {model_name} ...")
        overrides = dict(
            conf=SAM3_CONF, task="segment", mode="predict",
            model=str(model_name), imgsz=SAM3_IMGSZ,
            device=self.device, verbose=False, save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        print("[INFO] SAM3 ready.")

    # ----------------------------------------------------------
    # Module A — Sketch feature extraction
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
            raise ValueError("No contours found in sketch.")

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
    # Image enhancement
    # ----------------------------------------------------------
    @staticmethod
    def _apply_clahe(img_bgr, clip_limit=3.0, tile_grid=(8, 8)):
        lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    # ----------------------------------------------------------
    # Module B — Explore-style detection
    # ----------------------------------------------------------
    def detect_masks(self, img_bgr: np.ndarray) -> list[dict]:
        h, w     = img_bgr.shape[:2]
        total_px = h * w

        self.predictor.set_image(img_bgr)
        results = self.predictor(text=SAM3_PROMPTS)

        detections = []
        for r in results:
            if r.masks is None:
                continue
            masks_np = r.masks.data.cpu().numpy()
            confs    = (r.boxes.conf.cpu().numpy()
                        if r.boxes is not None and len(r.boxes.conf) == len(masks_np)
                        else np.ones(len(masks_np)))
            cls_ids  = (r.boxes.cls.cpu().numpy().astype(int)
                        if r.boxes is not None and len(r.boxes.cls) == len(masks_np)
                        else np.zeros(len(masks_np), int))

            for mask_arr, conf, cls_id in zip(masks_np, confs, cls_ids):
                if mask_arr.shape != (h, w):
                    mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_arr > 0.5

                # keep only the largest connected component; removes distant noise blobs in the same mask
                seg_tmp = mask_bool.astype(np.uint8) * 255
                cnts_tmp, _ = cv2.findContours(seg_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_tmp:
                    largest = max(cnts_tmp, key=cv2.contourArea)
                    seg_clean = np.zeros_like(seg_tmp)
                    cv2.drawContours(seg_clean, [largest], -1, 255, -1)
                    mask_bool = seg_clean > 0

                frac = mask_bool.sum() / total_px
                if not (MIN_MASK_FRACTION <= frac <= MAX_MASK_FRACTION):
                    continue
                prompt = SAM3_PROMPTS[cls_id] if cls_id < len(SAM3_PROMPTS) else f"class{cls_id}"
                detections.append({"prompt": prompt, "mask": mask_bool, "conf": float(conf)})

        print(f"[INFO] {len(detections)} mask(s) after area filter")

        # Per-prompt dedup: keep only the largest mask per prompt
        by_prompt = defaultdict(list)
        for det in detections:
            by_prompt[det["prompt"]].append(det)
        detections = []
        for prompt, dets in by_prompt.items():
            largest = max(dets, key=lambda d: d["mask"].sum())
            if len(dets) > 1:
                print(f"  [per-prompt] '{prompt}': {len(dets)} masks → keeping largest (conf={largest['conf']:.3f})")
            detections.append(largest)
        print(f"[INFO] {len(detections)} mask(s) after per-prompt dedup")

        # Negative filter
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
                before = len(detections)
                detections = [
                    det for det in detections
                    if not any(
                        np.logical_and(det["mask"].ravel(), nm).sum() /
                        (np.logical_or(det["mask"].ravel(), nm).sum() + 1e-9) > NEG_IOU_THRESH
                        for nm in neg_masks
                    )
                ]
                print(f"[INFO] Negative filter: removed {before - len(detections)} mask(s)")

        # IoU dedup (conf-priority)
        detections.sort(key=lambda x: x["conf"], reverse=True)
        kept = []
        for det in detections:
            flat = det["mask"].ravel()
            dup  = False
            for k in kept:
                inter = np.logical_and(flat, k["mask"].ravel()).sum()
                union = np.logical_or(flat, k["mask"].ravel()).sum()
                if union > 0 and inter / union > IOU_DEDUP:
                    dup = True
                    break
            if not dup:
                kept.append(det)

        print(f"[INFO] {len(kept)} unique mask(s) after IoU dedup (threshold={IOU_DEDUP})")
        return kept

    # ----------------------------------------------------------
    # Detection visualization (headless — saves, no plt.show)
    # ----------------------------------------------------------
    def save_detections(self, img_bgr: np.ndarray, detections: list[dict], out_path: Path):
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w     = img_rgb.shape[:2]
        BG_COLOR = np.array([245, 242, 230], dtype=np.float32)
        overlay  = np.full((h, w, 3), BG_COLOR, dtype=np.float32)

        legend_patches = []
        for i, det in enumerate(detections):
            color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
            m = det["mask"]
            overlay[m] = img_rgb.astype(np.float32)[m] * 0.5 + color * 0.5
            seg  = m.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, color.tolist(), 2)
            legend_patches.append(
                mpatches.Patch(color=color / 255,
                               label=f"[{i}] {det['prompt']}  {det['conf']:.2f}")
            )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        for i, det in enumerate(detections):
            seg  = det["mask"].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                Mc = cv2.moments(max(cnts, key=cv2.contourArea))
                if Mc["m00"] > 0:
                    cx = int(Mc["m10"] / Mc["m00"])
                    cy = int(Mc["m01"] / Mc["m00"])
                    cv2.putText(overlay, str(i), (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(overlay, str(i), (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        fig, (ax_orig, ax_det) = plt.subplots(1, 2, figsize=(22, 10))
        ax_orig.imshow(img_rgb);  ax_orig.set_title("Original", fontsize=11);  ax_orig.axis("off")
        ax_det.imshow(overlay);   ax_det.set_title(f"SAM3 — {len(detections)} masks", fontsize=11)
        ax_det.axis("off")
        ax_det.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.8)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Save] Detection preview → {out_path}")

    # ----------------------------------------------------------
    # Module D — Affine alignment
    # ----------------------------------------------------------
    def _compute_affine(self, sketch_features: dict, tgt_pts, tgt_mask_bool, img_shape) -> np.ndarray:
        sk_rect  = sketch_features["min_rect"]
        tgt_rect = cv2.minAreaRect(tgt_pts)
        tgt_ctr  = tgt_rect[0]
        sk_box   = cv2.boxPoints(sk_rect).astype(np.float32)
        tgt_box  = cv2.boxPoints(tgt_rect).astype(np.float32)
        sk_cnt   = sketch_features["contour"]

        tgt_mask = tgt_mask_bool.astype(np.uint8) * 255

        best_M, best_iou = None, -1.0
        for offset in range(4):
            src  = np.roll(sk_box, offset, axis=0)
            M, _ = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2), tgt_box.reshape(-1, 1, 2), method=cv2.LMEDS
            )
            if M is None:
                continue
            transformed_cnt = cv2.transform(sk_cnt.reshape(-1, 1, 2), M).astype(np.int32)
            sk_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(sk_mask, [transformed_cnt], -1, 255, -1)
            inter = cv2.bitwise_and(tgt_mask, sk_mask)
            union = cv2.bitwise_or(tgt_mask, sk_mask)
            iou   = cv2.countNonZero(inter) / (cv2.countNonZero(union) + 1e-5)
            if iou > best_iou:
                best_iou, best_M = iou, M

        if best_M is None:
            raise RuntimeError("estimateAffinePartial2D failed for all 4 orderings.")

        scale = float(np.sqrt(best_M[0, 0] ** 2 + best_M[0, 1] ** 2))
        angle = float(np.degrees(np.arctan2(best_M[1, 0], best_M[0, 0])))
        print(f"[Align] scale={scale:.3f}  angle={angle:.1f}°  "
              f"center=({tgt_ctr[0]:.1f},{tgt_ctr[1]:.1f})  IoU={best_iou:.4f}")
        return best_M

    @staticmethod
    def make_transparent(sketch_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
        rgba = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255 - gray
        return rgba

    def align_and_overlay(self, sketch_path, aerial_bgr, sketch_features,
                          merged_bool, tgt_pts, output_dir, suffix=""):
        h, w = aerial_bgr.shape[:2]
        M    = self._compute_affine(sketch_features, tgt_pts, merged_bool, (h, w))

        sketch_rgba = self.make_transparent(cv2.imread(str(sketch_path)))
        output_dir.mkdir(parents=True, exist_ok=True)

        warped_rgba = cv2.warpAffine(sketch_rgba, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255, 0))
        alpha_f = warped_rgba[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255.0
        result  = np.clip(
            aerial_bgr.astype(np.float32) * (1 - alpha_f) +
            warped_rgba[:, :, :3].astype(np.float32) * alpha_f,
            0, 255,
        ).astype(np.uint8)

        out_path = output_dir / f"alignment_result{suffix}.png"
        cv2.imwrite(str(out_path), result)
        print(f"[Save] Alignment result → {out_path}")
        return result

    # ----------------------------------------------------------
    # Full pipeline (single pair)
    # ----------------------------------------------------------
    def run(self, sketch_path: Path, aerial_path: Path, output_dir: Path, pair_idx: int = 0):
        suffix = f"_{pair_idx + 1}"

        print("\n" + "=" * 60)
        print(f"  PAIR {pair_idx + 1}  —  {sketch_path.name}  +  {aerial_path.name}")
        print("=" * 60)

        print("\n  MODULE A — Sketch Feature Extraction")
        sketch_features = self.extract_sketch_features(sketch_path)

        # Load & preprocess aerial
        img_bgr = cv2.imread(str(aerial_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load aerial: {aerial_path}")

        h, w  = img_bgr.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_IMAGE_DIM:
            scale        = MAX_IMAGE_DIM / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr      = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w         = new_h, new_w
            print(f"[WARN] Aerial resized to {new_w}×{new_h} (scale={scale:.3f})")

        if CLAHE_ENABLED:
            img_bgr = self._apply_clahe(img_bgr, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)

        print(f"\n  MODULE B — SAM3 Detection  prompts={SAM3_PROMPTS}")
        detections = self.detect_masks(img_bgr)

        if not detections:
            raise RuntimeError("No masks detected — try different prompts or lower SAM3_CONF.")

        # Bbox gap auto-merge
        def _bbox_gap(det_a, det_b):
            def _rect(det):
                seg = det["mask"].astype(np.uint8) * 255
                cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return cv2.boundingRect(max(cnts, key=cv2.contourArea)) if cnts else (0, 0, 0, 0)
            ax, ay, aw, ah     = _rect(det_a)
            bx, by, bw2, bh2   = _rect(det_b)
            gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw2))
            gap_y = max(0, max(ay, by) - min(ay + ah, by + bh2))
            return float(np.sqrt(gap_x ** 2 + gap_y ** 2))

        auto_merged = False
        if AUTO_MERGE_PROXIMITY > 0 and len(detections) >= 2:
            h_img, w_img = img_bgr.shape[:2]
            diag   = float(np.sqrt(w_img ** 2 + h_img ** 2))
            thresh = AUTO_MERGE_PROXIMITY * diag
            print(f"[AUTO] image diagonal={diag:.0f}px  merge threshold={thresh:.1f}px")
            pairs = []
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    gap = _bbox_gap(detections[i], detections[j])
                    print(f"  #{i} vs #{j}  bbox_gap={gap:.1f}px"
                          + ("  <-- auto-merge" if gap < thresh else ""))
                    if gap < thresh:
                        pairs.append((gap, i, j))
            if pairs:
                to_merge = set()
                for _, i, j in pairs:
                    to_merge.add(i)
                    to_merge.add(j)
                merged = detections[next(iter(to_merge))]["mask"].copy()
                for idx in list(to_merge)[1:]:
                    merged = np.logical_or(merged, detections[idx]["mask"])
                print(f"[AUTO] auto-merging candidates {sorted(to_merge)}")
                auto_merged = True

        # Auto-merge not triggered: single → pick it; multiple → merge all
        if not auto_merged:
            if len(detections) == 1:
                merged = detections[0]["mask"]
                print(f"[AUTO] only one candidate, auto-select #0 "
                      f"({detections[0]['prompt']}  conf={detections[0]['conf']:.3f})")
            else:
                # Save detection preview for inspection, then merge all
                preview_path = output_dir / f"detection_preview{suffix}.png"
                self.save_detections(img_bgr, detections, preview_path)
                merged = detections[0]["mask"].copy()
                for det in detections[1:]:
                    merged = np.logical_or(merged, det["mask"])
                labels = [f"#{i} {d['prompt']}" for i, d in enumerate(detections)]
                print(f"[AUTO] multiple candidates, merging all: {labels}")

        seg  = merged.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("Merged mask has no contour.")
        all_pts = np.concatenate(cnts)

        print("\n  MODULE D — Geometric Alignment & Overlay")
        result_img = self.align_and_overlay(
            sketch_path, img_bgr, sketch_features, merged, all_pts, output_dir, suffix
        )

        # Final comparison: 4-panel figure
        aerial_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay_vis = np.zeros_like(aerial_rgb)
        overlay_vis[merged] = [255, 165, 0]
        mask_vis = cv2.cvtColor(
            cv2.addWeighted(aerial_rgb, 0.6, overlay_vis, 0.4, 0), cv2.COLOR_RGB2BGR
        )
        cv2.drawContours(mask_vis, cnts, -1, (0, 140, 255), 2)

        mask_iso = np.full_like(img_bgr, 240)
        mask_iso[merged] = img_bgr[merged]
        cv2.drawContours(mask_iso, cnts, -1, (0, 140, 255), 2)

        x, y, bw, bh = cv2.boundingRect(all_pts)
        pad = int(max(bw, bh) * ZOOM_PAD_RATIO)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_bgr.shape[1], x + bw + pad), min(img_bgr.shape[0], y + bh + pad)
        zoom = result_img[y1:y2, x1:x2]

        sketch_bgr = cv2.imread(str(sketch_path))
        fig, axes  = plt.subplots(2, 2, figsize=(18, 14))
        panels = [
            (sketch_bgr, "① Original Sketch"),
            (mask_vis,   "② SAM3 Mask"),
            (mask_iso,   "③ Target Mask (isolated)"),
            (zoom,       "④ Alignment Zoom"),
        ]
        for ax, (img_panel, title) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(img_panel, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.suptitle(
            f"Building Alignment v5.2 — pair {pair_idx + 1} — prompts: {SAM3_PROMPTS}",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        fig_path = output_dir / f"final_comparison{suffix}.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Save] Final comparison → {fig_path}")

        return result_img


# ============================================================
# Entry point
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matcher = BuildingMatcher(model_name=SAM3_MODEL)

    ok, failed = [], []
    for idx, (sketch_stem, aerial_stem) in enumerate(SKETCH_AERIAL_PAIRS):
        sketch_path = BASE_DIR / sketch_stem
        aerial_path = BASE_DIR / aerial_stem

        missing = [p for p in [sketch_path, aerial_path] if not p.exists()]
        if missing:
            print(f"[SKIP] Pair {idx + 1} — missing files: {missing}")
            failed.append((idx + 1, "file not found"))
            continue

        try:
            matcher.run(sketch_path=sketch_path, aerial_path=aerial_path,
                        output_dir=OUTPUT_DIR, pair_idx=idx)
            ok.append(idx + 1)
        except Exception as exc:
            print(f"[ERROR] Pair {idx + 1} failed: {exc}")
            failed.append((idx + 1, str(exc)))

    print("\n" + "=" * 60)
    print(f"  DONE  {len(ok)} OK  /  {len(failed)} failed")
    if failed:
        for pair_n, reason in failed:
            print(f"    Pair {pair_n}: {reason}")
    print(f"  Output → {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
