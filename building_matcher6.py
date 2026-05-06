"""
Cross-modal Building Image Alignment System — v6.0
Extends v5.2 with two new pipeline modules:

  Module E — Multi-dimensional confidence score
             (IoU + HD95 + Chamfer + PoLiS, composite weighted score)
  Module F — Discrepancy detection
             (extra buildings in aerial not in sketch; missing parts in sketch not in aerial)

New outputs per pair:
  discrepancy_map_N.png  — aerial with color-coded diff overlays + legend
  final_comparison_N.png — 6-panel (adds ⑤ discrepancy map, ⑥ confidence bar chart)
"""

import sys
import matplotlib
matplotlib.use("Agg")   # headless — must be before any other matplotlib import

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path

from scipy.spatial import cKDTree

try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("[WARN] shapely not installed — PoLiS metric disabled (pip install shapely)")


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
OUTPUT_DIR = BASE_DIR / "output" / "pair_v6"

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

# ★ Contour smoothing & visualization
APPROX_EPSILON = 0.001
ZOOM_PAD_RATIO = 0.20

# ★ CLAHE
CLAHE_ENABLED    = False
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)

# ★ Module E — Confidence scoring weights (must sum to 1.0)
# Exponential decay: distance at which similarity = exp(-1) ≈ 0.37 is CONF_DECAY_FRAC × image diagonal
CONF_WEIGHT_IOU     = 0.30   # alignment IoU
CONF_WEIGHT_HD95    = 0.20   # 95th-percentile Hausdorff boundary similarity
CONF_WEIGHT_CHAMFER = 0.15   # average Chamfer boundary similarity
CONF_WEIGHT_POLIS   = 0.10   # PoLiS polygon-line similarity
CONF_WEIGHT_EXTRA   = 0.25   # extra-area penalty: more uncovered aerial area → lower score
CONF_DECAY_FRAC     = 0.05   # 5% of image diagonal = "half-score" distance

# ★ Multi-building handling
MULTI_BLDG_PROXIMITY     = 0.10   # bbox gap < this fraction of diagonal → "close" → merge all (same as v5.2)
MULTI_BLDG_IOU_THRESH    = 0.35   # merged IoU below this (secondary guard)
MULTI_BLDG_MISSING_THRESH = 0.10  # missing area > this fraction of merged building → likely unmatched building exists

# ★ Module F — Discrepancy filtering
DISCREP_MIN_AREA_FRAC   = 0.02   # ignore regions < 2% of max(sketch, aerial) building area
DISCREP_MIN_COMPACTNESS = 0.05   # isoperimetric quotient 4π·A/P² < this → discard slivers

COLORS = [
    (255,  60,  60), (60, 180, 255), ( 60, 220,  60), (255, 180,  30),
    (180,  60, 255), (255, 100, 180), ( 60, 220, 200), (200, 200,  60),
]


# ============================================================
# BuildingMatcher v6.0
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

        # Build warped sketch mask for Module E/F use
        warped_cnt  = cv2.transform(sketch_features["contour"].reshape(-1, 1, 2), M).astype(np.int32)
        warped_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(warped_mask, [warped_cnt], -1, 255, -1)

        return result, (warped_mask > 0)

    # ----------------------------------------------------------
    # Module E helpers — distance metrics
    # ----------------------------------------------------------
    @staticmethod
    def _boundary_points(mask_bool: np.ndarray) -> np.ndarray:
        """All boundary pixel coordinates as (N, 2) float32 array (row, col)."""
        seg = mask_bool.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.concatenate([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
        return pts[:, ::-1]   # (x,y) → (row, col)

    @staticmethod
    def _hd95(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        """95th-percentile symmetric Hausdorff distance in pixels."""
        if not len(pts_a) or not len(pts_b):
            return float("inf")
        d_ab = cKDTree(pts_b).query(pts_a, k=1)[0]
        d_ba = cKDTree(pts_a).query(pts_b, k=1)[0]
        return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))

    @staticmethod
    def _chamfer(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        """Symmetric Chamfer distance: average of mean nearest-neighbor distances."""
        if not len(pts_a) or not len(pts_b):
            return float("inf")
        d_ab = cKDTree(pts_b).query(pts_a, k=1)[0]
        d_ba = cKDTree(pts_a).query(pts_b, k=1)[0]
        return float((d_ab.mean() + d_ba.mean()) / 2.0)

    @staticmethod
    def _polis(mask_a: np.ndarray, mask_b: np.ndarray):
        """PoLiS metric: average vertex-to-nearest-edge distance, both directions.
        Returns None if shapely is unavailable or polygon extraction fails."""
        if not SHAPELY_AVAILABLE:
            return None

        def _to_poly(mask_bool):
            seg = mask_bool.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            cnt = max(cnts, key=cv2.contourArea)
            eps = APPROX_EPSILON * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, eps, True)
            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                return None
            poly = ShapelyPolygon(pts)
            return poly if poly.is_valid else poly.buffer(0)

        try:
            poly_a = _to_poly(mask_a)
            poly_b = _to_poly(mask_b)
            if poly_a is None or poly_b is None:
                return None

            boundary_b = poly_b.boundary
            boundary_a = poly_a.boundary

            def _mean_to_boundary(poly_src, bnd_tgt):
                coords = np.array(poly_src.exterior.coords)
                dists  = [Point(p).distance(bnd_tgt) for p in coords]
                return float(np.mean(dists)) if dists else float("inf")

            d_ab = _mean_to_boundary(poly_a, boundary_b)
            d_ba = _mean_to_boundary(poly_b, boundary_a)
            return float((d_ab + d_ba) / 2.0)
        except Exception as exc:
            print(f"[WARN] PoLiS computation failed: {exc}")
            return None

    # ----------------------------------------------------------
    # Multi-building selection helpers
    # ----------------------------------------------------------
    @staticmethod
    def _bbox_gap(mask_a: np.ndarray, mask_b: np.ndarray, h: int, w: int) -> float:
        """Normalised gap between the bounding boxes of two masks (0 = touching/overlapping)."""
        diag = float(np.sqrt(h ** 2 + w ** 2))

        def _bbox(m):
            seg = m.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            x, y, bw, bh = cv2.boundingRect(np.concatenate(cnts))
            return x, y, x + bw, y + bh   # left, top, right, bottom

        ba = _bbox(mask_a)
        bb = _bbox(mask_b)
        if ba is None or bb is None:
            return float("inf")
        gap_x = max(0, max(ba[0], bb[0]) - min(ba[2], bb[2]))
        gap_y = max(0, max(ba[1], bb[1]) - min(ba[3], bb[3]))
        return float(np.sqrt(gap_x ** 2 + gap_y ** 2)) / diag

    def _quick_alignment_score(self, sketch_features: dict, target_mask: np.ndarray,
                                img_shape: tuple, tgt_pts=None):
        """Run affine alignment silently; return (iou, extra_px, missing_px).
        tgt_pts: contour array passed to _compute_affine (drives minAreaRect).
                 If None, defaults to the largest contour of target_mask.
                 Must match the all_pts that will be used in align_and_overlay."""
        h, w = img_shape
        seg  = target_mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0, int(target_mask.sum()), 0
        if tgt_pts is None:
            tgt_pts = max(cnts, key=cv2.contourArea)
        try:
            M = self._compute_affine(sketch_features, tgt_pts, target_mask, img_shape)
        except RuntimeError:
            return 0.0, int(target_mask.sum()), 0
        warped_cnt  = cv2.transform(
            sketch_features["contour"].reshape(-1, 1, 2), M
        ).astype(np.int32)
        warped_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(warped_mask, [warped_cnt], -1, 255, -1)
        warped_bool = warped_mask > 0
        inter      = int(np.logical_and(target_mask, warped_bool).sum())
        union      = int(np.logical_or(target_mask,  warped_bool).sum())
        iou        = float(inter / (union + 1e-9))
        extra_bool   = np.logical_and(target_mask,               np.logical_not(warped_bool))
        missing_bool = np.logical_and(warped_bool, np.logical_not(target_mask))
        extra_px     = int(extra_bool.sum())
        missing_px   = int(missing_bool.sum())

        # Count distinct regions (same min-area filter as detect_discrepancies)
        building_area = max(int(target_mask.sum()), int(warped_bool.sum()))
        min_area      = DISCREP_MIN_AREA_FRAC * building_area

        def _count_regions(bool_mask):
            seg = bool_mask.astype(np.uint8) * 255
            n, _, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
            return sum(1 for i in range(1, n)
                       if stats[i, cv2.CC_STAT_AREA] >= min_area)

        n_extra   = _count_regions(extra_bool)
        n_missing = _count_regions(missing_bool)
        return iou, extra_px, missing_px, n_extra, n_missing

    def _select_target_mask(self, detections: list, sketch_features: dict,
                             h: int, w: int) -> tuple:
        """
        Choose which mask(s) the sketch aligns to.

        Strategy
        --------
        1. One detection  → use it directly.
        2. Multiple detections → always try merged mask first.
           Gap (CLOSE vs FAR) only controls how all_pts is built for minAreaRect:
             CLOSE (centroid gap < MULTI_BLDG_PROXIMITY): all contour pts → full footprint
             FAR:                                          largest contour only → avoid spanning two buildings
           After the merged trial, regardless of CLOSE/FAR:
             • IoU sufficient (≥ MULTI_BLDG_IOU_THRESH)   → use merged, no unmatched
             • missing ratio small (≤ MULTI_BLDG_MISSING_THRESH) → extra building alongside, use merged
             • IoU low AND missing large                   → sketch overflows merged bbox
               → try each mask individually, pick best (least extra+missing)
               → remaining masks become unmatched buildings

        Returns
        -------
        (target_mask, all_pts, unmatched_masks)
          target_mask     — bool H×W used for alignment + discrepancy
          all_pts         — contour array driving minAreaRect in _compute_affine
          unmatched_masks — list[bool H×W] for extra detected buildings (shown orange)
        """
        def _merge(dets):
            m = dets[0]["mask"].copy()
            for d in dets[1:]:
                m = np.logical_or(m, d["mask"])
            return m

        def _all_contour_pts(mask_bool):
            seg = mask_bool.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return np.concatenate(cnts) if cnts else None

        def _largest_cnt(mask_bool):
            seg = mask_bool.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return max(cnts, key=cv2.contourArea) if cnts else None

        if len(detections) == 1:
            tgt = detections[0]["mask"]
            return tgt, _largest_cnt(tgt), []

        n = len(detections)

        # Build proximity clusters via union-find on pairwise bbox gap
        parent = list(range(n))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x, y):
            parent[_find(x)] = _find(y)

        for i in range(n):
            for j in range(i + 1, n):
                g = self._bbox_gap(detections[i]["mask"], detections[j]["mask"], h, w)
                print(f"  [gap] mask #{i} vs #{j}: bbox_gap={g:.3f}"
                      + (" → same cluster" if g < MULTI_BLDG_PROXIMITY else ""))
                if g < MULTI_BLDG_PROXIMITY:
                    _union(i, j)

        # Group indices by cluster root
        from collections import defaultdict
        clusters = defaultdict(list)
        for i in range(n):
            clusters[_find(i)].append(i)
        cluster_list = list(clusters.values())
        print(f"[Multi-bldg] {len(cluster_list)} cluster(s): "
              + "  ".join(f"[{','.join(f'#{i}' for i in c)}]" for c in cluster_list))

        if len(cluster_list) == 1:
            # All masks in one cluster → merge all (v5.2 style)
            merged_all  = _merge(detections)
            merged_pts  = _all_contour_pts(merged_all)
            merged_iou, merged_extra, merged_missing, _, _ = self._quick_alignment_score(
                sketch_features, merged_all, (h, w), tgt_pts=merged_pts
            )
            building_area = int(merged_all.sum())
            missing_ratio = merged_missing / (building_area + 1e-9)
            print(f"[Multi-bldg] Single cluster merged trial: "
                  f"IoU={merged_iou:.3f}  missing={missing_ratio:.1%}")
            if missing_ratio <= MULTI_BLDG_MISSING_THRESH:
                return merged_all, merged_pts, []
            # Still too much missing even within one cluster → fall through to per-cluster scoring
            cluster_list = [[i] for i in range(n)]   # treat each mask as its own cluster
            print(f"[Multi-bldg] Missing too large — re-evaluating each mask individually")

        # Multiple clusters (or per-mask fallback): trial each cluster against sketch
        best_cluster_idx, best_score = 0, float("inf")
        for ci, indices in enumerate(cluster_list):
            cluster_dets  = [detections[i] for i in indices]
            cluster_mask  = _merge(cluster_dets)
            # CLOSE cluster → all contour pts; single mask → largest
            cluster_pts   = (_all_contour_pts(cluster_mask)
                             if len(indices) > 1 else _largest_cnt(cluster_mask))
            iou_c, _, _, n_extra_c, n_missing_c = self._quick_alignment_score(
                sketch_features, cluster_mask, (h, w), tgt_pts=cluster_pts
            )
            score_c = n_extra_c + n_missing_c
            print(f"  Cluster {ci} (masks {[f'#{i}' for i in indices]}): "
                  f"IoU={iou_c:.3f}  extra_regions={n_extra_c}  "
                  f"missing_regions={n_missing_c}  total={score_c}")
            if score_c < best_score:
                best_score, best_cluster_idx = score_c, ci

        best_indices    = cluster_list[best_cluster_idx]
        unmatched_indices = [i for ci, grp in enumerate(cluster_list)
                             if ci != best_cluster_idx for i in grp]
        best_dets       = [detections[i] for i in best_indices]
        target_mask     = _merge(best_dets)
        target_pts      = (_all_contour_pts(target_mask)
                           if len(best_indices) > 1 else _largest_cnt(target_mask))
        unmatched_masks = [detections[i]["mask"] for i in unmatched_indices]
        print(f"[Multi-bldg] Best cluster → {[f'#{i}' for i in best_indices]}  "
              f"(total_regions={best_score})  "
              f"unmatched: {[f'#{i}' for i in unmatched_indices]}")
        return target_mask, target_pts, unmatched_masks

    # ----------------------------------------------------------
    # Module E — Confidence scoring
    # ----------------------------------------------------------
    def compute_confidence_score(self, warped_sketch_mask: np.ndarray,
                                  aerial_mask: np.ndarray, img_shape: tuple) -> dict:
        H, W  = img_shape
        diag  = float(np.sqrt(H ** 2 + W ** 2))
        decay = CONF_DECAY_FRAC * diag

        # IoU
        inter = np.logical_and(warped_sketch_mask, aerial_mask).sum()
        union = np.logical_or(warped_sketch_mask,  aerial_mask).sum()
        iou   = float(inter / (union + 1e-9))

        # HD95 and Chamfer share the same boundary point sets
        pts_w = self._boundary_points(warped_sketch_mask)
        pts_a = self._boundary_points(aerial_mask)

        hd95_px    = self._hd95(pts_w, pts_a)
        chamfer_px = self._chamfer(pts_w, pts_a)
        sim_hd95    = float(np.exp(-hd95_px    / decay)) if np.isfinite(hd95_px)    else 0.0
        sim_chamfer = float(np.exp(-chamfer_px / decay)) if np.isfinite(chamfer_px) else 0.0

        # PoLiS (optional)
        polis_px  = None
        sim_polis = 0.0
        if CONF_WEIGHT_POLIS > 0:
            polis_px = self._polis(warped_sketch_mask, aerial_mask)
            if polis_px is not None:
                sim_polis = float(np.exp(-polis_px / decay))

        # Extra-area coverage: fraction of aerial mask NOT covered by sketch
        # Higher extra_ratio → more uncovered building area → lower confidence
        extra_pixels = int(np.logical_and(aerial_mask, np.logical_not(warped_sketch_mask)).sum())
        aerial_pixels = int(aerial_mask.sum())
        extra_ratio = float(extra_pixels / (aerial_pixels + 1e-9))
        sim_extra   = 1.0 - extra_ratio   # 1 = sketch covers everything, 0 = nothing covered

        # Effective weights (redistribute PoLiS weight if unavailable)
        if polis_px is None:
            base  = (CONF_WEIGHT_IOU + CONF_WEIGHT_HD95
                   + CONF_WEIGHT_CHAMFER + CONF_WEIGHT_EXTRA)
            w_iou = CONF_WEIGHT_IOU     / base
            w_hd  = CONF_WEIGHT_HD95    / base
            w_ch  = CONF_WEIGHT_CHAMFER / base
            w_po  = 0.0
            w_ex  = CONF_WEIGHT_EXTRA   / base
        else:
            w_iou = CONF_WEIGHT_IOU
            w_hd  = CONF_WEIGHT_HD95
            w_ch  = CONF_WEIGHT_CHAMFER
            w_po  = CONF_WEIGHT_POLIS
            w_ex  = CONF_WEIGHT_EXTRA

        composite = (w_iou * iou
                   + w_hd  * sim_hd95
                   + w_ch  * sim_chamfer
                   + w_po  * sim_polis
                   + w_ex  * sim_extra)

        hd95_str  = f"{hd95_px:.1f}px" if np.isfinite(hd95_px)    else "∞"
        ch_str    = f"{chamfer_px:.1f}px" if np.isfinite(chamfer_px) else "∞"
        polis_str = f"{polis_px:.1f}px" if polis_px is not None     else "N/A"
        print(f"[Confidence] composite={composite:.4f}  "
              f"IoU={iou:.4f}  HD95={hd95_str}  "
              f"Chamfer={ch_str}  PoLiS={polis_str}  "
              f"extra_ratio={extra_ratio:.3f}({extra_pixels}px)")

        return {
            "composite":   composite,
            "iou":         iou,
            "hd95_sim":    sim_hd95,
            "chamfer_sim": sim_chamfer,
            "polis_sim":   sim_polis,
            "extra_sim":   sim_extra,
            "weights":     (w_iou, w_hd, w_ch, w_po, w_ex),
            "raw": {
                "hd95_px":     hd95_px,
                "chamfer_px":  chamfer_px,
                "polis_px":    polis_px,
                "extra_ratio": extra_ratio,
                "extra_px":    extra_pixels,
            },
        }

    # ----------------------------------------------------------
    # Module F — Discrepancy detection
    # ----------------------------------------------------------
    def detect_discrepancies(self, warped_sketch_mask: np.ndarray,
                              aerial_mask: np.ndarray,
                              unmatched_masks: list = None) -> dict:
        extra        = np.logical_and(aerial_mask,        np.logical_not(warped_sketch_mask))
        missing      = np.logical_and(warped_sketch_mask, np.logical_not(aerial_mask))
        intersection = np.logical_and(aerial_mask,        warped_sketch_mask)

        # Combine unmatched building masks into one union overlay
        unmatched_combined = None
        if unmatched_masks:
            unmatched_combined = unmatched_masks[0].copy()
            for m in unmatched_masks[1:]:
                unmatched_combined = np.logical_or(unmatched_combined, m)

        building_area = max(int(aerial_mask.sum()), int(warped_sketch_mask.sum()))
        min_area      = DISCREP_MIN_AREA_FRAC * building_area

        def _filter_regions(region_mask: np.ndarray, region_type: str) -> list[dict]:
            seg = region_mask.astype(np.uint8) * 255
            num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
                seg, connectivity=8
            )
            regions    = []
            label_idx  = 0
            for i in range(1, num_labels):   # skip background (label 0)
                area_i = int(stats[i, cv2.CC_STAT_AREA])
                if area_i < min_area:
                    continue
                # Compactness (isoperimetric quotient): 4π·A/P²
                comp_mask = (label_map == i).astype(np.uint8) * 255
                cnts_c, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts_c:
                    continue
                cnt  = max(cnts_c, key=cv2.contourArea)
                perim = cv2.arcLength(cnt, closed=True)
                iq    = (4.0 * np.pi * area_i) / (perim ** 2 + 1e-9)
                if iq < DISCREP_MIN_COMPACTNESS:
                    continue
                label_idx += 1
                regions.append({
                    "label":       f"{region_type}_{label_idx}",
                    "mask":        (label_map == i),
                    "area":        area_i,
                    "bbox":        (
                        int(stats[i, cv2.CC_STAT_LEFT]),
                        int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]),
                        int(stats[i, cv2.CC_STAT_HEIGHT]),
                    ),
                    "centroid":    (float(centroids[i, 0]), float(centroids[i, 1])),
                    "compactness": float(iq),
                })
            return regions

        extra_regions   = _filter_regions(extra,   "extra")
        missing_regions = _filter_regions(missing, "missing")

        # Per-mask unmatched region metadata (bbox, centroid, area)
        unmatched_regions = []
        if unmatched_masks:
            for i, um in enumerate(unmatched_masks):
                seg_u  = um.astype(np.uint8) * 255
                cnts_u, _ = cv2.findContours(seg_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts_u:
                    continue
                all_pts_u           = np.concatenate(cnts_u)
                area_u              = int(um.sum())
                x_u, y_u, bw_u, bh_u = cv2.boundingRect(all_pts_u)
                unmatched_regions.append({
                    "label":    f"unmatched_{i + 1}",
                    "mask":     um,
                    "area":     area_u,
                    "bbox":     (x_u, y_u, bw_u, bh_u),
                    "centroid": (float(x_u + bw_u / 2), float(y_u + bh_u / 2)),
                })

        print(f"[Discrepancy] Extra regions: {len(extra_regions)}  "
              f"Missing regions: {len(missing_regions)}  "
              f"Unmatched buildings: {len(unmatched_regions)}")
        for r in extra_regions:
            print(f"  [extra]     {r['label']}  area={r['area']}px  IQ={r['compactness']:.3f}")
        for r in missing_regions:
            print(f"  [missing]   {r['label']}  area={r['area']}px  IQ={r['compactness']:.3f}")
        for r in unmatched_regions:
            print(f"  [unmatched] {r['label']}  area={r['area']}px")

        return {
            "extra":               extra,
            "missing":             missing,
            "intersection":        intersection,
            "extra_regions":       extra_regions,
            "missing_regions":     missing_regions,
            "unmatched_buildings": unmatched_combined,
            "unmatched_regions":   unmatched_regions,
        }

    # ----------------------------------------------------------
    # Discrepancy Map visualization
    # ----------------------------------------------------------
    def save_discrepancy_map(self, aerial_bgr: np.ndarray, discrepancies: dict,
                              confidence: dict, out_path: Path) -> np.ndarray:
        """Generate and save the discrepancy map figure. Returns RGB overlay array."""
        aerial_rgb = cv2.cvtColor(aerial_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        overlay    = aerial_rgb.copy()

        COLOR_MATCH     = np.array([  0, 200,   0], dtype=np.float32)
        COLOR_EXTRA     = np.array([  0, 220, 220], dtype=np.float32)
        COLOR_MISSING   = np.array([220,  60,  60], dtype=np.float32)
        COLOR_UNMATCHED = np.array([255, 165,   0], dtype=np.float32)

        m = discrepancies["intersection"]
        if m.any():
            overlay[m] = aerial_rgb[m] * 0.5 + COLOR_MATCH * 0.5
        m = discrepancies["extra"]
        if m.any():
            overlay[m] = aerial_rgb[m] * 0.5 + COLOR_EXTRA * 0.5
        m = discrepancies["missing"]
        if m.any():
            overlay[m] = aerial_rgb[m] * 0.5 + COLOR_MISSING * 0.5
        # Unmatched buildings rendered last (visual priority over extra/missing)
        m = discrepancies.get("unmatched_buildings")
        if m is not None and m.any():
            overlay[m] = aerial_rgb[m] * 0.5 + COLOR_UNMATCHED * 0.5

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        for reg in discrepancies["extra_regions"]:
            x, y, bw, bh = reg["bbox"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 200, 200), 2)
            cv2.putText(overlay, reg["label"], (x, max(y - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
        for reg in discrepancies["missing_regions"]:
            x, y, bw, bh = reg["bbox"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (220, 60, 60), 2)
            cv2.putText(overlay, reg["label"], (x, max(y - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 60, 60), 2)
        for reg in discrepancies.get("unmatched_regions", []):
            x, y, bw, bh = reg["bbox"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 140, 0), 3)
            cv2.putText(overlay, reg["label"], (x, max(y - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)

        n_unmatched = len(discrepancies.get("unmatched_regions", []))
        legend_patches = [
            mpatches.Patch(facecolor=(0, 200/255, 0),
                           label="Matching area"),
            mpatches.Patch(facecolor=(0, 220/255, 220/255),
                           label=f"Extra protrusion in aerial — {len(discrepancies['extra_regions'])} region(s)"),
            mpatches.Patch(facecolor=(220/255, 60/255, 60/255),
                           label=f"Sketch area missing from aerial — {len(discrepancies['missing_regions'])} region(s)"),
        ]
        if n_unmatched:
            legend_patches.append(
                mpatches.Patch(facecolor=(255/255, 165/255, 0),
                               label=f"Unmatched building(s) in aerial — {n_unmatched}")
            )

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(overlay)
        ax.set_title(f"⑤ Discrepancy Map  —  Confidence: {confidence['composite']:.4f}", fontsize=11)
        ax.axis("off")
        ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
                  framealpha=0.88, edgecolor="gray")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Save] Discrepancy map → {out_path}")

        return overlay   # RGB uint8 — reused as panel ⑤ in final comparison

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

        # ---- Target mask selection ----
        if len(detections) >= 2:
            preview_path = output_dir / f"detection_preview{suffix}.png"
            self.save_detections(img_bgr, detections, preview_path)

        labels = [f"#{i} {d['prompt']} conf={d['conf']:.2f}" for i, d in enumerate(detections)]
        print(f"\n  TARGET SELECTION — {len(detections)} candidate(s): {labels}")
        merged, all_pts, unmatched_masks = self._select_target_mask(
            detections, sketch_features, h, w
        )
        if all_pts is None:
            raise RuntimeError("Selected target mask has no contour.")
        if unmatched_masks:
            print(f"  → {len(unmatched_masks)} unmatched building(s) will be shown in orange")

        seg  = merged.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("Target mask has no contour.")
        if len(cnts) > 1:
            print(f"[Align] {len(cnts)} contours in target mask — "
                  f"alignment driven by largest ({cv2.contourArea(all_pts):.0f}px²)")

        print("\n  MODULE D — Geometric Alignment & Overlay")
        result_img, warped_sketch_mask = self.align_and_overlay(
            sketch_path, img_bgr, sketch_features, merged, all_pts, output_dir, suffix
        )

        print("\n  MODULE E — Confidence Scoring")
        confidence = self.compute_confidence_score(warped_sketch_mask, merged, (h, w))

        print("\n  MODULE F — Discrepancy Detection")
        discrepancies = self.detect_discrepancies(warped_sketch_mask, merged, unmatched_masks)

        # Discrepancy map (standalone file + returns overlay for comparison panel)
        disc_overlay = self.save_discrepancy_map(
            img_bgr, discrepancies, confidence,
            output_dir / f"discrepancy_map{suffix}.png"
        )

        # ---- Final comparison: 6-panel (3×2) figure ----
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

        fig = plt.figure(figsize=(18, 21))
        gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.12)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])

        ax1.imshow(cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title("① Original Sketch", fontsize=10); ax1.axis("off")

        ax2.imshow(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
        ax2.set_title("② SAM3 Mask", fontsize=10); ax2.axis("off")

        ax3.imshow(cv2.cvtColor(mask_iso, cv2.COLOR_BGR2RGB))
        ax3.set_title("③ Target Mask (isolated)", fontsize=10); ax3.axis("off")

        ax4.imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
        ax4.set_title("④ Alignment Zoom", fontsize=10); ax4.axis("off")

        ax5.imshow(disc_overlay)
        ax5.set_title("⑤ Discrepancy Map", fontsize=10); ax5.axis("off")
        disc_legend = [
            mpatches.Patch(facecolor=(0, 200/255, 0),           label="Match"),
            mpatches.Patch(facecolor=(0, 220/255, 220/255),     label=f"Extra ({len(discrepancies['extra_regions'])})"),
            mpatches.Patch(facecolor=(220/255, 60/255, 60/255), label=f"Missing ({len(discrepancies['missing_regions'])})"),
        ]
        if discrepancies.get("unmatched_regions"):
            disc_legend.append(
                mpatches.Patch(facecolor=(255/255, 165/255, 0),
                               label=f"Unmatched bldg ({len(discrepancies['unmatched_regions'])})")
            )
        ax5.legend(handles=disc_legend, loc="lower right", fontsize=8, framealpha=0.85)

        # Panel ⑥ — confidence bar chart
        raw        = confidence["raw"]
        bar_labels = ["IoU", "HD95\nsimilarity", "Chamfer\nsimilarity",
                      "PoLiS\nsimilarity", "Extra\ncoverage"]
        bar_values = [confidence["iou"],         confidence["hd95_sim"],
                      confidence["chamfer_sim"],  confidence["polis_sim"],
                      confidence["extra_sim"]]
        bar_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

        def _raw_label(key, unit="px"):
            val = raw[key]
            if val is None:
                return "N/A"
            return f"{val:.1f}{unit}" if np.isfinite(val) else "∞"

        raw_annotations = [
            f"IoU = {confidence['iou']:.4f}",
            f"HD95 = {_raw_label('hd95_px')}",
            f"Chamfer = {_raw_label('chamfer_px')}",
            f"PoLiS = {_raw_label('polis_px')}",
            f"extra = {raw['extra_ratio']:.1%}  ({raw['extra_px']}px)",
        ]

        bars = ax6.barh(bar_labels, bar_values, color=bar_colors,
                        alpha=0.85, edgecolor="gray", linewidth=0.5)
        for bar, ann in zip(bars, raw_annotations):
            ax6.text(bar.get_width() + 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"  {ann}", va="center", ha="left", fontsize=8)
        ax6.set_xlim(0, 1.35)
        ax6.set_xlabel("Similarity Score", fontsize=9)
        ax6.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax6.set_title(
            f"⑥ Confidence Score: {confidence['composite']:.4f}",
            fontsize=10, fontweight="bold"
        )

        plt.suptitle(
            f"Building Alignment v6.0 — pair {pair_idx + 1} — prompts: {SAM3_PROMPTS}",
            fontsize=12, y=1.005,
        )
        fig_path = output_dir / f"final_comparison{suffix}.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Save] Final comparison → {fig_path}")

        # Summary block
        raw = confidence["raw"]
        print(f"\n{'─'*52}")
        print(f"  PAIR {pair_idx+1} SUMMARY — {sketch_path.name} + {aerial_path.name}")
        print(f"  Composite confidence : {confidence['composite']:.4f}")
        print(f"  IoU                  : {confidence['iou']:.4f}  (weight {confidence['weights'][0]:.2f})")
        print(f"  HD95 similarity      : {confidence['hd95_sim']:.4f}  (raw: {_raw_label('hd95_px')})")
        print(f"  Chamfer similarity   : {confidence['chamfer_sim']:.4f}  (raw: {_raw_label('chamfer_px')})")
        print(f"  PoLiS similarity     : {confidence['polis_sim']:.4f}  (raw: {_raw_label('polis_px')})")
        print(f"  Extra coverage sim   : {confidence['extra_sim']:.4f}  "
              f"(extra={raw['extra_ratio']:.1%} of aerial = {raw['extra_px']}px)")
        print(f"  Extra regions (cyan)    : {len(discrepancies['extra_regions'])}")
        print(f"  Missing regions (red)   : {len(discrepancies['missing_regions'])}")
        print(f"  Unmatched bldgs (orange): {len(discrepancies.get('unmatched_regions', []))}")
        print(f"{'─'*52}")

        return {
            "result_img":         result_img,
            "warped_sketch_mask": warped_sketch_mask,
            "aerial_mask":        merged,
            "confidence":         confidence,
            "discrepancies":      discrepancies,
        }


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
