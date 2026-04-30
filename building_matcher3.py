"""
Cross-modal Building Image Alignment System — v3
Architecture: SAM2 AutomaticMaskGenerator → matchShapes selection → affine align → transparent overlay

V3 combines:
  - V1 核心：全图自动生成 mask + Hu 矩形状匹配（不依赖建筑位置）
  - V2 改进：透明 sketch 叠加、CLAHE 开关、集中调参区、干净可视化

No assumptions about building location or visual continuity.
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
    """Return the first existing file matching stem.{png,jpg,jpeg}."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = base_dir / (stem + ext)
        if p.exists():
            return p
    return base_dir / (stem + ".png")


# ============================================================
# Paths & Global Config
# ============================================================
BASE_DIR   = Path(r"D:\2026 myplan\SAMLearning")
MODEL_CKPT = BASE_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = BASE_DIR / "output"

MAX_IMAGE_DIM = 1024   # resize large images to prevent GPU OOM

# ------------------------------------------------------------------
# ★ 手动指定输入文件路径（支持 .png / .jpg / .jpeg）
#   设为 None → 自动在 BASE_DIR 下查找 sketch.* / aerial.*
# ------------------------------------------------------------------
# SKETCH_PATH: Path | None = "sketch_2.jpg"
# AERIAL_PATH: Path | None = "aerial_2.jpg"
SKETCH_PATH: Path | None = "sketch.png"
AERIAL_PATH: Path | None = "aerial.png"

# ------------------------------------------------------------------
# ★ 调参区 — 所有可调超参数集中于此，无需深入代码
# ------------------------------------------------------------------

# 面积过滤：候选 mask 占整图面积的比例范围
# 低于下限 → 碎片/小树丢弃；高于上限 → 背景/地面丢弃
MIN_MASK_FRACTION = 0.01
MAX_MASK_FRACTION = 0.70

# SAM2 AutomaticMaskGenerator 参数
AMG_POINTS_PER_SIDE  = 32    # 提示点网格密度，越大越细致但越慢；建议 16~64
AMG_PRED_IOU_THRESH  = 0.80  # mask 质量阈值，降低可保留更多候选
AMG_STABILITY_THRESH = 0.90  # mask 稳定性阈值

# 轮廓平滑容差（相对周长比例）
# 0.02 = 允许 2% 误差，消除树木锯齿；调大边缘更光滑但细节丢失
APPROX_EPSILON = 0.02

# 可视化第④格 zoom 的自适应留白比例（相对建筑最长边）
ZOOM_PAD_RATIO = 0.20

# CLAHE 图像增强
CLAHE_ENABLED    = True    # False → 跳过增强，直接用原图送入 SAM2
CLAHE_CLIP_LIMIT = 3.0    # 对比度放大上限
CLAHE_TILE_GRID  = (8, 8) # 局部直方图块大小

# 候选 mask 导出（用于调试 matchShapes 选择结果）
EXPORT_CANDIDATES   = True   # True → 每次运行后在 ai_test_crops/ 下保存切图
EXPORT_TOP_N        = 8      # 导出 shape_score 最低的前 N 个候选


# ============================================================
# BuildingMatcher v3
# ============================================================
class BuildingMatcher:
    """
    V3 pipeline:
      Module A — Sketch feature extraction
      Module B — SAM2 automatic mask generation + matchShapes selection
      Module D — Affine alignment & transparent overlay
    """

    def __init__(self, model_ckpt: Path, model_cfg: str):
        self.device = self._select_device()
        self._load_model(model_ckpt, model_cfg)

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1024 ** 3
            print(f"[GPU] {props.name}  VRAM: {vram:.1f} GB")
            if vram < 4:
                print("[WARN] Low VRAM — consider reducing MAX_IMAGE_DIM or AMG_POINTS_PER_SIDE.")
            return torch.device("cuda")
        print("[WARN] CUDA unavailable — running on CPU (slow).")
        return torch.device("cpu")

    def _load_model(self, model_ckpt: Path, model_cfg: str):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        print(f"[INFO] Loading SAM2 from {model_ckpt} ...")
        sam2 = build_sam2(model_cfg, str(model_ckpt),
                          device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model                  = sam2,
            points_per_side        = AMG_POINTS_PER_SIDE,
            pred_iou_thresh        = AMG_PRED_IOU_THRESH,
            stability_score_thresh = AMG_STABILITY_THRESH,
        )
        print("[INFO] Model ready.")

    # ----------------------------------------------------------
    # Module A — Sketch feature extraction
    # ----------------------------------------------------------
    def extract_sketch_features(self, sketch_path: Path) -> dict:
        """Load sketch, binarize, extract largest contour + geometric features."""
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
    # Image enhancement
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
    # Module B — Automatic mask generation + shape matching
    # ----------------------------------------------------------
    def segment_aerial(self, aerial_path: Path, sketch_features: dict):
        """
        Full-image SAM2 automatic segmentation followed by Hu-moment shape matching.

        Pipeline:
          1. Load & resize (OOM guard)
          2. Optional CLAHE enhancement
          3. SAM2 generates all masks in the image (no location assumption)
          4. Area fraction filter removes noise and background
          5. matchShapes ranks remaining masks by similarity to sketch contour
          6. Best match selected; contour smoothed with approxPolyDP
        """
        img_bgr = cv2.imread(str(aerial_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load aerial image: {aerial_path}")

        h, w  = img_bgr.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_IMAGE_DIM:
            scale    = MAX_IMAGE_DIM / max(h, w)
            new_w    = int(w * scale)
            new_h    = int(h * scale)
            img_bgr  = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w     = new_h, new_w
            print(f"[WARN] Aerial resized to {new_w}×{new_h} (scale={scale:.3f})")

        if CLAHE_ENABLED:
            img_bgr = self._apply_clahe(img_bgr, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)
            print(f"[INFO] CLAHE applied (clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID})")
        else:
            print("[INFO] CLAHE skipped")

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        total_px = h * w

        print("[INFO] Running SAM2 AutomaticMaskGenerator ...")
        raw_masks = self.mask_generator.generate(img_rgb)
        print(f"[INFO] SAM2 generated {len(raw_masks)} raw masks")

        # ---- Area fraction filter ----
        mask_list = []
        for m in raw_masks:
            frac = m["area"] / total_px
            if not (MIN_MASK_FRACTION <= frac <= MAX_MASK_FRACTION):
                continue
            seg = m["segmentation"].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            mask_list.append({
                "mask":      m["segmentation"],
                "contour":   cnt,
                "area":      m["area"],
                "iou_score": m["predicted_iou"],
            })

        if not mask_list:
            raise RuntimeError(
                f"No masks survived area filter ({MIN_MASK_FRACTION}–{MAX_MASK_FRACTION}). "
                "Try relaxing MIN_MASK_FRACTION / MAX_MASK_FRACTION."
            )
        print(f"[INFO] {len(mask_list)} masks passed area filter")

        # ---- matchShapes: rank by Hu-moment similarity to sketch ----
        # Hu moments are scale/rotation/translation invariant → cross-modal safe.
        # Lower score = more similar shape.
        sketch_cnt = sketch_features["contour"]
        for mc in mask_list:
            mc["shape_score"] = cv2.matchShapes(
                sketch_cnt, mc["contour"], cv2.CONTOURS_MATCH_I1, 0
            )
        mask_list.sort(key=lambda x: x["shape_score"])

        top3 = [f"{m['shape_score']:.4f}" for m in mask_list[:3]]
        print(f"[INFO] Top-3 shape scores: {top3}")

        # ---- Export candidate crops for visual inspection ----
        if EXPORT_CANDIDATES:
            crop_dir = OUTPUT_DIR / "ai_test_crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            # clear old crops first
            for old in crop_dir.glob("candidate_*.png"):
                old.unlink()

            for rank, mc in enumerate(mask_list[:EXPORT_TOP_N]):
                x, y, bw, bh = cv2.boundingRect(mc["contour"])
                pad = int(max(bw, bh) * ZOOM_PAD_RATIO)
                x1  = max(0, x - pad);                     y1 = max(0, y - pad)
                x2  = min(img_bgr.shape[1], x + bw + pad); y2 = min(img_bgr.shape[0], y + bh + pad)
                crop = img_bgr[y1:y2, x1:x2].copy()

                # draw contour on crop for reference
                shifted = mc["contour"] - np.array([x1, y1])
                cv2.drawContours(crop, [shifted], -1, (0, 140, 255), 2)

                label = (f"rank{rank}  shape={mc['shape_score']:.4f}"
                         f"  area={mc['area']}  iou={mc['iou_score']:.3f}")
                cv2.putText(crop, label, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(crop, label, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                fname = f"candidate_{rank:02d}_shape{mc['shape_score']:.4f}.png"
                cv2.imwrite(str(crop_dir / fname), crop)

            print(f"[Save] {min(EXPORT_TOP_N, len(mask_list))} candidate crops → {crop_dir}")

        best = mask_list[0]
        print(f"[INFO] Selected  shape_score={best['shape_score']:.4f}  "
              f"area={best['area']} px²  iou={best['iou_score']:.4f}")

        # ---- Smooth contour ----
        perimeter       = cv2.arcLength(best["contour"], True)
        epsilon         = APPROX_EPSILON * perimeter
        best["contour"] = cv2.approxPolyDP(best["contour"], epsilon, True)
        print(f"[INFO] Contour smoothed → {len(best['contour'])} vertices")

        return img_bgr, img_rgb, [best], scale

    # ----------------------------------------------------------
    # Module D — Geometric alignment & overlay
    # ----------------------------------------------------------
    def _compute_affine(self, sketch_features: dict, target_contour) -> np.ndarray:
        """
        Similarity transform: sketch space → aerial space.
        Tries all 4 corner orderings of minAreaRect to resolve 90°-ambiguity.
        """
        sk_rect  = sketch_features["min_rect"]
        tgt_rect = cv2.minAreaRect(target_contour)
        tgt_ctr  = tgt_rect[0]

        sk_box  = cv2.boxPoints(sk_rect).astype(np.float32)
        tgt_box = cv2.boxPoints(tgt_rect).astype(np.float32)

        best_M, best_err = None, float("inf")
        for offset in range(4):
            src  = np.roll(sk_box, offset, axis=0)
            M, _ = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2), tgt_box.reshape(-1, 1, 2), method=cv2.LMEDS
            )
            if M is None:
                continue
            src_h = np.hstack([src, np.ones((4, 1))])
            err   = float(np.sum(((M @ src_h.T).T - tgt_box) ** 2))
            if err < best_err:
                best_err, best_M = err, M

        if best_M is None:
            raise RuntimeError("estimateAffinePartial2D failed for all 4 corner orderings.")

        scale = float(np.sqrt(best_M[0, 0] ** 2 + best_M[0, 1] ** 2))
        angle = float(np.degrees(np.arctan2(best_M[1, 0], best_M[0, 0])))
        print(f"[Align] scale={scale:.3f}  angle={angle:.1f}°  "
              f"center=({tgt_ctr[0]:.1f},{tgt_ctr[1]:.1f})  reproj_err={best_err:.2f}")
        if not (0.05 < scale < 20.0):
            print(f"[WARN] Scale={scale:.3f} looks unreasonable — verify building is in aerial image.")
        return best_M

    @staticmethod
    def make_transparent(sketch_bgr: np.ndarray) -> np.ndarray:
        """White-background sketch → RGBA; white → alpha=0, dark lines → alpha=255."""
        gray = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
        rgba = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255 - gray
        return rgba

    def align_and_overlay(self, sketch_path, aerial_bgr, sketch_features, target_mask, output_dir):
        h, w = aerial_bgr.shape[:2]
        M    = self._compute_affine(sketch_features, target_mask["contour"])

        sketch_rgba = self.make_transparent(cv2.imread(str(sketch_path)))
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "sketch_transparent.png"), sketch_rgba)

        warped_rgba   = cv2.warpAffine(sketch_rgba, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=(255, 255, 255, 0))
        warped_binary = (warped_rgba[:, :, 3] > 30).astype(np.uint8) * 255

        alpha_f = warped_rgba[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255.0
        result  = np.clip(
            aerial_bgr.astype(np.float32) * (1 - alpha_f) +
            warped_rgba[:, :, :3].astype(np.float32) * alpha_f,
            0, 255,
        ).astype(np.uint8)

        out_path = output_dir / "alignment_result.png"
        cv2.imwrite(str(out_path), result)
        print(f"[Save] Alignment result → {out_path}")
        return result, warped_binary

    # ----------------------------------------------------------
    # Visualization — 4-panel figure
    # ----------------------------------------------------------
    def visualize_result(self, sketch_path, aerial_rgb, aerial_bgr, mask, result_img, output_dir):
        """
        Panel ① Original sketch
        Panel ② Aerial with SAM2 mask overlay + contour
        Panel ③ Target mask isolated
        Panel ④ Alignment zoom (sketch warped onto building, no orange border)
        """
        # ② mask overlay
        overlay = np.zeros_like(aerial_rgb)
        overlay[mask["mask"]] = [255, 165, 0]
        vis_bgr = cv2.cvtColor(
            cv2.addWeighted(aerial_rgb, 0.6, overlay, 0.4, 0), cv2.COLOR_RGB2BGR
        )
        cv2.drawContours(vis_bgr, [mask["contour"]], -1, (0, 140, 255), 2)
        cv2.imwrite(str(output_dir / "masks_preview.png"), vis_bgr)

        # ③ isolated mask
        mask_iso = np.full_like(aerial_bgr, 240)
        mask_iso[mask["mask"]] = aerial_bgr[mask["mask"]]
        cv2.drawContours(mask_iso, [mask["contour"]], -1, (0, 140, 255), 2)

        # ④ alignment zoom
        x, y, bw, bh = cv2.boundingRect(mask["contour"])
        pad = int(max(bw, bh) * ZOOM_PAD_RATIO)
        x1  = max(0, x - pad);                     y1 = max(0, y - pad)
        x2  = min(aerial_bgr.shape[1], x + bw + pad); y2 = min(aerial_bgr.shape[0], y + bh + pad)
        zoom = result_img[y1:y2, x1:x2]

        sketch_bgr = cv2.imread(str(sketch_path))
        fig, axes  = plt.subplots(2, 2, figsize=(18, 14))
        panels = [
            (sketch_bgr, "① Original Sketch"),
            (vis_bgr,    f"② SAM2 Mask  (shape={mask['shape_score']:.4f}  iou={mask['iou_score']:.3f})"),
            (mask_iso,   "③ Target Mask (isolated)"),
            (zoom,       "④ Alignment Zoom"),
        ]
        for ax, (img, title) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.suptitle("Cross-modal Building Alignment v3 — AutoMask + matchShapes",
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
        print("  MODULE B — SAM2 Auto Segmentation + Shape Matching")
        print("=" * 60)
        aerial_bgr, aerial_rgb, mask_list, scale = self.segment_aerial(aerial_path, sketch_features)
        target_mask = mask_list[0]

        print("\n" + "=" * 60)
        print("  MODULE D — Geometric Alignment & Overlay")
        print("=" * 60)
        result_img, _ = self.align_and_overlay(
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
    missing = [str(p) for p in [MODEL_CKPT, sketch_path, aerial_path] if not p.exists()]
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print(f"  • {m}")
        sys.exit(1)


def _resolve_path(p, base: Path) -> Path:
    """Convert str/Path to absolute Path; relative paths are resolved under base."""
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

    matcher = BuildingMatcher(model_ckpt=MODEL_CKPT, model_cfg=MODEL_CFG)
    matcher.run(sketch_path=sketch_path, aerial_path=aerial_path, output_dir=OUTPUT_DIR)
    print("\n[DONE] Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
