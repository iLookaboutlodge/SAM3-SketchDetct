"""
Cross-modal Building Image Alignment System — v2
Architecture: Center-Point SAM2 Prompt → single mask → affine align → overlay

Assumptions (v2):
  - Input aerial image is a Parcel Map; the target building is always at image center.
  - No human-in-the-loop needed: fully automated, single-click run.

Usage:
    Place sketch.(png|jpg) and aerial.(png|jpg) in D:\\2026 myplan\\SAMLearning\\
    Then run: python building_matcher2.py
"""

import sys
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Image discovery helper (supports .png / .jpg / .jpeg)
# ============================================================
def _find_image(base_dir: Path, stem: str) -> Path:
    """Return the first existing file matching stem.{png,jpg,jpeg}."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = base_dir / (stem + ext)
        if p.exists():
            return p
    return base_dir / (stem + ".png")   # fallback — will fail with a clear error


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
#   设为具体路径 → 直接使用该文件，忽略自动查找
# ------------------------------------------------------------------
SKETCH_PATH: Path | None = BASE_DIR / "sketch.png"
AERIAL_PATH: Path | None = BASE_DIR / "aerial.png"

# ------------------------------------------------------------------
# ★ 调参区 — 所有可调超参数集中于此，无需深入代码
# ------------------------------------------------------------------
# SAM2 多点提示：十字阵列各臂相对图像边长的偏移比例
# 0.20 = 中心点向四个方向各延伸 20%；调大可覆盖更大的建筑翼
PROMPT_OFFSET_RATIO = 0.10

# 轮廓多边形拟合容差（相对周长的比例）
# 0.02 = 允许 2% 误差，可消除树木锯齿；调大边缘更光滑但细节丢失更多
APPROX_EPSILON = 0.02

# 可视化第④格 zoom 的自适应留白比例（相对建筑最长边）
# 0.20 = 建筑周围留 20% 的环境上下文；建筑越大留白越大
ZOOM_PAD_RATIO = 0.20

# CLAHE 图像增强参数
CLAHE_ENABLED    = False   # False → 跳过增强，直接用原图送入 SAM2
CLAHE_CLIP_LIMIT = 3.0   # 对比度放大上限，调高增强更强但噪声更多
CLAHE_TILE_GRID  = (8, 8) # 局部直方图块大小


# ============================================================
# BuildingMatcher
# ============================================================
class BuildingMatcher:
    """
    Streamlined pipeline (v2):
      Module A — Sketch feature extraction
      Module B — SAM2 center-point segmentation (single mask, smoothed)
      Module D — Affine alignment & transparent overlay
    """

    def __init__(self, model_ckpt: Path, model_cfg: str):
        self.device = self._select_device()
        self._load_model(model_ckpt, model_cfg)

    # ----------------------------------------------------------
    # Setup helpers
    # ----------------------------------------------------------
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1024 ** 3
            print(f"[GPU] {props.name}  VRAM: {vram:.1f} GB")
            if vram < 4:
                print("[WARN] Low VRAM — consider reducing MAX_IMAGE_DIM if OOM occurs.")
            return torch.device("cuda")
        print("[WARN] CUDA unavailable — running on CPU (slow).")
        return torch.device("cpu")

    def _load_model(self, model_ckpt: Path, model_cfg: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print(f"[INFO] Loading SAM2 from {model_ckpt} ...")
        sam2 = build_sam2(model_cfg, str(model_ckpt),
                          device=self.device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(sam2)
        print("[INFO] Model ready.")

    # ----------------------------------------------------------
    # Module A — Sketch feature extraction
    # ----------------------------------------------------------
    def extract_sketch_features(self, sketch_path: Path) -> dict:
        """Load sketch, binarize, extract largest contour + geometric features."""
        img = cv2.imread(str(sketch_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load sketch: {sketch_path}")

        # Build single-channel grayscale
        if img.ndim == 2:
            gray = img
        elif img.shape[2] == 4:                      # RGBA
            alpha = img[:, :, 3]
            gray  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            if np.mean(gray[alpha < 10]) < 128:      # dark background → invert
                gray = cv2.bitwise_not(gray)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize: lines → white (255), background → black (0)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Close small gaps in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in sketch — check that lines are visible.")

        main_cnt = max(contours, key=cv2.contourArea)
        area     = cv2.contourArea(main_cnt)

        M = cv2.moments(main_cnt)
        if M["m00"] == 0:
            raise ValueError("Sketch contour has zero area.")

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        min_rect = cv2.minAreaRect(main_cnt)

        print(f"[Sketch] centroid=({cx:.1f}, {cy:.1f})  "
              f"angle={np.degrees(angle_rad):.1f}°  area={area:.0f} px²")

        return {
            "contour":   main_cnt,
            "binary":    binary,
            "centroid":  (cx, cy),
            "angle_deg": np.degrees(angle_rad),
            "hu":        cv2.HuMoments(M).flatten(),
            "min_rect":  min_rect,
            "area":      area,
        }

    # ----------------------------------------------------------
    # Image enhancement
    # ----------------------------------------------------------
    @staticmethod
    def _apply_clahe(img_bgr: np.ndarray,
                     clip_limit: float = 3,
                     tile_grid: tuple  = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
        L channel of the LAB color space.

        Why LAB:  operating on L only leaves hue and saturation untouched,
                  so SAM2's colour-based features are not distorted.
        clip_limit:  amplification cap — higher values = more contrast boost
                     but also more noise amplification (2.0 is a safe default).
        tile_grid:   size of each local histogram tile; 8×8 balances local
                     detail vs. over-segmentation artefacts.
        """
        lab              = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b          = cv2.split(lab)
        clahe            = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        lab_enhanced     = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ----------------------------------------------------------
    # Module B — SAM2 center-point segmentation
    # ----------------------------------------------------------
    def segment_aerial(self, aerial_path: Path, sketch_contour=None):
        """
        Use a single center-point prompt to extract the building at image center.

        Pipeline:
          1. Load & optionally resize image (OOM guard)
          2. CLAHE enhancement on L channel (boosts local contrast for SAM2)
          3. Feed image center as foreground + 4 corner background points
          4. Among the 3 candidate masks, pick the one whose contour best matches
             the sketch shape (matchShapes / Hu moments). Falls back to largest
             area when no sketch contour is supplied.
          5. Smooth the winning contour with approxPolyDP
          6. Return a single-element mask_list
        """
        img_bgr = cv2.imread(str(aerial_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load aerial image: {aerial_path}")

        h, w  = img_bgr.shape[:2]
        scale = 1.0

        # Resize if too large to prevent GPU OOM
        if max(h, w) > MAX_IMAGE_DIM:
            scale   = MAX_IMAGE_DIM / max(h, w)
            new_w   = int(w * scale)
            new_h   = int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w    = new_h, new_w
            print(f"[WARN] Aerial resized to {new_w}×{new_h} (scale={scale:.3f})")

        # CLAHE enhancement — improves local contrast before SAM2 inference
        if CLAHE_ENABLED:
            img_bgr = self._apply_clahe(img_bgr,
                                        clip_limit=CLAHE_CLIP_LIMIT,
                                        tile_grid=CLAHE_TILE_GRID)
            print(f"[INFO] CLAHE applied (clipLimit={CLAHE_CLIP_LIMIT}, tileGrid={CLAHE_TILE_GRID})")
        else:
            print("[INFO] CLAHE skipped (CLAHE_ENABLED=False)")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ---- Multi-point prompt: foreground cross + background corners ----
        # Foreground (label=1): center + 4 short-arm cardinal points.
        #   Short arms (10%) stay on the roof, not on surrounding trees.
        # Background (label=0): 4 image corner points.
        #   Explicitly tells SAM2 "the edges of the image are NOT the building",
        #   which prevents tree canopy from being swept into the mask.
        cx, cy = w // 2, h // 2
        dx     = int(w * PROMPT_OFFSET_RATIO)
        dy     = int(h * PROMPT_OFFSET_RATIO)
        margin = 10   # pixels from image border for corner bg points

        fg_pts = np.array([
            [cx,      cy     ],   # center
            [cx,      cy - dy],   # top arm
            [cx,      cy + dy],   # bottom arm
            [cx - dx, cy     ],   # left arm
            [cx + dx, cy     ],   # right arm
        ])
        bg_pts = np.array([
            [margin,     margin    ],   # top-left corner
            [w - margin, margin    ],   # top-right corner
            [margin,     h - margin],   # bottom-left corner
            [w - margin, h - margin],   # bottom-right corner
        ])

        prompt_pts    = np.vstack([fg_pts, bg_pts])
        prompt_labels = np.concatenate([
            np.ones (len(fg_pts), dtype=np.int32),   # foreground
            np.zeros(len(bg_pts), dtype=np.int32),   # background
        ])
        print(f"[INFO] SAM2 prompt: {len(fg_pts)} fg points (center+arms, offset={dx},{dy}) "
              f"+ {len(bg_pts)} bg corner points")

        self.predictor.set_image(img_rgb)

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords     = prompt_pts,
                        point_labels     = prompt_labels,
                        multimask_output = True,
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords     = prompt_pts,
                    point_labels     = prompt_labels,
                    multimask_output = True,
                )

        # ---- Select best mask ----
        # If sketch contour is available, rank by matchShapes (Hu-moment similarity).
        # Hu moments are scale/rotation/translation invariant → works cross-modally.
        # Fallback: largest area (original behaviour).
        areas = [np.count_nonzero(m) for m in masks]

        if sketch_contour is not None:
            shape_scores = []
            for m in masks:
                cnts_tmp, _ = cv2.findContours(
                    m.astype(np.uint8) * 255,
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                if not cnts_tmp:
                    shape_scores.append(float("inf"))
                    continue
                c = max(cnts_tmp, key=cv2.contourArea)
                shape_scores.append(
                    cv2.matchShapes(sketch_contour, c, cv2.CONTOURS_MATCH_I1, 0)
                )
            best_i = int(np.argmin(shape_scores))
            print(f"[INFO] matchShapes scores: {[f'{s:.4f}' for s in shape_scores]}")
            print(f"[INFO] Selected mask index={best_i}  "
                  f"shape_score={shape_scores[best_i]:.4f}  "
                  f"area={areas[best_i]} px²  iou={scores[best_i]:.4f}")
        else:
            best_i = int(np.argmax(areas))
            print(f"[INFO] No sketch — fallback to largest area, index={best_i}  "
                  f"area={areas[best_i]} px²")

        best_mask  = masks[best_i].astype(np.uint8) * 255
        best_score = float(scores[best_i])

        # ---- Extract & smooth contour ----
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("SAM2 produced a non-empty mask but no contour was found.")

        cnt       = max(cnts, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        epsilon   = APPROX_EPSILON * perimeter
        cnt       = cv2.approxPolyDP(cnt, epsilon, True)

        area = cv2.contourArea(cnt)
        print(f"[INFO] Contour smoothed → {len(cnt)} vertices  area={area:.0f} px²")

        # Wrap as a single-element list to keep downstream API compatible
        mask_list = [{
            "mask":      masks[best_i].astype(bool),
            "contour":   cnt,
            "area":      area,
            "iou_score": best_score,
        }]

        return img_bgr, img_rgb, mask_list, scale

    # ----------------------------------------------------------
    # Module D — Geometric alignment & overlay
    # ----------------------------------------------------------
    def _compute_affine(self, sketch_features: dict, target_contour) -> np.ndarray:
        """
        Build a 2×3 affine (similarity) matrix: sketch space → aerial space.

        Uses cv2.estimateAffinePartial2D on the 4 corners of both minAreaRects.
        Tries all 4 rotational orderings of the source corners and picks the one
        with minimum reprojection error — this resolves the 0°/90°/180°/270° ambiguity
        inherent in minAreaRect angles.
        """
        sk_rect  = sketch_features["min_rect"]
        tgt_rect = cv2.minAreaRect(target_contour)
        tgt_ctr  = tgt_rect[0]

        sk_box  = cv2.boxPoints(sk_rect).astype(np.float32)   # (4, 2)
        tgt_box = cv2.boxPoints(tgt_rect).astype(np.float32)  # (4, 2)

        best_M   = None
        best_err = float("inf")

        for offset in range(4):
            src = np.roll(sk_box, offset, axis=0)
            M, _ = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2),
                tgt_box.reshape(-1, 1, 2),
                method=cv2.LMEDS,
            )
            if M is None:
                continue
            # Reprojection error: Σ‖M·src_i − tgt_i‖²
            src_h = np.hstack([src, np.ones((4, 1))])
            pred  = (M @ src_h.T).T
            err   = float(np.sum((pred - tgt_box) ** 2))
            if err < best_err:
                best_err = err
                best_M   = M

        if best_M is None:
            raise RuntimeError("estimateAffinePartial2D failed for all 4 corner orderings.")

        scale = float(np.sqrt(best_M[0, 0] ** 2 + best_M[0, 1] ** 2))
        angle = float(np.degrees(np.arctan2(best_M[1, 0], best_M[0, 0])))
        print(f"[Align] scale={scale:.3f}  angle={angle:.1f}°  "
              f"target_center=({tgt_ctr[0]:.1f}, {tgt_ctr[1]:.1f})  "
              f"reproj_err={best_err:.2f}")
        if not (0.05 < scale < 20.0):
            print(f"[WARN] Scale={scale:.3f} looks unreasonable — "
                  "verify that the aerial image contains the correct building.")
        return best_M

    @staticmethod
    def make_transparent(sketch_bgr: np.ndarray) -> np.ndarray:
        """
        White-background sketch → RGBA with transparent background.
        White pixels (bright) → alpha=0, dark line pixels → alpha=255.
        """
        gray  = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
        alpha = 255 - gray
        rgba  = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        return rgba

    def align_and_overlay(self,
                          sketch_path:     Path,
                          aerial_bgr:      np.ndarray,
                          sketch_features: dict,
                          target_mask:     dict,
                          output_dir:      Path):
        """
        Warp sketch onto aerial image via the computed affine transform.
        Sketch white background is made transparent; black lines are composited
        directly over the aerial image so the underlying scene remains visible.

        Returns (result_bgr, warped_binary).
        """
        h, w = aerial_bgr.shape[:2]
        M    = self._compute_affine(sketch_features, target_mask["contour"])

        # Convert sketch to RGBA (white → transparent)
        sketch_bgr  = cv2.imread(str(sketch_path))
        sketch_rgba = self.make_transparent(sketch_bgr)

        # Save standalone transparent sketch for external use
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "sketch_transparent.png"), sketch_rgba)
        print(f"[Save] Transparent sketch → {output_dir / 'sketch_transparent.png'}")

        # Warp RGBA sketch into aerial space
        warped_rgba   = cv2.warpAffine(sketch_rgba, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=(255, 255, 255, 0))
        warped_binary = (warped_rgba[:, :, 3] > 30).astype(np.uint8) * 255

        # Alpha-composite: aerial (bottom) + sketch lines (top)
        alpha_f  = warped_rgba[:, :, 3].astype(np.float32) / 255.0
        alpha_f  = alpha_f[:, :, np.newaxis]
        aerial_f = aerial_bgr.astype(np.float32)
        sketch_f = warped_rgba[:, :, :3].astype(np.float32)

        result = np.clip(aerial_f * (1 - alpha_f) + sketch_f * alpha_f, 0, 255).astype(np.uint8)

        out_path = output_dir / "alignment_result.png"
        cv2.imwrite(str(out_path), result)
        print(f"[Save] Alignment result → {out_path}")
        return result, warped_binary

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------
    def visualize_result(self,
                         sketch_path:  Path,
                         aerial_rgb:   np.ndarray,
                         aerial_bgr:   np.ndarray,
                         mask:         dict,
                         result_img:   np.ndarray,
                         output_dir:   Path) -> np.ndarray:
        """
        Save masks_preview.png and a 4-panel final_comparison.png.

        Panel ① Original sketch
        Panel ② Aerial with SAM2 mask overlay
        Panel ③ Target mask isolated (building cropped out)
        Panel ④ Alignment zoom (sketch warped onto building)
        """
        # ── Mask overlay (panel ②) ──────────────────────────────
        overlay     = np.zeros_like(aerial_rgb)
        overlay[mask["mask"]] = [255, 165, 0]   # orange fill
        vis = cv2.addWeighted(aerial_rgb, 0.6, overlay, 0.4, 0)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.drawContours(vis_bgr, [mask["contour"]], -1, (0, 140, 255), 2)

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "masks_preview.png"
        cv2.imwrite(str(mask_path), vis_bgr)
        print(f"[Save] Mask preview → {mask_path}")

        # ── Isolated mask (panel ③) ─────────────────────────────
        mask_iso = np.full_like(aerial_bgr, 240)
        mask_iso[mask["mask"]] = aerial_bgr[mask["mask"]]
        cv2.drawContours(mask_iso, [mask["contour"]], -1, (0, 140, 255), 2)

        # ── Alignment zoom (panel ④) ────────────────────────────
        x, y, bw, bh = cv2.boundingRect(mask["contour"])
        pad = int(max(bw, bh) * ZOOM_PAD_RATIO)
        x1  = max(0, x - pad);            y1 = max(0, y - pad)
        x2  = min(aerial_bgr.shape[1], x + bw + pad)
        y2  = min(aerial_bgr.shape[0], y + bh + pad)
        zoom = result_img[y1:y2, x1:x2]

        # ── 4-panel figure ──────────────────────────────────────
        sketch_bgr = cv2.imread(str(sketch_path))
        fig, axes  = plt.subplots(2, 2, figsize=(18, 14))
        panels = [
            (sketch_bgr, "① Original Sketch"),
            (vis_bgr,    f"② SAM2 Mask  (score={mask['iou_score']:.3f})"),
            (mask_iso,   "③ Target Mask (isolated)"),
            (zoom,       "④ Alignment Zoom"),
        ]
        for ax, (img, title) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.suptitle("Cross-modal Building Alignment v2 — Center-Point SAM2",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        fig_path = output_dir / "final_comparison.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"[Save] Final comparison → {fig_path}")
        plt.show()
        return vis_bgr

    # ----------------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------------
    def run(self, sketch_path: Path, aerial_path: Path, output_dir: Path):
        print("\n" + "=" * 60)
        print("  MODULE A — Sketch Feature Extraction")
        print("=" * 60)
        sketch_features = self.extract_sketch_features(sketch_path)

        print("\n" + "=" * 60)
        print("  MODULE B — SAM2 Center-Point Segmentation")
        print("=" * 60)
        aerial_bgr, aerial_rgb, mask_list, scale = self.segment_aerial(
            aerial_path, sketch_contour=sketch_features["contour"]
        )

        # mask_list always has exactly 1 element in v2
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
            sketch_path, aerial_rgb, aerial_bgr,
            target_mask, result_img, output_dir
        )

        return result_img


# ============================================================
# Entry point
# ============================================================
def _check_prerequisites(sketch_path: Path, aerial_path: Path):
    missing = []
    for p in [MODEL_CKPT, sketch_path, aerial_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print(f"  • {m}")
        print(f"\nExpected layout:\n  {BASE_DIR}/")
        print("  ├── sketch.(png|jpg)")
        print("  ├── aerial.(png|jpg)")
        print("  └── checkpoints/sam2.1_hiera_large.pt")
        sys.exit(1)


def main():
    # 优先使用手动指定路径，否则自动查找
    sketch_path = SKETCH_PATH if SKETCH_PATH is not None else _find_image(BASE_DIR, "sketch")
    aerial_path = AERIAL_PATH if AERIAL_PATH is not None else _find_image(BASE_DIR, "aerial")

    _check_prerequisites(sketch_path, aerial_path)

    print(f"[INFO] Sketch : {sketch_path}")
    print(f"[INFO] Aerial : {aerial_path}")

    matcher = BuildingMatcher(model_ckpt=MODEL_CKPT, model_cfg=MODEL_CFG)
    matcher.run(sketch_path=sketch_path, aerial_path=aerial_path, output_dir=OUTPUT_DIR)

    print("\n[DONE] Pipeline completed. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
