"""
Cross-modal Building Image Alignment System (PoC)
Aligns architectural sketch (Sketch) with aerial/bird's-eye view image (Aerial View).

Usage:
    Place sketch.png and aerial.png in D:\\2026 myplan\\SAMLearning\\
    Then run: python building_matcher.py
"""

import sys
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths & Global Config
# ============================================================
BASE_DIR       = Path(r"D:\2026 myplan\SAMLearning")
MODEL_CKPT     = BASE_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
MODEL_CFG      = "configs/sam2.1/sam2.1_hiera_l.yaml"  # l = large, relative to sam2 package
SKETCH_PATH    = BASE_DIR / "sketch.jpg"
AERIAL_PATH    = BASE_DIR / "aerial.jpg"
OUTPUT_DIR     = BASE_DIR / "output"

MAX_IMAGE_DIM      = 1024   # max pixel dimension to avoid OOM
MIN_MASK_AREA      = 200    # absolute lower bound (px²)
MIN_MASK_FRACTION  = 0.01   # mask must be ≥ 1% of aerial image area
MAX_MASK_FRACTION  = 0.60   # mask must be ≤ 60% of aerial image area (skip sky/ground)

# Set to an integer to skip auto-matching and force-use a specific mask index.
# Run once with None to see the labeled masks_preview.png, then set the correct index.
TARGET_MASK_IDX    = None


# ============================================================
# BuildingMatcher
# ============================================================
class BuildingMatcher:
    """End-to-end pipeline: sketch → SAM2 segmentation → shape match → aligned overlay."""

    def __init__(self, model_ckpt: Path, model_cfg: str):
        self.device = self._select_device()
        self._load_model(model_ckpt, model_cfg)

    # ----------------------------------------------------------
    # Setup helpers
    # ----------------------------------------------------------
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1024**3
            print(f"[GPU] {props.name}  VRAM: {vram:.1f} GB")
            if vram < 4:
                print("[WARN] Low VRAM — consider reducing MAX_IMAGE_DIM if you hit OOM.")
            return torch.device("cuda")
        else:
            print("[WARN] CUDA unavailable — running on CPU (slow).")
            return torch.device("cpu")

    def _load_model(self, model_ckpt: Path, model_cfg: str):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        print(f"[INFO] Loading SAM2 from {model_ckpt} ...")
        self.sam2 = build_sam2(model_cfg, str(model_ckpt), device=self.device, apply_postprocessing=False)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.90,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=MIN_MASK_AREA,
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

        # Build single-channel grayscale
        if img.ndim == 2:
            gray = img
        elif img.shape[2] == 4:          # RGBA — transparent background
            alpha = img[:, :, 3]
            gray  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            # Make sure lines are dark on light background
            if np.mean(gray[alpha < 10]) < 128:   # background is dark → invert
                gray = cv2.bitwise_not(gray)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize: lines become white (255), background black (0)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Close small gaps in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in sketch — check that lines are visible.")

        # Keep the largest contour as the main building outline
        main_cnt = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(main_cnt)

        M  = cv2.moments(main_cnt)
        if M["m00"] == 0:
            raise ValueError("Sketch contour has zero area.")

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Principal axis angle from second-order central moments
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        hu = cv2.HuMoments(M).flatten()
        min_rect = cv2.minAreaRect(main_cnt)

        print(f"[Sketch] centroid=({cx:.1f}, {cy:.1f})  "
              f"angle={np.degrees(angle_rad):.1f}°  area={area:.0f} px²")

        return {
            "contour":  main_cnt,
            "binary":   binary,
            "centroid": (cx, cy),
            "angle_deg": np.degrees(angle_rad),
            "hu":        hu,
            "min_rect":  min_rect,
            "area":      area,
        }

    # ----------------------------------------------------------
    # Module B — SAM2 automatic segmentation
    # ----------------------------------------------------------
    def segment_aerial(self, aerial_path: Path):
        """Run SAM2 AutomaticMaskGenerator; return image, masks list, scale factor."""
        img_bgr = cv2.imread(str(aerial_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load aerial image: {aerial_path}")

        h, w   = img_bgr.shape[:2]
        scale  = 1.0

        # Resize if too large to prevent GPU OOM
        if max(h, w) > MAX_IMAGE_DIM:
            scale  = MAX_IMAGE_DIM / max(h, w)
            new_w  = int(w * scale)
            new_h  = int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[WARN] Aerial image resized {w}×{h} → {new_w}×{new_h} (scale={scale:.3f})")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"[INFO] Running SAM2 on {img_rgb.shape[1]}×{img_rgb.shape[0]} image …")

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_masks = self.mask_generator.generate(img_rgb)
            else:
                raw_masks = self.mask_generator.generate(img_rgb)

        print(f"[INFO] SAM2 generated {len(raw_masks)} raw masks.")

        # Convert each mask to OpenCV contour, filter by area fraction
        img_area  = img_bgr.shape[0] * img_bgr.shape[1]
        area_min  = max(MIN_MASK_AREA, img_area * MIN_MASK_FRACTION)
        area_max  = img_area * MAX_MASK_FRACTION
        print(f"[INFO] Area filter: {area_min:.0f} – {area_max:.0f} px²  "
              f"(image area={img_area} px²)")

        mask_list = []
        for m in raw_masks:
            seg = m["segmentation"].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)

            # 【多边形拟合】拉平树木毛刺，保留建筑直角与凹陷
            # epsilon = 2% 周长：足以消除植被锯齿，同时保留建筑转角
            perimeter  = cv2.arcLength(cnt, True)
            epsilon    = 0.02 * perimeter
            cnt        = cv2.approxPolyDP(cnt, epsilon, True)

            a   = cv2.contourArea(cnt)
            if not (area_min <= a <= area_max):
                continue
            M_  = cv2.moments(cnt)
            hu_ = cv2.HuMoments(M_).flatten() if M_["m00"] > 0 else np.zeros(7)
            mask_list.append({
                "mask":      m["segmentation"],
                "contour":   cnt,
                "area":      a,
                "iou_score": m.get("predicted_iou", 0.0),
                "hu":        hu_,
            })

        print(f"[INFO] {len(mask_list)} valid masks retained after area filtering.")
        return img_bgr, img_rgb, mask_list, scale

    # ----------------------------------------------------------
    # Module C — Shape matching (Hu moments via matchShapes)
    # ----------------------------------------------------------
    def match_shapes(self, sketch_features: dict, mask_list: list):
        """
        Compare sketch contour against all aerial masks using Hu-moment distance.
        Returns index of best match, its score, and all scores.
        """
        sk_cnt    = sketch_features["contour"]
        best_score = float("inf")
        best_idx   = -1
        scores     = []

        for i, mc in enumerate(mask_list):
            score = cv2.matchShapes(sk_cnt, mc["contour"], cv2.CONTOURS_MATCH_I1, 0)
            scores.append(score)
            if score < best_score:
                best_score = score
                best_idx   = i

        if best_idx < 0:
            raise ValueError("No masks to match against — segmentation may have failed.")

        # Print top-5 candidates so user can audit
        ranked = sorted(enumerate(scores), key=lambda x: x[1])
        print("[Match] Top-5 candidates (idx, score, area, iou):")
        for rank, (idx, sc) in enumerate(ranked[:5]):
            mc = mask_list[idx]
            marker = " ← BEST" if idx == best_idx else ""
            print(f"  #{rank+1}  mask[{idx:2d}]  score={sc:.6f}  "
                  f"area={mc['area']:6.0f} px²  iou={mc['iou_score']:.3f}{marker}")

        return best_idx, best_score, scores

    # ----------------------------------------------------------
    # Module C-ext — Export crops for external AI/human review
    # ----------------------------------------------------------
    def export_ai_test_crops(self,
                             aerial_bgr: np.ndarray,
                             mask_list:  list,
                             output_dir: Path) -> Path:
        """
        Crop each valid Mask out of the aerial image and save as individual PNGs,
        so an external LMM / human judge can pick the best candidate.

        Layout:  output_dir/ai_test_crops/candidate_0.png, candidate_1.png, …

        Each crop:
          - Bounding rect of the mask contour, expanded by PAD=30 px (clamped to image edge)
          - Orange outline of the mask contour drawn in crop-local coordinates
          - White index label in the top-left corner for easy identification
        """
        if not mask_list:
            print("[ERROR] 未发现任何有效建筑 Mask，无法导出切图。")
            sys.exit(1)

        crop_dir = output_dir / "ai_test_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        img_h, img_w = aerial_bgr.shape[:2]
        PAD = 30

        # 按面积降序取前 6 个，面积越大越可能是主体建筑
        top6 = sorted(mask_list, key=lambda m: m["area"], reverse=True)[:6]
        print(f"[Crop] 从 {len(mask_list)} 个 Mask 中取面积最大的 6 张 → {crop_dir}")
        for i, mc in enumerate(top6):
            x, y, bw, bh = cv2.boundingRect(mc["contour"])

            # Expand bounding rect by PAD, clamp to image boundary
            x1 = max(0, x - PAD)
            y1 = max(0, y - PAD)
            x2 = min(img_w, x + bw + PAD)
            y2 = min(img_h, y + bh + PAD)

            crop = aerial_bgr[y1:y2, x1:x2].copy()

            # Index label: black background rectangle + white text
            label = f"#{i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(crop, (4, 4), (4 + tw + 4, 4 + th + 6), (0, 0, 0), -1)
            cv2.putText(crop, label, (6, 4 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imwrite(str(crop_dir / f"candidate_{i}.png"), crop)

        print(f"[Crop] 完成，共 6 张已保存。")
        # 返回 crop_dir 和 top6，供调用方将用户输入的切图序号映射回 mask_list
        return crop_dir, top6

    # ----------------------------------------------------------
    # Module D — Geometric alignment & overlay
    # ----------------------------------------------------------
    def _compute_affine(self, sketch_features: dict, target_contour) -> np.ndarray:
        """
        Build a 2×3 affine matrix that maps sketch space → aerial space.

        Strategy: use cv2.estimateAffinePartial2D on the 4 corners of both
        minAreaRects.  This solves for the optimal similarity transform
        (rotation + uniform scale + translation) in one shot, instead of
        hand-decomposing angle / scale / translation separately.

        Because boxPoints can return corners in 4 different rotational orderings
        relative to the target rect, we try all 4, pick the one with the lowest
        reprojection error — this also resolves the 0°/90°/180°/270° ambiguity.
        """
        sk_rect  = sketch_features["min_rect"]
        tgt_rect = cv2.minAreaRect(target_contour)
        tgt_ctr  = tgt_rect[0]

        sk_box  = cv2.boxPoints(sk_rect).astype(np.float32)   # (4, 2)
        tgt_box = cv2.boxPoints(tgt_rect).astype(np.float32)  # (4, 2)

        best_M   = None
        best_err = float("inf")

        for offset in range(4):
            src = np.roll(sk_box, offset, axis=0)   # shift corner ordering
            M, _ = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2),
                tgt_box.reshape(-1, 1, 2),
                method=cv2.LMEDS
            )
            if M is None:
                continue
            # Reprojection error: Σ‖M·src_i − tgt_i‖²
            src_h = np.hstack([src, np.ones((4, 1))])   # (4, 3) homogeneous
            pred  = (M @ src_h.T).T                      # (4, 2)
            err   = float(np.sum((pred - tgt_box) ** 2))
            if err < best_err:
                best_err = err
                best_M   = M

        if best_M is None:
            raise RuntimeError(
                "estimateAffinePartial2D failed for all 4 corner orderings. "
                "Check that the sketch and target contours are valid."
            )

        scale = float(np.sqrt(best_M[0, 0] ** 2 + best_M[0, 1] ** 2))
        angle = float(np.degrees(np.arctan2(best_M[1, 0], best_M[0, 0])))
        print(f"[Align] scale={scale:.3f}  angle={angle:.1f}°  "
              f"target_center=({tgt_ctr[0]:.1f}, {tgt_ctr[1]:.1f})  "
              f"reproj_err={best_err:.2f}")
        if not (0.05 < scale < 20.0):
            print(f"[WARN] Scale={scale:.3f} looks unreasonable — "
                  "check masks_preview.png and MIN_MASK_FRACTION.")
        return best_M

    @staticmethod
    def make_transparent(sketch_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a white-background black-line sketch to RGBA with transparent background.
        White (bright) pixels → alpha=0 (transparent)
        Dark pixels           → alpha=255 (opaque, keeps original color)
        """
        gray  = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
        alpha = 255 - gray          # white→0, black→255
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
        Warp sketch onto aerial image.
        Returns (result_bgr, warped_binary) where result_bgr uses line-only overlay
        so the aerial image is always visible underneath.
        """
        h, w = aerial_bgr.shape[:2]
        M    = self._compute_affine(sketch_features, target_mask["contour"])

        # Load original sketch and convert to transparent RGBA
        sketch_bgr  = cv2.imread(str(sketch_path))
        sketch_rgba = self.make_transparent(sketch_bgr)   # white→transparent, black→opaque

        # Save standalone transparent sketch for external use
        output_dir.mkdir(parents=True, exist_ok=True)
        trans_path = output_dir / "sketch_transparent.png"
        cv2.imwrite(str(trans_path), sketch_rgba)
        print(f"[Save] Transparent sketch → {trans_path}")

        # Warp the RGBA sketch into aerial image space
        warped_rgba   = cv2.warpAffine(sketch_rgba, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=(255, 255, 255, 0))
        warped_binary = (warped_rgba[:, :, 3] > 30).astype(np.uint8) * 255

        # ---- Alpha composite: aerial (bottom) + transparent sketch (top) ----
        alpha_f = warped_rgba[:, :, 3].astype(np.float32) / 255.0   # 0..1
        alpha_f = alpha_f[:, :, np.newaxis]

        aerial_f  = aerial_bgr.astype(np.float32)
        # Keep sketch lines in their original color (black) for clarity
        sketch_f  = warped_rgba[:, :, :3].astype(np.float32)

        result = np.clip(aerial_f * (1 - alpha_f) + sketch_f * alpha_f, 0, 255).astype(np.uint8)

        out_path = output_dir / "alignment_result.png"
        cv2.imwrite(str(out_path), result)
        print(f"[Save] Alignment result → {out_path}")
        return result, warped_binary

    # ----------------------------------------------------------
    # Visualization helpers
    # ----------------------------------------------------------
    def visualize_masks(self,
                        aerial_rgb:  np.ndarray,
                        mask_list:   list,
                        best_idx:    int,
                        scores:      list,
                        output_dir:  Path) -> np.ndarray:
        """Render all SAM masks as a coloured overlay; highlight best match."""
        vis     = aerial_rgb.copy()
        overlay = np.zeros_like(vis)
        cmap    = matplotlib.colormaps.get_cmap("tab20")

        for i, mc in enumerate(mask_list):
            color = (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8).tolist()
            overlay[mc["mask"]] = color

        vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        # Draw index label on every mask centroid
        for i, mc in enumerate(mask_list):
            M_ = cv2.moments(mc["contour"])
            if M_["m00"] > 0:
                lx = int(M_["m10"] / M_["m00"])
                ly = int(M_["m01"] / M_["m00"])
            else:
                lx, ly = mc["contour"][0][0]
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_bgr, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
            cv2.putText(vis_bgr, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Highlight best-match contour
        if 0 <= best_idx < len(mask_list):
            best_cnt = mask_list[best_idx]["contour"]
            cv2.drawContours(vis_bgr, [best_cnt], -1, (0, 0, 255), 3)
            box = cv2.boxPoints(cv2.minAreaRect(best_cnt)).astype(np.int32)
            cv2.drawContours(vis_bgr, [box], -1, (0, 255, 255), 2)
            sc_text = f"[{best_idx}] score={scores[best_idx]:.4f}" if scores else f"[{best_idx}] manual"
            x, y    = best_cnt[0][0]
            cv2.putText(vis_bgr, sc_text, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "masks_preview.png"
        cv2.imwrite(str(mask_path), vis_bgr)
        print(f"[Save] Mask preview → {mask_path}")
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
        print("  MODULE B — Aerial Image Segmentation (SAM2)")
        print("=" * 60)
        aerial_bgr, aerial_rgb, mask_list, scale = self.segment_aerial(aerial_path)

        if not mask_list:
            print("[ERROR] SAM2 produced no usable masks. "
                  "Try lowering pred_iou_thresh or reducing the image.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("  MODULE C — 候选切图导出 & AI/Human 裁判")
        print("=" * 60)
        if TARGET_MASK_IDX is not None:
            # 快速跳过：已预设序号，直接使用，跳过切图导出与交互
            if not (0 <= TARGET_MASK_IDX < len(mask_list)):
                print(f"[ERROR] TARGET_MASK_IDX={TARGET_MASK_IDX} 超出范围 "
                      f"(有效范围 0–{len(mask_list)-1})，请检查 masks_preview.png。")
                sys.exit(1)
            best_idx   = TARGET_MASK_IDX
            best_score = 0.0
            scores     = []
            print(f"[Match] 预设序号覆盖 → mask #{best_idx}  "
                  f"area={mask_list[best_idx]['area']:.0f} px²")
        else:
            # 导出 top6 切图，挂起等待 AI/人工裁判输入切图序号（0–5）
            crop_dir, top6 = self.export_ai_test_crops(aerial_bgr, mask_list, output_dir)

            while True:
                raw = input(
                    f"\n切图已导出至 {crop_dir}。\n"
                    f"请查看 candidate_0.png ~ candidate_5.png，"
                    f"输入最佳切图序号（0–5，输入 q 退出）：\n> "
                ).strip()

                if raw.lower() == "q":
                    print("[INFO] 用户选择退出。")
                    sys.exit(0)

                try:
                    crop_idx = int(raw)
                    if 0 <= crop_idx <= 5:
                        break
                    print(f"[WARN] 序号 {crop_idx} 超出范围，请输入 0–5。")
                except ValueError:
                    print("[WARN] 请输入有效的整数序号，或输入 q 退出。")

            # 将切图序号映射回 mask_list 中的真实索引（用对象身份比对）
            selected_mask = top6[crop_idx]
            best_idx  = next(i for i, m in enumerate(mask_list) if m is selected_mask)
            best_score = 0.0
            scores     = []
            print(f"[Match] AI/Human 选择切图 #{crop_idx} → mask_list[{best_idx}]  "
                  f"area={selected_mask['area']:.0f} px²")

        print("\n" + "=" * 60)
        print("  MODULE D — Geometric Alignment & Overlay")
        print("=" * 60)
        result_img, warped_binary = self.align_and_overlay(
            sketch_path, aerial_bgr, sketch_features,
            mask_list[best_idx], output_dir
        )

        print("\n" + "=" * 60)
        print("  VISUALIZATION")
        print("=" * 60)
        mask_vis = self.visualize_masks(aerial_rgb, mask_list, best_idx, scores, output_dir)

        # ── 4-panel comparison ──────────────────────────────────
        sketch_bgr = cv2.imread(str(sketch_path))

        # Panel 3: isolated target mask on white background
        mask_iso = np.full_like(aerial_bgr, 240)
        mask_iso[mask_list[best_idx]["mask"]] = aerial_bgr[mask_list[best_idx]["mask"]]
        cv2.drawContours(mask_iso, [mask_list[best_idx]["contour"]], -1, (0, 140, 255), 2)

        # Panel 4: zoom into the bounding rect of the target mask (+30% margin)
        x, y, bw, bh = cv2.boundingRect(mask_list[best_idx]["contour"])
        pad = int(max(bw, bh) * 0.30)
        x1  = max(0, x - pad);  y1 = max(0, y - pad)
        x2  = min(aerial_bgr.shape[1], x + bw + pad)
        y2  = min(aerial_bgr.shape[0], y + bh + pad)
        zoom = result_img[y1:y2, x1:x2]

        match_title = (f"Manual override: mask #{best_idx}" if TARGET_MASK_IDX is not None
                       else f"Auto match: #{best_idx}  score={best_score:.4f}")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        panels = [
            (sketch_bgr,                    "① Original Sketch"),
            (mask_vis,                      f"② All SAM2 Masks\n{match_title}"),
            (mask_iso,                      "③ Target Mask (isolated)"),
            (zoom,                          "④ Alignment Zoom\n(cyan=sketch  orange=mask boundary)"),
        ]
        for ax, (img, title) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.suptitle("Cross-modal Building Alignment — PoC", fontsize=13, y=1.01)
        plt.tight_layout()
        fig_path = output_dir / "final_comparison.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"[Save] Final comparison → {fig_path}")

        plt.show()
        return result_img


# ============================================================
# Entry point
# ============================================================
def _check_prerequisites():
    missing = []
    for p in [MODEL_CKPT, SKETCH_PATH, AERIAL_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print(f"  • {m}")
        print("\nExpected layout:")
        print(f"  {BASE_DIR}/")
        print(f"  ├── sketch.png        ← architectural sketch")
        print(f"  ├── aerial.png        ← aerial / bird's-eye image")
        print(f"  └── checkpoints/")
        print(f"      └── sam2.1_hiera_large.pt")
        sys.exit(1)


def main():
    _check_prerequisites()

    matcher = BuildingMatcher(
        model_ckpt=MODEL_CKPT,
        model_cfg=MODEL_CFG,
    )

    matcher.run(
        sketch_path=SKETCH_PATH,
        aerial_path=AERIAL_PATH,
        output_dir=OUTPUT_DIR,
    )

    print("\n[DONE] Pipeline completed. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
