"""
SAM3 探索脚本 — 看看模型都能识别什么

用法：
  1. 修改下面的 IMAGE_PATH 和 PROMPTS
  2. python sam3_explore.py
  3. 查看弹窗和 output/sam3_explore.png
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================
# ★ 你只需要改这里
# ============================================================
IMAGE_PATH = Path(r"D:\2026 myplan\SAMLearning\aerial_2.jpg")
MODEL_PATH = Path(r"D:\2026 myplan\SAMLearning\checkpoints\sam3.pt")
OUTPUT_DIR = Path(r"D:\2026 myplan\SAMLearning\output")

PROMPTS = [
    # "rooftop",
    # "balcony",
    # "terrace",
    # "the entire building footprint",
    # "tree",
    #"road",
    # "deck",
    # "car",
    # "swimming pool",
    # "outdoor building seating area",
    # "wooden backyard deck",
    # "garden",
    "building",
    "outdoor building seating area",
    # "terrace"
]

# 负向 prompt：检测到这些 object 后，凡是与它们 IoU 超过 NEG_IOU_THRESH 的正向结果都丢弃
NEGATIVE_PROMPTS = [
    "car",
    "tree",
    "road",
    # "terrace"
]

IMGSZ         = 640  # None = 自动使用图片较长边；或手动指定如 640 / 1024
CONF          = 0.10
IOU_DEDUP     = 0.5   # 两个正向 mask 的 IoU 超过此值视为重复，保留置信度高的那个
NEG_IOU_THRESH = 0.3  # 正向 mask 与任一负向 mask 重叠超过此值则被过滤
# ============================================================

COLORS = [
    (255,  60,  60), (60, 180, 255), ( 60, 220,  60), (255, 180,  30),
    (180,  60, 255), (255, 100, 180), ( 60, 220, 200), (200, 200,  60),
]


def load_predictor(imgsz: int):
    from ultralytics.models.sam import SAM3SemanticPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {torch.cuda.get_device_properties(0).name}  {vram:.1f} GB")
    print(f"[INFO] imgsz={imgsz}")
    overrides = dict(
        conf=CONF, task="segment", mode="predict",
        model=str(MODEL_PATH), imgsz=imgsz, device=device,
        verbose=False, save=False,
    )
    return SAM3SemanticPredictor(overrides=overrides), device


def run(predictor, img_bgr: np.ndarray) -> list[dict]:
    """返回所有检测到的 mask，每条记录含 prompt/mask/conf/bbox。"""
    h, w = img_bgr.shape[:2]
    predictor.set_image(img_bgr)
    results = predictor(text=PROMPTS)

    detections = []
    for r in results:
        if r.masks is None:
            continue
        masks_np = r.masks.data.cpu().numpy()       # [N, H, W]
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
            if not mask_bool.any():
                continue
            prompt = PROMPTS[cls_id] if cls_id < len(PROMPTS) else f"class{cls_id}"
            detections.append({"prompt": prompt, "mask": mask_bool, "conf": float(conf)})

    print(f"[INFO] {len(detections)} mask(s) detected across {len(PROMPTS)} prompts")

    # --- Negative prompt 过滤 ---
    if NEGATIVE_PROMPTS:
        predictor.set_image(img_bgr)
        neg_masks = []
        for r in predictor(text=NEGATIVE_PROMPTS):
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
                    (inter := np.logical_and(det["mask"].ravel(), nm).sum()) /
                    (np.logical_or(det["mask"].ravel(), nm).sum() + 1e-9) > NEG_IOU_THRESH
                    for nm in neg_masks
                )
            ]
            print(f"[INFO] Negative filter: removed {before - len(detections)} mask(s) "
                  f"(prompts={NEGATIVE_PROMPTS}, iou_thresh={NEG_IOU_THRESH})")

    # --- IoU 去重：置信度高的优先，重叠太大的后来者丢弃 ---
    detections.sort(key=lambda x: x["conf"], reverse=True)
    kept = []
    for det in detections:
        flat = det["mask"].ravel()
        duplicate = False
        for k in kept:
            inter = np.logical_and(flat, k["mask"].ravel()).sum()
            union = np.logical_or(flat, k["mask"].ravel()).sum()
            if union > 0 and inter / union > IOU_DEDUP:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    print(f"[INFO] {len(kept)} unique mask(s) after IoU dedup (threshold={IOU_DEDUP})")
    return kept


def visualize(img_bgr: np.ndarray, detections: list[dict], out_path: Path):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    # 米色背景，只在 mask 区域叠色，非 mask 区域保持干净
    BG_COLOR = np.array([245, 242, 230], dtype=np.float32)  # 米色
    overlay  = np.full((h, w, 3), BG_COLOR, dtype=np.float32)

    legend_patches = []
    for i, det in enumerate(detections):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        m = det["mask"]
        # 只混合 mask 区域：原图 50% + 颜色 50%
        overlay[m] = img_rgb.astype(np.float32)[m] * 0.5 + color * 0.5

        seg  = det["mask"].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color.tolist(), 2)

        label = f"[{i}] {det['prompt']}  {det['conf']:.2f}"
        legend_patches.append(
            mpatches.Patch(color=color / 255, label=label)
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # 在图上标序号
    for i, det in enumerate(detections):
        seg  = det["mask"].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            M = cv2.moments(max(cnts, key=cv2.contourArea))
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(i), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(overlay, str(i), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    fig, (ax_orig, ax_det) = plt.subplots(1, 2, figsize=(22, 10))
    ax_orig.imshow(img_rgb)
    ax_orig.set_title("Original", fontsize=11)
    ax_orig.axis("off")
    ax_det.imshow(overlay)
    ax_det.set_title(f"SAM3 — {len(detections)} masks", fontsize=11)
    ax_det.axis("off")
    ax_det.legend(handles=legend_patches, loc="upper right",
                  fontsize=8, framealpha=0.8)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"[Save] → {out_path}")
    plt.show()


def main():
    img_bgr = cv2.imread(str(IMAGE_PATH))
    if img_bgr is None:
        raise FileNotFoundError(IMAGE_PATH)

    h, w = img_bgr.shape[:2]
    imgsz = IMGSZ if IMGSZ is not None else max(h, w)

    predictor, _ = load_predictor(imgsz)
    detections   = run(predictor, img_bgr)

    if not detections:
        print("[WARN] No masks returned — try different prompts or lower CONF.")
        return

    for i, det in enumerate(detections):
        print(f"  [{i}] {det['prompt']:<45} conf={det['conf']:.3f}")

    visualize(img_bgr, detections, OUTPUT_DIR / "sam3_explore.png")


if __name__ == "__main__":
    main()
