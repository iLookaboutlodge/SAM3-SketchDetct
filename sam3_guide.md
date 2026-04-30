SAM 3 是一个统一的基础模型，用于图像和视频中的可提示分割。它能够利用文本或视觉提示（例如点、框和掩码）来检测、分割和跟踪对象。与前代SAM 2相比，SAM 3 引入了穷举分割由简短文本短语或示例指定的开放词汇概念所有实例的功能。与以往的工作不同，SAM 3 可以处理规​​模更大的开放词汇提示集。在我们包含 27 万个独特概念（比现有基准测试多 50 多倍）的全新SA-CO 基准测试中，SAM 3 的性能达到了人类水平的75-80%。

拥抱脸🤗应用程序

基本用法
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]

官方代码已在sam3 代码库中公开发布。

与变形金刚一起使用🤗
SAM3 - 图像的可提示概念分割 (PCS)
SAM3 对图像执行可提示概念分割 (PCS)，以文本和/或图像示例作为提示，并返回图像中所有匹配对象实例的分割掩码。

纯文本提示
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Segment using text prompt
inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

您可以使用如下所示的简单辅助函数来显示掩码：

import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

然后，您可以保存生成的合成图像或在笔记本中显示它：

overlay_masks(image, results["masks"])

单个边界框提示
使用边界框分割对象：

# Box in xyxy format: [x1, y1, x2, y2] in pixel coordinates
# Example: laptop region
box_xyxy = [100, 150, 500, 450]
input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
input_boxes_labels = [[1]]  # 1 = positive box

inputs = processor(
    images=image,
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

多个提示框（正面和负面）
使用多个分别带有正面和负面标签的方框来完善概念：

# Load kitchen image
kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
kitchen_image = Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")

# Define two positive boxes (e.g., dial and button on oven)
# Boxes are in xyxy format [x1, y1, x2, y2] in pixel coordinates
box1_xyxy = [59, 144, 76, 163]  # Dial box
box2_xyxy = [87, 148, 104, 159]  # Button box
input_boxes = [[box1_xyxy, box2_xyxy]]
input_boxes_labels = [[1, 1]]  # Both positive

inputs = processor(
    images=kitchen_image,
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
overlay_masks(kitchen_image, results["masks"])

组合提示（文本 + 否定框）
使用文字提示和否定性视觉提示来完善概念：

# Segment "handle" but exclude the oven handle using a negative box
text = "handle"
# Negative box covering oven handle area (xyxy): [40, 183, 318, 204]
oven_handle_box = [40, 183, 318, 204]
input_boxes = [[oven_handle_box]]

inputs = processor(
    images=kitchen_image,
    text=text,
    input_boxes=input_boxes,
    input_boxes_labels=[[0]],  # 0 = negative (exclude this region)
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
# This will segment pot handles but exclude the oven handle

基于文本提示的批量推理
批量处理多张带有不同文本提示的图片：

cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
images = [
    Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
    Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
]

text_prompts = ["ear", "dial"]

inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results for both images
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)

print(f"Image 1: {len(results[0]['masks'])} objects found")
print(f"Image 2: {len(results[1]['masks'])} objects found")

批量混合提示
同一批次中的不同图像使用不同的提示类型：

# Image 1: text prompt "laptop"
# Image 2: visual prompt (dial box)
box2_xyxy = [59, 144, 76, 163]

inputs = processor(
    images=images,
    text=["laptop", None],  # Only first image has text
    input_boxes=[None, [box2_xyxy]],  # Only second image has box
    input_boxes_labels=[None, [1]],  # Positive box for second image
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results for both images
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)
# Both images processed in single forward pass

语义分割输出
SAM3 除了实例掩码外，还提供语义分割：

inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Instance segmentation masks
instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

# Semantic segmentation (single channel)
semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

print(f"Instance masks: {instance_masks.shape}")
print(f"Semantic segmentation: {semantic_seg.shape}")

SAM3 视频 - 视频的可提示概念分割 (PCS)
SAM3 Video 对视频执行可提示概念分割 (PCS)，以文本作为提示，检测并跟踪视频帧中所有匹配的对象实例。

预加载视频推理
使用文本提示处理所有帧都已可用的视频：

from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch

device = Accelerator().device
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompt to detect and track objects
text = "person"
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# Process all frames in the video
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=50
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")
Processed 51 frames

# Access results for a specific frame
frame_0_outputs = outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
print(f"Scores: {frame_0_outputs['scores'].tolist()}")
print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks shape: {frame_0_outputs['masks'].shape}")

流媒体视频推理
对于实时应用，Transformer 实现的 SAM3 视频支持在视频帧到达时立即进行处理：

# Initialize session for streaming
streaming_inference_session = processor.init_video_session(
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompt
text = "person"
streaming_inference_session = processor.add_text_prompt(
    inference_session=streaming_inference_session,
    text=text,
)

# Process frames one by one (streaming mode)
streaming_outputs_per_frame = {}
for frame_idx, frame in enumerate(video_frames[:50]):  # Process first 50 frames
    # First, process the frame using the processor
    inputs = processor(images=frame, device=device, return_tensors="pt")
...
    # Process frame using streaming inference - pass the processed pixel_values
    model_outputs = model(
        inference_session=streaming_inference_session,
        frame=inputs.pixel_values[0],  # Provide processed frame - this enables streaming mode
        reverse=False,
    )
...
    # Post-process outputs with original_sizes for proper resolution handling
    processed_outputs = processor.postprocess_outputs(
        streaming_inference_session,
        model_outputs,
        original_sizes=inputs.original_sizes,  # Required for streaming inference
    )
    streaming_outputs_per_frame[frame_idx] = processed_outputs
...
    if (frame_idx + 1) % 10 == 0:
        print(f"Processed {frame_idx + 1} frames...")

print(f"✓ Streaming inference complete! Processed {len(streaming_outputs_per_frame)} frames")
✓ Streaming inference complete! Processed 50 frames

# Access results
frame_0_outputs = streaming_outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects in first frame")
print(f"Boxes are in XYXY format (absolute pixel coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks are at original video resolution: {frame_0_outputs['masks'].shape}")

⚠️ **关于流式推理质量的说明**：流式推理会禁用热启动启发式算法（该算法用于移除不匹配和重复的对象），因为这些算法需要访问后续帧才能做出明智的决策。与预加载视频推理相比，这可能会导致更多的误报和重复对象轨迹。为了获得最佳结果，请在所有帧都可用时使用预加载视频推理。
SAM3 Tracker - 图像可提示视觉分割 (PVS)
Sam3Tracker 对图像执行可提示视觉分割 (PVS)，它接受交互式视觉提示（点、框、掩码），并根据每个提示分割特定的对象实例。它是 SAM2 的升级版本，在保持相同 API 的同时，性能得到了提升，因此可以无缝替代 SAM2 工作流程。

使用流水线自动生成掩模
from transformers import pipeline

generator = pipeline("mask-generation", model="facebook/sam3", device=0)
image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
outputs = generator(image_url, points_per_batch=64)

len(outputs["masks"])  # Number of masks generated

基本图像分割
单点点击
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from accelerate import Accelerator
import torch
from PIL import Image
import requests

device = Accelerator().device

model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

# The model outputs multiple mask predictions ranked by quality score
print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")

多点改进
# Add both positive and negative points to refine the mask
input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
input_labels = [[[1, 1]]]  # Both positive clicks

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

边界框输入
# Define bounding box as [x_min, y_min, x_max, y_max]
input_boxes = [[[75, 275, 1725, 850]]]

inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

多对象分割
# Define points for two different objects
input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
input_labels = [[[1], [1]]]  # Positive clicks for both objects

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

# Each object gets its own mask
masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects

批量推理
# Load multiple images
image_urls = [
    "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
]
raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

# Single point per image
input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
input_labels = [[[1]], [[1]]]  # Positive clicks for both images

inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

# Post-process masks for each image
all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")

SAM3 Tracker Video - 视频可提示视觉分割 (PVS)
Sam3TrackerVideo 对视频执行可提示视觉分割 (PVS)，它接受交互式视觉提示（点、框、遮罩），并根据每个提示在视频帧中跟踪特定的对象实例。它是 SAM2 Video 的升级版本，在保持相同 API 的同时，性能得到了提升，因此可以无缝替代 SAM2 Video 的工作流程。

基础视频跟踪
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator
import torch

device = Accelerator().device
model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    dtype=torch.bfloat16,
)

# Add click on first frame to select object
ann_frame_idx = 0
ann_obj_id = 1
points = [[[[210, 350]]]]
labels = [[[1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=points,
    input_labels=labels,
)

# Segment the object on the first frame (optional, you can also propagate the masks through the video directly)
outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)
video_res_masks = processor.post_process_masks(
    [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
)[0]
print(f"Segmentation shape: {video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

# Propagate through the entire video
video_segments = {}
for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    video_segments[sam3_tracker_video_output.frame_idx] = video_res_masks

print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 180 frames

多目标视频跟踪
同时跟踪视频帧中的多个对象：

# Reset for new tracking session
inference_session.reset_inference_session()

# Add multiple objects on the first frame
ann_frame_idx = 0
obj_ids = [2, 3]
input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
input_labels = [[[1], [1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=obj_ids,
    input_points=input_points,
    input_labels=input_labels,
)

# Get masks for both objects on first frame (optional, you can also propagate the masks through the video directly)
outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)

# Propagate both objects through video
video_segments = {}
for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    video_segments[sam3_tracker_video_output.frame_idx] = {
        obj_id: video_res_masks[i]
        for i, obj_id in enumerate(inference_session.obj_ids)
    }

print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames

流媒体视频推理
对于实时应用，Sam3TrackerVideo 支持在视频帧到达时立即进行处理：

# Initialize session for streaming
inference_session = processor.init_video_session(
    inference_device=device,
    dtype=torch.bfloat16,
)

# Process frames one by one
for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
    inputs = processor(images=frame, device=device, return_tensors="pt")
...
    if frame_idx == 0:
        # Add point input on first frame
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=1,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
            original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
        )
...
    # Process current frame
    sam3_tracker_video_output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
...
    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
    )[0]
    print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")