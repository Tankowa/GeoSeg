# CUDA_VISIBLE_DEVICES=1 python process_annotations_clip_surverg.py --annotations_dir /data/jianglifan/sam_new/RS_Earth2_Benchmark/annotations --api_base_url http://localhost:8010 --output_dir ./11111
# CUDA_VISIBLE_DEVICES=6 python process_annotations_clip_surverg.py --annotations_dir /data/jianglifan/sam_new/RS_ReasonSeg_Benchmark/annotations --api_base_url http://localhost:8011
# conda activate flow_grpo
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
import requests
from typing import Optional, Tuple, List
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import SAM
import tempfile
import torch
import clip_surgery
import clip  # 系统的clip包，用于SAM3 prompt方法
import argparse
import time

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from datetime import datetime

# Timing configuration: measure per-item processing time for the first N items
MAX_TIMING_ITEMS = 20
_timing_item_times = []
_total_items_timed = 0

# API 服务器配置（默认值，可通过命令行参数覆盖）
API_BASE_URL = "http://localhost:8010"  # 修改为你的 API 服务器地址
API_ENDPOINT = f"{API_BASE_URL}/generate"

# 初始化 SAM3 predictor（全局变量，避免重复初始化）
_sam3_predictor = None
_sam3_model = None
_clip_model = None
_clip_preprocess = None
_clip_device = None

BICUBIC = InterpolationMode.BICUBIC

def get_sam3_model():
    """获取或初始化 SAM3 模型（用于点提示分割）"""
    global _sam3_model
    if _sam3_model is None:
        print("Initializing SAM3 model for point prompts...")
        sam3_model_path = "/data/jianglifan/sam_new/checkpoints/sam3.pt"
        _sam3_model = SAM(sam3_model_path)
        print("SAM3 model initialized.")
    return _sam3_model

def get_sam3_predictor():
    """获取或初始化 SAM3 predictor（用于文本提示分割）"""
    global _sam3_predictor
    if _sam3_predictor is None:
        print("Initializing SAM3 predictor for text prompts...")
        sam3_model_path = "/data/jianglifan/sam_new/checkpoints/sam3.pt"
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=sam3_model_path,
            half=True,  # Use FP16 for faster inference
            save=False,  # 我们自己保存结果
        )
        _sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3 predictor initialized.")
    return _sam3_predictor

def get_clip_model():
    """获取或初始化 CLIP Surgery 模型"""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        print("Initializing CLIP Surgery model...")
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {_clip_device}")
        _clip_model, _ = clip_surgery.load("CS-ViT-B/16", device=_clip_device)
        _clip_model.eval()
        
        # 高分辨率预处理 (512x512)
        _clip_preprocess = Compose([
            Resize((512, 512), interpolation=BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        print("CLIP Surgery model initialized.")
    return _clip_model, _clip_preprocess, _clip_device

def parse_qwen_response(response_text):
    """解析 Qwen 的响应，提取边界框和 prompt（center是可选的，如果没有则从bbox计算）"""
    result = {
        'bbox': None,
        'center': None,
        'prompt': None
    }
    
    # 首先尝试找到完整的 JSON 对象（可能包含嵌套结构）
    # 尝试多种 JSON 匹配模式（不要求center字段）
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"bbox"[^{}]*(?:\{[^{}]*\}[^{}]*)*"prompt"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\{[^}]*"bbox"[^}]*"prompt"[^}]*\}',
        r'\{.*?"bbox".*?"prompt".*?\}',
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                json_str = match.group()
                # 清理可能的代码块标记
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'```\s*', '', json_str)
                parsed = json.loads(json_str)
                if 'bbox' in parsed:
                    result['bbox'] = parsed['bbox']
                if 'center' in parsed:
                    result['center'] = parsed['center']
                if 'prompt' in parsed:
                    result['prompt'] = parsed['prompt']
                # 只要bbox和prompt都有值就返回（center是可选的）
                if result['bbox'] and result['prompt']:
                    return result
            except json.JSONDecodeError:
                continue
    
    # 如果完整 JSON 解析失败，尝试分别提取各个字段
    # 尝试提取边界框坐标
    bbox_patterns = [
        r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r'bbox[:\s]*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?',
        r'边界框[:\s]*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?',
        r'\((\d+)\s*,\s*(\d+)\)\s*到\s*\((\d+)\s*,\s*(\d+)\)',
    ]
    for pattern in bbox_patterns:
        match = re.search(pattern, response_text)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            result['bbox'] = [x1, y1, x2, y2]
            break
    
    # 尝试提取中心点坐标
    center_patterns = [
        r'"center"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
        r'center[:\s]*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?',
        r'中心点[:\s]*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?',
        r'中心[:\s]*\((\d+)\s*,\s*(\d+)\)',
    ]
    for pattern in center_patterns:
        match = re.search(pattern, response_text)
        if match:
            cx, cy = map(int, match.groups())
            result['center'] = [cx, cy]
            break
    
    # 尝试提取 prompt
    prompt_patterns = [
        r'"prompt"\s*:\s*"([^"]+)"',
        r'"prompt"\s*:\s*\'([^\']+)\'',
        r'prompt[:\s]*["\']([^"\']+)["\']',
        r'分割提示[:\s]*["\']([^"\']+)["\']',
        r'SAM3[:\s]*["\']([^"\']+)["\']',
    ]
    for pattern in prompt_patterns:
        match = re.search(pattern, response_text)
        if match:
            result['prompt'] = match.group(1).strip()
            break
    
    return result

def draw_bbox_on_image(image_path, bbox, center, output_path):
    """在图像上绘制边界框（不绘制中心点）"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        # 绘制边界框（红色）
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        # 标注坐标
        draw.text((x1, y1 - 20), f"({x1}, {y1})", fill='red')
        draw.text((x2, y2 + 5), f"({x2}, {y2})", fill='red')
    
    # 不再绘制中心点
    
    img.save(output_path)
    print(f"Saved annotated image to: {output_path}")

def extract_keypoints_with_clip_surgery(cropped_image: np.ndarray, text_prompt: str, threshold: float = 0.8) -> Tuple[List, List]:
    """
    使用 CLIP Surgery 在裁剪图像上提取关键点
    
    Args:
        cropped_image: 裁剪的图像（numpy array, BGR格式）
        text_prompt: 文本提示
        threshold: 关键点提取阈值
    
    Returns:
        points: 关键点列表 [[x1, y1], [x2, y2], ...]
        labels: 标签列表 [1, 0, 1, ...] (1=前景, 0=背景)
    """
    try:
        clip_model, clip_preprocess, clip_device = get_clip_model()
        
        # 将BGR转换为RGB，然后转换为PIL Image
        rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # 预处理图像
        image_tensor = clip_preprocess(pil_image).unsqueeze(0).to(clip_device)
        
        # 获取图像尺寸
        h, w = cropped_image.shape[:2]
        
        # 提取关键点
        texts = [text_prompt]
        with torch.no_grad():
            # 1. 提取图像特征
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 2. 提取文本特征（使用提示词集成）
            text_features = clip_surgery.encode_text_with_prompt_ensemble(clip_model, texts, clip_device)
            
            # 3. 提取冗余特征（从空字符串）
            redundant_features = clip_surgery.encode_text_with_prompt_ensemble(clip_model, [""], clip_device)
            
            # 4. 应用 CLIP 特征手术
            similarity = clip_surgery.clip_feature_surgery(
                image_features,
                text_features,
                redundant_features
            )[0]  # 取第一个批次
            
            # 5. 从相似度图提取关键点
            points, labels = clip_surgery.similarity_map_to_points(
                similarity[1:, 0],  # 跳过 CLS token，取第一个文本
                (h, w),  # 裁剪图像的尺寸
                t=threshold
            )
        
        # 转换为列表格式
        points_list = [[int(p[0]), int(p[1])] for p in points]
        labels_list = [int(l) for l in labels]
        
        print(f"Extracted {len(points_list)} keypoints from CLIP Surgery (threshold={threshold})")
        positive_count = sum(labels_list)
        print(f"  Positive points: {positive_count}, Negative points: {len(labels_list) - positive_count}")
        
        return points_list, labels_list
        
    except Exception as e:
        print(f"Error extracting keypoints with CLIP Surgery: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def segment_with_sam3(image_path, bbox, sam3_prompt, keypoints_viz_path=None):
    """
    使用 CLIP Surgery 提取关键点，然后用 SAM3 点提示对 bbox 区域进行分割
    
    Args:
        image_path: 原始图像路径
        bbox: 边界框 [x1, y1, x2, y2]
        sam3_prompt: SAM3 文本提示（用于CLIP Surgery提取关键点）
        keypoints_viz_path: 关键点可视化图像保存路径（可选）
    
    Returns:
        mask: 分割掩膜（numpy array，与原始图像尺寸相同）
        points: 提取的关键点列表
        labels: 关键点标签列表
    """
    if not bbox or not sam3_prompt:
        print("Warning: Missing bbox or sam3_prompt, skipping SAM3 segmentation")
        return None, [], []
    
    temp_crop_path = None
    try:
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 裁剪 bbox 区域
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # 如果裁剪区域过小（特别是高度或宽度为 1 像素），Ultralytics 在内部 padding/crop 时
        # 很容易出现 top >= bottom，导致 scale_masks 里得到 H=0 的 mask。
        # 这里不直接丢弃该样本，而是以 bbox 中心为基准，适度扩展 bbox，保证有一个最小尺寸。
        MIN_SIZE = 16  # 经验值，保证后续插值不会出 H=0
        bw = x2 - x1
        bh = y2 - y1
        if bw < MIN_SIZE or bh < MIN_SIZE:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half_w = MIN_SIZE // 2
            half_h = MIN_SIZE // 2
            new_x1 = max(0, cx - half_w)
            new_y1 = max(0, cy - half_h)
            new_x2 = min(w, cx + half_w)
            new_y2 = min(h, cy + half_h)
            print(f"Info: Expanding small bbox {bbox} to [{new_x1}, {new_y1}, {new_x2}, {new_y2}] "
                  f"for stable SAM3 segmentation.")
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
        
        cropped_image = image[y1:y2, x1:x2]
        cropped_h, cropped_w = cropped_image.shape[:2]
        
        # print(f"Cropped region: {x1}, {y1}, {x2}, {y2} (size: {cropped_w}x{cropped_h})")
        
        # 保存临时裁剪图像用于 SAM3（使用临时文件避免并发冲突）
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix='sam3_crop_')
        temp_crop_path = temp_file.name
        temp_file.close()
        cv2.imwrite(temp_crop_path, cropped_image)
        
        # 步骤1: 使用 CLIP Surgery 在裁剪区域内提取关键点
        print(f"Extracting keypoints with CLIP Surgery using prompt: '{sam3_prompt}'")
        points, labels = extract_keypoints_with_clip_surgery(cropped_image, sam3_prompt, threshold=0.8)
        
        if len(points) == 0:
            print("Warning: No keypoints extracted by CLIP Surgery, skipping segmentation")
            return None, [], []
        
        # 步骤1.5: 可视化关键点并保存（如果提供了输出路径）
        if keypoints_viz_path:
            try:
                # 复制裁剪图像用于可视化
                vis_image = cropped_image.copy()
                
                # 在图像上绘制关键点
                for i, (point, label) in enumerate(zip(points, labels)):
                    x, y = int(point[0]), int(point[1])
                    # 正样本点用橙色，负样本点用蓝色 (BGR格式)
                    # 正样本点（前景）: (0, 165, 255) = 橙色
                    # 负样本点（背景）: (255, 165, 0) = 蓝色
                    color_point = (0, 165, 255) if label == 1 else (255, 165, 0)  # BGR格式
                    cv2.circle(vis_image, (x, y), 5, color_point, -1)  # 实心圆
                    cv2.circle(vis_image, (x, y), 8, color_point, 2)  # 外圈
                
                # 保存可视化图像
                cv2.imwrite(keypoints_viz_path, vis_image)
                print(f"Saved keypoints visualization to: {keypoints_viz_path}")
            except Exception as e:
                print(f"Warning: Failed to save keypoints visualization: {e}")
        
        # 步骤2: 只使用正样本点（label=1），过滤掉负样本点
        # 过滤出正样本点
        positive_points = [p for p, l in zip(points, labels) if l == 1]

        # 如果 CLIP Surgery 没有给出正样本点，可以在裁剪图的中心附近人工生成若干正样本点，
        # 避免因为关键点缺失而完全跳过该目标，同时满足你“在 bbox 中心随机生成多个像素”的需求。
        if len(positive_points) == 0:
            print("Warning: No positive keypoints found from CLIP Surgery, "
                  "generating synthetic foreground points around bbox center.")
            ch, cw = cropped_image.shape[:2]
            cx, cy = cw // 2, ch // 2
            synthetic_points = []
            # 在中心附近 3x3 的随机轻微扰动网格上生成点（全部视为前景）
            offsets = [(-2, -2), (0, -2), (2, -2),
                       (-2, 0),  (0, 0),  (2, 0),
                       (-2, 2),  (0, 2),  (2, 2)]
            for dx, dy in offsets:
                px = max(0, min(cw - 1, cx + dx))
                py = max(0, min(ch - 1, cy + dy))
                synthetic_points.append([int(px), int(py)])
            positive_points = synthetic_points
            # 同时把 points/labels 也更新，便于后续 JSON 记录
            points = positive_points
            labels = [1] * len(positive_points)
        
        # 将点转换为 SAM3 需要的格式（只使用正样本点，所有点的label都是1）
        # SAM3 需要的格式: 
        # - 单点: points=[x, y], labels=[1]
        # - 多点: points=[[x1, y1], [x2, y2], ...], labels=[1, 1, 1, ...]
        if len(positive_points) == 1:
            sam_points = positive_points[0]  # [x, y]
            sam_labels = [1]  # 只有正样本点，label都是1
        else:
            sam_points = positive_points  # [[x1, y1], [x2, y2], ...]
            sam_labels = [1] * len(positive_points)  # 所有点的label都是1
        
        print(f"Using {len(positive_points)} positive keypoints for SAM3 segmentation (filtered from {len(points)} total keypoints)")
        
        # 步骤3: 使用 SAM3 进行点分割（在裁剪图像上）
        print("Running SAM3 segmentation with point prompts...")
        sam3_model = get_sam3_model()
        
        results = sam3_model.predict(
            source=temp_crop_path,
            points=sam_points,
            labels=sam_labels
        )
        
        # 获取结果
        result = results[0]
        masks = getattr(result, "masks", None)
        
        if masks is None or masks.data is None:
            print("Warning: SAM3 did not detect any masks")
            return None, points, labels
        
        # 获取掩膜数据
        mask_data = masks.data
        try:
            mask_np = mask_data.cpu().numpy()
        except Exception:
            mask_np = np.array(mask_data)
        
        # 合并所有掩膜（如果有多个，合并它们）
        if mask_np.ndim == 3:
            # 如果有多个掩膜，合并它们
            cropped_mask = (mask_np.max(axis=0) > 0).astype(np.uint8)
        else:
            cropped_mask = (mask_np > 0).astype(np.uint8)
        
        # 调整掩膜尺寸以匹配裁剪区域（如果需要）
        mh, mw = cropped_mask.shape[:2]
        if (mh, mw) != (cropped_h, cropped_w):
            cropped_mask = cv2.resize(cropped_mask, (cropped_w, cropped_h), interpolation=cv2.INTER_NEAREST)
        
        # 步骤4: 将裁剪区域的掩膜还原到原始图像尺寸
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = cropped_mask
        
        print(f"SAM3 segmentation completed. Mask shape: {full_mask.shape}")
        return full_mask, points, labels
        
    except Exception as e:
        print(f"Error during SAM3 segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []
    finally:
        # 确保临时文件总是被清理
        if temp_crop_path and os.path.exists(temp_crop_path):
            try:
                os.remove(temp_crop_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_crop_path}: {e}")

def segment_with_sam3_prompt(image_path, bbox, sam3_prompt):
    """
    使用 SAM3 文本提示对 bbox 区域进行分割
    
    Args:
        image_path: 原始图像路径
        bbox: 边界框 [x1, y1, x2, y2]
        sam3_prompt: SAM3 文本提示
    
    Returns:
        mask: 分割掩膜（numpy array，与原始图像尺寸相同）
    """
    if not bbox or not sam3_prompt:
        print("Warning: Missing bbox or sam3_prompt, skipping SAM3 prompt segmentation")
        return None
    
    temp_crop_path = None
    try:
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 裁剪 bbox 区域
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # 与点提示同样逻辑：如果 bbox 过小，则围绕中心扩展到一个最小尺寸，
        # 避免 Ultralytics 在 scale_masks 中出现 H=0。
        MIN_SIZE = 16
        bw = x2 - x1
        bh = y2 - y1
        if bw < MIN_SIZE or bh < MIN_SIZE:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half_w = MIN_SIZE // 2
            half_h = MIN_SIZE // 2
            new_x1 = max(0, cx - half_w)
            new_y1 = max(0, cy - half_h)
            new_x2 = min(w, cx + half_w)
            new_y2 = min(h, cy + half_h)
            print(f"Info: Expanding small bbox {bbox} to [{new_x1}, {new_y1}, {new_x2}, {new_y2}] "
                  f"for stable SAM3 prompt segmentation.")
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
        
        cropped_image = image[y1:y2, x1:x2]
        cropped_h, cropped_w = cropped_image.shape[:2]
        
        print(f"Cropped region for SAM3 prompt: {x1}, {y1}, {x2}, {y2} (size: {cropped_w}x{cropped_h})")
        
        # 保存临时裁剪图像用于 SAM3（使用临时文件避免并发冲突）
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix='sam3_prompt_crop_')
        temp_crop_path = temp_file.name
        temp_file.close()
        cv2.imwrite(temp_crop_path, cropped_image)
        
        # 获取 SAM3 predictor
        predictor = get_sam3_predictor()
        
        # 设置图像
        predictor.set_image(temp_crop_path)
        
        # 使用文本提示进行分割
        print(f"Running SAM3 prompt segmentation with prompt: '{sam3_prompt}'")
        results = predictor(text=[sam3_prompt])
        
        # 获取结果
        result = results[0]
        masks = getattr(result, "masks", None)
        
        if masks is None or masks.data is None:
            print("Warning: SAM3 prompt did not detect any masks")
            return None
        
        # 获取掩膜数据
        mask_data = masks.data
        try:
            mask_np = mask_data.cpu().numpy()
        except Exception:
            mask_np = np.array(mask_data)
        
        # 合并所有掩膜（如果有多个，合并它们）
        if mask_np.ndim == 3:
            # 如果有多个掩膜，合并它们
            cropped_mask = (mask_np.max(axis=0) > 0).astype(np.uint8)
        else:
            cropped_mask = (mask_np > 0).astype(np.uint8)
        
        # 调整掩膜尺寸以匹配裁剪区域（如果需要）
        mh, mw = cropped_mask.shape[:2]
        if (mh, mw) != (cropped_h, cropped_w):
            cropped_mask = cv2.resize(cropped_mask, (cropped_w, cropped_h), interpolation=cv2.INTER_NEAREST)
        
        # 将裁剪区域的掩膜还原到原始图像尺寸
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = cropped_mask
        
        print(f"SAM3 prompt segmentation completed. Mask shape: {full_mask.shape}")
        return full_mask
        
    except Exception as e:
        print(f"Error during SAM3 prompt segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 确保临时文件总是被清理
        if temp_crop_path and os.path.exists(temp_crop_path):
            try:
                os.remove(temp_crop_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_crop_path}: {e}")

def save_segmentation_result(image_path, mask, output_path, alpha=0.5):
    """
    保存分割结果到图像（保存为0-1二值mask）
    
    Args:
        image_path: 原始图像路径
        mask: 分割掩膜（numpy array）
        output_path: 输出图像路径
        alpha: 叠加透明度（0-1，已废弃，保留以兼容）
    """
    if mask is None:
        print("Warning: No mask to save")
        return
    
    # 读取原始图像以获取尺寸
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # 确保掩膜尺寸匹配
    h, w = image.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 转换为0-1二值mask（值为0或1）
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 保存为二值图像：0保持为0（黑色），1转换为255（白色）以便可视化
    binary_mask_vis = binary_mask * 255
    cv2.imwrite(output_path, binary_mask_vis)
    # print(f"Saved segmentation result to: {output_path}")

def save_combined_segmentation_result(image_path, mask_clip_surgery, mask_sam3_prompt, output_path, alpha=0.5):
    """
    保存合并的分割结果到图像（保存为0-1二值mask）
    - 如果其中一个mask为空，则直接使用另一个
    - 否则显示交集区域：两种分割方法都检测到的区域
    
    Args:
        image_path: 原始图像路径
        mask_clip_surgery: CLIP Surgery分割掩膜（numpy array）
        mask_sam3_prompt: SAM3 prompt分割掩膜（numpy array）
        output_path: 输出图像路径
        alpha: 叠加透明度（0-1，已废弃，保留以兼容）
    """
    if mask_clip_surgery is None and mask_sam3_prompt is None:
        print("Warning: No masks to save")
        return
    
    # 读取原始图像以获取尺寸
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # 确保掩膜尺寸匹配并转换为二值掩膜
    mask_clip_valid = None
    if mask_clip_surgery is not None:
        mh, mw = mask_clip_surgery.shape[:2]
        if (mh, mw) != (h, w):
            mask_clip_surgery = cv2.resize(mask_clip_surgery, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_clip_valid = (mask_clip_surgery > 0).astype(np.uint8)
        # 检查是否为空（全0）
        if np.sum(mask_clip_valid) == 0:
            mask_clip_valid = None
    
    mask_sam3_valid = None
    if mask_sam3_prompt is not None:
        mh, mw = mask_sam3_prompt.shape[:2]
        if (mh, mw) != (h, w):
            mask_sam3_prompt = cv2.resize(mask_sam3_prompt, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_sam3_valid = (mask_sam3_prompt > 0).astype(np.uint8)
        # 检查是否为空（全0）
        if np.sum(mask_sam3_valid) == 0:
            mask_sam3_valid = None
    
    # 决定使用哪个掩膜
    if mask_clip_valid is None and mask_sam3_valid is None:
        print("Warning: Both masks are empty")
        return
    elif mask_clip_valid is None:
        # 只有 SAM3 prompt 有效，直接使用它
        final_mask = mask_sam3_valid
    elif mask_sam3_valid is None:
        # 只有 CLIP Surgery 有效，直接使用它
        final_mask = mask_clip_valid
    else:
        # 两个都有效，取交集
        final_mask = ((mask_clip_valid == 1) & (mask_sam3_valid == 1)).astype(np.uint8)
    
    # 保存为二值图像：0保持为0（黑色），1转换为255（白色）以便可视化
    final_mask_vis = final_mask * 255
    cv2.imwrite(output_path, final_mask_vis)
    # print(f"Saved combined segmentation result to: {output_path}")
    # print(f"  Final mask area: {np.sum(final_mask)} pixels")

def process_annotation_file(json_path, base_output_dir):
    """处理单个 JSON 注释文件
    
    Args:
        json_path: JSON注释文件路径
        base_output_dir: 基础输出目录（已包含时间戳）
    """
    global _timing_item_times, _total_items_timed

    print(f"\n{'='*60}")
    print(f"Processing: {json_path}")
    print(f"{'='*60}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从JSON文件路径中提取类别名（例如：Urban.json -> Urban）
    category = Path(json_path).stem
    
    for idx, item in enumerate(data):
        print(f"\n--- Processing item {idx + 1}/{len(data)} ---")
        
        # 从JSON item中获取必要信息
        item_id = item.get('id', 'N/A')
        difficulty = item.get('difficulty', 'N/A')
        question = item.get('question', 'N/A')
        image_path = item.get('file_name', 'N/A')
        
        print(f"ID: {item_id}")
        print(f"Difficulty: {difficulty}")
        print(f"Question: {question}")
        print(f"Image: {image_path}")
        
        # 为每个item创建对应的输出目录：{category}/{difficulty}/{id}/
        item_output_dir = base_output_dir / category / difficulty / str(item_id)
        item_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Item output directory: {item_output_dir}")

        # 断点续传：如果该 item 的结果 JSON 已存在，则认为已经处理完成，直接跳过
        result_json_path = item_output_dir / "result.json"
        if result_json_path.exists():
            print(f"Result JSON already exists for item {item_id}, skip (resume).")
            continue
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # 读取图像尺寸
        img = Image.open(image_path)
        img_width, img_height = img.size
        print(f"Image size: {img_width} x {img_height}")

        # Timing: measure per-item processing time (pipeline) for the first MAX_TIMING_ITEMS items
        timing_this_item = _total_items_timed < MAX_TIMING_ITEMS
        if timing_this_item:
            item_start_time = time.time()
        else:
            item_start_time = None
        
        # 通过 API 调用 Qwen 模型
        print("Sending request to Qwen API...")
        try:
            api_request = {
                "image_path": image_path,
                "question": question,
                "image_width": img_width,
                "image_height": img_height,
                "max_new_tokens": 1024
            }
            
            print(f"API Request: {json.dumps(api_request, indent=2, ensure_ascii=False)}")
            
            response_obj = requests.post(API_ENDPOINT, json=api_request, timeout=300)
            response_obj.raise_for_status()
            
            result = response_obj.json()
            response = result.get('response', '')
            status = result.get('status', 'unknown')
            
            # print(f"\nAPI Status: {status}")
            # print(f"Qwen Response:\n{response}")
            
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to API server at {API_BASE_URL}")
            print("Please make sure the API server is running. Start it with: python qwen_api_server.py")
            continue
        except requests.exceptions.Timeout:
            print("Error: API request timed out")
            continue
        except requests.exceptions.HTTPError as e:
            print(f"Error: API request failed with status {e.response.status_code}: {e.response.text}")
            continue
        except Exception as e:
            print(f"Error: Failed to call API: {e}")
            continue
        
        # 解析响应
        parsed_result = parse_qwen_response(response)
        # print(f"\nParsed Results:")
        # print(f"  BBox: {parsed_result['bbox']}")
        # print(f"  Center: {parsed_result['center']}")
        # print(f"  Prompt: {parsed_result['prompt']}")
        
        # 如果解析失败，尝试直接提取 JSON
        if not all(parsed_result.values()):
            print("Warning: Failed to parse some fields, trying alternative parsing...")
            # 尝试找到 JSON 块
            json_blocks = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
            for block in json_blocks:
                try:
                    parsed = json.loads(block)
                    if 'bbox' in parsed:
                        parsed_result['bbox'] = parsed['bbox']
                    if 'center' in parsed:
                        parsed_result['center'] = parsed['center']
                    if 'prompt' in parsed:
                        parsed_result['prompt'] = parsed['prompt']
                    print(f"Successfully parsed JSON block: {parsed}")
                    break
                except:
                    continue
        
        # 验证和调整坐标
        if parsed_result['bbox']:
            x1, y1, x2, y2 = parsed_result['bbox']
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))
            parsed_result['bbox'] = [x1, y1, x2, y2]
            # print(f"Adjusted BBox: {parsed_result['bbox']}")
        
        # 如果没有中心点，从边界框计算
        if not parsed_result['center'] and parsed_result['bbox']:
            x1, y1, x2, y2 = parsed_result['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            parsed_result['center'] = [cx, cy]
            # print(f"Calculated center from bbox: {parsed_result['center']}")
        
        # 扩展bbox坐标（左上角扩展20%，右下角扩展10%）
        expanded_bbox = None
        if parsed_result['bbox']:
            print("\n--- Expanding bbox coordinates ---")
            x1, y1, x2, y2 = parsed_result['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # 左上角向左上方扩展20%
            expand_x1 = int(width * 0.2)
            expand_y1 = int(height * 0.2)
            # 右下角向右下方扩展10%
            expand_x2 = int(width * 0.1)
            expand_y2 = int(height * 0.1)
            
            # 扩展后的坐标
            expanded_x1 = max(0, x1 - expand_x1)
            expanded_y1 = max(0, y1 - expand_y1)
            expanded_x2 = min(img_width, x2 + expand_x2)
            expanded_y2 = min(img_height, y2 + expand_y2)
            
            expanded_bbox = [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
            # print(f"Original bbox: [{x1}, {y1}, {x2}, {y2}]")
            # print(f"Expanded bbox: [{expanded_x1}, {expanded_y1}, {expanded_x2}, {expanded_y2}]")
        
        # 保存带标注的图像（使用扩展后的bbox，如果存在）
        output_image_path = item_output_dir / "annotated.png"
        bbox_for_annotation = expanded_bbox if expanded_bbox else parsed_result['bbox']
        # 计算扩展后的中心点
        if expanded_bbox:
            ex1, ey1, ex2, ey2 = expanded_bbox
            expanded_center = [(ex1 + ex2) // 2, (ey1 + ey2) // 2]
        else:
            expanded_center = parsed_result['center']
        draw_bbox_on_image(image_path, bbox_for_annotation, expanded_center, str(output_image_path))
        
        # 使用两种方法进行分割
        # 使用扩展后的bbox进行分割（如果存在），否则使用原始bbox
        bbox_for_segmentation = expanded_bbox if expanded_bbox else parsed_result['bbox']
        
        # 方法1: 基于CLIP Surgery的分割方法
        mask_clip_surgery = None
        points = []
        labels = []
        if bbox_for_segmentation and parsed_result['prompt']:
            print("\n--- Running SAM3 segmentation with CLIP Surgery ---")
            # 设置关键点可视化输出路径
            keypoints_viz_path = item_output_dir / "keypoints.png"
            mask_clip_surgery, points, labels = segment_with_sam3(
                image_path, 
                bbox_for_segmentation, 
                parsed_result['prompt'],
                keypoints_viz_path=str(keypoints_viz_path)
            )
        
        # 方法2: 基于SAM3 prompt的分割方法
        mask_sam3_prompt = None
        if bbox_for_segmentation and parsed_result['prompt']:
            print("\n--- Running SAM3 segmentation with text prompt ---")
            mask_sam3_prompt = segment_with_sam3_prompt(
                image_path,
                bbox_for_segmentation,
                parsed_result['prompt']
            )
        
        # 保存合并的分割结果图像
        if mask_clip_surgery is not None or mask_sam3_prompt is not None:
            combined_segmentation_path = item_output_dir / "segmented_combined.png"
            save_combined_segmentation_result(
                image_path, 
                mask_clip_surgery, 
                mask_sam3_prompt, 
                str(combined_segmentation_path)
            )
        else:
            print("Warning: No segmentation masks generated")
        
        # 可选：保存单独的分割结果（用于对比）
        if mask_clip_surgery is not None:
            segmentation_output_path = item_output_dir / "segmented_clip_surgery.png"
            save_segmentation_result(image_path, mask_clip_surgery, str(segmentation_output_path))
            # print(f"Saved CLIP Surgery segmentation result to: {segmentation_output_path}")
        
        if mask_sam3_prompt is not None:
            segmentation_output_path = item_output_dir / "segmented_sam3_prompt.png"
            save_segmentation_result(image_path, mask_sam3_prompt, str(segmentation_output_path))
            # print(f"Saved SAM3 prompt segmentation result to: {segmentation_output_path}")
        
        # 使用原始bbox（不扩展）进行分割并保存
        mask_not_bbox = None
        if parsed_result['bbox'] and parsed_result['prompt']:
            print("\n--- Running segmentation with original bbox (not expanded) ---")
            # 使用CLIP Surgery方法进行分割（使用原始bbox）
            mask_not_bbox, _, _ = segment_with_sam3(
                image_path, 
                parsed_result['bbox'],  # 使用原始bbox，不扩展
                parsed_result['prompt'],
                keypoints_viz_path=None  # 不保存关键点可视化
            )
            
            if mask_not_bbox is not None:
                segmentation_output_path = item_output_dir / "segmented_not_bbox.png"
                save_segmentation_result(image_path, mask_not_bbox, str(segmentation_output_path))
                print(f"Saved not-expanded bbox segmentation result to: {segmentation_output_path}")
            else:
                print("Warning: Failed to generate segmentation with original bbox")
        
        # 使用CLIP Surgery的结果作为主要结果（用于JSON保存）
        mask = mask_clip_surgery
        
        # 保存结果到 JSON
        result_data = {
            'id': item_id,
            'difficulty': difficulty,
            'file_name': image_path,
            'question': question,
            'qwen_response': response,
            'bbox': parsed_result['bbox'],
            'expanded_bbox': expanded_bbox,
            'center': parsed_result['center'],
            'sam3_prompt': parsed_result['prompt'],
            'segmentation_clip_surgery_success': mask_clip_surgery is not None,
            'segmentation_sam3_prompt_success': mask_sam3_prompt is not None,
            'keypoints_count': len(points),
            'keypoints': points if points else None,
            'keypoint_labels': labels if labels else None
        }
        
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        # print(f"Saved result JSON to: {result_json_path}")

        # Finish timing for this item if enabled
        if timing_this_item and item_start_time is not None:
            item_end_time = time.time()
            elapsed = item_end_time - item_start_time
            _timing_item_times.append(elapsed)
            _total_items_timed += 1
            print(
                f"[Timing] Item {item_id} processing time: {elapsed:.4f} seconds "
                f"({_total_items_timed}/{MAX_TIMING_ITEMS})"
            )
        
        print(f"\n{'='*60}")

def main():
    """主函数：处理所有注释文件"""
    global API_BASE_URL, API_ENDPOINT
    
    parser = argparse.ArgumentParser(description='Process annotation files with CLIP Surgery and SAM3')
    parser.add_argument(
        '--annotations_dir',
        type=str,
        default="/data/jianglifan/sam_new/RS_ReasonSeg_Benchmark/annotations",
        help='Path to the annotations directory containing JSON files (default: /data/jianglifan/sam_new/RS_ReasonSeg_Benchmark/annotations)'
    )
    parser.add_argument(
        '--api_base_url',
        type=str,
        default="http://localhost:8010",
        help='Base URL of the API server (default: http://localhost:8010)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save results. If specified and already exists, processing will resume by skipping finished items.'
    )
    
    args = parser.parse_args()
    
    # 设置 API 服务器地址
    API_BASE_URL = args.api_base_url
    API_ENDPOINT = f"{API_BASE_URL}/generate"
    print(f"Using API server: {API_BASE_URL}")
    
    annotations_dir = Path(args.annotations_dir)
    
    # 检查目录是否存在
    if not annotations_dir.exists():
        print(f"Error: Annotations directory does not exist: {annotations_dir}")
        return
    
    if not annotations_dir.is_dir():
        print(f"Error: Path is not a directory: {annotations_dir}")
        return
    
    print(f"Using annotations directory: {annotations_dir}")
    
    # 获取所有 JSON 文件
    json_files = list(annotations_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {annotations_dir}")
        return
    
    print(f"Found {len(json_files)} annotation file(s)")

    # 预先初始化重型模型（CLIP Surgery / SAM3），避免它们的一次性加载时间被计入前几个样本的计时
    # 这些初始化属于环境准备，不算到核心 pipeline 时间里
    try:
        print("Initializing CLIP/SAM3 models (not counted in timing)...")
        # CLIP Surgery 模型
        get_clip_model()
        # SAM3 点提示与文本提示模型
        get_sam3_model()
        get_sam3_predictor()
        print("Model initialization finished.")
    except Exception as e:
        print(f"Warning: Failed to pre-initialize models: {e}")
    
    # 决定输出目录：
    # - 如果用户指定了 --output_dir，则使用该目录（可用于断点续传）
    # - 否则为本次运行创建一个新的带时间戳的目录
    if args.output_dir is not None:
        base_output_dir = Path(args.output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = annotations_dir.parent / f"qwen_results_{timestamp}"
        base_output_dir.mkdir(exist_ok=True)
    print(f"Base output directory: {base_output_dir}")
    print(f"All results will be saved to: {base_output_dir}")
    
    for json_file in json_files:
        process_annotation_file(json_file, base_output_dir)
    
    # Report average processing time over the measured items (up to MAX_TIMING_ITEMS)
    if _timing_item_times:
        avg_time = sum(_timing_item_times) / len(_timing_item_times)
        print(
            f"[Timing] Average processing time over {len(_timing_item_times)} items: "
            f"{avg_time:.4f} seconds"
        )
    
    print("\n" + "="*60)
    print("All processing completed!")
    print(f"Results saved to: {base_output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

