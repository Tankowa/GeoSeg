import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
import requests
from typing import Optional
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import tempfile

# API 服务器配置
API_BASE_URL = "http://localhost:8010"  # 修改为你的 API 服务器地址
API_ENDPOINT = f"{API_BASE_URL}/generate"

# 初始化 SAM3 predictor（全局变量，避免重复初始化）
_sam3_predictor = None

def get_sam3_predictor():
    """获取或初始化 SAM3 predictor"""
    global _sam3_predictor
    if _sam3_predictor is None:
        print("Initializing SAM3 predictor...")
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model="checkpoints/sam3.pt",
            half=True,  # Use FP16 for faster inference
            save=False,  # 我们自己保存结果
        )
        _sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3 predictor initialized.")
    return _sam3_predictor

def parse_qwen_response(response_text):
    """解析 Qwen 的响应，提取边界框、中心点和 prompt"""
    result = {
        'bbox': None,
        'center': None,
        'prompt': None
    }
    
    # 首先尝试找到完整的 JSON 对象（可能包含嵌套结构）
    # 尝试多种 JSON 匹配模式
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"bbox"[^{}]*(?:\{[^{}]*\}[^{}]*)*"center"[^{}]*(?:\{[^{}]*\}[^{}]*)*"prompt"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\{[^}]*"bbox"[^}]*"center"[^}]*"prompt"[^}]*\}',
        r'\{.*?"bbox".*?"center".*?"prompt".*?\}',
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
                if all(result.values()):
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
    """在图像上绘制边界框和中心点"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        # 绘制边界框（红色）
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        # 标注坐标
        draw.text((x1, y1 - 20), f"({x1}, {y1})", fill='red')
        draw.text((x2, y2 + 5), f"({x2}, {y2})", fill='red')
    
    if center:
        cx, cy = center
        # 绘制中心点（蓝色）
        radius = 5
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill='blue', outline='blue')
        draw.text((cx + 10, cy - 10), f"Center: ({cx}, {cy})", fill='blue')
    
    img.save(output_path)
    print(f"Saved annotated image to: {output_path}")

def segment_with_sam3(image_path, bbox, sam3_prompt):
    """
    使用 SAM3 对 bbox 区域进行分割
    
    Args:
        image_path: 原始图像路径
        bbox: 边界框 [x1, y1, x2, y2]
        sam3_prompt: SAM3 文本提示
    
    Returns:
        mask: 分割掩膜（numpy array，与原始图像尺寸相同）
    """
    if not bbox or not sam3_prompt:
        print("Warning: Missing bbox or sam3_prompt, skipping SAM3 segmentation")
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
        
        cropped_image = image[y1:y2, x1:x2]
        cropped_h, cropped_w = cropped_image.shape[:2]
        
        print(f"Cropped region: {x1}, {y1}, {x2}, {y2} (size: {cropped_w}x{cropped_h})")
        
        # 保存临时裁剪图像用于 SAM3（使用临时文件避免并发冲突）
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix='sam3_crop_')
        temp_crop_path = temp_file.name
        temp_file.close()
        cv2.imwrite(temp_crop_path, cropped_image)
        
        # 获取 SAM3 predictor
        predictor = get_sam3_predictor()
        
        # 设置图像
        predictor.set_image(temp_crop_path)
        
        # 使用文本提示进行分割
        print(f"Running SAM3 segmentation with prompt: '{sam3_prompt}'")
        results = predictor(text=[sam3_prompt])
        
        # 获取结果
        result = results[0]
        masks = getattr(result, "masks", None)
        
        if masks is None or masks.data is None:
            print("Warning: SAM3 did not detect any masks")
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
        
        print(f"SAM3 segmentation completed. Mask shape: {full_mask.shape}")
        return full_mask
        
    except Exception as e:
        print(f"Error during SAM3 segmentation: {e}")
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
    保存分割结果到图像
    
    Args:
        image_path: 原始图像路径
        mask: 分割掩膜（numpy array）
        output_path: 输出图像路径
        alpha: 叠加透明度（0-1）
    """
    if mask is None:
        print("Warning: No mask to save")
        return
    
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # 确保掩膜尺寸匹配
    h, w = image.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 生成可视化结果：用红色半透明覆盖分割区域
    overlay = image.copy()
    color = np.array([0, 0, 255], dtype=np.uint8)  # BGR格式的红色
    overlay[mask == 1] = color
    
    # 半透明叠加
    vis_result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # 保存结果
    cv2.imwrite(output_path, vis_result)
    print(f"Saved segmentation result to: {output_path}")

def process_annotation_file(json_path):
    """处理单个 JSON 注释文件"""
    print(f"\n{'='*60}")
    print(f"Processing: {json_path}")
    print(f"{'='*60}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录
    output_dir = Path(json_path).parent.parent / "qwen_results"
    output_dir.mkdir(exist_ok=True)
    
    for idx, item in enumerate(data):
        print(f"\n--- Processing item {idx + 1}/{len(data)} ---")
        print(f"ID: {item.get('id', 'N/A')}")
        print(f"Question: {item.get('question', 'N/A')}")
        print(f"Image: {item.get('file_name', 'N/A')}")
        
        image_path = item['file_name']
        question = item['question']
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # 读取图像尺寸
        img = Image.open(image_path)
        img_width, img_height = img.size
        print(f"Image size: {img_width} x {img_height}")
        
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
            
            print(f"\nAPI Status: {status}")
            print(f"Qwen Response:\n{response}")
            
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
        print(f"\nParsed Results:")
        print(f"  BBox: {parsed_result['bbox']}")
        print(f"  Center: {parsed_result['center']}")
        print(f"  Prompt: {parsed_result['prompt']}")
        
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
            print(f"Adjusted BBox: {parsed_result['bbox']}")
        
        # 如果没有中心点，从边界框计算
        if not parsed_result['center'] and parsed_result['bbox']:
            x1, y1, x2, y2 = parsed_result['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            parsed_result['center'] = [cx, cy]
            print(f"Calculated center from bbox: {parsed_result['center']}")
        
        # 保存带标注的图像
        base_name = Path(image_path).stem
        output_image_path = output_dir / f"{base_name}_annotated.png"
        draw_bbox_on_image(image_path, parsed_result['bbox'], parsed_result['center'], str(output_image_path))
        
        # 使用 SAM3 进行分割
        mask = None
        if parsed_result['bbox'] and parsed_result['prompt']:
            print("\n--- Running SAM3 segmentation ---")
            mask = segment_with_sam3(image_path, parsed_result['bbox'], parsed_result['prompt'])
        
        # 保存分割结果图像
        if mask is not None:
            segmentation_output_path = output_dir / f"{base_name}_segmented.png"
            save_segmentation_result(image_path, mask, str(segmentation_output_path))
        else:
            print("Warning: No segmentation mask generated, skipping segmentation visualization")
        
        # 保存结果到 JSON
        result_data = {
            'id': item.get('id'),
            'file_name': image_path,
            'question': question,
            'qwen_response': response,
            'bbox': parsed_result['bbox'],
            'center': parsed_result['center'],
            'sam3_prompt': parsed_result['prompt'],
            'segmentation_success': mask is not None
        }
        
        result_json_path = output_dir / f"{base_name}_result.json"
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"Saved result JSON to: {result_json_path}")
        
        print(f"\n{'='*60}")

def main():
    """主函数：处理所有注释文件"""
    annotations_dir = Path("/data/jianglifan/sam_new/test_sam/annotations")
    
    # 获取所有 JSON 文件
    json_files = list(annotations_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {annotations_dir}")
        return
    
    print(f"Found {len(json_files)} annotation file(s)")
    
    for json_file in json_files:
        process_annotation_file(json_file)
    
    print("\n" + "="*60)
    print("All processing completed!")
    print("="*60)

if __name__ == "__main__":
    main()

