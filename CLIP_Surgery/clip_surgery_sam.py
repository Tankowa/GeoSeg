import clip_surgery
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from segment_anything import sam_model_registry, SamPredictor
from datetime import datetime

BICUBIC = InterpolationMode.BICUBIC

# 初始化全局变量
_device = None
_clip_model = None
_clip_preprocess = None
_sam = None
_predictor = None

def init_models():
    """初始化 CLIP Surgery 和 SAM 模型"""
    global _device, _clip_model, _clip_preprocess, _sam, _predictor
    
    if _clip_model is None:
        print("Initializing CLIP Surgery model...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _ = clip_surgery.load("CS-ViT-B/16", device=_device)
        _clip_model.eval()
        
        # Preprocess for higher resolution (512x512)
        _clip_preprocess = Compose([
            Resize((512, 512), interpolation=BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        print("CLIP Surgery model initialized.")
    
    if _sam is None:
        print("Initializing SAM model...")
        sam_checkpoint = "/data/jianglifan/sam_new/checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        _sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        _sam.to(device=_device)
        _predictor = SamPredictor(_sam)
        print("SAM model initialized.")
    
    return _device, _clip_model, _clip_preprocess, _predictor

def segment_with_clip_surgery_sam(image_path, text_prompt, output_path, threshold=0.8):
    """
    使用 CLIP Surgery 提取关键点，然后用 SAM 进行分割
    
    Args:
        image_path: 图像路径
        text_prompt: 文本提示（从 question 中获取）
        output_path: 输出黑白二值图像路径（白色区域为分割结果）
        threshold: 关键点提取阈值
    
    Returns:
        success: 是否成功分割
    """
    try:
        device, clip_model, clip_preprocess, predictor = init_models()
        
        # 读取图像
        pil_img = Image.open(image_path).convert('RGB')
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image = clip_preprocess(pil_img).unsqueeze(0).to(device)
        
        # 设置 SAM predictor 的图像
        predictor.set_image(np.array(pil_img))
        
        # 使用 CLIP Surgery 提取关键点并分割
        with torch.no_grad():
            # CLIP architecture surgery acts on the image encoder
            image_features = clip_model.encode_image(image)  # Image resized to 512
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Prompt ensemble for text features with normalization
            # 使用 question 作为文本提示
            text_features = clip_surgery.encode_text_with_prompt_ensemble(
                clip_model, [text_prompt], device
            )
            
            # 提取冗余特征（从空字符串）
            redundant_features = clip_surgery.encode_text_with_prompt_ensemble(
                clip_model, [""], device
            )
            
            # Apply feature surgery
            similarity = clip_surgery.clip_feature_surgery(
                image_features, 
                text_features,
                redundant_features
            )[0]
            
            # 从相似度图提取关键点（只处理第一个文本，即我们的 question）
            points, labels = clip_surgery.similarity_map_to_points(
                similarity[1:, 0],  # 跳过 CLS token，取第一个文本
                cv2_img.shape[:2],
                t=threshold
            )
            
            if len(points) == 0:
                print(f"Warning: No keypoints extracted for prompt: '{text_prompt}'")
                return False
            
            # 确保 points 和 labels 是正确的格式
            # points 应该是 (N, 2) 形状的 numpy 数组，格式为 [[x1, y1], [x2, y2], ...]
            # labels 应该是 (N,) 形状的 numpy 数组，格式为 [1, 0, 1, ...]
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            # 验证和调整形状
            if points_array.ndim != 2 or points_array.shape[1] != 2:
                print(f"Warning: Unexpected points shape: {points_array.shape}, expected (N, 2)")
                return False
            
            if labels_array.ndim != 1:
                print(f"Warning: Unexpected labels shape: {labels_array.shape}, expected (N,)")
                # 尝试 flatten
                labels_array = labels_array.flatten()
            
            if len(points_array) != len(labels_array):
                print(f"Warning: Mismatch between points ({len(points_array)}) and labels ({len(labels_array)})")
                return False
            
            # 确保至少有一个点
            if len(points_array) == 0:
                print(f"Warning: No valid points after processing")
                return False
            
            # 确保 labels 是一维数组
            if labels_array.ndim != 1:
                labels_array = labels_array.flatten()
            
            # 打印调试信息
            print(f"Points shape: {points_array.shape}, Labels shape: {labels_array.shape}")
            print(f"Number of points: {len(points_array)}, Positive: {np.sum(labels_array == 1)}, Negative: {np.sum(labels_array == 0)}")
            
            # 使用 SAM 进行分割
            # SAM 期望 point_coords 是 (N, 2) 形状，point_labels 是 (N,) 形状
            masks, scores, logits = predictor.predict(
                point_labels=labels_array,
                point_coords=points_array,
                multimask_output=True
            )
            
            # 选择得分最高的掩膜
            mask = masks[np.argmax(scores)]
            mask = mask.astype('uint8')
            
            # 保存黑白二值图像（白色区域为分割结果）
            # 将 mask 转换为 0-255 范围，白色(255)为分割区域，黑色(0)为背景
            binary_mask = (mask * 255).astype(np.uint8)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存黑白二值图像
            cv2.imwrite(output_path, binary_mask)
            print(f"Saved segmentation mask to: {output_path}")
            
            return True
            
    except Exception as e:
        print(f"Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_annotation_file(json_path, base_output_dir):
    """处理单个 JSON 注释文件
    
    Args:
        json_path: JSON注释文件路径
        base_output_dir: 基础输出目录（已包含时间戳）
    """
    print(f"\n{'='*60}")
    print(f"Processing: {json_path}")
    print(f"{'='*60}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从JSON文件路径中提取类别名（例如：Nature.json -> Nature）
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
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # 进行分割
        output_mask_path = item_output_dir / "segmentation_mask.png"
        success = segment_with_clip_surgery_sam(
            image_path,
            question,  # 使用 question 作为文本提示
            str(output_mask_path),
            threshold=0.8
        )
        
        if success:
            print(f"Segmentation completed successfully for item {item_id}")
        else:
            print(f"Segmentation failed for item {item_id}")
        
        print(f"\n{'='*60}")

def main():
    """主函数：处理所有注释文件"""
    annotations_dir = Path("/data/jianglifan/sam_new/RS_ReasonSeg_Benchmark/annotations")
    
    # 获取所有 JSON 文件
    json_files = list(annotations_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {annotations_dir}")
        return
    
    print(f"Found {len(json_files)} annotation file(s)")
    
    # 一次运行只创建一个时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = annotations_dir.parent / f"clip_surgery_sam_results_{timestamp}"
    base_output_dir.mkdir(exist_ok=True)
    print(f"Base output directory: {base_output_dir}")
    print(f"All results will be saved to: {base_output_dir}")
    
    for json_file in json_files:
        process_annotation_file(json_file, base_output_dir)
    
    print("\n" + "="*60)
    print("All processing completed!")
    print(f"Results saved to: {base_output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
