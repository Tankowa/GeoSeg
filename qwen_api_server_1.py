"""
Qwen3-VL API 服务器
部署 Qwen 模型并提供 API 接口
"""
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import logging
import numpy as np
from PIL import Image
import json
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型和处理器
model = None
processor = None
model_path = "/data/jianglifan/checkpoints/Qwen/Qwen3-VL-32B-Instruct"

app = FastAPI(title="Qwen3-VL API Server", version="1.0.0")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageQuestionRequest(BaseModel):
    """图像和问题的请求模型"""
    image_path: str
    question: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    max_new_tokens: Optional[int] = 1024

class QwenResponse(BaseModel):
    """Qwen 响应模型"""
    response: str
    status: str

class EvaluationRequest(BaseModel):
    """评估请求模型"""
    image_path: str
    gt_mask_path: str
    pred_mask_path: str
    class_name: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    max_new_tokens: Optional[int] = 2048

class EvaluationResponse(BaseModel):
    """评估响应模型"""
    faithfulness: float  # 语义准确性 (1-5)
    localization: float  # 边界贴合度 (1-5)
    robustness: float    # 环境鲁棒性 (1-5)
    overlap: float       # 地理重合度/IoU (1-5)
    status: str

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, processor
    logger.info("Loading Qwen 32B model and processor...")
    logger.info(f"Model path: {model_path}")
    try:
        # default: Load the model on the available device(s)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Model loaded successfully!")
        # 打印模型设备分配信息
        if hasattr(model, 'hf_device_map'):
            logger.info(f"Model device map: {model.hf_device_map}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "message": "Qwen3-VL API Server",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/generate", response_model=QwenResponse)
async def generate_response(request: ImageQuestionRequest):
    """
    生成响应端点
    接收图像路径和问题，返回 Qwen 的响应
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 读取图像尺寸（如果未提供）
        from PIL import Image
        img = Image.open(request.image_path)
        img_width, img_height = img.size
        
        # 如果请求中提供了尺寸，使用请求中的尺寸
        if request.image_width:
            img_width = request.image_width
        if request.image_height:
            img_height = request.image_height
        
        # 构建 prompt
        prompt_text = f"""Carefully analyze this image and answer the question. You must respond with ONLY a valid JSON object.

        ### STRATEGIC GUIDELINES:
        1. **Global Coordinate Alignment**: Before defining the box, mentally calibrate the image origin [0,0] at the top-left and the maximum dimensions [{img_width}, {img_height}] at the bottom-right. Ensure the horizontal (x) and vertical (y) coordinates are not shifted.
        2. **Total Entity Inclusion**: The bounding box must encapsulate the **entire physical structure** of the object. Do not truncate edges, limbs, or extensions (e.g., if a bridge, include the start and end ramps; if a plane, include the wingtips).

        ### EXECUTION STEPS:
        1. Scan the full image to understand the context.
        2. Identify the target based on the question: "{request.question}"
        3. Perform a "Boundary Check": Zoom into the target's edges to ensure the [x1, y1, x2, y2] values truly touch the outermost pixels of the entity.

        Image dimensions: {img_width} x {img_height} pixels

        ### JSON RESPONSE FORMAT:
        {{
            "bbox": [x1, y1, x2, y2],
            "prompt": "entity_name"
        }}

        Where:
        - "bbox": [x1, y1, x2, y2] where x1,y1 is Top-Left and x2,y2 is Bottom-Right. 
        **CRITICAL**: Avoid coordinate drift. Double-check that the x-coordinates are not shifted to the right. Ensure the box is wide and tall enough to cover the WHOLE object.
        - "prompt": A simple English noun/phrase for SAM3 (e.g., "bridge", "aircraft", "stadium").

        Respond with ONLY the JSON object."""
        
        # 准备消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": request.image_path,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 处理输入
        logger.info(f"Processing request for image: {request.image_path}")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到正确的设备（对于多卡模型，输入应该放在第一个设备上）
        # 获取模型的主要设备
        if hasattr(model, 'device'):
            main_device = model.device
        elif hasattr(model, 'hf_device_map') and model.hf_device_map:
            # 获取第一个设备的键（通常是 "model" 或 "model.embed_tokens"）
            first_key = list(model.hf_device_map.keys())[0]
            main_device = torch.device(model.hf_device_map[first_key])
        else:
            # 默认使用 cuda:0（当设置了 CUDA_VISIBLE_DEVICES=6,7 时，cuda:0 对应物理 GPU 6）
            main_device = torch.device("cuda:0")
        
        inputs = {k: v.to(main_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 生成响应
        generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0] if output_text else ""
        
        return QwenResponse(
            response=response,
            status="success"
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

def load_mask(path: str) -> np.ndarray:
    """加载掩码图像"""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return arr > 0  # binary mask

def compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """计算 IoU (Overlap)"""
    gt = gt.astype(bool)
    if pred.shape != gt.shape:
        h, w = gt.shape
        pred_img = Image.fromarray(pred.astype(np.uint8) * 255)
        pred_resized = pred_img.resize((w, h), resample=Image.NEAREST)
        pred = np.array(pred_resized) > 0
    pred = pred.astype(bool)
    
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(intersection / union) if union > 0 else 0.0

def parse_score_from_response(response_text: str, metric_name: str) -> float:
    """从响应文本中解析评分（1-5 整数评分）"""
    # 尝试多种格式提取评分
    patterns = [
        rf'"{metric_name}"\s*:\s*([1-5])',
        rf'"{metric_name.lower()}"\s*:\s*([1-5])',
        rf'{metric_name}\s*[:=]\s*([1-5])',
        rf'{metric_name.lower()}\s*[:=]\s*([1-5])',
        # 也支持浮点数格式（如 4.5），但会四舍五入到最近的整数
        rf'"{metric_name}"\s*:\s*([0-9.]+)',
        rf'"{metric_name.lower()}"\s*:\s*([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # 确保分数在 1-5 范围内
                if score < 1.0:
                    score = 1.0
                elif score > 5.0:
                    score = 5.0
                # 四舍五入到最近的整数
                score = round(score)
                return float(max(1, min(5, score)))
            except ValueError:
                continue
    
    # 如果无法解析，返回默认值 3（中等评分）
    logger.warning(f"Could not parse {metric_name} from response, using default 3")
    return 3.0

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_segmentation(request: EvaluationRequest):
    """
    评估分割结果端点
    从四个方面评估：Faithfulness, Localization, Robustness, Overlap
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 读取图像和掩码
        img = Image.open(request.image_path)
        img_width, img_height = img.size
        
        if request.image_width:
            img_width = request.image_width
        if request.image_height:
            img_height = request.image_height
        
        gt_mask = load_mask(request.gt_mask_path)
        pred_mask = load_mask(request.pred_mask_path)
        
        # 确保掩码尺寸与图像尺寸匹配
        img_array = np.array(img.convert("RGB"))
        img_h, img_w = img_array.shape[:2]
        
        # 调整 GT 掩码尺寸
        if gt_mask.shape != (img_h, img_w):
            gt_mask_img = Image.fromarray(gt_mask.astype(np.uint8) * 255)
            gt_mask_resized = gt_mask_img.resize((img_w, img_h), resample=Image.NEAREST)
            gt_mask = np.array(gt_mask_resized) > 0
        
        # 调整预测掩码尺寸
        if pred_mask.shape != (img_h, img_w):
            pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
            pred_mask_resized = pred_mask_img.resize((img_w, img_h), resample=Image.NEAREST)
            pred_mask = np.array(pred_mask_resized) > 0
        
        # 计算 Overlap (IoU) - 这是可以直接计算的（使用调整后的掩码）
        overlap_iou = compute_iou(gt_mask, pred_mask)
        # 将 IoU (0-1) 转换为 1-5 评分
        # IoU 0.0-0.2 -> 1分, 0.2-0.4 -> 2分, 0.4-0.6 -> 3分, 0.6-0.8 -> 4分, 0.8-1.0 -> 5分
        if overlap_iou < 0.2:
            overlap_score = 1.0
        elif overlap_iou < 0.4:
            overlap_score = 2.0
        elif overlap_iou < 0.6:
            overlap_score = 3.0
        elif overlap_iou < 0.8:
            overlap_score = 4.0
        else:
            overlap_score = 5.0
        
        # 准备掩码可视化（用于 MLLM 评估）
        # 创建叠加图像：原图 + GT掩码（绿色） + 预测掩码（红色）
        gt_vis = np.zeros_like(img_array)
        pred_vis = np.zeros_like(img_array)
        
        # GT 掩码用绿色显示
        gt_vis[gt_mask] = [0, 255, 0]
        # 预测掩码用红色显示
        pred_vis[pred_mask] = [255, 0, 0]
        
        # 叠加：绿色=GT，红色=预测，黄色=重叠
        overlay = img_array.copy()
        overlay = np.where(gt_vis > 0, gt_vis * 0.5 + overlay * 0.5, overlay)
        overlay = np.where(pred_vis > 0, pred_vis * 0.5 + overlay * 0.5, overlay)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # 保存临时可视化图像
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        vis_path = os.path.join(temp_dir, f"eval_vis_{os.getpid()}.png")
        Image.fromarray(overlay).save(vis_path)
        
        # 构建评估 prompt
        prompt_text = f"""You are an expert in remote sensing image analysis. Evaluate the segmentation quality of a {request.class_name} in this remote sensing image.

The image shows:
- Original image (background)
- Green overlay: Ground truth mask (correct segmentation)
- Red overlay: Predicted mask (model's segmentation)
- Yellow areas: Overlapping regions (where both masks agree)

Image dimensions: {img_width} x {img_height} pixels
Class: {request.class_name}

Evaluate the segmentation from FOUR aspects and provide integer scores from 1 to 5 for each metric. Use the following scoring standards:

## Scoring Standards (1-5 scale):

**1. Faithfulness (语义准确性)**: Does the predicted mask correctly identify the {request.class_name}?
- **Score 5 (Excellent)**: Perfectly correct identification, no confusion with other classes
- **Score 4 (Good)**: Mostly correct, minor confusion with similar classes
- **Score 3 (Fair)**: Generally correct but some confusion with related classes
- **Score 2 (Poor)**: Significant confusion, partially wrong class identification
- **Score 1 (Very Poor)**: Completely wrong class, major misidentification

**2. Localization (边界贴合度)**: Does the predicted mask precisely follow the complex edges?
- **Score 5 (Excellent)**: Boundaries perfectly match ground truth, no rounded corners or overflow
- **Score 4 (Good)**: Boundaries mostly accurate, minor deviations at complex edges
- **Score 3 (Fair)**: Generally follows boundaries but noticeable deviations or slight overflow
- **Score 2 (Poor)**: Significant boundary misalignment, obvious rounded corners or overflow
- **Score 1 (Very Poor)**: Severe boundary errors, completely misaligned edges

**3. Robustness (环境鲁棒性)**: Can the segmentation resist interference from clouds, shadows, seasonal changes, or similar textures?
- **Score 5 (Excellent)**: Highly robust, unaffected by environmental variations or similar textures
- **Score 4 (Good)**: Mostly robust, minor sensitivity to environmental factors
- **Score 3 (Fair)**: Moderate robustness, some sensitivity to clouds/shadows/similar textures
- **Score 2 (Poor)**: Low robustness, easily confused by environmental variations
- **Score 1 (Very Poor)**: Very fragile, severely affected by clouds, shadows, or similar textures

**4. Overlap (地理重合度)**: Pixel-level IoU (Intersection over Union) between predicted and ground truth masks.
- **Score 5 (Excellent)**: IoU ≥ 0.8, excellent pixel-level overlap
- **Score 4 (Good)**: IoU 0.6-0.8, good overlap with minor differences
- **Score 3 (Fair)**: IoU 0.4-0.6, moderate overlap, noticeable differences
- **Score 2 (Poor)**: IoU 0.2-0.4, poor overlap, significant differences
- **Score 1 (Very Poor)**: IoU < 0.2, very poor overlap, minimal agreement

Note: The actual calculated IoU is {overlap_iou:.4f}, which corresponds to a score of {int(overlap_score)} for reference.

Respond with ONLY a valid JSON object in this format (use integer scores 1-5):
{{
    "faithfulness": <integer 1-5>,
    "localization": <integer 1-5>,
    "robustness": <integer 1-5>,
    "overlap": <integer 1-5>
}}"""
        
        # 准备消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": vis_path,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 处理输入
        logger.info(f"Evaluating segmentation for class: {request.class_name}")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到正确的设备
        if hasattr(model, 'device'):
            main_device = model.device
        elif hasattr(model, 'hf_device_map') and model.hf_device_map:
            first_key = list(model.hf_device_map.keys())[0]
            main_device = torch.device(model.hf_device_map[first_key])
        else:
            main_device = torch.device("cuda:0")
        
        inputs = {k: v.to(main_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 生成响应
        generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response_text = output_text[0] if output_text else ""
        
        # 清理临时文件
        try:
            if os.path.exists(vis_path):
                os.remove(vis_path)
        except:
            pass
        
        # 解析评分（1-5 整数评分）
        faithfulness = parse_score_from_response(response_text, "faithfulness")
        localization = parse_score_from_response(response_text, "localization")
        robustness = parse_score_from_response(response_text, "robustness")
        # Overlap 优先使用从 IoU 转换的评分，也可以从响应中解析作为参考
        overlap_from_response = parse_score_from_response(response_text, "overlap")
        # 如果 MLLM 给出的 overlap 评分与基于 IoU 的评分差异超过 1 分，记录警告
        if abs(overlap_from_response - overlap_score) > 1.0:
            logger.warning(f"Overlap score mismatch: IoU-based={int(overlap_score)}, MLLM={int(overlap_from_response)}, IoU={overlap_iou:.4f}")
        
        return EvaluationResponse(
            faithfulness=faithfulness,
            localization=localization,
            robustness=robustness,
            overlap=overlap_score,  # 使用基于 IoU 转换的评分
            status="success"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    # 运行服务器
    # 默认运行在 0.0.0.0:8000
    uvicorn.run(
        "qwen_api_server:app",
        host="0.0.0.0",
        port=8011,
        log_level="info"
    )

