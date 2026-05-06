"""
Qwen3-VL API 服务器
部署 Qwen 模型并提供 API 接口
"""
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import logging

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

if __name__ == "__main__":
    # 运行服务器
    # 默认运行在 0.0.0.0:8000
    uvicorn.run(
        "qwen_api_server:app",
        host="0.0.0.0",
        port=8011,
        log_level="info"
    )

