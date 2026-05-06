"""
使用 CLIP Surgery 模型从高分辨率图像中提取关键点
然后将关键点和文本 prompt 一起送入 SAM3 进行分割
"""
import clip_surgery as clip
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from ultralytics import SAM

BICUBIC = InterpolationMode.BICUBIC

# ==================== 初始化 ====================
# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载 CLIP Surgery 模型
print("加载 CLIP Surgery 模型...")
model, preprocess = clip.load("CS-ViT-B/16", device=device)
model.eval()

# 高分辨率预处理 (512x512)
preprocess = Compose([
    Resize((512, 512), interpolation=BICUBIC), 
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), 
              (0.26862954, 0.26130258, 0.27577711))
])

# ==================== 配置参数 ====================
# 图像路径
image_path = "/data/jianglifan/sam_new/CLIP_Surgery/demo.jpg"  # 修改为你的图像路径

# 查询文本
query_text = 'person'  # 修改为你想要查询的文本

# SAM3 模型路径
sam3_model_path = "/data/jianglifan/sam_new/checkpoints/sam3.pt"  # 修改为你的 SAM3 模型路径

# 关键点提取阈值
threshold = 0.8

# ==================== 加载图像 ====================
print(f"加载图像: {image_path}")
pil_img = Image.open(image_path)
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)
print(f"图像尺寸: {pil_img.size}")

# ==================== 初始化 SAM3 ====================
print("加载 SAM3 模型...")
sam3_model = SAM(sam3_model_path)

# ==================== 设置查询文本 ====================
texts = [query_text]
print(f"查询文本: {query_text}")

# ==================== 提取关键点 ====================
with torch.no_grad():
    # 1. 提取图像特征
    print("提取图像特征...")
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 2. 提取文本特征（使用提示词集成）
    print("提取文本特征...")
    text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)
    
    # 3. 提取冗余特征（从空字符串）
    print("提取冗余特征...")
    redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)
    
    # 4. 应用 CLIP 特征手术
    print("应用 CLIP 特征手术...")
    similarity = clip.clip_feature_surgery(
        image_features, 
        text_features, 
        redundant_features
    )[0]  # 取第一个批次
    
    # 5. 从相似度图提取关键点
    print("从相似度图提取关键点...")
    points, labels = clip.similarity_map_to_points(
        similarity[1:, 0],  # 跳过 CLS token，取第一个文本
        cv2_img.shape[:2],  # 原始图像尺寸
        t=threshold  # 阈值，用于选择高置信度区域
    )
    
    # ==================== 打印关键点信息 ====================
    print("\n" + "="*50)
    print(f"文本查询: '{texts[0]}'")
    print(f"提取到的关键点数量: {len(points)}")
    print("="*50)
    
    # 统计正负样本点
    positive_points = [p for i, p in enumerate(points) if labels[i] == 1]
    negative_points = [p for i, p in enumerate(points) if labels[i] == 0]
    
    print(f"\n正样本点数量: {len(positive_points)}")
    print(f"负样本点数量: {len(negative_points)}")
    
    print("\n--- 正样本点 (前景点) ---")
    for i, (x, y) in enumerate(positive_points):
        print(f"  点 {i+1}: ({x}, {y})")
    
    print("\n--- 负样本点 (背景点) ---")
    for i, (x, y) in enumerate(negative_points):
        print(f"  点 {i+1}: ({x}, {y})")
    
    print("\n--- 所有关键点详细信息 ---")
    for i, (point, label) in enumerate(zip(points, labels)):
        point_type = "正样本(前景)" if label == 1 else "负样本(背景)"
        print(f"  点 {i+1}: 坐标=({point[0]}, {point[1]}), 类型={point_type}")
    
    print("\n" + "="*50)
    print("关键点提取完成！")
    print("="*50)

# ==================== 准备 SAM3 输入 ====================
print("\n" + "="*50)
print("准备 SAM3 输入...")

# 将点转换为 SAM3 需要的格式
# SAM3 需要的格式: 
# - 单点: points=[x, y], labels=[1]
# - 多点: points=[[x1, y1], [x2, y2], ...], labels=[1, 0, 1, ...]

if len(points) == 1:
    # 单点格式
    sam_points = [int(points[0][0]), int(points[0][1])]
    sam_labels = [int(labels[0])]
    print(f"使用单点提示: {sam_points}, 标签: {sam_labels}")
else:
    # 多点格式: [[x1, y1], [x2, y2], ...]
    sam_points = [[int(p[0]), int(p[1])] for p in points]
    sam_labels = [int(l) for l in labels]
    print(f"使用多点提示: {len(sam_points)} 个点")
    print(f"  正样本点: {sum(sam_labels)} 个")
    print(f"  负样本点: {len(sam_labels) - sum(sam_labels)} 个")

# ==================== 使用 SAM3 进行分割 ====================
print("\n" + "="*50)
print("使用 SAM3 进行分割...")
print(f"输入: 图像={image_path}")
print(f"  点提示: {len(sam_points) if isinstance(sam_points[0], list) else 1} 个点")
print(f"  文本提示: '{query_text}'")

# 尝试使用点和文本提示进行分割
results = None
method_used = None


# 方式2: 仅使用点提示
try:
    results = sam3_model.predict(
        source=image_path,
        points=sam_points,
        labels=sam_labels
    )
    method_used = "点提示"
    print(f"✓ 使用 {method_used} 进行分割成功")
    print(f"  注意: 文本提示 '{query_text}' 未使用，仅使用点提示")
except Exception as e2:
    print(f"  尝试点提示失败: {str(e2)[:150]}")
    raise ValueError("SAM3 分割失败，请检查模型和输入参数")

print(f"使用的分割方法: {method_used}")

# ==================== 处理分割结果 ====================
print("\n处理分割结果...")
result = results[0]

# 获取掩膜
masks = getattr(result, "masks", None)
if masks is None or masks.data is None:
    print("警告: 未检测到掩膜")
    # 尝试显示结果
    if hasattr(result, "show"):
        result.show()
    print("分割完成，但无法获取掩膜数据")
else:
    # 获取掩膜数据
    mask_data = masks.data
    try:
        mask_np = mask_data.cpu().numpy()
    except Exception:
        mask_np = np.array(mask_data)
    
    # 合并所有掩膜（如果有多个，合并它们）
    if mask_np.ndim == 3:
        # 如果有多个掩膜，合并它们
        combined_mask = (mask_np.max(axis=0) > 0).astype(np.uint8)
    else:
        combined_mask = (mask_np > 0).astype(np.uint8)
    
    # 调整掩膜尺寸以匹配原图
    h, w = cv2_img.shape[:2]
    mh, mw = combined_mask.shape[:2]
    if (mh, mw) != (h, w):
        combined_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # ==================== 可视化结果 ====================
    print("生成可视化结果...")
    
    # 创建可视化图像：在原图上叠加掩膜
    overlay = cv2_img.copy()
    color = np.array([0, 255, 255], dtype=np.uint8)  # BGR格式的青色
    overlay[combined_mask == 1] = color
    
    # 半透明叠加
    alpha = 0.5
    vis_result = cv2.addWeighted(cv2_img, 1 - alpha, overlay, alpha, 0)
    
    # 在可视化图像上绘制关键点
    for i, (point, label) in enumerate(zip(points, labels)):
        x, y = int(point[0]), int(point[1])
        # 正样本点用蓝色，负样本点用橙色
        color_point = (255, 102, 0) if label == 1 else (0, 102, 255)  # BGR格式
        cv2.circle(vis_result, (x, y), 5, color_point, -1)  # 实心圆
        cv2.circle(vis_result, (x, y), 8, color_point, 2)  # 外圈
    
    # ==================== 保存结果 ====================
    output_path = f"segmentation_result_{query_text}.png"
    cv2.imwrite(output_path, vis_result)
    print(f"\n分割结果已保存到: {output_path}")
    
    # 保存掩膜
    mask_output_path = f"mask_{query_text}.png"
    cv2.imwrite(mask_output_path, combined_mask * 255)
    print(f"掩膜已保存到: {mask_output_path}")

print("\n" + "="*50)
print("处理完成！")
print("="*50)
