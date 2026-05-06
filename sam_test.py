import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="checkpoints/sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=False,  # 我们自己保存结果
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# 设置图片路径
# image_path = "/data/raid5/jlf2/sam-new/LoveDA/Test/Test/Rural/images_png/4230.png"

image_path = "/data/jianglifan/sam_new/test_sam/images/Urban/Level1/5206.png"

# 读取原始图片
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"无法读取图片: {image_path}")

# Set image for segmentation
predictor.set_image(image_path)

# 直接使用文本提示指向海拔相对较高的楼盘
results = predictor(text=["higher buildings"])

# 获取结果
result = results[0]
masks = getattr(result, "masks", None)

if masks is None or masks.data is None:
    print("未检测到高层建筑")
    exit()

# 获取掩膜数据
mask_data = masks.data
try:
    mask_np = mask_data.cpu().numpy()
except Exception:
    mask_np = np.array(mask_data)

# 合并所有掩膜（如果有多个，合并它们）
if mask_np.ndim == 3:
    # 如果有多个掩膜，合并它们
    building_mask = (mask_np.max(axis=0) > 0).astype(np.uint8)
else:
    building_mask = (mask_np > 0).astype(np.uint8)

# 调整掩膜尺寸以匹配原图
h, w = image.shape[:2]
mh, mw = building_mask.shape[:2]
if (mh, mw) != (h, w):
    building_mask = cv2.resize(building_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# 生成可视化结果：用红色半透明覆盖高层建筑区域
overlay = image.copy()
color = np.array([0, 0, 255], dtype=np.uint8)  # BGR格式的红色
overlay[building_mask == 1] = color

# 半透明叠加
alpha = 0.5
vis_result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# 保存结果
output_path = "high_building_segmentation.png"
cv2.imwrite(output_path, vis_result)
print(f"分割结果已保存到: {output_path}")