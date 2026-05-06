# CLIP Surgery Demo 使用说明

本文档详细说明了 `demo.py` 文件中每个步骤的操作流程和功能。

## 目录
1. [环境初始化](#1-环境初始化)
2. [原始 CLIP 预测展示](#2-原始-clip-预测展示)
3. [CLIP Surgery 改进](#3-clip-surgery-改进)
4. [高分辨率 CLIP Surgery](#4-高分辨率-clip-surgery)
5. [单个文本的 CLIP Surgery](#5-单个文本的-clip-surgery)
6. [CLIP Surgery 指导 SAM 分割](#6-clip-surgery-指导-sam-分割)
7. [单个文本的 CLIP Surgery + SAM](#7-单个文本的-clip-surgery--sam)
8. [组合目标的 CLIP Surgery + SAM](#8-组合目标的-clip-surgery--sam)

---

## 1. 环境初始化

### 1.1 导入依赖库
```python
import clip, torch, cv2, numpy, PIL, matplotlib, torchvision
from segment_anything import sam_model_registry, SamPredictor
```
- **clip**: CLIP 模型库，用于图像-文本匹配
- **torch**: PyTorch 深度学习框架
- **cv2**: OpenCV，用于图像处理
- **PIL**: Python 图像处理库
- **segment_anything**: Meta 的 SAM（Segment Anything Model）分割模型

### 1.2 初始化 CLIP 模型
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/16", device=device)
model.eval()
```
- **操作**: 检测可用设备（优先使用 GPU），加载 CLIP 的 ViT-B/16 模型
- **输出**: 模型设置为评估模式

### 1.3 定义图像预处理
```python
preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC), 
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), 
              (0.26862954, 0.26130258, 0.27577711))
])
```
- **操作**: 定义图像预处理流程
  - 调整大小为 224×224
  - 转换为张量
  - 使用 CLIP 的标准归一化参数

### 1.4 加载和预处理图像
```python
pil_img = Image.open("demo.jpg")
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)
```
- **操作**: 
  - 使用 PIL 加载图像
  - 转换为 OpenCV 格式（BGR）
  - 预处理并添加批次维度，移至指定设备

### 1.5 定义文本标签
```python
all_texts = ['airplane', 'bag', 'bed', ...]  # 59 个类别
target_texts = ['bench', 'person', 'ground', 'building']
```
- **操作**: 定义所有可能的文本标签和目标标签
- **用途**: `all_texts` 用于完整预测，`target_texts` 用于可视化特定类别

---

## 2. 原始 CLIP 预测展示

### 2.1 提取图像特征
```python
image_features = model.encode_image(image)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```
- **操作**: 
  - 使用 CLIP 图像编码器提取特征
  - L2 归一化特征向量

### 2.2 提取文本特征（提示词集成）
```python
text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
```
- **操作**: 使用提示词集成方法提取文本特征
- **原理**: 通过多个提示词模板（如 "a photo of {text}"）增强文本表示

### 2.3 计算相似度图
```python
features = image_features @ text_features.t()
similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])
```
- **操作**: 
  - 计算图像特征与文本特征的相似度矩阵
  - `features[:, 1:, :]` 跳过 CLS token，只使用图像 patch tokens
  - 将相似度映射到原始图像尺寸

### 2.4 可视化相似度图
```python
for b in range(similarity_map.shape[0]):
    for n in range(similarity_map.shape[-1]):
        if all_texts[n] not in target_texts:
            continue
        vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis = cv2_img * 0.4 + vis * 0.6
        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(vis)
        plt.show()
```
- **操作**: 
  - 对每个目标类别生成热力图
  - 使用 JET 颜色映射
  - 将热力图与原图叠加（40% 原图 + 60% 热力图）
  - 显示可视化结果

**注意**: 原始 CLIP 的预测通常存在噪声和反向激活问题。

---

## 3. CLIP Surgery 改进

### 3.1 加载 CLIP Surgery 模型
```python
model, preprocess = clip.load("CS-ViT-B/16", device=device)
model.eval()
```
- **操作**: 加载经过架构手术改进的 CLIP 模型（CS-ViT-B/16）
- **改进**: 模型架构已修改，减少冗余特征

### 3.2 提取特征
```python
image_features = model.encode_image(image)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
```
- **操作**: 与步骤 2 相同，但使用改进后的模型

### 3.3 应用特征手术
```python
similarity = clip.clip_feature_surgery(image_features, text_features)
similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
```
- **操作**: 
  - 应用 CLIP Surgery 的特征手术操作
  - 移除冗余特征，提高定位精度
  - 生成改进的相似度图

### 3.4 可视化结果
- **操作**: 与步骤 2.4 相同，但结果更清晰、噪声更少

**改进效果**: 相比原始 CLIP，CLIP Surgery 能提供更准确、更清晰的定位结果。

---

## 4. 高分辨率 CLIP Surgery

### 4.1 更新预处理为高分辨率
```python
preprocess = Compose([
    Resize((512, 512), interpolation=BICUBIC), 
    ToTensor(),
    Normalize(...)
])
image = preprocess(pil_img).unsqueeze(0).to(device)
```
- **操作**: 将输入图像分辨率从 224×224 提升到 512×512
- **优势**: 更高分辨率能捕获更多细节，提高定位精度

### 4.2 执行高分辨率推理
```python
image_features = model.encode_image(image)  # 512×512 输入
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
similarity = clip.clip_feature_surgery(image_features, text_features)
similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
```
- **操作**: 与步骤 3 相同，但使用更高分辨率的输入
- **注意**: 此预处理设置将用于后续所有步骤

### 4.3 可视化结果
- **操作**: 显示高分辨率下的相似度图，通常比 224×224 更精确

---

## 5. 单个文本的 CLIP Surgery

### 5.1 定义单个文本
```python
texts = ['shoes']
```
- **操作**: 定义单个查询文本，不依赖固定标签集

### 5.2 提取冗余特征
```python
redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)
```
- **操作**: 从空字符串提取冗余特征
- **用途**: 用于移除与文本无关的通用特征

### 5.3 应用特征手术（带冗余特征）
```python
text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)
similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
```
- **操作**: 
  - 提取目标文本特征
  - 应用特征手术，同时移除冗余特征
  - 适用于单个文本查询，无需固定标签集

### 5.4 可视化结果
- **操作**: 显示单个文本的定位结果

**应用场景**: 当需要查询不在预定义标签集中的对象时。

---

## 6. CLIP Surgery 指导 SAM 分割

### 6.1 初始化 SAM 模型
```python
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(np.array(pil_img))
```
- **操作**: 
  - 加载 SAM 的 ViT-H 模型
  - 创建预测器并设置输入图像
  - SAM 用于基于点提示的分割

### 6.2 执行 CLIP Surgery 推理
```python
image_features = model.encode_image(image)  # 512×512
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
similarity = clip.clip_feature_surgery(image_features, text_features)[0]
```
- **操作**: 与步骤 3 相同，但取第一个批次 `[0]`

### 6.3 从相似度图提取点提示
```python
for n in range(similarity.shape[-1]):
    if all_texts[n] not in target_texts:
        continue
    points, labels = clip.similarity_map_to_points(
        similarity[1:, n], cv2_img.shape[:2], t=0.8
    )
```
- **操作**: 
  - 对每个目标类别，从相似度图提取点坐标和标签
  - `t=0.8` 是阈值，用于选择高置信度区域
  - `points`: 点坐标列表
  - `labels`: 点标签（1=正样本，0=负样本）

### 6.4 SAM 分割
```python
masks, scores, logits = predictor.predict(
    point_labels=labels, 
    point_coords=np.array(points), 
    multimask_output=True
)
mask = masks[np.argmax(scores)]
```
- **操作**: 
  - 使用 CLIP Surgery 提取的点作为 SAM 的提示
  - SAM 生成多个候选掩码
  - 选择得分最高的掩码

### 6.5 可视化分割结果
```python
vis = cv2_img.copy()
vis[mask > 0] = vis[mask > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
for i, [x, y] in enumerate(points):
    cv2.circle(vis, (x, y), 3, 
               (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)
```
- **操作**: 
  - 在原图上叠加分割掩码（青色高亮）
  - 绘制点提示：蓝色=正样本，橙色=负样本

**工作流程**: 文本 → CLIP Surgery → 相似度图 → 点提示 → SAM → 分割掩码

---

## 7. 单个文本的 CLIP Surgery + SAM

### 7.1 定义单个文本
```python
texts = ['bench']
```

### 7.2 提取特征并应用手术
```python
image_features = model.encode_image(image)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)
redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)
similarity = clip.clip_feature_surgery(
    image_features, text_features, redundant_features
)[0]
```
- **操作**: 与步骤 5 相同，但用于 SAM 分割

### 7.3 提取点并执行 SAM 分割
```python
points, labels = clip.similarity_map_to_points(
    similarity[1:, 0], cv2_img.shape[:2], t=0.8
)
masks, scores, logits = predictor.predict(
    point_labels=labels, 
    point_coords=np.array(points), 
    multimask_output=True
)
mask = masks[np.argmax(scores)]
```
- **操作**: 与步骤 6.3-6.4 相同，但针对单个文本

### 7.4 可视化结果
- **操作**: 显示单个文本的分割结果

**应用场景**: 查询单个对象并获取精确分割掩码。

---

## 8. 组合目标的 CLIP Surgery + SAM

### 8.1 定义组合文本
```python
text = 'person+bench'
```
- **操作**: 使用 "+" 连接多个目标
- **注意**: 不使用完整句子，避免明显文本占主导

### 8.2 提取特征并应用手术
```python
image_features = model.encode_image(image)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)
text_features = clip.encode_text_with_prompt_ensemble(
    model, text.split('+'), device
)
sm = clip.clip_feature_surgery(
    image_features, text_features, redundant_features
)[0, 1:, :]
```
- **操作**: 
  - 将组合文本拆分为多个文本
  - 对每个文本应用特征手术
  - 得到多个相似度图

### 8.3 归一化并合并特征
```python
sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (
    sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0]
)
sm_mean = sm_norm.mean(-1, keepdim=True)
```
- **操作**: 
  - 对每个相似度图进行 min-max 归一化
  - 计算平均相似度图（用于负样本点）

### 8.4 提取组合点提示
```python
# 从平均图获取负样本点
p, l = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], t=0.8)
num = len(p) // 2
points = p[num:]  # 负样本在后半部分
labels = [l[num:]]

# 从各个单独图获取正样本点
for i in range(sm.shape[-1]):
    p, l = clip.similarity_map_to_points(sm[:, i], cv2_img.shape[:2], t=0.8)
    num = len(p) // 2
    points = points + p[:num]  # 正样本在前半部分
    labels.append(l[:num])
labels = np.concatenate(labels, 0)
```
- **操作**: 
  - 从平均相似度图提取负样本点（背景区域）
  - 从每个目标的相似度图提取正样本点（前景区域）
  - 合并所有点及其标签

### 8.5 SAM 分割和可视化
```python
masks, scores, logits = predictor.predict(
    point_labels=labels, 
    point_coords=np.array(points), 
    multimask_output=True
)
mask = masks[np.argmax(scores)]
```
- **操作**: 使用组合点提示进行 SAM 分割
- **可视化**: 显示包含多个目标的分割结果

**应用场景**: 同时分割图像中的多个相关对象。

---

## 总结

### 核心流程
1. **文本 → 特征**: 使用 CLIP 将文本编码为特征向量
2. **图像 → 特征**: 使用 CLIP 将图像编码为特征向量
3. **特征手术**: 移除冗余特征，提高定位精度
4. **相似度图**: 计算图像区域与文本的相似度
5. **点提取**: 从相似度图提取高置信度点
6. **SAM 分割**: 使用点提示进行精确分割

### 关键改进
- **CLIP Surgery**: 通过架构修改和特征手术减少噪声
- **高分辨率**: 使用 512×512 输入提高精度
- **冗余特征移除**: 使用空字符串特征移除无关信息
- **组合目标**: 支持多个目标的联合分割

### 注意事项
- 需要准备 `demo.jpg` 作为输入图像
- 需要下载 SAM 模型权重文件 `sam_vit_h_4b8939.pth`
- 某些情况下，即使点提示准确，SAM 的分割结果仍可能需要改进
