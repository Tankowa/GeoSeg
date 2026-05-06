import numpy as np
from pycocotools import mask as maskUtils

# 你的数据
rle_obj = {
    "size": [600, 600],
    "counts": "mmj53bb0h0[O1N2O0000000001O0O2O2N[Pm4"
}

# 解码成二进制掩码 (0和1)
binary_mask = maskUtils.decode(rle_obj)

# binary_mask 现在是一个 shape 为 (600, 600) 的 numpy 数组
print(f"Mask 形状: {binary_mask.shape}")
print(f"包含的目标像素点数量: {np.sum(binary_mask)}")