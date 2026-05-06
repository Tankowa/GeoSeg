import os
import json
import shutil

import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image


def load_annotations(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def iter_annotations(ann_path):
    """
    Stream annotations if possible (ijson), otherwise fallback to json.load.
    This avoids loading the whole huge JSON list into memory.
    """
    try:
        import ijson  # type: ignore

        with open(ann_path, "rb") as f:
            # train_ann.json is a top-level list -> iterate each item
            for item in ijson.items(f, "item"):
                yield item
    except Exception:
        # Fallback (may be slow / high memory for huge files)
        yield from load_annotations(ann_path)


class JsonArrayWriter:
    """
    Write a JSON array incrementally:
    [
      {...},
      {...}
    ]
    """

    def __init__(self, fp):
        self.fp = fp
        self.first = True
        self.fp.write("[\n")

    def write_item(self, obj):
        if not self.first:
            self.fp.write(",\n")
        json.dump(obj, self.fp, ensure_ascii=False)
        self.first = False

    def close(self):
        self.fp.write("\n]\n")
        self.fp.flush()


def rles_to_binary_mask(rles, expected_size=None):
    """
    rles: list of RLE dicts, or a single RLE dict.
    returns: (H, W) uint8 array with values 0/1
    """
    if isinstance(rles, dict):
        rles = [rles]

    # decode all instance masks
    decoded = maskUtils.decode(rles)

    # pycocotools returns (H,W) for single, (H,W,N) for list
    if decoded.ndim == 2:
        bin_mask = decoded.astype(np.uint8)
    else:
        bin_mask = (decoded.max(axis=2)).astype(np.uint8)

    if expected_size is not None:
        h, w = expected_size
        if bin_mask.shape[0] != h or bin_mask.shape[1] != w:
            raise ValueError(
                f"Mask size {bin_mask.shape} does not match expected size {(h, w)}"
            )

    return bin_mask


def save_binary_mask_png(binary_mask, save_path):
    """
    binary_mask: (H,W) uint8, values {0,1}
    saves: single-channel PNG with values {0,255}
    """
    mask_img = (binary_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img, mode="L")
    mask_pil.save(save_path)


def main():
    # 源数据集（你现在的 RS_Earth2）
    src_root = "/data/jianglifan/sam_new/RS_Earth2"
    src_ann_path = os.path.join(src_root, "annotations", "train_ann.json")
    src_img_dir = os.path.join(src_root, "images")

    # 目标数据集（模仿 RS_ReasonSeg_Benchmark 的组织，但只有 Level1，不分四个大类）
    dst_root = "/data/jianglifan/sam_new/RS_Earth2_Benchmark"
    dst_img_dir = os.path.join(dst_root, "images", "Level1")
    dst_mask_dir = os.path.join(dst_root, "masks", "Level1")
    dst_ann_dir = os.path.join(dst_root, "annotations")
    dst_ann_path = os.path.join(dst_ann_dir, "RS_Earth2.json")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    processed = 0

    # 边处理边写 JSON，避免最后一次性 dump 巨大 list
    with open(dst_ann_path, "w", encoding="utf-8") as f:
        writer = JsonArrayWriter(f)

        for ann in iter_annotations(src_ann_path):
            if ann.get("split") != "train":
                continue

            image_name = ann.get("image_name")
            if not image_name:
                continue

            src_img_path = os.path.join(src_img_dir, image_name)
            if not os.path.exists(src_img_path):
                print(f"Image not found, skip: {src_img_path}")
                continue

            mask_list = ann.get("mask", [])
            if not mask_list:
                print(f"No mask for image {image_name}, skip.")
                continue

            h, w = ann.get("size", [None, None])
            if h is None or w is None:
                print(f"No size info for image {image_name}, skip.")
                continue

            try:
                binary_mask = rles_to_binary_mask(mask_list, expected_size=(h, w))
            except Exception as e:
                print(f"Failed to decode mask for {image_name}: {e}")
                continue

            stem, ext = os.path.splitext(image_name)
            # 输出文件名：图片沿用原名；mask 统一为 png
            dst_img_name = f"{stem}{ext}"
            dst_mask_name = f"{stem}.png"

            # 复制/保存到新目录
            dst_img_path = os.path.join(dst_img_dir, dst_img_name)
            if not os.path.exists(dst_img_path):
                # 直接复制更快（不重新解码/编码）
                shutil.copy2(src_img_path, dst_img_path)

            dst_mask_path = os.path.join(dst_mask_dir, dst_mask_name)
            save_binary_mask_png(binary_mask, dst_mask_path)

            # 生成 annotations 样本（对齐 README 的字段命名）
            sample = {
                "id": str(processed).zfill(6),
                "file_name": os.path.join("Level1", dst_img_name),
                "mask_path": os.path.join("Level1", dst_mask_name),
                "height": int(h),
                "width": int(w),
                "difficulty": "Level1",
                # RS_Earth2 里没有 question 字段，这里用 description 作为 question
                "question": ann.get("description", ""),
            }
            writer.write_item(sample)

            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} samples...")

        writer.close()

    print(f"Done. Total converted samples: {processed}")
    print(f"Output dataset root: {dst_root}")
    print(f"Annotations: {dst_ann_path}")


if __name__ == "__main__":
    main()

