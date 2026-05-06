<div align="center">

# GeoSeg: Training-Free Reasoning-Driven Segmentation in Remote Sensing Imagery


<div>
  <strong>Lifan Jiang</strong> &nbsp;
  <strong>Yuhang Pei</strong> &nbsp;
  <strong>Boxi Wu</strong> &nbsp;
  <strong>Yan Zhao</strong> &nbsp;
  <br>
  <strong>Tianrun Wu</strong> &nbsp;
  <strong>Shulong Yu</strong> &nbsp;
  <strong>Lihui Zhang</strong> &nbsp;
  <strong>Deng Cai</strong>
</div>

<div>
  State Key Lab of CAD&CG, Zhejiang University
</div>

<div>
  <em>Submitted to ECCV 2026</em>
</div>

<br>

<p>
  <a href="https://arxiv.org/abs/2603.03983">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  &nbsp;&nbsp;
  <a href="https://tankowa.github.io/GeoSeg.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-20BEFF?style=flat&logo=google-chrome&logoColor=white" alt="Project Page">
  </a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/qingjiu151/GeoSeg">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-111111?style=flat&logo=huggingface&logoColor=FFD21E" alt="Dataset">
  </a>
</p>

</div>

---

## 🚧 Coming Soon

The project page, paper, and dataset for **GeoSeg** are currently being prepared and will be released soon.

Please stay tuned!

## 📌 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{jiang2026geosegtrainingfreereasoningdrivensegmentation,
      title={GeoSeg: Training-Free Reasoning-Driven Segmentation in Remote Sensing Imagery},
      author={Lifan Jiang and Yuhang Pei and Boxi Wu and Yan Zhao and Tianrun Wu and Shulong Yu and Lihui Zhang and Deng Cai},
      year={2026},
      eprint={2603.03983},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.03983},
}
```

---

## 运行与复现

本 **GitHub 仓库** 仅收录 **GeoSeg 本方法的推理与测评服务相关代码**（见下表）。**不包含** 各对比方法（第三方 baseline）的实现与训练代码；本地若存在 `baselines/`、`sam_mllm/`、`GLaMM-RefSeg/`、`S-VICE/` 等目录，已由根目录 **`.gitignore`** 排除，请勿使用 `git add -f` 强行提交。

**`experiments/`**（像素 / VLM 指标聚合脚本）同样由 **`.gitignore`** 排除、**默认不上传**；请在本地保留该目录，若需对外分发指标脚本可单独打包或使用 `git add -f` 仅加入个别文件。

**建议纳入版本控制的目录与文件：**

| 路径 | 说明 |
|------|------|
| `CLIP_Surgery/` | CLIP Surgery + SAM3 推理流水线（含 `clip_surgery/` 包与 `process_annotations_clip_surverg.py` 等） |
| `qwen_api_server.py` | Qwen3-VL 服务：`/generate`（结构化落地）与 `/evaluate`（MLLM 测评） |
| `qwen_api_server_*.py`、`start_api_server*.sh` | 若你有多端口/多卡实例，可按需一并提交 |
| `start_api_server.sh` | 启动 API 的示例脚本 |
| `readme.md`、`.gitignore` | 说明与忽略规则 |

**不包含在发布范围内：** `experiments/`、`baselines/`、`sam_mllm/`、各 `RS_*Benchmark*/` 数据树、权重目录 `checkpoints/` 等；数据集与权重请按外链（如 HuggingFace）或自备路径配置。

下文说明如何启动 Qwen API、运行推理脚本，以及测评逻辑与本地 `experiments/` 脚本对应关系；若你已将指标脚本强制纳入 Git，参数名以各文件内 `argparse` 为准。

### 环境与前置条件

- 建议使用 Conda（`start_api_server.sh` 中示例环境名为 `flow_grpo`，可按你的环境修改脚本）。
- **Qwen API**（`qwen_api_server.py`）：需在脚本内配置可用的 Qwen3-VL 模型路径（`model_path` 等），并安装 `fastapi`、`uvicorn`、`modelscope`、对应 PyTorch 等依赖。
- **分割流水线**（`CLIP_Surgery/process_annotations_clip_surverg.py`）：需 PyTorch、`ultralytics`、OpenAI CLIP 包、`CLIP_Surgery/clip_surgery` 包，以及本机上的 **SAM3 权重**（代码中为固定路径，上传仓库后请自行放到该路径或按需改代码中的路径）。
- **数据**：`--annotations_dir` 下为若干 `*.json`；每条样本的 `file_name` 须指向可读图像路径。

### 1. 启动 Qwen API 服务（生成 + 测评共用）

#### 方式 A：`start_api_server.sh`（示例）

在仓库根目录执行（首次请根据本机修改脚本内的 Conda 路径、`conda activate` 环境名、`CUDA_VISIBLE_DEVICES` 等）：

```bash
chmod +x start_api_server.sh
./start_api_server.sh
```

脚本默认会执行：`python qwen_api_server.py`。默认服务地址为 `http://localhost:8010`（以脚本内打印为准）。

#### 方式 B：直接运行 Python

在已激活的 Conda 环境中：

```bash
cd /path/to/sam_new
python qwen_api_server.py
```

服务提供例如 **`/generate`**（供分割流水线调用）与 **`/evaluate`**（供 MLLM 测评脚本调用）等接口；具体路由以 `qwen_api_server.py` 内 FastAPI 定义为准。交互式文档一般为 `http://localhost:8010/docs`。

### 2. 运行 CLIP Surgery + SAM3 注释处理流水线

在另一终端中（需已安装依赖，且 API 已启动），进入 `CLIP_Surgery` 目录后执行：

```bash
cd /path/to/sam_new/CLIP_Surgery
CUDA_VISIBLE_DEVICES=0 python process_annotations_clip_surverg.py \
  --annotations_dir /path/to/your/annotations \
  --api_base_url http://localhost:8010 \
  --output_dir /path/to/output_dir
```

参数说明（与代码一致）：

| 参数 | 含义 |
|------|------|
| `--annotations_dir` | 存放标注 `*.json` 的目录。 |
| `--api_base_url` | Qwen API 根地址（不含路径后缀）；脚本会请求 `{api_base_url}/generate`。 |
| `--output_dir` | 结果输出目录；若省略，则在 `annotations_dir` 的父目录下创建带时间戳的 `qwen_results_*` 目录。 |

### 3. 测评与指标脚本（本地 `experiments/`，默认不随本仓库推送）

以下路径指 **你工作区中的 `experiments/`**（已被 `.gitignore` 忽略）。作用是对 **已生成的掩码目录** 与标注做聚合统计；脚本名中的 “baseline” 表示「多组预测结果文件夹」的对比入口，**不是** 向仓库引入第三方 baseline 模型代码。

#### 3.1 像素级指标（IoU、Dice、Acc、Prec、Rec、Spec、F1、BF 及平均排名）

| 脚本 | 用途 |
|------|------|
| `experiments/baseline_score.py` | RS_ReasonSeg 风格：多组 `--method NAME ROOT_DIR PATTERN` 指向各方法预测目录，计算像素指标。 |
| `experiments/baseline_score_RS_Earth.py` | RS_Earth 风格，接口类似。 |
| `experiments/ablation.py` / `experiments/ablation_RS_Earth.py` | 对 GeoSeg 多 variant 掩码路径做消融统计。 |

参数名以各文件内 `argparse` 为准（例如 `--ann-dir`、`--output`、`--method`）。

#### 3.2 MLLM 主观评分（Faithfulness、Localization、Robustness、Overlap）

- **服务端**：`qwen_api_server.py` 的 **`/evaluate`**（读图与 GT/预测掩码，解析 1–5 分；Overlap 与 IoU 分档见该文件实现）。
- **客户端批处理**：`experiments/baseline_score_RS_Earth_vlm.py`、`experiments/baseline_score_vlm.py` 及带 `qwen_8B` 的变体；通过 **`--api-url`**（如 `http://localhost:8010/evaluate`）调用上述接口。

**不包含于本仓库：** 各 baseline 方法自身的训练/推理工程（例如原 `sam_mllm/`、`baselines/` 下第三方代码）；若需与论文表格完全一致，请在本地另行准备对应方法的预测掩码目录，再交给 `experiments/*.py` 做数值汇总。
