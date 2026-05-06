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
  <em>Submitted to NeurIPS 2026</em>
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


### Environment and prerequisites

- Conda is recommended (the example env name in `start_api_server.sh` is `flow_grpo`; adjust the script to match your setup).
- **Qwen API** (`qwen_api_server.py`): set a valid Qwen3-VL `model_path` (and related settings) inside the script; install `fastapi`, `uvicorn`, `modelscope`, and a matching PyTorch stack.
- **Segmentation pipeline** (`CLIP_Surgery/process_annotations_clip_surverg.py`): requires PyTorch, `ultralytics`, the OpenAI CLIP package, the `CLIP_Surgery/clip_surgery` package, and **SAM3 weights** on disk (the script uses a fixed path in code—place weights there after clone, or edit the path in code).
- **Data:** `--annotations_dir` should contain `*.json` files; each item’s `file_name` must point to a readable image path.

### 1. Start the Qwen API server (generation + evaluation)

#### Option A: `start_api_server.sh` (example)

From the repository root (edit Conda paths, `conda activate` env name, `CUDA_VISIBLE_DEVICES`, etc. on first use):

```bash
chmod +x start_api_server.sh
./start_api_server.sh
```

By default this runs `python qwen_api_server.py`. The service URL is usually `http://localhost:8010` (see what the script prints).

#### Option B: Run Python directly

With your Conda environment activated:

```bash
cd /path/to/sam_new
python qwen_api_server.py
```

The server exposes routes such as **`/generate`** (for the segmentation pipeline) and **`/evaluate`** (for MLLM evaluation clients); see FastAPI definitions in `qwen_api_server.py`. Interactive docs are typically at `http://localhost:8010/docs`.

### 2. Run the CLIP Surgery + SAM3 annotation pipeline

In another terminal (dependencies installed, API running), from `CLIP_Surgery`:

```bash
cd /path/to/sam_new/CLIP_Surgery
CUDA_VISIBLE_DEVICES=0 python process_annotations_clip_surverg.py \
  --annotations_dir /path/to/your/annotations \
  --api_base_url http://localhost:8010 \
  --output_dir /path/to/output_dir
```

Arguments (unchanged from code):

| Argument | Meaning |
|----------|---------|
| `--annotations_dir` | Directory containing annotation `*.json` files. |
| `--api_base_url` | Base URL of the Qwen API (no path suffix); the script calls `{api_base_url}/generate`. |
| `--output_dir` | Output directory; if omitted, a timestamped `qwen_results_*` folder is created under the parent of `annotations_dir`. |

### 3. Evaluation and metric scripts (local `experiments/`, not pushed by default)

The paths below refer to **`experiments/` in your workspace** (ignored by `.gitignore`). They aggregate statistics over **generated mask folders** and annotations. The word “baseline” in script names means **multiple prediction-root folders** for comparison, **not** that third-party baseline model code is vendored into this repo.

#### 3.1 Pixel-level metrics (IoU, Dice, Acc, Prec, Rec, Spec, F1, BF, and average rank)

| Script | Role |
|--------|------|
| `experiments/baseline_score.py` | RS_ReasonSeg-style: multiple `--method NAME ROOT_DIR PATTERN` entries over prediction roots; pixel metrics. |
| `experiments/baseline_score_RS_Earth.py` | RS_Earth-style; similar interface. |
| `experiments/ablation.py` / `experiments/ablation_RS_Earth.py` | Ablation over GeoSeg mask-path variants. |

Argument names follow each file’s `argparse` (e.g. `--ann-dir`, `--output`, `--method`).

#### 3.2 MLLM scores (Faithfulness, Localization, Robustness, Overlap)

- **Server:** `qwen_api_server.py` route **`/evaluate`** (reads image and GT/pred masks, parses 1–5 scores; Overlap bins vs. IoU are implemented in that file).
- **Client batch jobs:** `experiments/baseline_score_RS_Earth_vlm.py`, `experiments/baseline_score_vlm.py`, and `qwen_8B` variants; call via **`--api-url`** (e.g. `http://localhost:8010/evaluate`).

**Not part of this repository:** training/inference codebases for other methods (e.g. third-party code under `sam_mllm/`, `baselines/`). To match paper tables, prepare each method’s prediction mask directories locally, then run the corresponding `experiments/*.py` scripts for numeric aggregation.
