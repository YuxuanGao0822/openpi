# PickXtimes：单张 A100 80G，π0.5 baseline 与 T²Mem

服务器仓库为 `/data1/workspace/gaoyuxuan/openpi`，物理 GPU 编号为 **6**。所有 GPU 命令均设置 `CUDA_VISIBLE_DEVICES=6`；此时 Python 内部的 `cuda:0` 对应物理 GPU 6。先训练 baseline，再训练 T²Mem，最后依次评测两种模型。

| 实验 | 数据与初始化 | 训练设置 | 输出目录 |
|---|---|---|---|
| π0.5 baseline | PickXtimes 训练集，π0.5-base | 无记忆 Stage 1；20,000 updates，lr 2e-5，有效 batch 64 帧 | `runs/pickxtimes/pi05_baseline` |
| T²Mem | 同一训练集、同一归一化；初始化自 baseline 的 step 20,000 | 500 memory / 500 policy updates 交替；有效 batch 32 条序列 | `runs/pickxtimes/t2mem` |

Stage 1 使用原生 π0.5 discrete-state prompt、VLM rank-16 LoRA 和完整 action expert；评测时关闭 interface 和 fast memory。T²Mem 使用论文的 18 层、16 个 interface tokens、1024 维记忆，H64/E32、写入 stride16。单卡每次处理一帧／一条序列，通过梯度累积保持有效 batch 64／32，不改变模型宽度、动作长度或历史梯度。

论文附录明确给出 Stage 1 的 20k 和 Stage 2 的 500/500 交替周期，未披露主结果的 Stage 2 总步数。本入口默认采用论文诊断的 **2,000-step Stage 2 预算**。这是单任务复现实验的预算；不能将其结果当作论文完整主表复现。[方法对应及原文未披露项](t2mem_reproduction.md)保留在通用手册中。

## 1. 同步代码与安装服务器环境

本地 Mac 修改后按你的工作流提交并上传 GitHub；服务器执行以下命令。本次未在 Mac 安装或运行训练环境。

```bash
cd /data1/workspace/gaoyuxuan/openpi
git pull --ff-only
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv pip install -e .
uv pip install h5py==3.13.0 safetensors
TRANSFORMERS_PATH=$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')
cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_PATH/"
```

若重新同步环境覆盖了 transformers，重新执行 patch 的最后两行。后续命令均从上述仓库根目录执行。

## 2. 只下载和处理 PickXtimes

```bash
cd /data1/workspace/gaoyuxuan/openpi
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'Yinpei/robomme_data_h5',
    repo_type='dataset',
    allow_patterns=['record_dataset_PickXtimes.h5.tar.xz'],
    local_dir='data/robomme_data_h5',
)
PY
uv run python scripts/t2mem/prepare_data.py \
  --input data/robomme_data_h5 --output data/t2mem_pickxtimes \
  --tasks PickXtimes
```

`--tasks PickXtimes` 同时筛选解压、episode 转换和归一化统计。两种模型共用 `data/t2mem_pickxtimes/norm_stats.json`。输入仅包含两路 RGB、7 关节角及夹爪状态和任务指令；训练目标为 8 维绝对关节角／夹爪动作。保留原始轨迹，训练时构造 64-step chunks。

## 3. 下载并转换 π0.5-base

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python - <<'PY'
from pathlib import Path
from openpi.shared.download import maybe_download
path = maybe_download('gs://openpi-assets/checkpoints/pi05_base')
Path('checkpoints').mkdir(exist_ok=True)
Path('checkpoints/pi05_base_path.txt').write_text(str(path))
PY
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "$(cat checkpoints/pi05_base_path.txt)" \
  --config-name pi05_libero \
  --output-path checkpoints/pi05_base_pytorch --precision bfloat16
```

这里的 `pi05_libero` 只选择 π0.5 结构，实际加载的是 `pi05_base` 权重。

## 4. 在服务器验证数值与 GPU 入口

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run pytest -q src/openpi/t2mem/memory_test.py scripts/t2mem/summarize_test.py
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/t2mem/smoke.py \
  --task PickXtimes --data data/t2mem_pickxtimes \
  --base-weights checkpoints/pi05_base_pytorch/model.safetensors \
  --activation-checkpointing
```

第一条检查记忆更新、外层梯度、reset 和结果汇总协议。第二条加载真实模型，检查三种训练阶段的前向／反向、参数组、历史写入梯度和 H64 动作采样；打印显存峰值。该短序列检查通过后，正式训练中的长前缀仍需要观察实际显存占用。

## 5. 依次训练两种模型

先训练 π0.5 baseline：

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 bash scripts/t2mem/pickxtimes_a100.sh baseline
```

训练结束得到 `runs/pickxtimes/pi05_baseline/step_020000.pt`。然后从这份权重初始化 T²Mem 的 policy，记忆模块采用新的初始化：

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 bash scripts/t2mem/pickxtimes_a100.sh t2mem
```

默认得到 `runs/pickxtimes/t2mem/step_002000.pt`。两个模型的训练 seed 都为 0，每 500 steps 保存一次完整 checkpoint。`recipe.json` 记录实际配置，日志包含 `sequence_frames_max` 和 `peak_allocated_gib`。

如需自行指定 Stage 2 总预算，设置 `T2MEM_UPDATES`，例如 4,000 steps：

```bash
CUDA_VISIBLE_DEVICES=6 T2MEM_UPDATES=4000 \
  bash scripts/t2mem/pickxtimes_a100.sh t2mem
```

此命令是从 baseline **重新初始化**的训练；续训使用下一节的 `--resume`。改变总预算后，评测服务的 checkpoint 路径也对应改变。

单卡默认开启 non-reentrant activation checkpointing，保留完整 chronological prefix 和二阶梯度。若服务器实测长序列超出 80G，可开启保存反向张量到 CPU：

```bash
CUDA_VISIBLE_DEVICES=6 ACTIVATION_OFFLOAD=1 \
  bash scripts/t2mem/pickxtimes_a100.sh t2mem
```

该选项使用 [PyTorch `save_on_cpu`](https://docs.pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.save_on_cpu)，保留原有计算图，增加 CPU 内存和传输开销；默认关闭。不会自动截断序列、detach 记忆、降低 batch 或替换优化算法。本地静态检查无法确定完整训练的显存和速度。

## 6. 断点续训

以下示例从已保存的 Stage 2 step 1,000 继续到累计 step 2,000；使用实际存在的 checkpoint。`--updates` 表示累计目标步数。

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/t2mem/train.py \
  --stage alternating --resume runs/pickxtimes/t2mem/step_001000.pt \
  --data data/t2mem_pickxtimes --tasks PickXtimes \
  --output runs/pickxtimes/t2mem --updates 2000 --batch-size 32 \
  --seed 0 --activation-checkpointing
```

需要 CPU offload 时追加 `--activation-offload`。续训会恢复 optimizer 和 RNG；保持单卡、数据目录、任务、seed 和监督 stride。baseline 续训同理，使用 `--stage stage1`、baseline checkpoint、`--batch-size 64` 和 `--updates 20000`。

## 7. 准备 RoboMME 仿真环境

策略使用 openpi 环境，仿真使用官方 benchmark 的独立环境。首次安装：

```bash
cd /data1/workspace/gaoyuxuan/openpi
mkdir -p external
git clone https://github.com/RoboMME/robomme_benchmark.git external/robomme_benchmark
git -C external/robomme_benchmark checkout fae9f7300347f1146c2ae4844faa4eef7e38098e
cd external/robomme_benchmark
uv sync --frozen
uv pip install -e .
uv pip install -e ../../packages/openpi-client
cd /data1/workspace/gaoyuxuan/openpi
```

已有可用的 RoboMME 环境可复用，并安装当前仓库的 `packages/openpi-client`。Vulkan 配置见 [官方安装说明](https://github.com/RoboMME/robomme_benchmark#-installation)。

## 8. 评测 π0.5 baseline

训练停止后，终端 A 启动 baseline 服务：

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/t2mem/serve.py \
  --checkpoint runs/pickxtimes/pi05_baseline/step_020000.pt --port 8000
```

终端 B 使用同一张 GPU 6 运行仿真，依次评测 seeds 0、1、2：

```bash
cd /data1/workspace/gaoyuxuan/openpi
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=6 external/robomme_benchmark/.venv/bin/python \
    examples/robomme/eval_t2mem.py --host localhost --port 8000 \
    --tasks PickXtimes --seed "$seed" \
    --output "runs/pickxtimes/evaluation/pi05_seed${seed}.json"
done
```

策略与仿真此时共同使用 GPU 6，只加载一个模型。baseline 自动使用无记忆推理；结果文件记录 `method: pi05_baseline`。

## 9. 评测 T²Mem 并比较

baseline 评测结束后，在终端 A 按 Ctrl+C 停止 baseline 服务，再启动 T²Mem：

```bash
cd /data1/workspace/gaoyuxuan/openpi
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/t2mem/serve.py \
  --checkpoint runs/pickxtimes/t2mem/step_002000.pt --port 8000
```

终端 B：

```bash
cd /data1/workspace/gaoyuxuan/openpi
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=6 external/robomme_benchmark/.venv/bin/python \
    examples/robomme/eval_t2mem.py --host localhost --port 8000 \
    --tasks PickXtimes --seed "$seed" \
    --output "runs/pickxtimes/evaluation/t2mem_seed${seed}.json"
done
uv run python scripts/t2mem/summarize.py \
  runs/pickxtimes/evaluation/pi05_seed0.json \
  runs/pickxtimes/evaluation/pi05_seed1.json \
  runs/pickxtimes/evaluation/pi05_seed2.json \
  --tasks PickXtimes --compare \
  runs/pickxtimes/evaluation/t2mem_seed0.json \
  runs/pickxtimes/evaluation/t2mem_seed1.json \
  runs/pickxtimes/evaluation/t2mem_seed2.json \
  > runs/pickxtimes/evaluation/comparison.json
cat runs/pickxtimes/evaluation/comparison.json
```

每模型共 50 个 test episodes × 3 seeds = **150 次 rollout**；每 seed 为 26 easy、12 medium、12 hard。两者共享 test 环境种子、evaluation seeds、H64/E32 和训练集归一化。统计只采用 terminal success，输出各模型 PickXtimes 成功率和 T²Mem − baseline 的百分点差值。单任务输出的 `AVG_percent` 也是 PickXtimes 成功率；`suite_success_percent` 为空，不代表 Counting 四任务或完整 RoboMME 平均分。

数据、权重、训练输出和 benchmark checkout 留在服务器，均在 gitignore 范围内。**本次验证仅为 Mac 上的静态检查；本页中的数值测试、模型训练和仿真命令由服务器执行。**
