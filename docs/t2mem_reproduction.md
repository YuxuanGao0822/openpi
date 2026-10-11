# T²Mem：openpi π0.5 复现与 RoboMME 运行手册

**当前实验：单张 A100 80G、物理 GPU 6、只训练 PickXtimes，并比较 π0.5 baseline。请使用 [专用服务器手册](pickxtimes_a100.md)，路径为 `/data1/workspace/gaoyuxuan/openpi`。本页保留方法对应和完整 benchmark 的通用配置。**

本实现依据用户提供的 **2609.36720v1：T²Mem: Learning Test-Time Memory for Robotics**，以当前仓库的 `PI0Pytorch`、PaliGemma 和 Gemma action expert 为骨干。模型、数据处理、训练、推理和 RoboMME test split 评测已经接通。Mac 上仅作静态验证；下面的依赖安装、数值测试、训练、推理和仿真命令全部在 Linux GPU 服务器执行。

作者的 [项目页](https://yzliu84.github.io/T2MEM-project/) 在 2026-10-11 标注 Code Coming soon。本实现是从原文重建的方法实现，不是作者尚未发布的代码。实验分数须由服务器训练与评测产生。

## 方法对应

| 原文 | 实现位置与行为 |
|---|---|
| §3.3 / App. A observation-grounded interface | `src/openpi/t2mem/model.py`：18 层、16×1024 interface tokens；VL 只看 VL，interface queries 只看有效 VL keys；连续状态和动作仅在 AE suffix 中 |
| 层间对应 | interface 层 l 的输出由层 l+1 消费；18 个模块全部分配，最后一个不会影响当前动作，17 层有效 |
| Eq. 5–8 | 每层独立 biased Q/K/V；16 heads、64→256→64 exact GeLU；Q/K/V RMSNorm，Q/K interleaved RoPE；位置为 interface token index；初始融合强度 0.01 |
| Eq. 10–15 | `memory.py`：每头方向／位移对齐与重建残差、自适应 β、J_f g 曲率校准、试探步与 learned softplus multiplier；停止 β／校准梯度，保留更新梯度和二阶导数 |
| fast state | FP32 weights+biases，episode reset 到学习的 W₀；函数式更新，不改模型参数，不做额外 inner-gradient clipping |
| App. B Stage 1 | 20,000 updates；VLM rank-16 LoRA、完整 AE、action/time projections；2e-5，global effective batch 64；无记忆 |
| App. B Stage 2A | 冻结 policy，训练 interface tokens、Q/K/V adapters、memory projections、W₀、ρ 和 α；7.5e-5，gate 5e-4；global effective batch 32 sequences |
| App. B Stage 2B | 冻结 memory 慢参数，更新 VLM LoRA、AE、action/time/proprio projections；3e-5；fast state 仍按观测更新 |
| 交替 | 500 memory updates → 500 policy updates，分别保留两个 optimizer 状态 |
| Eq. 16 | 64-step flow matching；每 chunk 按有效未来动作×8 个监督维度归一化；视频演示不计动作 loss，episode 尾部不重复最后动作充当有效监督 |
| 推理顺序 | 当前观测的所有读取使用 pre-write memory；写入结果在后续观测生效；denoising 复用当前观测的 VL/interface cache，不逐次写入 |
| App. B/C 环境时序 | H64/E32，physical stride16；执行中点观测可以仅写记忆；demonstration 使用独立 indexed timeline |
| App. C 主评测 | 官方 test split、绝对 joint_angle、50 episodes/task ×3 evaluation seeds；只使用 terminal success，按任务宏平均 |

新增参数计数（不包含 VLM LoRA）：

| 部分 | 参数数 |
|---|---:|
| 18 层 biased 1024→1024 Q/K/V | 56,678,400 |
| 18×16 个 fast MLP 初始化 | 9,529,344 |
| rank-16 interface Q/K/V attention adapters | 1,622,016 |
| interface embeddings、fusion gates、ρ、32→1024 proprio projection | 68,626 |
| 合计 | **67,898,386** |

单 episode fast state 为 9,529,344 个 FP32 标量，即 36.35 MiB，和附录一致。

## 原文未完全公开的实现设置

这些设置会影响数值结果，记录在这里和每次运行的 `recipe.json` 中：

- **Interface 的共享权重细节**：原文给出 rank-16 adapters 和参数分项，但没有独立 interface FFN 的完整参数表。本实现从参数数目推导：共享 AE 的 Q/K/V 基础投影、O projection 与 FFN，interface Q/K/V 增加各自的 rank-16 residual adapters；interface 使用不依赖 flow-time 的无参数 RMSNorm。1,622,016 的 adapter 数量恰好对应 Q/K/V 三组。论文没有唯一确定这个共享方案，最终可与作者代码核对。
- **VLM LoRA**：使用 openpi 的 rank16/scale1 思路，作用于 VL Transformer 的 Q/K/V/O 和 FFN gate/up/down；视觉 encoder 和 base VL weights 固定。
- **Stage 1 的状态输入**：沿用原生 π0.5 的 discrete state prompt。Stage 2 将状态移到单独连续 proprio suffix token，确保记忆没有直接状态 token 入口；原文没有给出 Stage 1 与 Stage 2 的 tokenizer 模板。
- **总训练长度与序列采样**：附录给出 Stage 1 20k 和交替周期 500/500，没有给出主表最终的交替总步数、窗口长度、每个序列的监督采样密度及最终 checkpoint 选择规则。因此 `--updates` 必须显式给出。本实现均匀选一个任务，再采样该任务 episode 和 execution endpoint；每次从 W₀ 重放完整前缀，不 detach、不截断 BPTT；演示处理写入观测，执行按 `--supervise-stride` 监督，默认32。改变该参数仍使用相同的物理 stride16 写入规则。
- **Demonstration 写入相位**：默认 2,18,34,…；stride4 时为 2,6,10,…，与 Appendix D 明示的诊断时序相符；主实验只说明 stride16，没有单独明确 phase。
- **Optimizer 等**：原文主 recipe 没有给出 optimizer betas/weight decay/warm-up/EMA。使用 AdamW 默认 betas、weight_decay0、恒定论文学习率，无 EMA、无 outer clipping。denoising 默认10步，沿用 openpi。图像使用 openpi 的224×224 resize-with-pad，不添加未说明的随机增强。
- **Global batch 与 microbatch**：为完整保留序列图，每 GPU 串行处理一个序列再累积梯度，数学上的 effective batch 为32；Stage1 同样累积到64 frames，未照搬论文8 frames×8的物理 microbatch形状。支持 torchrun DDP；world size 必须整除 global batch。
- **评测采样**：每 episode 重置 fast state 和动作采样 RNG 到指定 evaluation seed，benchmark 环境种子始终由 test metadata 指定。默认1300 execution steps 沿用官方 policy evaluator；论文未列出这个上限。正确/空演示和禁用写入保持同样的调用时序。

本实现没有加入动作裁剪、记忆衰减、额外辅助 loss、外部 VLM、subgoal 监督、重试训练、跳过异常样本或异常评测计作失败等策略。仅保留必需的 mask、公式中的数值下界和训练阶段检查。

## GitHub → 服务器工作流

在本地查看并提交本次文件即可；数据、模型、训练输出留在服务器，`data/`、`checkpoints/`、`runs/`、`external/` 已被 gitignore 排除。服务器拉取代码：

```bash
git pull --ff-only
```

以下命令从服务器上的 openpi 仓库根目录运行。第一次安装：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv pip install -e .
# h5py / safetensors 在现有锁文件中也有依赖；显式确认用于数据与权重加载。
uv pip install h5py==3.13.0 safetensors
# openpi 的 PyTorch adaRMSNorm 依赖仓库自带的 transformers patch。
TRANSFORMERS_PATH=$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')
cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_PATH/"
```

如果重新同步环境覆盖了 transformers，重新执行最后两行。

## 数据与 π0.5 初始化

只下载官方 **训练** HDF5。未使用 subgoal、目标位置或 task-progress 作为模型输入：

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Yinpei/robomme_data_h5', repo_type='dataset', local_dir='data/robomme_data_h5')
PY
uv run python scripts/t2mem/prepare_data.py \
  --input data/robomme_data_h5 --output data/t2mem
```

此脚本解压官方 `.h5.tar.xz`，按真实 timestep 顺序保留 demonstration/execution；输出 episode NPZ、manifest 和 execution frames 的 q01/q99。状态为7关节角+夹爪第一维宽度，动作是7绝对关节角+夹爪命令（-1闭合，+1张开）。不要使用官方已处理的20-step action chunks替代这里的64-step chunks。

下载并转换 **π0.5-base**，不要换成 DROID/LIBERO 已微调模型。下面只使用 `pi05_libero` 配置选择 π0.5 的结构；加载的权重仍为 `pi05_base`：

```bash
uv run python - <<'PY'
from pathlib import Path
from openpi.shared.download import maybe_download
path = maybe_download('gs://openpi-assets/checkpoints/pi05_base')
Path('checkpoints').mkdir(exist_ok=True)
Path('checkpoints/pi05_base_path.txt').write_text(str(path))
PY
uv run python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "$(cat checkpoints/pi05_base_path.txt)" \
  --config-name pi05_libero \
  --output-path checkpoints/pi05_base_pytorch --precision bfloat16
```

## 服务器验证

Mac 上已验证语法、Ruff 和 diff whitespace，没有导入或执行模型。以下数值测试留给服务器；第一条检查 fast-memory 的显式梯度与 autograd/JVP 参考的一致性、外层二阶梯度、episode reset 和 per-example write mask：

```bash
uv run pytest -q src/openpi/t2mem/memory_test.py
CUDA_VISIBLE_DEVICES=0 uv run python scripts/t2mem/smoke.py \
  --base-weights checkpoints/pi05_base_pytorch/model.safetensors \
  --data data/t2mem --activation-checkpointing
```

第二条加载真实 π0.5，在 Stage1/2A/2B 分别做短序列前向／反向，核对新增参数量、冻结组和历史写入的梯度，再执行一次H64采样。随后可用训练命令的 `--updates 1 --batch-size 1` 验证入口（Stage1 smoke输出不是Stage2正式初始化）。

## 训练

Stage 1（以下以4卡示例；也可将 `--nproc-per-node` 改为整除64/32的卡数）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
  scripts/t2mem/train.py --stage stage1 \
  --base-weights checkpoints/pi05_base_pytorch/model.safetensors \
  --data data/t2mem --output runs/t2mem/stage1 \
  --updates 20000 --batch-size 64 --seed 0 --activation-checkpointing
```

Stage 2：先用**原文诊断明确采用的2000-step预算**跑两个完整交替周期。2000不是论文主表最终预算；主表预算需要作者补充，或在验证集确定训练预算并报告：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
  scripts/t2mem/train.py --stage alternating \
  --init runs/t2mem/stage1/step_020000.pt \
  --data data/t2mem --output runs/t2mem/alternating \
  --updates 2000 --batch-size 32 --seed 0 --activation-checkpointing
```

单任务诊断可在两阶段命令中均追加 `--tasks VideoUnmask`；论文 online-memory VideoUnmask 诊断使用stride4，Stage2需追加 `--write-stride 4`。普通主实验使用默认stride16。

完整序列反向图会占用较多显存；实现支持 non-reentrant activation checkpointing，不为节省显存自动丢弃历史梯度或缩短前缀。这里没有给出未经服务器实测的显存/速度承诺。

断点续训保留两个 optimizer 状态及各 rank RNG；保持原运行的卡数、tasks、seed、supervise stride 和数据目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
  scripts/t2mem/train.py --stage alternating \
  --resume runs/t2mem/alternating/step_002000.pt \
  --data data/t2mem --output runs/t2mem/alternating \
  --updates 4000 --batch-size 32 --seed 0 --activation-checkpointing
```

`--updates` 是目标累计步数，不是额外步数。`--init` 从Stage1仅载入base policy+VL LoRA，Stage2新增参数保持fresh初始化。

## RoboMME 评测

策略模型与 simulator 使用**两个 Python 环境**，避免 PyTorch/JAX/ManiSkill 版本互相覆盖。服务器第一次准备 simulator：

```bash
mkdir -p external
git clone https://github.com/RoboMME/robomme_benchmark.git external/robomme_benchmark
git -C external/robomme_benchmark checkout fae9f7300347f1146c2ae4844faa4eef7e38098e
cd external/robomme_benchmark
uv sync --frozen
uv pip install -e .
uv pip install -e ../../packages/openpi-client
cd ../..
```

若原官方环境已有可用安装，直接复用其 Python 环境，安装本仓库 `packages/openpi-client` 即可。Vulkan/NVIDIA 配置按 [RoboMME 官方安装说明](https://github.com/RoboMME/robomme_benchmark#-installation) 处理。

终端 A：使用openpi环境提供策略。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/t2mem/serve.py \
  --checkpoint runs/t2mem/alternating/step_002000.pt --port 8000
```

终端 B：使用 simulator 环境运行，预测64、执行32、写入stride由server metadata读取。这里只发送演示观测和执行步中点，不使用Oracle subgoals：

```bash
CUDA_VISIBLE_DEVICES=1 external/robomme_benchmark/.venv/bin/python \
  examples/robomme/eval_t2mem.py --host localhost --port 8000 \
  --seed 0 --output runs/t2mem/evaluation/step002000_seed0.json
CUDA_VISIBLE_DEVICES=1 external/robomme_benchmark/.venv/bin/python \
  examples/robomme/eval_t2mem.py --host localhost --port 8000 \
  --seed 1 --output runs/t2mem/evaluation/step002000_seed1.json
CUDA_VISIBLE_DEVICES=1 external/robomme_benchmark/.venv/bin/python \
  examples/robomme/eval_t2mem.py --host localhost --port 8000 \
  --seed 2 --output runs/t2mem/evaluation/step002000_seed2.json
uv run python scripts/t2mem/summarize.py \
  runs/t2mem/evaluation/step002000_seed0.json \
  runs/t2mem/evaluation/step002000_seed1.json \
  runs/t2mem/evaluation/step002000_seed2.json
```

先检查一项任务时追加 `--tasks VideoUnmask`。主表必须跑完16项，每项50个 test episodes，difficulty应为26 easy+12 medium+12 hard，三evaluation seeds共150个rollouts；summary会拒绝不完整或重复的rollout文件。评测执行错误直接报错，不当作失败成功率样本。

Stage1 baseline可改server checkpoint为 `runs/t2mem/stage1/step_020000.pt`。它会自动关闭interface与fast-memory，仍用H64/E32。写入消融使用serve的 `--disable-writes`；非自适应消融从Stage2新训练，使用train的 `--non-adaptive`（保留曲率校准）；空演示在eval追加 `--demo-content empty`（保留learned W₀和同样infer cadence）。这些入口不声称已重建论文中未发布的50组 counterfactual paired episodes。

## 来源与状态

- [论文 v1](https://arxiv.org/abs/2609.36720v1)：方法、附录与训练/评测协议；实现以用户提供的本地PDF为依据。
- [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark)，接口核对commit `fae9f7300347f1146c2ae4844faa4eef7e38098e`。
- [RoboMME policy learning](https://github.com/RoboMME/robomme_policy_learning)，数据/动作/normalization核对commit `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`；没有把其其他记忆方法当作T²Mem。
- [官方训练数据](https://huggingface.co/datasets/Yinpei/robomme_data_h5)。

**验证状态：本地静态检查；未运行依赖环境、数值测试、GPU模型、训练或simulator。** 数值正确性与效果以服务器输出为准，原文未披露设置列于上文。
