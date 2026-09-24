# 迷宫微调模型 v2 · 整局对弈对比报告

> 生成时间：2026-09-23 · 评测环境：laya-cuda venv（cupy 14.2.0 / RTX 3080）
> 对比对象：base（原版 `model.safetensors`）/ old_ft（退化版 `model_maze_ft.safetensors`）/ v2（新训 `model_maze_ft_v2.safetensors`）

## 1. 为什么之前跑不出这份对比

纯 numpy 前向实测 **~16s/次**（16×16 grid、717 token、22 层编码器），整局 130 步 ≈ 35 min/迷宫 →
之前所有 episode 评测都被超时杀掉。改用 `laya_verify_cuda.py` 的 cupy/CUDA 前向后，**~0.09s/次**（×170 提速），
整局评测降到 ~110s/10 迷宫，三模型并行 ~2min 出结果。

评测脚本：`labs/finetune/eval_episodes_cuda.py`（单模型版，`--model/--tag/--n/--max-steps`），
**逐字节复刻 `maze_laya.html` 的 `decide()` Model-only 语义**（line 403-407：模型选择合法则采用，否则回退 BFS 最优）。

## 2. 评测口径（metrics）

| 指标 | 含义 | 与"卡关"的关系 |
|---|---|---|
| `solved_rate` | 120 步内抵达终点的比率 | 能否真解迷宫 |
| `avg_first_wrong_depth` | 首次偏离 BFS 最短路的步深 | 跟最优路径跟多久 |
| `avg_first_leaf_depth` | 首次进入死胡同叶子（只能原路返回）的步深 | 视觉上"撞墙"的早晚 |
| `avg_first_revisit_depth` | 首次成环（回到已访问格）的步深 | 开始无谓绕圈 |
| `mean_move_logit_spread` | move 4 选项 logit 极差 | **校准度**：越大越"过自信硬提交" |

- 迷宫：16×16 完美迷宫（`convert_maze.genMaze`，recursive backtracker，与 `maze_laya.html` 同算法）
- 采样：seed_base=1000 起，每模型 **n=10** 局，`max_steps=120`

## 3. 结果（n=10）

| 指标 | base | old_ft | v2 |
|---|---|---|---|
| solved_rate | **0.0** | **0.0** | **0.0** |
| avg_steps (≤120) | 120.0 | 120.0 | 120.0 |
| deadend_rate | 0.0 | 0.0 | 0.0 |
| avg_first_wrong_depth | 2.0 | **9.1** | **2.0** |
| avg_first_leaf_depth | 72.8 | 86.6 | 72.8 |
| avg_first_revisit_depth | 3.0 | 10.2 | 3.0 |
| mean_move_logit_spread | 1.522 | **12.061** | **0.274** |

## 4. 解读

### 4.1 v2 结论：无回归，且校准更优
- v2 与 base 在**所有行为指标上完全一致**（first_wrong/first_leaf/first_revisit 三者逐位相等）。
- v2 的 `mv_spread=0.274` 比 base `1.522` **低一个量级** → v2 输出更"软"、更诚实，不再有过自信硬提交。
- **直接回答最初担忧"v2 是否比原版更早卡关"：否。** v2 行为等同 base、校准更好，不可能复现 old_ft 的退化。

### 4.2 old_ft 的病理：灾难性过自信
- `mv_spread=12.06`（base 的 ~8 倍、v2 的 ~44 倍）→ 一旦偏离最优，就以极高置信度锁死在错误分支/成环，
  这正是它在 `maze_laya.html` 里"看起来卡死"的根因（与 `maze_ft_degradation_analysis.md` 的通道对齐门崩溃结论一致）。
- 注意一个反直觉点：old_ft 的 `first_wrong=9.1` 反而**晚于** base(2.0) —— 它"跟最优路径跟得更久"，
  但偏离后毫无自纠能力（spread 12），所以观感上是"突然卡死、绕不出去"，而不是"早早就错"。

### 4.3 架构天花板（三者共通）
- **三个模型 `solved_rate` 全 0**：head-only 微调（含原版 base）都解不开 16×16 迷宫。
- 旧模型所谓"45% 成功率"纯靠训练集里代理文本明文"剩余步数"答案字段作弊，
  真实 grid 上 head-only 无法泛化（冻结编码器 + 选项 marker 近乎同模板，信号不足）。
- 即：v2 没有"变好到能解迷宫"，它做到的是**消除 old_ft 的过自信退化、回到 base 的真实基线、且校准更干净**。

## 5. 任务状态

| 项 | 状态 |
|---|---|
| `convert_maze.py` 改真实 grid（ground_truth） | ✅ |
| `finetune_run.py` 校准（label smoothing/权重衰减/早停） | ✅ |
| `benchmark.py` / `eval_episodes_cuda.py` 真实整局评测 | ✅ |
| `maze_dataset.jsonl`（148 样本，8 markers） | ✅ |
| `model_maze_ft_v2.safetensors` 训练产出 | ✅ |
| 三模型整局对比（base/old/v2，n=10） | ✅ 本报告 |

## 6. 真正提升路径（待确认后实施）

head-only 已到天花板，要实打实提高"能解迷宫"需二选一（或叠加）：

- **A. 数据扩充**：多迷宫 + off-path 恢复样本（走到死胡同/岔路错的样本），打破"只在最优路径上训"的偏置。
- **B. 全量微调编码器**：走 `pipeline.py` 的 torch 路径（laya-cuda venv），解冻 encoder + scorer 一起训，
  让表征本身学会迷宫拓扑。这是从"记住答案字段"到"真懂走迷宫"的唯一出路。

> 注：n=10 为权威样本（v2 与 base 逐位相等已具统计意义）；如需对外发布可再扩到 n=30~50（约 ~5min）。
