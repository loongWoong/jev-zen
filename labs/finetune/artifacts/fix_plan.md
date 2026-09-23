# 微调数据集 / 微调管线 / benchmark 修正方案（准确版）

> 目标：让 `model_maze_ft` 学会**真实迷宫导航**，而不是记忆训练集里的"剩余步数字段"。
> 本文给出三个文件（convert_maze.py / finetune_run.py / benchmark.py）的**逐处修正**，
> 每个改动都对应已读代码里的真实函数与行号，可直接落地。

---

## 0. 总体策略（三段式，按优先级）

| 优先级 | 文件 | 修正点 | 解决的问题 |
|---|---|---|---|
| **P0** | `convert_maze.py` | 用轨迹**直接重建迷宫拓扑**，抛弃 `find_seed()` 种子穷举 | 训练-服务分布失配（根因） |
| **P0** | `convert_maze.py` | 同时喂 `move` + `closeness` 两问 | 推理发两问、训练只喂一问的失配 |
| **P1** | `finetune_run.py` | 加 **spread 惩罚 + 权重衰减 + 留出验证集早停** | 通道对齐门从 WARN→FAIL、过自信死胡同 |
| **P2** | `benchmark.py` | 新增 **真实 grid 整局可解率** 评测（替代同分布逐步一致率） | 0.453 误导性指标 |
| 信息 | `align.py` | Phase-0 对齐仅作参考（见 §5 说明） | cfg 维度不足以过通道门 |

---

## 1. P0 · 修正数据集：抛弃 `find_seed`，从轨迹直接重建网格

### 1.1 为什么 `find_seed()` 必然失败（已定位真实机制）

`convert_maze.py` `find_seed()`（line 240-262）在匹配前先过滤：

```python
if bfsDist(maze, START, GOAL) != optimal_target:
    continue
```

`optimal_target` 来自轨迹 metrics 的 `"BFS 最短"`（line 281-285），缺失时默认 `148`。
而 `benchmark_before.json` 的 148 是**轨迹步数**，不是 BFS 最短距离。一旦 `optimal_target`
与真实迷宫的 `bfsDist` 不等（绝大多数种子都不等），**所有种子被过滤掉 → 返回 None → reconstructed**。
即使把 `--search` 调到 200 万也救不回来，因为问题在过滤条件，不在搜索范围。

### 1.2 准确修复：轨迹的每一步移动已经编码了迷宫拓扑

`reconstruct_positions()`（line 216-237）已经解析出每步落点 `pos[i]=(r,c)` 与 `applied[i]` 方向。
`pos[i-1] →(d) pos[i]` 成立 ⇒ 单元 `cur` 与 `nxt` 之间的墙是**开通**的。
由此可**不依赖任何 seed / RNG / BFS 过滤**，逐边重建出与推理时完全一致的迷宫。

在 `convert_maze.py` 中新增（放在 `find_seed` 之后即可）：

```python
OBIT = {"up": 2, "down": 1, "left": 8, "right": 4}

def reconstruct_maze_from_trace(pos, applied, W=W, H=H):
    """从轨迹的 (落点序列 + 每步方向) 直接重建迷宫 link 数组。
    不依赖 seed / RNG / BFS 过滤，保证与 maze_laya.html 的真实网格逐边一致。"""
    link = [0] * (W * H)
    for i in range(1, len(pos)):
        d = applied[i - 1]
        cur = pos[i - 1][0] * W + pos[i - 1][1]
        nxt = pos[i][0] * W + pos[i][1]
        link[cur] |= BIT[d]
        link[nxt] |= OBIT[d]
    return {"W": W, "H": H, "link": link}
```

### 1.3 改写 `main()` 的状态构造分支（line 301-336）

把 `found = find_seed(...)` 整段替换为直接重建，且**用已移植的 `text()`**（line 173-199，
与 `maze_laya.html` `SCENE.text()` 逐字节一致）生成真实 grid 文本：

```python
pos, _applied = reconstruct_positions(trace)
optimal_target = None
for ln in trace:
    for m in ln.get("metrics", []):
        if m.get("k") == "BFS 最短":
            optimal_target = int(m["v"])
optimal_target = optimal_target or len(pos) - 1

print("[convert] 从轨迹直接重建迷宫拓扑（无需 seed 穷举）…")
maze = reconstruct_maze_from_trace(pos, _applied)          # ← 关键
recon_opt = bfsDist(maze, START, GOAL)
method = "ground_truth"
print(f"[convert] 重建迷宫 bfs_shortest={recon_opt}（轨迹声明 {optimal_target}）"
      f"{' ✅' if recon_opt == optimal_target else ' ⚠️不一致，请核对轨迹'}")

_, tok, pred = build(args.model)
questions = [move_question(), closeness_question()]        # ← 见 §1.4
samples = []
for idx, ln in enumerate(trace, start=1):
    state_pos = pos[idx - 1]
    rd = ln.get("rd") or {}
    sp = state_pos[0] * W + state_pos[1]
    lm = legal(maze, sp)
    stxt = text(maze, sp, idx - 1, recon_opt, lm)          # ← 真实 grid ASCII
    sample = {"scene": "maze", "seed": 0, "step": idx,
              "state": stxt, "questions": questions}
    labels = derive_labels([Question(**q) for q in questions], rd)
    sample["labels"] = labels
    sample = tokenize_sample(pred, sample)
    if "move" not in sample["labels"] or "key" not in sample["labels"]["move"]:
        raise RuntimeError(f"step {idx} 缺少 move 标签：{sample['labels']}")
    samples.append(sample)
```

> `meta` 里把 `method` 写死 `"ground_truth"`，`seed` 写 `null`（拓扑来自轨迹，无种子）。
> 验证：`maze_convert_meta.json` 的 `method` 应为 `ground_truth`，且 `recon_opt == optimal_target`。

### 1.4 补 `closeness` 问题（与 `maze_laya.html` `SCENE.questions()` 对齐，line 1008-1010）

在 `convert_maze.py` `move_question()` 后加：

```python
def closeness_question():
    return {
        "id": "closeness", "type": "score",
        "instructions": "How far is the agent from the goal?",
        "criteria": ["at the goal", "close", "halfway", "far"],
    }
```

`dataset.derive_labels`（line 54-64）已支持 `type=="score"` → 产出 `expected_value`，
`tokenize_sample` 也已能 tokenize 多问题。这样训练集的分布与推理（发两问）一致。

### 1.5 可选增强：off-path 恢复样本（P2，防"一错就崩"）

仅沿教师最优路径训练，模型不会"走错后挽回"。在生成完 on-path 样本后，额外：
对每步状态随机选一个**非最优**合法方向走一步，再用移植的 `bfsDist` 重算剩余步并继续沿最优走，
把"偏离状态→最优恢复"的样本一并写入。这一步让模型在 Model-only 模式下偏离路径时仍能拉回。
（实现见 §4 的 `rollout_recovery`，可独立脚本追加，不阻塞 P0。）

---

## 2. P1 · 修正微调管线：给打分头加"校准围栏"

`finetune_run.py` 当前 `lr=0.02 × 80 epoch` 在 148 样本上把 `max_logit_spread` 从 `5e-3` 推到 `2.0`
（line 242-264 的训练循环 + `Adam` line 116-133）。需要三道围栏：

### 2.1 spread 惩罚项（直接压制过自信）

在训练循环里给每个样本的损失加一项：

```python
# 在 scorer_forward 之后
spread = logits.max() - logits.min()
pen = max(0.0, spread - SPREAD_CAP)          # SPREAD_CAP 取 0.3（接近基线 5e-3，留出判别空间）
tot_loss += LAMBDA * pen
# 反向：relu(spread-CAP) 对 logits 的梯度只作用在 max/min 两项
if spread > SPREAD_CAP:
    dlogits[int(np.argmax(logits))] += LAMBDA
    dlogits[int(np.argmin(logits))] -= LAMBDA
```

`SPREAD_CAP=0.3, LAMBDA=0.02`。这把 logit 极差锁在 ~0.3 内，避免冲到 2.0。

### 2.2 权重衰减 + 降 LR

- `args.lr` 默认 `0.02 → 0.003`（148 样本不需要大 LR）。
- `Adam.step` 之后加衰减：`for k in P: P[k] = P[k] - upd[k] - args.lr * args.wd * P[k]`
  （`--wd` 默认 `1e-4`）。
- 加命令行：`--calib-penalty`（默认 0.02）、`--spread-cap`（默认 0.3）、`--wd`（默认 1e-4）。

### 2.3 留出验证集 + 早停（防 on-path 过拟合假装有效）

- 载入样本后按 `step` 奇偶或前 80% 切分：`train_idx / val_idx`（如 120 训练 / 28 验证）。
- 每个 epoch 末用验证集算 `val_spread = max over val of (max(logits)-min(logits))` 与 `val_acc`。
- 若 `val_spread > FAIL_SPREAD`（如 0.5）**或** `val_acc` 连续 N 个 epoch 不升 → 还原最佳 `P` 并 `break`。
- 最佳 `P` 以"`val_spread < cap` 且 `val_acc` 最高"为准，而非最后一轮。

> 改动都在 `finetune_run.py` 的 `main()`（line 164-302）：切分 markers_list/gold_idx、改训练循环、加早停、改保存逻辑。

---

## 3. P2 · 修正 benchmark：真实 grid 整局可解率

`benchmark.py` 的 `eval_label_accuracy`（line 30-59）在 tokenized 数据集上测 top-1——一旦数据集是
reconstructed 就是测代理拟合。新增 `eval_episodes`，**用真实迷宫整局对弈**直接回答"卡关早晚"：

```python
def eval_episodes(model_path, n_mazes=20, seed_base=1000, size=16):
    """真实 grid 整局评测：模拟 Model-only 对弈，测可解率/步数/死胡同/首次错步深度。"""
    st, tok, pred = build(model_path)
    from convert_maze import genMaze, mulberry32, legal, bfsDist, text, START, GOAL, W, H
    solved, steps_list, deadends, first_wrong = [], [], [], []
    for k in range(n_mazes):
        maze = genMaze(W, H, mulberry32((seed_base + k) * 2654435761 & 0xFFFFFFFF))
        pos = START; steps = 0; optimal = bfsDist(maze, START, GOAL); fw = None
        while pos != GOAL and steps < W * H * 4:
            lm = legal(maze, pos)
            if not lm:
                deadends.append(1); break
            stxt = text(maze, pos, steps, optimal, lm)
            qs = [move_question()]
            out = pred.model.forward(*pred.build(stxt, qs)[c] for c in ("ids","marker_pos","marker_type"))
            # 取 move 的 4 选项 argmax
            logits = out["logits"][:4]
            choice = ["up","down","left","right"][int(np.argmax(logits))]
            # 教师最优
            best = min(lm, key=lambda d: bfsDist(maze,
                pos//W + DIR[d][0] ... , GOAL))   # 见下方注
            if fw is None and choice != best:
                fw = steps
            pos = step_in_maze(maze, pos, choice); steps += 1
        else:
            if pos == GOAL: solved.append(1)
        steps_list.append(steps); first_wrong.append(fw if fw is not None else steps)
    return {"n": n_mazes, "solved_rate": mean(solved), "avg_steps": mean(steps_list),
            "deadend_rate": mean(deadends), "avg_first_wrong_depth": mean(first_wrong)}
```

> 注：`best` 计算直接用 `bfsDist(maze, 邻居, GOAL)` 取最小，等价于 `SCENE.rule()` 的 bfs 分支；
> `step_in_maze` 按方向更新 pos（与 `maze_laya.html` `step()` 同语义）。这两个小工具函数加在
> `convert_maze.py` 里即可复用。

`run()` 里把通道门 FAIL 作为硬告警，并在报告加 `"episodes": eval_episodes(...)`。
**判断修复是否有效的金标准**：`solved_rate` 提升 + `avg_first_wrong_depth` 比原版**更大**（卡关更晚）
+ 通道门**不退化**（spread 仍 < 0.5）。

---

## 4. 端到端执行顺序（命令序列）

```bash
# 0) （信息性）Phase-0 对齐，看当前通道门落在哪（cfg 维度不足以过门，仅记录）
python labs/finetune/align.py --model model.safetensors

# 1) P0 重建真实 grid 数据集（method 应为 ground_truth，且发两问）
python labs/finetune/convert_maze.py \
    --trace labs/finetune/artifacts/benchmark_before.json \
    --out    labs/finetune/artifacts/maze_dataset.jsonl

# 2) P1 带校准围栏训练（输出 v2，避免覆盖旧产物）
python labs/finetune/finetune_run.py \
    --dataset labs/finetune/artifacts/maze_dataset.jsonl \
    --out-model labs/finetune/artifacts/model_maze_ft_v2.safetensors \
    --epochs 80 --lr 0.003 --wd 1e-4 --calib-penalty 0.02 --spread-cap 0.3 --early-stop

# 3) P2 真实 grid 整局评测（对比原版与 v2）
python labs/finetune/benchmark.py --model model.safetensors          --tag before_v2
python labs/finetune/benchmark.py --model labs/finetune/artifacts/model_maze_ft_v2.safetensors --tag after_v2
```

---

## 5. 关于 `align.py`（Phase-0）的定位

`alignment_report.json` 的 `note` 已明确：可经 cfg 切换的维度
（`type_emb_stage/site`、`temperature` 置换、`max_prefixes`）**不足以让 channel_probe 通过**；
真实未对齐在 **prompt 模板 / head 读出位置**，需 monkeypatch 或 head 重训练。
因此：

- `align.py` 仍跑，但结论当作"已知背景"，**不阻塞 P0/P1**。
- 真正的校准保证来自 §2 的 spread 惩罚（把极差锁在 0.3 内），它比依赖对齐搜索更可控。
- 若后续想根治，再走 `align.py --try-template` 的 prompt 模板 hook 或 head 全量重训（`pipeline.py` 的 torch 路径），
  那是独立的大工程，不在本次 P0/P1/P2 范围内。

---

## 6. 验收口径（如何确认修好了）

| 检查项 | 修复前 | 修复后期望 |
|---|---|---|
| `maze_convert_meta.json` 的 `method` | `reconstructed` | **`ground_truth`** |
| 训练 `state` 内容 | 含 `per-move remaining-steps (teacher rd)` | **真实 grid ASCII**（与 `maze_laya.html` 一致） |
| `benchmark_after` 通道门 | FAIL (spread 2.0) | **不退化**（spread < 0.5） |
| 真实 grid 整局 `solved_rate` | 低（且很快死胡同） | **更高** |
| `avg_first_wrong_depth` | 很小（早卡关） | **更大**（卡关更晚） |
| `benchmark` 0.453 指标 | 误导性（同分布代理） | 被真实可解率取代 |

> 一句话：**P0 让训练分布 = 推理分布；P1 让校准不崩；P2 用整局可解率替代逐步一致率来诚实度量。**
