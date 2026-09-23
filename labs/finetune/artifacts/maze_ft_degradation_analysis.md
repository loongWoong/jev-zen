# 迷宫微调模型（model_maze_ft）性能退化根因分析报告

> 现象：微调后的 `model_maze_ft.safetensors` 经 `maze_laya.html` 的 **Model-only** 模式测试，
> **卡关位置比微调前的原版 `model.safetensors` 还早（死胡同更早、更深）**。
> 本报告给出结论、根因、证据与修复路线。

---

## 0. 结论（TL;DR）

微调**没有学会走迷宫**，而是**记住了训练集里教师标注的"剩余步数"字段**——而这个字段在真实推理时根本不存在。

三个叠加的失败因素：

1. **训练-服务分布失配（主因）**：`convert_maze.py` 的 `find_seed()` 没找回原始迷宫，
   退回 `method="reconstructed"`，训练 `state` 用的是**合成的代理文本**（位置+legal+教师每步剩余步数），
   而 `maze_laya.html` 推理时喂的是**真实迷宫 grid ASCII**（`.`/`#`/`A`/`G` 全网格）。
   打分头在代理文本上学到的"看剩余步数字段答题"模式，在真实 grid 上无对应信号 → 实际等于没学会。
2. **通道对齐门崩溃（决定"更早死胡同"）**：head-only 微调把 `max_logit_spread` 从 `5.04e-03`
   （近均匀、WARN）推到 `2.00`（过自信、FAIL）。Model-only 模式（见 §3.2）**直接采用模型选择**，
   过自信的错误方向被坚定执行 → 一错就扎进死胡同走廊，比近均匀的基线更早卡死。
3. **误导性 benchmark 掩盖了退化**：`benchmark_after_ft.json` 的 `0.453` 准确率是在**同一份 reconstructed 分布**上测的，
   测的是"代理拟合度"而非"真实迷宫能力"。诚实数字是 `benchmark_live.json` 的 `0.351`，仍只是逐步一致率，非整局可解率。

---

## 1. 复现场景与关键证据

用户命令（注意见 §6 的 `-p` 笔误）：

```
python laya_verify_cuda.py -m labs/finetune/artifacts/model_maze_ft.safetensors -p 8771
```

测试方式：`maze_laya.html` → 策略选 **Model**（只用模型，不混 rule 窄带）。

| 指标 | 原版 `model.safetensors` | 微调 `model_maze_ft.safetensors` | 来源文件 |
|---|---|---|---|
| 通道状态 | WARN (2/3) | **FAIL (0/3)** | `benchmark_before_ft.json` / `benchmark_after_ft.json` |
| `max_logit_spread` | `0.005037` | **`1.998535`** | 同上 |
| `mean_max_prob` | `0.333956` | **`0.60854`** | 同上 |
| choice top-1 准确率（同分布集） | `0.1149` | `0.4527` | 同上 |
| choice top-1 准确率（真实部署 live） | — | `0.3514` | `benchmark_live.json` |
| choice top-1 训练前→后 | `0.115` → `0.453` | — | `finetune_result.json` |
| 数据集 `method` | — | **`reconstructed`** | `finetune_result.json` / `maze_convert_meta.json` |

> 注意：`benchmark_after_ft.json` 的 `0.4527` 与 `finetune_result.json` 的 `0.453` 一致，
> 因为它们测的是**同一个 reconstructed 训练分布**——这是"拟合代理"的指标，不是"真实导航"的指标。

---

## 2. 根因一：训练-服务分布失配（主因）

### 2.1 训练侧喂的是"代理文本"，不是真网格

`labs/finetune/convert_maze.py`：

- `find_seed()`（line 240-262）穷举 seed 匹配 A* 路径；**失败则返回 `None`**。
- `main()`（line 290-298）：`found = find_seed(...)` → `None` 时 `method = "reconstructed"`，`maze = None`。
- 退化分支（line 312-321）构造的 `state` 文本是：

  ```
  Perfect maze replay (grid not recovered).
  current pos=(r,c). steps_taken=k. bfs_shortest=N.
  legal_moves: up, down, left, right
  per-move remaining-steps (teacher rd):
    up -> remaining <rem_up>
    down -> remaining <rem_down>
    ...
  ```

  其中 `rem = {k: int(-r["score"]) for r in rd.get("ranks", [])}`（line 315），
  `r["score"]` 来自 `SCENE.rule()` 的 `-bfsDist`（见 `maze_laya.html` line 1041），
  **即每个方向的 BFS 剩余步数——这本身就是"答案"**。

  → 训练集把答案以明文形式写进了 `state`。打分头只要学会"选 remaining 最小的那个方向"就能拿高分，
  根本不需要理解网格拓扑。

### 2.2 推理侧喂的是真实 grid ASCII

`maze_laya.html` `SCENE.text()`（line 969-998）构造的是**完整迷宫网格**：

```
Perfect maze 16x16 (no loops). start=(0,0) goal=(15,15).
current pos row r col c. steps_taken=k. bfs_shortest=N.
legal_moves: up, down, left, right
grid (A = agent, G = goal, # = wall), cell + right-wall:
.#.#.#....
.#.#.#.#..
A.#.#.#.G.
...（16 行完整网格）
```

`decide()`（line 365-366）把这段**真实 grid 文本** + 问题列表发给 `/api/predict`。

### 2.3 失配后果

- 训练：`state` 含 `per-move remaining-steps (teacher rd)` 字段 → 模型学会了"读剩余步数字段"。
- 推理：`state` 是真实 grid，**没有那个字段** → 模型赖以答题的信号消失。
- 结果：在真实 grid 上，打分头对迷宫结构几乎无判别力，`benchmark_live` 仅 `0.351`（勉强高于随机 0.25），
  且这一致率是建立在"沿教师最优路径的 148 个状态"上测的——一旦模型**自己**走错一步离开该路径，
  就进入训练时从未见过的 off-path 状态，信号彻底归零（见 §3.2）。

> 验证依据：`finetune_result.json` → `"method": "reconstructed（seed 未在 0..200000 + 常见种子中匹配到，无法恢复原始迷宫网格）"`，
> `"state": "用轨迹字段重建：位置 + legal + 各方向剩余步（决策等价代理表示，非原始 grid ASCII）"`。

---

## 3. 根因二：通道对齐门崩溃 → 过自信死胡同（决定"更早卡关"）

### 3.1 校准从"近均匀"变成"过自信"

| | `max_logit_spread` | `mean_max_prob` | 通道 |
|---|---|---|---|
| 微调前 | `0.005037` | `0.334` | WARN (2/3) |
| 微调后 | **`1.998535`** | **`0.609`** | **FAIL (0/3)** |

（`finetune_result.json`：`alignment_before` WARN / `5.04e-03` → `alignment_after` FAIL / `2.00e+00`。）

单任务 head-only 训练放大了 4 个选项间的 logit 差异，把原本近均匀的跨问题通道标定**打爆**成过自信。

### 3.2 Model-only 模式直接执行模型选择 → 错就扎进死胡同

`maze_laya.html` `decide()` 的策略分支（line 402-426）：

```js
if (strategy === 'model' && modelChoice !== null) {
  source = 'model';
  if (legal.indexOf(modelChoice) >= 0) applied = modelChoice;   // 直接用模型选择
  else { fallback = true; applied = rd.choice; }
}
```

`mix` 模式有 rule 候选窄带约束（line 407-425，只在 band 内破平，模型选不到带外危险动作）；
但 **Model-only 模式无此安全带**，`applied = modelChoice` 直接落子。

叠加 §2.3 的 off-path 信号归零 + §3.1 的过自信：

- 微调模型在真实 grid 上实际逐步准确率仅 ~35%，即 **~65% 的路口选错方向**；
- 因其 `max_prob≈0.61`、logit spread≈2.0，**它对自己选错的方向也高度自信**，坚定地扎进死胡同走廊；
- 一旦离开教师路径（极易发生），进入未见过的 off-path 状态，打分头无信号、仍过自信地乱指 → 迅速耗尽步数或在死胡同尽头卡死。

相比之下，原版模型 `max_logit_spread≈5e-3`、`mean_max_prob≈0.334`（≈随机均匀），
虽也常选错，但**不"自信地"朝错误方向一根筋走**——
在 Model-only 下它更像是低幅度的随机游走，反而比"过自信的错"更不容易在开局就扎进深层死胡同。
这正是"微调后比微调前更早卡关"的直接机制。

---

## 4. 根因三（加剧因素）：离线轨迹只在最优路径 + 只喂 move 问题

- **训练状态全部在教师最优路径上**（`benchmark_before.json` 的 148 步 A* 回溯轨迹）。
  模型从未见过"走错后如何挽回"的状态，因此一旦偏离路径就无能为力——与 §3.2 的 off-path 崩溃互为因果。
- **只喂了 `move` 问题**：`convert_maze.py` `questions = [move_question()]`（line 303），
  而 `maze_laya.html` `SCENE.questions()`（line 1000-1012）实际发送 **`move`(choice) + `closeness`(score)** 两个问题。
  共享打分头在推理时多处理了一个训练时没见过的 `closeness` 文本（虽 `decide` 只用 `qs[0].id`=move 决策，line 379-384，
  但仍是分布外的额外负担）。
- 标签来源本身一致（`dataset.py` `derive_labels`：`labels.move.key = rd.choice`，即教师 A* 选择），
  **不是**标签错误；问题出在 `state` 文本与推理分布不一致。

---

## 5. 关键证据清单（带路径）

| 证据 | 路径 / 位置 |
|---|---|
| 数据集 `method=reconstructed`、state 为代理文本 | `labs/finetune/artifacts/finetune_result.json` (line 9, 12) |
| 训练 state 含教师剩余步数字段 | `labs/finetune/convert_maze.py` (line 312-321, 315) |
| `find_seed()` 失败→reconstructed 退化 | `labs/finetune/convert_maze.py` (line 240-262, 290-298) |
| 推理用真实 grid ASCII | `maze_laya.html` `SCENE.text()` (line 969-998) |
| 推理发送 `move`+`closeness` 两问 | `maze_laya.html` `SCENE.questions()` (line 1000-1012) |
| Model-only 直接采用模型选择（无安全带） | `maze_laya.html` `decide()` (line 403-406) |
| 对齐门 WARN→FAIL、spread 5e-3→2.0 | `benchmark_before_ft.json` / `benchmark_after_ft.json` / `finetune_result.json` (line 32-33) |
| 同分布测 0.453 vs 真实部署 0.351 | `benchmark_after_ft.json` (line 32) / `benchmark_live.json` (line 32) |

---

## 6. ⚠️ 命令行笔误：`-p 8771` 不会被识别

`laya_verify_cuda.py` 仅定义了长选项（line 1415）：

```python
ap.add_argument("--port", type=int, default=0, help="起 HTTP 服务（复用原 index.html）")
```

**没有 `-p` 短选项**。所以 `-p 8771` 会报 `unrecognized arguments: -p 8771`。
正确写法是 `--port 8771`。

> 推论：你贴出的这条命令本身应已报错；`maze_laya.html` 实际能测通，
> 说明当时已有某个 `/api/predict` 服务在运行（可能是另一个进程或一次成功起的 `--port`）。
> 复现与分析时请用 `--port`。

---

## 7. 修复路线（按 P0/P1/P2）

### P0 — 先拿到"真实分布"的训练/评测数据（不做这个，微调必崩）
- **恢复原始迷宫 grid**：重跑 `convert_maze.py` 前，先定位真实 seed/RNG。
  - 当前 `find_seed()` 默认 `search=200000`（line 273）未命中 → 要么真实 seed 超出范围，要么
    `maze_laya.html` 实际生成迷宫用的 RNG/尺寸/起点与 `convert_maze.py` 复刻的不完全一致。
  - 排查点：确认 `maze_laya.html` 的 `genMaze`/`mulberry32`/起点/尺寸与 `convert_maze.py`（line 41-95）逐行一致；
    必要时把 `search` 上限调大（如 `2_000_000`）或换用 `maze_laya.html` 真实运行导出的 trace。
  - 目标：让 `maze_convert_meta.json` 的 `method` 变为 **`ground_truth`**，使训练 `state` = 真实 `SCENE.text()`（grid ASCII）。
- **加一个"真实 grid"评测脚本**：用真实 grid 文本评估整局可解率（到达 goal 的步数/占比），
  替代当前只在教师轨迹上测逐步一致率的 `benchmark.py`，避免再次被 0.453 误导。

### P1 — 微调前先做 Phase-0 通道对齐
- `finetune_result.json` 的 `interpretation` 已指明：单任务 head 训练会牺牲通用跨问题校准。
- 在 head FT 之前，先在 **prompt 模板 / head 读出位置** 维度做对齐搜索（参考 `alignment_report.json` 的 25 配置结论：
  cfg 可切换维度不足以过通道探针，真实未对齐在 prompt 模板 / head 读出位置），
  把通道门从 WARN/FAIL 拉到稳定 PASS，再跑 head-only FT，避免 spread 飙到 2.0。

### P2 — 训练与推理对齐 + 评测口径统一
- 训练时一并喂 `move`+`closeness`（与 `SCENE.questions()` 对齐），或对 closeness 单独处理，避免分布外问题。
- 加入 **off-path 恢复样本**：在教师轨迹的每个状态施加一次随机扰动再续走，让模型学到"走错后如何挽回"，
  而非只在最优路径上过拟合。
- 报告口径：区分"同分布逐步一致率"与"真实 grid 整局可解率"，后者才是用户关心的"卡关早晚"。

---

## 8. 一句话总结

> `model_maze_ft` 在 **代理文本（含答案字段）** 上过拟合、在 **真实 grid** 上零迁移，
> 且 head-only 微调把校准从近均匀打爆成过自信；Model-only 模式又**直接执行**这个过自信的错误选择——
> 于是它比近均匀的基线**更早、更坚定地扎进死胡同**。修复优先级：P0 先恢复真实 grid 数据并建真实评测，
> P1 先做通道对齐，P2 再补问题/状态分布对齐。
