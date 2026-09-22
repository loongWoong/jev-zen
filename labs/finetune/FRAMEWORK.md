# Laya 决策模型微调框架设计

> 版本 v0.1 · 2026-09-22 · 基于 `laya_verify.py` 真实 API 面落地
> 配套脚手架：`labs/finetune/{align,dataset,pipeline,benchmark}.py`

---

## 0. 一句话结论

**在「决策通道未对齐」之前，任何微调都是空中楼阁。** 当前 `laya_verify.py` 自检报告：语义项 `BLOCKED`、选项间 `logit_spread ≤ 5.04e-03`（`Verifier.channel_probe` 未通过）。这意味着 `<mask>` 位置的读出值尚未承载真实的 typed-question 语义，监督信号本身就是噪声。

因此本框架把 **Phase 0（通道对齐）** 作为前置硬门，数据集 / 微调管线 / benchmark 三者都挂在它后面。对齐一旦通过，后三步即可在同一套张量（`ids / marker_pos / marker_type`）与同一套 loss 上闭环。

---

## 1. 现状与已知未知（设计的事实底座）

### 1.1 已确认的事实（来自 `laya_verify.py` 反推 + 官方 config.json）
| 维度 | 值 | 来源 |
|---|---|---|
| 主干 | mmBERT-base 风格编码器，22 层，hidden=768，12 heads，GeGLU inter=1152 | 张量形状 |
| 决策头 | `scorer: 768→768→768→1`（index 2 = 激活）；`act_head: 768+4→256→2`；`type_emb[3,768]`；`temperature[F32,3]` | 张量形状 |
| 位置编码 | RoPE，`rope_theta=160000`（官方 config.json，旧默认 10000 偏差 16 倍） | `DEFAULT_CFG` |
| 注意力 | `alternating3`（每 3 层 1 层 global，其余 sliding 半径 64） | config.json `layer_types` |
| Norm | encoder 1e-5 无 bias；head 1e-5 有 bias | 张量有无 bias |
| 推理范式 | 非自回归；`<bos> state <sep> [Question: … <option> <mask> … <sep>] × N`；**单次前向**回答全部 `<mask>` | `Predictor.build` / `forward_passes=1` |
| 权重 | `model.safetensors` 614 MiB / 170 张量 / 321.9M 参数 | `SafeTensors` |

### 1.2 已知未知（= 微调前必须锁定的 3 个变量）
1. **prompt 模板的精确措辞**（`<Question: …>` 前后、`<option>` 与 `<mask>` 的相对位置、`max_prefixes: 6` 的语义）。
2. **`type_emb` 注入位置**：`type_emb_stage ∈ {input, pre_head}`，`type_emb_site ∈ {markers, all}`。
3. **head 读出语义**：`act_head` 的 4 个标量特征（`type one-hot(3) + log1p(n_opt)/log64`）与 head 激活（`gelu` 假设）。

附带次级未知：primitive→temperature 索引顺序（假设 `choice=0/score=1/noul=2`）。

> 这些正是 `ASSUMPTIONS` 列表里标「权重未体现」的项。它们不是超参，是**结构前提**——错了，loss 永远不降。

---

## 2. 框架总览

```
labs/finetune/
├── FRAMEWORK.md        # 本文档
├── align.py            # Phase 0：通道自动对齐（网格/BO 搜索 3 未知，复用 channel_probe）
├── dataset.py          # 数据集：16 场景 + 2048 → (state, typed_questions, rule 教师标签)
├── pipeline.py         # 微调管线：head-only(numpy) / full(torch+CUDA) 两后端
├── benchmark.py        # 评测：对齐门 + 标签准确率 + 任务迁移(goal-rate) + 延迟
└── artifacts/          # 对齐报告 / 数据集 / 检查点 / 评测报告（运行时生成）
```

数据流：
```
SCENE.init/legal/step/goalOk  ──┐
SCENE.text / questions / rule  ─┴─► dataset.py ─► JSONL + tokenized cache(ids/marker_pos/marker_type)
                                        │
align.py (Phase 0) ── 锁定 3 未知 ──► 写入 config overlay ──┐
                                        │                    │
                                        └──► pipeline.py ◄───┘
                                                │ 训练
                                                ▼
                                        benchmark.py ──► before/after 报告
```

---

## 3. 数据集（dataset.py）

### 3.1 出处与教师信号
- **场景源**：`labs/` 下 16 个游戏场景（snake / tetris / minesweeper / sokoban / puzzle15 / lightsout / tictactoe / connect4 / othello / maze / flappy / breakout / elevator / traffic / stock）+ `2048_laya.html` 主场景。
- **场景接口**（来自 `labs/core.js` 末尾约定，已逐文件确认）：
  - `init(seed, sopt) → state`
  - `legal(state) → [key,…]`
  - `step(state, key, sopt) → {info}`（就地推进）
  - `text(state, legal, sopt) → state 文本`（即喂给模型的 `state`）
  - `questions(state, legal, sopt) → [{id,type,instructions,criteria}]`（typed primitives）
  - `rule(state, legal, sopt) → {choice, ranks:[{key,label,score,detail,risky}], note}`（**教师标签**）
  - `goalOk(state, sopt) → bool`（批量验证目标）
  - `maxSteps(sopt) → int`
- **教师 = rule 策略**：`rule.choice` 给出每个 `choice` 问题的 gold 选项键；`ranks` 给出每个选项的软分数。

### 3.2 样本 schema（与模型输入严格同构）
```json
{
  "scene": "snake", "seed": 7, "step": 12,
  "state": "<SCENE.text 输出>",
  "questions": [ {"id":"q0","type":"choice","instructions":"…","criteria":{"UP":"上","DOWN":"下"}} ],
  "labels": {
    "q0": {"type":"choice","key":"UP","rank_score":0.81},
    "q1": {"type":"score","expected_value":0.62},
    "q2": {"type":"noul","p_yes":0.15}
  },
  "alignment_tag": {"channel_probe_status":"PASS","max_logit_spread":0.31}
}
```

### 3.3 标签派生
| primitive | 标签 | 公式 |
|---|---|---|
| `choice` | gold 选项键 + 排序分 | `key = rule.choice`；`rank_score = softmax(ranks.score)` |
| `score` | 期望值 | `EV = Σ p_i·s_i`，`p_i` 取自 `ranks` 归一化 |
| `noul` | P(yes) | `p_yes = risky?0.15:0.85`（rule 暴露风险标志时） |

### 3.4 生成协议
- 对每个 `scene × seed`（种子网格，例 200 种子 × 上限 `maxSteps`）跑 `playEpisode` 式循环（`init→legal→questions→rule→step→goalOk`）。
- 每步若 `questions` 非空则产出 1 条样本；按场景配额上限防止分布失衡。
- **切分按 seed 不按 step**（防泄漏）；**留出 2 个场景（建议 `stock`、`traffic`）作 OOD 泛化测试集**。

### 3.5 分词缓存（关键）
每个样本经 `Predictor.build(state, questions)` 预分词，落盘 `ids / marker_pos / marker_type / spans`：
- 训练直接读张量，避免重复编码；
- **把训练时使用的 prompt 模板冻结进缓存**——对齐阶段选定的模板就钉死在数据集上，pipeline 不必再猜。

### 3.6 对齐闸门
数据集携带生成时的 `Verifier.channel_probe` 结果。若 `status != PASS`，数据集标记为 `quarantined`：允许训练但报告显著标注「标签可能位置偏置」。

---

## 4. 微调管线（pipeline.py）

### 4.1 双后端
| 后端 | 训练范围 | 依赖 | 定位 |
|---|---|---|---|
| `head-only` | 仅 `scorer` / `act_head` / `type_emb` / `temperature`，编码器冻结 | numpy（手工 autodiff / 有限差分） | **对齐恢复 + 快速实验**；头仅 ~1.5M 参数，CPU 可跑 |
| `full` | 编码器 + 头 端到端 | PyTorch + CUDA（RTX 3080，py3.12 venv） | 对齐通过后的**质量提升** |

`full` 后端必须把 `laya_verify` 的 RoPE(theta=160000)、`alternating3`、GeGLU 用 torch 重实现，并与 `laya_verify_cuda.py` 前向做**数值对齐测试**（逐张量误差 < 1e-4）后才能训练。

### 4.2 Loss（在 `<mask>` 位置计算）
```
L_choice = CE( scorer_logits / T[primitive] , one_hot(gold_key) )
L_score  = MSE( scorer_logits_EV , EV(ranks) )  + λ·ranking_KL
L_noul   = BCE( act_head_yes_prob , p_yes )
L = w_c·L_choice + w_s·L_score + w_n·L_noul
```
可选 **rule 策略软标签蒸馏**（KD），在小数据 regime 稳定训练。

### 4.3 超参探针空间（= Phase 0 锁定的 3 未知，作为可搜索维）
- prompt 模板： `{canonical, +variant_A, +variant_B}`（枚举候选模板）
- `type_emb_stage ∈ {input, pre_head}`，`type_emb_site ∈ {markers, all}`
- `head_activation ∈ {gelu, relu, tanh-gelu}`
- primitive→temperature 索引置换（6 种）

搜索由 `align.py` 驱动，评分 = `channel_probe`（`hits==n && max_logit_spread > 1e-2`）。**胜出配置**写入 `config overlay`（随检查点保存），成为数据集冻结模板与训练默认。

### 4.4 检查点
- 存增量或全量 `safetensors`；
- 附带 `config.json` overlay 记录「已解析的 3 未知」，保证可复现。

---

## 5. Benchmark（benchmark.py）

### 5.1 对齐门（前置）
`Verifier.channel_probe()` 必须 `PASS`，否则下游指标不可信，直接报错退出。

### 5.2 指标
| 指标 | 来源 | 说明 |
|---|---|---|
| 对齐状态 | `Verifier.channel_probe` | 硬门：`hits==n && max_logit_spread>1e-2` |
| 延迟 | `Verifier.benchmark` | 1/5/10 问题，ms/问题 |
| 标签准确率 | 测试集 top-1 vs `rule.choice` | 微调目标代理 |
| 任务迁移 | 各场景 goal-rate | 复用 `labs/core.js` `runBatch` 逻辑（Python 重实现 / `node` 子进程驱动 `<scene>_laya.html` 的 Scene 导出） |
| 泛化 | 2 个留出场景单独报 | OOD |

### 5.3 对比维度
- 策略：`rule` vs `model` vs `mix`（沿 `core.js` 三策略）。
- 时间：微调**前 vs 后** → 输出 before/after 增量表（JSON + markdown）。

---

## 6. 落地优先级（P0/P1/P2）

- **P0**：`align.py` 跑通 Phase 0，给出当前 `channel_probe` 实测（含 3 未知的网格扫描结果）。这是全框架的闸门。
- **P1**：`dataset.py` 接 16 场景产出 JSONL + 分词缓存；`benchmark.py` 复用 `channel_probe` + `benchmark` 出 before 基线。
- **P2**：`pipeline.py` 先 `head-only` 跑通端到端（对齐恢复），再上 `full` torch+CUDA 质量提升；OOD 泛化测试。

---

## 7. 运行（真实环境）
```bash
# 模型在 Downloads/jev-zen，worktree 内无权重
PY=~/.workbuddy/binaries/python/envs/default/bin/python
MODEL=/c/Users/Administrator/Downloads/jev-zen/model.safetensors

# Phase 0
$PY labs/finetune/align.py --model $MODEL --grid small

# 数据集
$PY labs/finetune/dataset.py --model $MODEL --scenes snake,tetris --seeds 200

# 评测基线
$PY labs/finetune/benchmark.py --model $MODEL

# 微调（head-only）
$PY labs/finetune/pipeline.py --model $MODEL --backend head-only --epochs 8
```
