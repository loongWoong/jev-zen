# Laya 迷宫决策实验分析报告

## 1. 实验目的

本次实验通过构造 10×10 无环 Perfect Maze，测试 Laya 类型化决策模型在**连续状态、环境约束和多步路径选择**场景下的决策能力。

实验重点观察三个问题：

1. Laya 能否根据当前状态正确选择下一步动作；
2. Laya 输出的概率是否能够反映决策可信度；
3. 当决策需要连续执行多个步骤时，模型是否能够避免回退、重复访问和死循环。

实验同时用于验证一个更基础的假设：

> **Laya 更适合作为“System-1 快速决策器”，还是可以独立承担需要历史记忆和多步规划的 Agent 决策任务。**

---

## 2. Laya 决策机制概述

Laya 的核心形式并不是传统语言生成，而是：

```text
State + Typed Questions
        ↓
Encoder
        ↓
Decision Head
        ↓
Structured Decision
        +
Probability / Confidence
```

官方将 Laya 定义为非自回归 System-1 决策模型，支持 `choice`、`score`、`noul` 三类决策原语。对于 `choice`，模型对每个候选选项计算分数，再形成候选概率分布，而不是生成自然语言答案。其公开架构采用 ModernBERT 编码器和专门的 decision head。

因此，Laya 的核心职责更接近：

> **给定当前 State，对预先定义好的有限候选答案快速做一次决策。**

它本身并不等价于搜索算法、路径规划器或者带长期记忆的 Agent Loop。

---

## 3. 本次实验场景

实验环境：

```text
Perfect maze 10x10
无环
start = (0,0)
goal  = (9,9)
```

实验过程中某一步的状态：

```text
current pos = (1,4)
steps_taken = 23
bfs_shortest = 70
legal_moves = down, right
```

模型需要回答：

```text
Question:
Choose the direction that follows the shortest path to the goal.

Options:
up
down
left
right
```

同时附加一个：

```text
score:
How far is the agent from the goal?

0 = at the goal
1 = close
2 = halfway
3 = far
```

---

## 4. 实验核心结果

### 4.1 模型在当前状态下高置信度选择了非法动作

模型输出：

```text
up      0.952685
down    0.042085
left    0.004940
right   0.000289
```

confidence：

```text
0.952685
```

但是输入 State 已经明确告诉模型：

```text
legal_moves: down, right
```

也就是说：

```text
up = 非法动作
left = 非法动作
```

模型却以 **95.27%** 的概率选择 `up`。

这是本次实验最关键的结果。

它说明当前 checkpoint 在该任务上并没有把 `legal_moves` 建立成可靠的决策约束。

因此：

> **Laya 输出的 probability 是模型决策分布的置信程度，不等价于环境意义上的“动作合法性概率”。**

这一区分对于后续工程设计非常重要。

---

### 4.2 模型决策不是随机，而是“高度确定地错”

本次 logits：

```text
up     -49.1940
down   -58.5528
left   -64.9797
right  -73.4929
```

logit spread：

```text
24.298893
```

up 和 down 的 logit 差：

```text
9.3588
```

结合当前 temperature=3，模型仍然形成了明显的概率优势：

```text
up ≈ 95.3%
down ≈ 4.2%
```

因此，这个实验不是：

```text
模型不知道答案
→ 随便猜
```

而更接近：

```text
模型形成了稳定而错误的决策边界
→ 并且非常自信
```

所以问题的核心不是简单的 confidence 不足，而是：

> **Decision Boundary 与迷宫任务的正确决策边界不一致。**

---

### 4.3 连续决策形成了明显的 2-cycle 死循环

历史记录：

```text
#20 down
#21 up
#22 down
#23 up
#24 down
```

对应位置：

```text
(1,4) ↔ (2,4)
```

形成：

```text
(1,4)
   ↓
(2,4)
   ↑
(1,4)
   ↓
(2,4)
   ↑
...
```

这是典型的 **2-cycle**。

更重要的是，这不是一次偶发错误，而是稳定重复：

```text
up
down
up
down
up
down
```

因此可认为模型已经进入一个稳定局部策略：

```text
π((1,4)) = up
π((2,4)) = down
```

只要 State 的核心语义保持不变，模型就会持续复现这一策略。

---

## 5. 为什么会出现死循环

### 5.1 当前模型本质上更接近一个无记忆策略 π(s)

从 Agent 执行角度看，当前流程近似：

```text
State_t
   ↓
Laya
   ↓
Action_t
   ↓
Environment
   ↓
State_t+1
   ↓
Laya
   ↓
Action_t+1
```

其核心近似为：

```text
Action = π(State)
```

而不是：

```text
Action = π(State, History)
```

迷宫任务实际上要求考虑：

```text
当前状态
+
已经访问过的位置
+
上一动作
+
已经探索过的分支
+
已经确认的死路
+
当前路径
```

即：

```text
Action =
π(
  current,
  visited,
  path,
  dead_ends,
  previous_state,
  goal
)
```

因此，如果两次来到 `(1,4)` 时，模型看到的主要 State 基本一致，那么它很容易产生相同决策：

```text
f(State_(1,4)) → up
```

然后重新进入 `(2,4)`：

```text
f(State_(2,4)) → down
```

最终形成循环。

---

### 5.2 “steps_taken” 不是有效的长期记忆

当前 State 中有：

```text
steps_taken=23
bfs_shortest=70
```

但这两个数字无法直接告诉模型：

```text
我刚才走过了哪里？
哪个方向已经被证明失败？
这里是不是刚刚回退过？
这个分支已经尝试过几次？
```

因此：

```text
steps_taken
```

只能表示“走了多少步”，不能表达：

```text
探索状态
```

二者不是同一个概念。

---

### 5.3 legal_moves 被作为文本输入，而不是 Runtime 硬约束

当前 prompt 中：

```text
legal_moves: down, right
```

本质上只是输入文本的一部分。

它并没有进入一个：

```text
Action Validator
```

例如：

```text
if action not in legal_moves:
    reject
```

因此 Laya 可以从概率上选择：

```text
up = 95%
```

而 Runtime 仍然接受这个动作。

这暴露出一个非常明确的架构问题：

> **概率决策与环境约束没有形成闭环。**

Laya 本身只能提供：

```text
decision + probability
```

不能自动保证：

```text
decision ∈ legal_action_space
```

---

## 6. closeness 实验进一步证明了当前任务不适配

模型对：

```text
How far is the agent from the goal?
```

输出：

```text
score = 0
probability = [1,0,0,0]
entropy ≈ 0
```

对应定义：

```text
0 = at the goal
1 = close
2 = halfway
3 = far
```

但实际：

```text
current = (1,4)
goal    = (9,9)
```

并且：

```text
bfs_shortest = 70
```

显然不是：

```text
at the goal
```

这说明模型不仅没有正确解决路径选择问题，对当前 State 的距离/阶段判断也没有学到可靠映射。

这一点符合 Laya 的公开能力定位：`score` 是 ordinal decision，而不是通用数值推理或几何距离计算器。公开 benchmark 中，fine-tuned Laya 的 `score` 表现也明显弱于 `noul`，说明 ordinal score 本身就是相对独立的能力。

---

## 7. 为什么 Laya 不微调时尤其容易出现这种结果

Laya 官方公开 benchmark 给出了一个非常重要的证据：

```text
random baseline       0.318
base Laya              0.362
base multilingual      0.342
fine-tuned Laya        0.766
```

在该 typed-decisions benchmark 上，官方明确指出：

> base checkpoint 并不是一个 zero-shot 通用决策器，而是一个用于进一步领域 specialization 的基础模型。

即：

```text
Laya Base
    ↓
学习“如何做 typed decision”
```

并不等于：

```text
Laya Base
    ↓
自动掌握任意业务领域的决策知识
```

领域微调之后，模型才逐渐学会：

```text
State
→
领域语义
→
候选选项
→
正确 decision distribution
```

官方当前公开的 fine-tuned checkpoint 在四类 typed-decision 工作流上的准确率约为：

```text
invoice processing       0.804
security incidents       0.766
customer service         0.764
agent-trace observability 0.730
```

而对应的基础模型则接近随机水平。

因此，本次迷宫实验的结果不能简单解释为：

> “Laya 的 System-1 不会做推理。”

更准确的解释是：

> **当前 Laya checkpoint 没有经过迷宫这一决策分布的领域训练，因此并没有学到“迷宫状态 → 合法动作 → 最短路径动作”这一映射。**

---

# 8. 本次实验揭示的能力边界

可以把 Laya 的能力边界划分为四个层次。

| 能力     | 当前结论       | 说明                |
| ------ | ---------- | ----------------- |
| 有限候选分类 | 较适合        | `choice` 的核心能力    |
| 一次状态决策 | 适合，但依赖领域训练 | State → Decision  |
| 概率性决策  | 适合，但需要校准   | confidence 不等于真实性 |
| 长时序规划  | 不应独立承担     | 缺乏显式搜索/长期记忆机制     |

因此：

```text
                    Laya
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    Classification  Routing      One-step Decision
        │             │             │
        └─────────────┼─────────────┘
                      ↓
                  强项领域


        Long-horizon Planning
                 ↓
              非核心能力
                 ↓
        需要 Planner / Runtime
```

---

# 9. 本次实验最重要的结论

本次实验至少验证了五点。

### 结论一：Laya 可以非常自信地做出错误决策

实验中：

```text
illegal action = up
confidence = 95.27%
```

因此：

```text
high confidence ≠ correct
```

概率必须经过领域校准，并且必须受到环境规则验证。

---

### 结论二：Laya 本身不是搜索算法

它没有天然解决：

```text
DFS
BFS
A*
backtracking
visited-set
dead-end detection
```

这类算法问题。

把一个需要多步搜索的问题直接压缩成：

```text
“下一步选择哪个 option”
```

并不能自动获得规划能力。

---

### 结论三：连续 Agent 任务需要 State Memory

真正的 State 至少应该包含：

```text
current
goal
visited
path
previous
dead_ends
constraints
```

否则：

```text
A → B → A → B
```

是完全可能发生的。

---

### 结论四：Probability 与 Environment Validity 是两层不同的东西

应该明确区分：

```text
Laya:
P(action | state)

Runtime:
is action valid?
```

二者不能混为一谈。

---

### 结论五：该实验反而验证了“System-1 + Runtime”组合的合理性

实验结果并没有否定 Laya，而是进一步明确了它应该处于什么位置：

```text
LLM / Planner
      ↓
规划、推理、复杂决策
      ↓
Laya / System-1
      ↓
快速、结构化、有限空间决策
      ↓
Runtime
      ↓
约束验证、状态更新、执行
```

---

# 10. 推荐的工程使用方式

不建议：

```text
Laya
 ↓
直接执行动作
```

推荐：

```text
                    Planner / LLM
                         ↓
                  Candidate Decisions
                         ↓
                ┌─────────────────┐
                │      Laya       │
                │ System-1        │
                │                 │
                │ choice          │
                │ score           │
                │ noul            │
                └────────┬────────┘
                         ↓
                 Decision + Confidence
                         ↓
                ┌─────────────────┐
                │ Runtime         │
                │                 │
                │ Constraint      │
                │ Validation      │
                │ State Update    │
                └────────┬────────┘
                         ↓
                      Execute
```

例如迷宫：

```text
Laya:
move = up
confidence = 0.95

Runtime:
up ∉ legal_moves

→ reject
→ re-decide / fallback
```

而不是：

```text
0.95
↓
直接执行
```

---

# 11. 对 Ontology Decision Model 的启示

本次实验对“参考 Laya 思路构建领域本体决策模型”的价值非常大。

建议不要复制成：

```text
Ontology Laya
=
模型直接决定一切
```

而应该设计成：

```text
               Ontology Decision Model
                         │
              Decision + Confidence
                         ↓
                Ontology Runtime
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    Schema Check      Rule Check      State Check
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                      Execute
```

例如：

```text
Question:
选择 Query Pattern

Options:
A passage_basic
B passage_split_anomaly
C settlement
D gantry_trace
```

模型：

```text
B = 0.93
```

然后 Ontology Runtime 验证：

```text
B 是否适用于当前 Object？
B 是否满足 Relation？
B 是否需要字段？
B 是否允许当前用户？
B 是否满足 Query Pattern 前置条件？
```

最终才执行：

```text
B
```

因此：

> **Laya 负责“哪个选项最可能合适”，Ontology Runtime 负责“这个选项是否真的合法、可执行”。**

这比单纯复制 Laya 更适合本体智能平台。

---

# 12. 后续实验建议

下一轮不建议直接继续增加迷宫规模，而建议做四组对照。

## 实验 A：纯 State

```text
current
goal
grid
legal_moves
```

观察基础决策能力。

## 实验 B：加入历史

```text
current
goal
grid
legal_moves
visited
previous
path
```

验证死循环是否明显下降。

## 实验 C：Laya + Runtime Validator

```text
Laya
 ↓
Action
 ↓
Validator
 ↓
Reject invalid
 ↓
Retry
```

验证“模型 + 确定性 Runtime”是否可以消除非法动作。

## 实验 D：领域微调

构造：

```text
State
+
Question
+
Candidate Options
+
Gold Decision Distribution
```

针对迷宫专门微调 Laya，再比较：

```text
Base
vs
Fine-tuned
vs
Fine-tuned + Runtime
```

最终评价：

```text
Decision Accuracy
Illegal Action Rate
Cycle Rate
Goal Success Rate
Average Steps
Confidence Calibration
Brier Score
Latency
```

其中最值得关注的是：

```text
Illegal Action Rate
Cycle Rate
Goal Success Rate
```

因为这三个指标比单纯的分类 Accuracy 更能反映 Agent 实际执行能力。

---

# 13. 后续思考

本次实验最大的价值并不是证明“Laya 不会走迷宫”，而是帮助明确一个更基础的架构原则：

> **System-1 决策模型不应该被当成完整 Agent。**

更合理的分工是：

```text
LLM / Planner
负责：
理解
推理
规划
生成候选方案

Laya / System-1
负责：
快速选择
分类
路由
评分
置信度判断

Runtime
负责：
约束
验证
状态管理
执行
事务
```

这三个部分结合以后：

```text
复杂问题
   ↓
LLM Planner

有限决策
   ↓
Laya

确定性执行
   ↓
Ontology Runtime
```

会形成一个更稳定的 Agent 闭环。

对于本体平台，这个模型尤其适合：

```text
NL
 ↓
Intent Decision
 ↓
Object Decision
 ↓
Relation Decision
 ↓
Query Pattern Decision
 ↓
Function / Workflow Decision
 ↓
Ontology Runtime Validation
 ↓
Execute
```

其中：

```text
Decision Model
```

解决：

> **“应该选什么？”**

而：

```text
Ontology Runtime
```

解决：

> **“这个选择是否符合本体约束，并且能不能执行？”**

这比让一个 LLM 直接完成：

```text
理解 → 决策 → 生成 DSL → 执行
```

更加容易控制、验证和观测。

---

## 14. 最终判断

本次实验可以归纳为一句话：

> **Laya 能提供高速、结构化、带概率的 System-1 决策，但在未经领域训练且缺乏历史状态与运行时约束的情况下，它不能可靠承担需要长期记忆、回溯和搜索的连续规划任务。**

迷宫实验中的：

```text
非法动作：
up = 95.27%

连续循环：
up → down → up → down

距离判断：
at goal = 100%
```

共同说明：

```text
高置信度
    ≠
环境正确性

单步 Decision
    ≠
长期 Planning

Probability
    ≠
Constraint Validation
```

因此，最值得继续验证的方向不是“如何让 Laya 单独解决迷宫”，而是：

```text
System-1 Decision Model
          +
Domain Knowledge
          +
Deterministic Runtime
          ↓
Reliable Agent Execution
```

这也是将 Laya 思路迁移到**领域本体决策模型**时最值得保留的核心思想。
