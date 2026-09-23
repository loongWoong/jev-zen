# Laya 架构解析：ModernBERT + 2-layer decision head + option marker scorer

> 实测环境：`http://127.0.0.1:8771`，权重 `model_maze_ft.safetensors`（321,908,998 参数，22 层编码器 + 2 层决策头），
> 词表 BPE 256,000（`faithful=true`），纯 numpy fp32 重建前向，不依赖 torch。
> 实测时间：2026-09-23。

---

## 一、这套架构要解决的问题

大模型做「简单抉择」有一个结构性浪费：答案是 `billing` 四个字符，却要跑一次自回归生成、
烧掉几百 token。Laya 的做法是把抉择从**生成问题**改造成**打分问题**：

- 不生成任何文本 → 没有可解析的输出，也不存在"生成了一半的幻觉"
- 一次前向同时回答所有问题 → 边际成本随问题数**次线性**增长
- 输出天然带概率 → 可以直接接阈值、接规则、接双阈值护栏

所以它不是"小号 LLM"，而是一个 **System-1 分类器**：非自回归、离散输出、可校准。

---

## 二、三段结构分别做什么

<svg viewBox="0 0 680 566" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Laya 决策架构总览</title>
<desc>state 与 questions 组装成带 option marker 的序列，经 22 层 ModernBERT 编码、2 层决策头，在 marker 位置打分得到 choice / score / noul 三类输出</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<g>
<rect x="60" y="40" width="280" height="56" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="200" y="60" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle" dominant-baseline="central">state：任意文本或对象</text>
<text x="200" y="78" font-family="sans-serif" font-size="12" fill="#185FA5" text-anchor="middle" dominant-baseline="central">对象渲染成 k: v 逐行</text>
</g>
<g>
<rect x="340" y="40" width="280" height="56" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="480" y="60" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle" dominant-baseline="central">questions × N（typed）</text>
<text x="480" y="78" font-family="sans-serif" font-size="12" fill="#185FA5" text-anchor="middle" dominant-baseline="central">choice / score / noul</text>
</g>
<path d="M200 96 L200 114 L340 114 L340 124" fill="none" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrow)"/>
<path d="M480 96 L480 114 L340 114" fill="none" stroke="#185FA5" stroke-width="1.5"/>
<g>
<rect x="140" y="126" width="400" height="64" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="148" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle" dominant-baseline="central">Prompt 组装 + option marker 展开</text>
<text x="340" y="170" font-family="sans-serif" font-size="12" fill="#185FA5" text-anchor="middle" dominant-baseline="central">每选项一段文本 + 一个 marker 占位，全拼进一条序列</text>
</g>
<path d="M340 190 L340 208" fill="none" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrow)"/>
<g>
<rect x="110" y="210" width="460" height="76" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="232" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle" dominant-baseline="central">ModernBERT 编码器 · 22 层 · hidden 768 · 12 头</text>
<text x="340" y="252" font-family="sans-serif" font-size="12" fill="#185FA5" text-anchor="middle" dominant-baseline="central">alternating3：每 3 层 1 层全局 + 2 层滑窗（半径 64）</text>
<text x="340" y="270" font-family="sans-serif" font-size="12" fill="#185FA5" text-anchor="middle" dominant-baseline="central">GeGLU 中间维 1152，RoPE θ=160000，fp32 纯 numpy</text>
</g>
<path d="M340 286 L340 304" fill="none" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrow)"/>
<g>
<rect x="110" y="306" width="460" height="60" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="326" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle" dominant-baseline="central">2-layer decision head</text>
<text x="340" y="348" font-family="sans-serif" font-size="12" fill="#185FA5" text-anchor="middle" dominant-baseline="central">整序列 self-attention + gelu，对 marker 位置做二次混合</text>
</g>
<path d="M340 366 L340 384" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<g>
<rect x="110" y="386" width="460" height="76" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="408" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">Option marker scorer</text>
<text x="340" y="428" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">取 marker 位置 hidden → 线性打分 → 每选项一个 logit</text>
<text x="340" y="446" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">同问题 marker 集合内 softmax，三类各用一个温度</text>
</g>
<path d="M240 462 L240 480" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<path d="M340 462 L340 480" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<path d="M440 462 L440 480" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<g>
<rect x="60" y="482" width="180" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="150" y="502" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">choice</text>
<text x="150" y="520" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">argmax + 置信 + 排序</text>
</g>
<g>
<rect x="250" y="482" width="180" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="502" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">score</text>
<text x="340" y="520" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">档位期望值 + 熵</text>
</g>
<g>
<rect x="440" y="482" width="180" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="530" y="502" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">noul</text>
<text x="530" y="520" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">noul = P(yes)</text>
</g>
</svg>

### 1. ModernBERT 编码器（22 层）——只做理解，不做输出

| 参数 | 值 | 工程含义 |
|---|---|---|
| hidden / layers / heads | 768 / 22 / 12 | 约 3.2 亿参数，BERT-base 量级的**编码器** |
| 注意力 | `alternating3` | 每 3 层 1 层全局 + 2 层滑窗（半径 64），长文本省算力 |
| FFN | GeGLU，中间维 1152 | `Wi` 输出 2304 = 2×1152，与 HF 官方实现一致 |
| 位置编码 | RoPE θ=160000 | 支持长上下文（max_len 1024） |
| 输出 | 全序列 hidden states | **关键：它不为某个 token 单独输出，全部信息留在序列里** |

注意这里是**双向编码器**（BERT 系），不是 causal LM。这决定了它能看到 marker 前后的全部上下文，
也决定了它天生不适合做生成——它压根没有 LM head。

### 2. 2-layer decision head ——把"读位置"变成"读关系"

编码器输出的 hidden 是逐 token 的。如果直接在 marker 位置接一个线性层打分，每个 marker 只能
用自己的向量，**marker 之间无法互相比较**——但选择题的本质恰恰是"这几个选项里哪个最合适"。

2 层 transformer 的作用就是让 marker 位置之间做一次 self-attention，使每个 marker 的表示知道
"其他选项长什么样"，再做归一化。它是一个**决策前的比较层**，不是特征提取层。

> **本机实测的一个硬事实**：这个 head 也是当前链路里最脆弱的一环。
> 只改读出位置、其余不动：走 `enc → head layers → x[marker]` 时 `logit_spread = 0.0107`、
> `confidence = 0.2515`（4 选项 = 数学上完美均匀）；直接 `enc → x[marker]` 跳过这两层时
> `spread = 4.72`、`confidence = 0.924`。**差 440 倍。**
> 原因是整序列 self-attention 会把 marker 位置的个性抹平。修掉它必要，但不充分。

### 3. Option marker scorer ——整套架构的核心创新

它换掉了 LLM 的"生成答案"，改成"给选项打分"：

```
<bos> [state tokens……] <eos> [Q1: 指令 + 选项文本 <m1> <m2> <m3> <m4>] [Q2: 指令 <m5> <m6>] …
```

<svg viewBox="0 0 680 372" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Option marker scorer 原理图</title>
<desc>所有问题拼进同一条 token 序列，每个选项后跟一个 marker 占位，编码器一次前向后在 marker 位置读出 logit，同一问题的 marker 集合内做 softmax</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="40" y="30" font-family="sans-serif" font-size="14" font-weight="500" fill="#2C2C2A" dominant-baseline="central">一条 token 序列（一次前向）</text>
<rect x="40" y="46" width="44" height="34" rx="5" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="62" y="63" font-family="sans-serif" font-size="12" fill="#2C2C2A" text-anchor="middle" dominant-baseline="central">bos</text>
<rect x="88" y="46" width="150" height="34" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="163" y="63" font-family="sans-serif" font-size="12" fill="#0C447C" text-anchor="middle" dominant-baseline="central">state tokens ……</text>
<rect x="242" y="46" width="44" height="34" rx="5" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="264" y="63" font-family="sans-serif" font-size="12" fill="#2C2C2A" text-anchor="middle" dominant-baseline="central">eos</text>
<rect x="290" y="46" width="120" height="34" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="350" y="63" font-family="sans-serif" font-size="12" fill="#0C447C" text-anchor="middle" dominant-baseline="central">Q1 指令 + 选项文本</text>
<rect x="416" y="46" width="18" height="34" rx="4" fill="#1D9E75" stroke="#0F6E56" stroke-width="0.5"/>
<rect x="440" y="46" width="18" height="34" rx="4" fill="#1D9E75" stroke="#0F6E56" stroke-width="0.5"/>
<rect x="464" y="46" width="18" height="34" rx="4" fill="#1D9E75" stroke="#0F6E56" stroke-width="0.5"/>
<rect x="488" y="46" width="18" height="34" rx="4" fill="#1D9E75" stroke="#0F6E56" stroke-width="0.5"/>
<rect x="518" y="46" width="86" height="34" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="561" y="63" font-family="sans-serif" font-size="12" fill="#0C447C" text-anchor="middle" dominant-baseline="central">Q2 指令…</text>
<rect x="608" y="46" width="18" height="34" rx="4" fill="#1D9E75" stroke="#0F6E56" stroke-width="0.5"/>
<rect x="630" y="46" width="18" height="34" rx="4" fill="#1D9E75" stroke="#0F6E56" stroke-width="0.5"/>
<text x="440" y="92" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">marker 占位 ×N</text>
<text x="625" y="92" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">×M</text>
<path d="M300 30 L300 22 L470 22" fill="none" stroke="#5F5E5A" stroke-width="0.5" stroke-dasharray="3 3"/>
<text x="474" y="22" font-family="sans-serif" font-size="12" fill="#5F5E5A" dominant-baseline="central">type_emb 在编码前注入 marker（choice/score/noul）</text>
<path d="M425 80 L425 118" fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="3 3"/>
<path d="M449 80 L449 118" fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="3 3"/>
<path d="M473 80 L473 118" fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="3 3"/>
<path d="M497 80 L497 118" fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="3 3"/>
<path d="M617 80 L617 118" fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="3 3"/>
<path d="M639 80 L639 118" fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="340" y="120" width="240" height="30" rx="5" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="460" y="135" font-family="sans-serif" font-size="12" fill="#085041" text-anchor="middle" dominant-baseline="central">hidden[marker] → 线性层 → 每选项 1 个 logit</text>
<path d="M460 150 L460 170" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<rect x="340" y="172" width="240" height="34" rx="5" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="460" y="189" font-family="sans-serif" font-size="12" fill="#085041" text-anchor="middle" dominant-baseline="central">logit ÷ 温度 → softmax</text>
<path d="M400 206 L400 232" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<path d="M560 206 L560 232" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow)"/>
<rect x="290" y="234" width="220" height="48" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="400" y="250" font-family="sans-serif" font-size="13" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">Q1 的 4 个 marker 内归一化</text>
<text x="400" y="270" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">choice：argmax = billing，p = 0.94</text>
<rect x="530" y="234" width="110" height="48" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="585" y="250" font-family="sans-serif" font-size="13" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">Q2 的 2 个</text>
<text x="585" y="270" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">noul = P(yes)</text>
<text x="40" y="258" font-family="sans-serif" font-size="12" fill="#5F5E5A" dominant-baseline="central">问题之间</text>
<text x="40" y="274" font-family="sans-serif" font-size="12" fill="#5F5E5A" dominant-baseline="central">各自独立归一化</text>
<rect x="40" y="304" width="600" height="48" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="320" font-family="sans-serif" font-size="13" font-weight="500" fill="#2C2C2A" text-anchor="middle" dominant-baseline="central">加问题 = 加序列长度，不加前向次数（forward_passes 恒为 1）</text>
<text x="340" y="338" font-family="sans-serif" font-size="12" fill="#5F5E5A" text-anchor="middle" dominant-baseline="central">实测：1 问 51 ms · 5 问 108 ms · 10 问 228 ms（本机 CPU numpy）</text>
</svg>

机制要点：

1. **每个选项后挂一个 marker 占位**（`<mask>` 位置），该位置的 hidden 就是"这个选项的表示"。
2. **type_emb 在编码前注入 marker**，告诉模型这个 marker 属于 `choice` / `score` / `noul`
   （`type_emb_stage = input`）。三种 primitive 共用一套权重，靠 type 区分行为。
3. **线性层把 hidden 压成一个标量 logit**，一个选项一个分数，没有词表 softmax。
4. **softmax 只在同一个问题的 marker 集合内做**，问题之间互不干扰。
5. **三个 primitive 各用一个温度**（`temperature = [1,1,1]`，权重内未拟合）。

由此得到三个直接推论：

- **加问题 = 加序列长度，不加前向次数**。`forward_passes` 恒为 1。实测 1 问 51 ms、
  5 问 108 ms、10 问 228 ms —— 10 个问题只花 4.5 倍时间，而不是 10 倍。
- **选项数有硬预算**。选项共享 `head_max_len`（英文 192 / 多语 256 token），
  77 个选项摊到每标签 3–4 token，文本互相不可区分，准确率断崖。**选项 ≤ 20**。
- **`logit_spread` 是通道健康的唯一可信读数**。同一问选项间的 logit 极差，字面可判定任务上
  应 `≫ 1e-2`；若只有 `1e-3` 量级，说明决策头压根没读到 marker 上下文。

---

## 三、特性：它强在哪

| 特性 | 机制来源 | 实测数字 |
|---|---|---|
| 零 token 成本 | 自托管本地前向 | 0 token，纯 numpy |
| 低且可预测延迟 | 非自回归，一次前向 | 17–260 ms（本机 CPU），无网络抖动 |
| 批量近乎免费 | 一次前向回答全部问题 | 10 问 228 ms vs 单问 51 ms |
| 输出可直接消费 | 概率 / 排序 / 期望值 | 无需解析、无需正则、无需重试 |
| 可用于护栏 | 双阈值 + P(yes) | `p<0.2` 放行、`p>0.8` 拦截、中间升级 LLM |
| 数据不出机 | 纯本地前向 | 隐私敏感场景的独立卸载理由 |
| 可诊断 | `logit_spread` / `entropy_norm` | 能区分"没意见"和"很有把握地答错" |

其中**可诊断性**是被低估的一条：LLM 给你一段错话，你无法判断它是"不确定"还是" confidently wrong"；
Laya 的 `entropy_norm ≈ 1` 明确告诉你"没形成意见"，`spread` 低明确告诉你"通道没增益"。

---

## 四、能力边界：它弱在哪（本机实测证据）

<svg viewBox="0 0 680 348" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Laya 当前权重的能力域分层</title>
<desc>以迷宫微调权重为中心，能力域分三层：网格空间状态决策可用、结构化枚举决策须门禁验证、开放语义语言理解实测不可用</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<ellipse cx="180" cy="176" rx="150" ry="98" fill="none" stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="4 4"/>
<ellipse cx="180" cy="176" rx="106" ry="70" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<ellipse cx="180" cy="176" rx="60" ry="40" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="180" y="168" font-family="sans-serif" font-size="13" font-weight="500" fill="#085041" text-anchor="middle" dominant-baseline="central">微调域</text>
<text x="180" y="186" font-family="sans-serif" font-size="12" fill="#0F6E56" text-anchor="middle" dominant-baseline="central">网格 · 迷宫</text>
<text x="180" y="120" font-family="sans-serif" font-size="12" fill="#5F5E5A" text-anchor="middle" dominant-baseline="central">结构化枚举决策</text>
<text x="180" y="240" font-family="sans-serif" font-size="12" fill="#5F5E5A" text-anchor="middle" dominant-baseline="central">开放语义 · 语言理解</text>
<path d="M318 130 L350 116" fill="none" stroke="#5F5E5A" stroke-width="0.5" stroke-dasharray="3 3"/>
<path d="M280 152 L350 176" fill="none" stroke="#5F5E5A" stroke-width="0.5" stroke-dasharray="3 3"/>
<path d="M232 200 L350 236" fill="none" stroke="#5F5E5A" stroke-width="0.5" stroke-dasharray="3 3"/>
<g>
<rect x="350" y="90" width="290" height="52" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="366" y="108" font-family="sans-serif" font-size="13" font-weight="500" fill="#2C2C2A" dominant-baseline="central">外层 · 不可用</text>
<text x="366" y="126" font-family="sans-serif" font-size="12" fill="#5F5E5A" dominant-baseline="central">4 种选项顺序全部选 red，与内容无关</text>
</g>
<g>
<rect x="350" y="150" width="290" height="52" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="366" y="168" font-family="sans-serif" font-size="13" font-weight="500" fill="#2C2C2A" dominant-baseline="central">中层 · 须门禁验证</text>
<text x="366" y="186" font-family="sans-serif" font-size="12" fill="#5F5E5A" dominant-baseline="central">choice / score / noul，spread ≥ 0.05 才可用</text>
</g>
<g>
<rect x="350" y="210" width="290" height="52" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="366" y="228" font-family="sans-serif" font-size="13" font-weight="500" fill="#085041" dominant-baseline="central">内层 · 当前权重能力域</text>
<text x="366" y="246" font-family="sans-serif" font-size="12" fill="#0F6E56" dominant-baseline="central">网格状态有响应，档位判断命中</text>
</g>
<rect x="40" y="288" width="600" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="308" font-family="sans-serif" font-size="12" fill="#2C2C2A" text-anchor="middle" dominant-baseline="central">当前服务加载 model_maze_ft.safetensors（321.9M 参数，迷宫微调），不是通用语义检查点</text>
</svg>

### 4.1 当前门禁状态：FAIL

```
[PASS] health / tokenizer_faithful / weights_loaded
[FAIL] probe_argmax       期望 blue，得到 red（p=0.8000）
[PASS] probe_confidence   0.8000 >= 0.6
[PASS] probe_logit_spread 41.42993 >= 0.05
结论：通道未通过门禁 —— 禁止作为决策依据，请回退大模型
```

对比 2026-09-22：`logit_spread` 从 `0.00556` → `41.43`（+7400 倍），**读出位置的缺陷已被修掉**，
但语义仍然是错的。这正好印证了技能里那句铁律：**增益 ≠ 正确**。

### 4.2 根因：权重是迷宫微调版，不是通用语义模型

`GET /api/status` 显示当前加载的是 **`model_maze_ft.safetensors`**。这解释了恒选 red：
颜色语义对它是 OOD，它只输出微调域内的常量偏好。

位置偏置对照（同一 state，只改选项顺序）：

| 选项顺序（blue 位置） | argmax | p | logit_spread |
|---|---|---|---|
| blue,green,red,yellow（1） | red | 0.9934 | 51.19 |
| green,blue,red,yellow（2） | red | 0.8000 | 41.43 |
| green,red,yellow,blue（4） | red | 0.9723 | 49.46 |
| red,yellow,blue,green（3） | red | 0.9970 | 47.46 |

**四种顺序全部输出 red** → 既不是位置偏置，也不是内容响应，是内容无关的常量偏置。

in-domain 对照（2048 网格 state，11 空格）：
`move` → down（p=0.7497, spread=24.51，非最优但非均匀）；
`full`(score) → 0.42/2.0，argmax_level=0「ample space」**命中**。

→ 网格/空间类状态上通道确实有信息；通用语言语义上不可用。

### 4.3 结构性边界（与当前 bug 无关，是架构决定的）

| 边界 | 原因 | 表现 |
|---|---|---|
| 不能生成文本 | 只有 scorer，无 LM head | 写不了理由、引不了原文 |
| 不能开放推理 | 答案空间必须预先枚举 | 想不到的选项不会被创造出来 |
| `score` 最弱 | 序数关系难学 | 模型卡实测 SST-5 仅 0.372，5 档以上不可用 |
| 选项 >20 崩 | 共享 head_max_len 预算 | 每标签 3–4 token，文本不可区分 |
| 长文本截断 | max_len 1024 | `meta.truncated` 为 true 时结论已失真 |
| 概率过自信 | 温度未在权重内拟合 | 必须自 refit（官方 refit 后 ECE 0.466 → 0.081） |
| 多语未验证 | 本机仅 multilingual 检查点 | 非拉丁脚本先查 `tokenizer.faithful` 与 `routing.script` |
| 无自我修正 | 一次前向定终身 | 没有 CoT、没有反思、没有第二次机会 |

**最需要警惕的一条**：它坏的时候**不报错**。均匀分布下 argmax 仍然会给出一个答案，
`confidence ≈ 1/选项数` 看起来"不确定"——但 97.5% 的情况会选同一个方向（由 prompt 的 token
几何决定，而非输入内容）。所以**任何 ad-hoc 指标（准确率、命中率）在小样本下都会 ±10~20pp 乱摆**，
只能用 `logit_spread` + 平凡探针判通道。

---

## 五、落地建议

**1. 门禁是强制的，不是建议**
```bash
python laya_client.py gate        # 退出码 0 才继续
python laya_client.py ask --payload req.json
```
门禁只在会话首次跑一遍（服务重启后失效）。FAIL 时**显式回退大模型并告知用户**，切勿静默降级。

**2. 卸载判据**
高频 + 同构 + 选项有限 + 只需离散结论 + 错了能回退 → 卸载；
开放生成、长文推理、需要理由、错了有后果 → 留大模型。

**3. 风险收口的标准姿势：模型只有破平权，没有否决权**
规则主判 → 规则筛出候选 → 模型只在候选内破平 → 非法输出回退规则并标 `fallback: true`。
这样把模型风险限制在"两个都不错时选哪个"，而不是"要不要这么做"。

**4. 待办（按当前权重状态）**
- **P0**：确认要用哪个权重。`model_maze_ft` 只适合网格/空间状态决策；若要做工单分流类语义任务，
  必须换回通用检查点，并连带重做 prompt 模板对齐。
- **P1**：`max_prefixes=6` + prefix slice 切分规则反推（多轮训练结构），这是当前语义未对齐的主嫌疑。
- **P2**：head 层作用域改为"state 作 KV、marker 作 Q"的 cross-attention，避免整序列抹平。
- **P3**：拿到信号后再做温度校准。

---

## 六、一句话总结

**ModernBERT 负责理解，2-layer head 负责让选项互相比较，option marker scorer 负责把"生成"
换成"打分"**——代价是彻底放弃开放生成能力，换来 0 token、毫秒级、可校准、可诊断的离散决策。
它适合当**规则引擎的边角料分类器**，不适合当**理解者**；而当它失效时，它依然会自信地给出一个答案，
所以必须用 `logit_spread` 门禁而不是准确率来判断能不能用。
