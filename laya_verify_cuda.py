#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 决策模型 · CUDA 加速版 (laya-verify-cuda)
=============================================================================

与 `laya_verify.py` 的关系
--------------------------
本文件**不复制**模型定义，而是 `import laya_verify as L` 后把前向换到 GPU：

    · SafeTensors / tokenizer / Question / PRESETS / Verifier / HTTP Handler
      全部复用原模块 —— 因此验证报告、路由元数据、prompt 组装、温度语义
      与 CPU 版**逐字节一致**，不存在"两套实现各自漂移"的问题。
    · 只新增一个后端：CudaLayaModel 覆写 LayaModel 的全部前向方法。

依赖
----
    cupy-cuda13x（numpy 1:1 API）+ 本机 CUDA 驱动。本机实测：
        RTX 3080 / 12 GB / sm_86 / driver CUDA 13.3 / nvcc 13.3
    无 cupy 时本脚本会直接报错退出（不会静默降级到 CPU —— 静默降级会让
    "快了多少"变成一个说不清的数）。

    本机环境（已建好的隔离 venv，直接用这个跑）：
        C:/Users/Administrator/.workbuddy/binaries/python/envs/laya-cuda/Scripts/python.exe
        pip install numpy cupy-cuda13x        # CUDA 12.x 驱动改用 cupy-cuda12x

加速从哪来（先量化，再动手）
----------------------------
模型是 22 层 encoder，hidden=768。每层 4 个 GEMM：

    Wqkv 768→2304 · Wo 768→768 · Wi 768→2304 · Wo2 1152→768
    = 5.01 M MAC/token/层 → 10.0 MFLOP/token/层 → 22 层 = 220 MFLOP/token

T=139 tokens（模型卡示例）→ 约 30.6 GFLOP/次前向，其中 **>99% 在这 88 个 GEMM**。
CPU(numpy fp32) 实测单次 300~3000 ms（抖动取决于机器负载）。

本机 RTX 3080 实测（T=139，warm-up 后中位数）：

    纯 GEMM（88 个）       fp32 4.37 ms   fp16 2.15 ms
    单层全部 op            fp32 ~0.9 ms（其中 GEMM 仅 0.20 ms）
    单次前向（eager）      fp32 20~25 ms
    单次前向（batch=16）   9.5 ms/条      batch=64 → 9.2 ms/条（B≈16 后饱和）

也就是说：**GPU 版 20 ms 里只有约 4.4 ms 是真在计算，剩下 ~16 ms 是 cupy 的
Python 调度开销**（实测单个 cupy op 空转 14 µs、reduce 36 µs、softmax 4 连 op
113 µs；22 层 × 约 22 个 op ≈ 500 次调用）。所以优化方向按收益排序是：
    P0 权重常驻显存（避免每前向重新 H2D 441 MB —— 这一项如果不做，
       每次前向光拷贝权重就几百 ms，GPU 等于白上）
    P0 融合 kernel（RawKernel 把 layernorm 3 pass 压成 1 次 launch；
       RoPE/GeGLU/残差同理；失败则自动降级到 cupy 表达式）
    P1 batch（`predict_batch`，padding + attention mask；吞吐 39 → 109 条/s）
    ✗ fp16（实测**反而更慢**：29.9 ms vs 22.3 ms，且 logit 相对差 18.6%
      —— 每层多出的 fp32↔fp16 转换开销吃掉了 GEMM 收益，故默认不开）
    ✗ CUDA Graph（代码保留，但 cupy 捕获时抛
      "calling cuBLAS API during stream capture is currently unsupported"，
      已自动回退 eager。要吃 Graph 的收益得转 torch）

数值一致性
----------
GPU 不是"换个设备重算一遍"就完事：cuBLAS 与 OpenBLAS 的归约顺序不同，
fp32 下必然有 ~1e-4 量级相对偏差。所以本脚本提供 `--selftest`，用**同一组
输入**跑 CPU 与 GPU，给出 logits 最大绝对差 / 相对差 / top-1 是否一致。
fp32 默认档实测：相对差 9.4e-7 / 6.7e-6，逐题 top-1 全一致。

调试过程中真实踩到并修掉的三个 kernel bug（都靠"CPU/GPU 逐层比对"定位）：
    1. layernorm 用 __shfl_down_sync 做 block reduce —— warp shuffle 只在
       32 lane 内有效，blockDim=256 时均值只统计了 1/8 的数据；
    2. RoPE 的 rotate_half 配号写反（前半应为 -x[i+half]，后半为 +x[i-half]）；
    3. RoPE 的 cos/sin 索引按 half 而不是 dh 取（相位错半个周期）。
这三个错都不会报错，只会让输出"看着像那么回事但数值全错" —— 所以一致性
校验不是装饰，是这个脚本的必需品。

时间测量的坑
------------
CUDA 是异步的：`cp.matmul` 返回时 kernel 还没跑完。所有计时点前必须
`cp.cuda.get_current_stream().synchronize()`，否则测到的是 launch 时间，
会得到"0.3 ms"这种自欺欺人的数。

数值一致性
----------
GPU 不是"换个设备重算一遍"就完事：cuBLAS 与 OpenBLAS 的归约顺序不同，
fp32 下必然有 ~1e-4 量级相对偏差。所以本脚本提供 `--selftest`，用**同一组
输入**跑 CPU 与 GPU，给出 logits 最大绝对差 / 相对差 / top-1 是否一致。
fp32 默认档要求：相对差 < 1e-3 且逐题 top-1 一致。

用法
----
    python laya_verify_cuda.py --bench             # CPU vs GPU 延迟 + 一致性对比
    python laya_verify_cuda.py --selftest          # 用 GPU 后端跑原版全套验证报告
    python laya_verify_cuda.py --dtype fp16        # fp16 TensorCore 档
    python laya_verify_cuda.py --bench --batch 8   # 批量吞吐
    python laya_verify_cuda.py --port 8772         # 起服务（复用原 index.html）

    # --model：指定加载哪个模型（文件 / 目录 / glob / 省略后缀 / 相对路径都行）
    python laya_verify_cuda.py --list
    python laya_verify_cuda.py -m labs/finetune/artifacts/model_maze_ft.safetensors --bench
    python laya_verify_cuda.py -m ./models/maze/ --selftest      # 目录 → 找 model.safetensors
    python laya_verify_cuda.py -m "labs/**/model*.safetensors"   # glob（须唯一命中）
    python laya_verify_cuda.py -m other.safetensors --strict     # 配置/分词器不回落脚本目录
    python laya_verify_cuda.py -m other.safetensors --tokenizer /path/tokenizer.json
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import re
import statistics
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import cupy as cp
except Exception as e:  # noqa: BLE001
    sys.stderr.write(
        "[laya-cuda] 无法导入 cupy：%s\n"
        "  安装：pip install cupy-cuda13x   （本机 CUDA 13.3；CUDA 12.x 驱动用 cupy-cuda12x）\n"
        "  本脚本不会静默降级到 CPU —— 否则「快了多少」会变成说不清的数。\n" % e)
    raise SystemExit(2)

import laya_verify as L  # noqa: E402  复用：SafeTensors / tokenizer / Question / PRESETS / Verifier

DEFAULT_MODEL = L.DEFAULT_MODEL


# =============================================================================
# 0. 模型定位：--model 解析（文件 / 目录 / glob / 省略后缀 / 相对路径）
# =============================================================================
# 为什么值得单独一层：`--model` 原来只是把字符串直接丢给 `os.path.isfile`，于是
#   · 给目录（HF 风格的 checkpoint 目录）→ 报"找不到权重文件"
#   · 给相对路径 → 只在 CWD 下有解，脚本目录之外的模型必须写全路径
#   · 给错路径 → 报错里没有任何线索说明"附近有哪些模型可用"
# 更隐蔽的一条在下游：`route()["repo"]` 的名字、`_cfg_dirs()` 的 config.json 回退
# 目录、`find_tokenizer()` 的 tokenizer 回退目录，三处全都锚在**脚本目录的默认
# 权重**上。也就是说哪怕加载的是 B 模型，报告里的 repo 名字、架构配置、分词器仍
# 可能来自 A —— 这种"文件就在手边、脚本却在用另一个模型的东西"是最难排查的偏差。
# 所以解析完成后必须把这三个锚点重钉到真正加载的模型上（见 build()）。

_MODEL_SUFFIX = ".safetensors"
_SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "site-packages",
              ".workbuddy", ".idea", ".vscode"}


def _norm(p: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(p))))


def model_search_dirs() -> List[str]:
    """解析相对路径 / 扫描候选模型时的搜索根，按优先级排列。"""
    out: List[str] = []
    for d in [os.environ.get("LAYA_MODEL_DIR"), os.getcwd(), HERE]:
        if not d:
            continue
        a = _norm(d)
        if a not in out:
            out.append(a)
    return out


def _scan_models(roots: Sequence[str], depth: int = 3) -> List[str]:
    """在 roots 下（限深 depth）收集 *.safetensors 的绝对路径。"""
    found: List[str] = []
    for root in roots:
        if not os.path.isdir(root) or os.path.dirname(root) == root:
            continue                      # 不递归盘符根 / 家目录根
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            if rel != "." and rel.count(os.sep) + 1 >= depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if fn.lower().endswith(_MODEL_SUFFIX):
                    found.append(_norm(os.path.join(dirpath, fn)))
    return found


def list_models() -> List[str]:
    """本机可见的模型清单（去重 + 按路径排序）。"""
    seen, out = set(), []
    for p in sorted(_scan_models(model_search_dirs())):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def show_models(spec: Optional[str] = None) -> None:
    ms = list_models()
    print(f"[laya-cuda] 搜索目录：{' | '.join(model_search_dirs())}")
    if not ms:
        print("[laya-cuda] 未发现任何 *.safetensors。"
              "可用 --model 直接给绝对路径，或设 $LAYA_MODEL_DIR 指定搜索根。")
        return
    cur: Optional[str] = None
    if spec:
        try:
            cur = resolve_model(spec)
        except SystemExit:
            cur = None
    elif DEFAULT_MODEL:
        cur = _norm(DEFAULT_MODEL)
    print(f"[laya-cuda] 发现 {len(ms)} 个权重：")
    marked = False
    for p in ms:
        st = os.stat(p)
        mark = "*" if p == cur else " "
        marked = marked or mark == "*"
        try:
            rel = os.path.relpath(p, os.getcwd())
        except ValueError:
            rel = ""
        print(f"  {mark} {p}")
        print(f"      {st.st_size / 1e6:8.0f} MB   "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"
              f"   {rel if not rel.startswith('..') else ''}")
    if marked:
        print("  （* = 当前 --model 解析到的文件）")
    elif cur:
        print(f"  （当前目标不在上述搜索范围内：{cur}）")


def _pick_in_dir(d: str) -> Optional[str]:
    """目录 → 权重文件。优先 model.safetensors；其次唯一的 *.safetensors。"""
    pref = os.path.join(d, "model" + _MODEL_SUFFIX)
    if os.path.isfile(pref):
        return _norm(pref)
    hits = sorted(f for f in os.listdir(d)
                  if f.lower().endswith(_MODEL_SUFFIX) and os.path.isfile(os.path.join(d, f)))
    return _norm(os.path.join(d, hits[0])) if len(hits) == 1 else None


def resolve_model(spec: Optional[str] = None) -> str:
    """把 `--model` 的取值解析成唯一的 .safetensors 绝对路径。

    接受：文件 / 目录 / glob（* ? [ ]）/ 省略 .safetensors 后缀。
    相对路径按 CWD → 脚本目录 → $LAYA_MODEL_DIR 依次尝试，全部失败则带候选清单报错。
    """
    if not spec:
        spec = DEFAULT_MODEL
    raw = os.path.expandvars(os.path.expanduser(str(spec)))
    roots = [raw] if os.path.isabs(raw) else \
        [os.path.join(d, raw) for d in model_search_dirs()] + [raw]

    for r in roots:
        if any(ch in r for ch in "*?["):        # glob：必须唯一命中
            hits = sorted({_norm(h) for h in glob.glob(r, recursive=True)
                           if os.path.isfile(h) and h.lower().endswith(_MODEL_SUFFIX)})
            if len(hits) == 1:
                return hits[0]
            if hits:
                raise SystemExit(f"[laya-cuda] glob 命中 {len(hits)} 个权重，请明确指定：\n    "
                                 + "\n    ".join(hits))
            continue
        for cand in (r, r + _MODEL_SUFFIX):     # 文件（自动补后缀）
            if os.path.isfile(cand):
                return _norm(cand)
        if os.path.isdir(r):                    # 目录 → 里面的权重
            got = _pick_in_dir(r)
            if got:
                return got
            hits = sorted(f for f in os.listdir(r) if f.lower().endswith(_MODEL_SUFFIX))
            raise SystemExit(f"[laya-cuda] 目录 {r} 内的权重不唯一（或为空）：\n    "
                             + ("\n    ".join(hits) if hits else "(无 *.safetensors)")
                             + "\n  请用 --model 明确指定其中一个。")
    raise SystemExit(_not_found_msg(str(spec)))


def _not_found_msg(spec: str) -> str:
    msg = [f"[laya-cuda] 找不到权重文件: {spec}"]
    avail = list_models()
    if avail:
        msg.append("  本机可见的模型（同 --list）：")
        msg += [f"    {p}" for p in avail[:12]]
        if len(avail) > 12:
            msg.append(f"    …另有 {len(avail) - 12} 个")
    else:
        msg.append("  未在搜索目录下发现任何 *.safetensors："
                   + " | ".join(model_search_dirs()))
    msg.append("  可指定：文件 / 目录（含 model.safetensors）/ glob / 省略 .safetensors 后缀")
    return "\n".join(msg)


def resolve_cfg_dirs(model_path: str, override: Optional[str],
                     strict: bool) -> Tuple[List[str], str]:
    """config.json / rl_agent_config.json 的查找目录（有序）。

    优先级：--config > 模型目录 > $LAYA_CONFIG_DIR > 脚本目录（--strict 时去掉最后一项）。
    注意原实现是「模型目录 → 脚本目录」，即模型目录没有 config.json 时**静默**套用脚本
    目录那份 —— 换模型时这等于拿 A 的架构参数去读 B 的权重，故新增 --strict 可关闭。
    """
    src: List[str] = []
    if override:
        p = _norm(override)
        d = p if os.path.isdir(p) else os.path.dirname(p)
        if not os.path.isdir(d):
            raise SystemExit(f"[laya-cuda] --config 指向的目录不存在: {d}")
        return [d], "--config (%s)" % d
    pairs = [(os.path.dirname(model_path), "模型目录")]
    env = os.environ.get("LAYA_CONFIG_DIR")
    if env and os.path.isdir(_norm(env)):
        pairs.append((_norm(env), "$LAYA_CONFIG_DIR"))
    if not strict:
        pairs.append((HERE, "脚本目录·回退"))
    dirs: List[str] = []
    for d, tag in pairs:
        if d in dirs:                    # 模型就在脚本目录时会命中同一目录 → 合并标注
            i = dirs.index(d)
            short = tag.split("·")[0]
            if short not in src[i]:
                src[i] = f"{src[i]}/{short}"
            continue
        dirs.append(d)
        src.append(tag)
    return dirs, " → ".join(src)


def resolve_tokenizer(model_path: str, override: Optional[str],
                      strict: bool) -> Tuple[Optional[str], str]:
    """tokenizer.json 的路径与出处。优先级：--tokenizer > 模型目录 > $LAYA_TOKENIZER
    > 脚本目录（--strict 时去掉最后一项）。找不到返回 (None, 说明)。"""
    if override:
        p = _norm(override)
        if os.path.isdir(p):
            p = os.path.join(p, "tokenizer.json")
        if not os.path.isfile(p):
            raise SystemExit(f"[laya-cuda] --tokenizer 指向的文件不存在: {p}")
        return p, "--tokenizer"
    p = os.path.join(os.path.dirname(model_path), "tokenizer.json")
    if os.path.isfile(p):
        return p, "模型目录"
    env = os.environ.get("LAYA_TOKENIZER")
    if env and os.path.isfile(_norm(env)):
        return _norm(env), "$LAYA_TOKENIZER"
    if not strict:
        p = os.path.join(HERE, "tokenizer.json")
        if os.path.isfile(p):
            return p, "脚本目录·回退"
    return None, "未找到 → 字节级兜底（可跑通，语义无效）"


@contextlib.contextmanager
def cfg_dirs_override(dirs: Sequence[str]):
    """临时替换 laya_verify 的 config 查找目录列表（apply_official_config 内部用它）。

    只改一个模块级函数、调用完立即还原，因此不会给 CPU 版留下状态残留。
    """
    orig = L._cfg_dirs
    L._cfg_dirs = lambda model_path="": list(dirs)      # type: ignore[assignment]
    try:
        yield
    finally:
        L._cfg_dirs = orig


def preflight(path: str) -> Dict[str, Any]:
    """加载前的张量清单体检：把"权重不完整"变成一句人话，而不是深处的 KeyError。"""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[laya-cuda] 不是合法的 safetensors 文件：{path}\n  {e}")
    keys = [k for k in hdr if k != "__metadata__"]
    n_layers = sum(1 for k in keys
                   if re.match(r"^encoder\.layers\.\d+\.attn\.Wqkv\.weight$", k))
    has_emb = "encoder.embeddings.tok_embeddings.weight" in hdr
    has_scorer = any(k.startswith("scorer.") for k in keys)

    if not has_emb or n_layers == 0:
        raise SystemExit(
            f"[laya-cuda] 权重不完整：{path}\n"
            f"  张量 {len(keys)} 个，其中 encoder 层 {n_layers} 层"
            f"{'、tok_embeddings 缺失' if not has_emb else ''}。\n"
            "  这看起来是**增量/仅头部**的权重（例如只保存 scorer）。本脚本要求完整\n"
            "  checkpoint（含全部 encoder.* 与 scorer.*）；请改用完整权重，或先把增量\n"
            "  合并进基座再加载。")
    if not has_scorer:
        sys.stderr.write(
            f"[laya-cuda] [!] 警告：{os.path.basename(path)} 里没有 scorer.* 张量 ——\n"
            "  决策头缺失，logits 将无意义（会以 0 维 scorer 走空路径）。\n")
    return {"keys": len(keys), "n_layers": n_layers, "has_scorer": has_scorer}


# =============================================================================
# 1. 融合 kernel（RawKernel；编译失败自动降级，不阻断）
# =============================================================================
# 为什么值得写：cupy 表达式不融合，一个 layernorm 会展开成
# mean → sub → square → mean → sqrt → div → mul(→add) 共 5~7 次 launch。
# 22 层 × 2 处 = 44 次 × 6 ≈ 264 次 launch，在小 T 下这就是主要开销。
# 融合成 1 次后，单前向 launch 数从 ~300 降到 ~120。

_LN_SRC = r"""
extern "C" __global__
void ln_fwd(const float* __restrict__ x, const float* __restrict__ w,
            const float* __restrict__ b, float* __restrict__ y,
            int D, float eps, int has_bias) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    const float* xr = x + (long long)row * D;
    float* yr = y + (long long)row * D;

    float s1 = 0.f, s2 = 0.f;   // sum(x), sum(x^2)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = xr[i];
        s1 += v;
        s2 += v * v;
    }
    // 第一级：warp 内 butterfly 归约（shfl 只在 32 lane 内有效，
    // 不能拿它跨 warp 归约 —— 这是最初版本的 bug：blockDim=256 时
    // shfl_down(off>=32) 读的是自己，均值只统计了 1/8 的数据）
    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        s2 += __shfl_xor_sync(0xffffffffu, s2, off);
    }
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if (nwarps > 1) {
        // 第二级：跨 warp 走 shared memory
        if (lane == 0) { smem[wid * 2] = s1; smem[wid * 2 + 1] = s2; }
        __syncthreads();
        if (wid == 0) {
            float a = (lane < nwarps) ? smem[lane * 2] : 0.f;
            float c = (lane < nwarps) ? smem[lane * 2 + 1] : 0.f;
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_xor_sync(0xffffffffu, a, off);
                c += __shfl_xor_sync(0xffffffffu, c, off);
            }
            if (lane == 0) { smem[0] = a; smem[1] = c; }
        }
        __syncthreads();
        s1 = smem[0]; s2 = smem[1];
    }
    float mean = s1 / (float)D;
    float var = s2 / (float)D - mean * mean;
    if (var < 0.f) var = 0.f;
    float inv = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = (xr[i] - mean) * inv;
        yr[i] = v * w[i] + (has_bias ? b[i] : 0.f);
    }
}
"""

_GEGLU_SRC = r"""
extern "C" __global__
void geglu_fwd(const float* __restrict__ g, float* __restrict__ out,
               int T, int inter) {
    // g: [T, 2*inter]，前 inter 为 gate、后 inter 为 value（mlp_gate=first）
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)T * inter;
    if (idx >= total) return;
    int t = (int)(idx / inter);
    int j = (int)(idx - (long long)t * inter);
    float a = g[(long long)t * 2 * inter + j];
    float b = g[(long long)t * 2 * inter + inter + j];
    // exact GELU: 0.5a(1+erf(a/sqrt2))
    float av = fabsf(a) / 1.41421356f;
    float tt = 1.0f / (1.0f + 0.3275911f * av);
    float erf = 1.0f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt
                 - 0.284496736f) * tt + 0.254829592f) * tt * __expf(-av * av);
    erf = a < 0.f ? -erf : erf;
    out[idx] = 0.5f * a * (1.0f + erf) * b;
}
"""

_ROPE_SRC = r"""
extern "C" __global__
void rope_fwd(const float* __restrict__ qk, const float* __restrict__ cos,
              const float* __restrict__ sin, float* __restrict__ out,
              long long total, int half, int T) {
    // qk/out 布局: [2, nh, T, dh]（q 与 k 拼在一起，一次 launch 处理完）
    // cos/sin 布局: [T, dh]，且 emb=[f,f] 前后两半相同 → 索引恒用 pos*dh+j
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dh = 2 * half;
    int j = (int)(idx % (long long)half);
    bool second_half = (idx % (long long)dh) >= half;
    long long base = idx - (idx % (long long)dh);
    // rotate_half(x) = concat([-x[half:], x[:half]])
    //   → i <  half : 配对是 -x[i+half]
    //   → i >= half : 配对是 +x[i-half]   （两个分支符号相反，写反则 RoPE 相位错半周）
    float v = second_half ? qk[base + half + j] : qk[base + j];
    float o = second_half ? qk[base + j] : -qk[base + half + j];
    int pos = (int)((idx / (long long)dh) % (long long)T);
    float c = cos[(long long)pos * dh + j];
    float s = sin[(long long)pos * dh + j];
    out[idx] = v * c + o * s;
}
"""


class FusedKernels:
    """编译融合 kernel；任一编译失败则该项降级为 cupy 表达式（不阻断）。"""

    def __init__(self):
        self.ln = None
        self.geglu = None
        self.rope = None
        self.notes: List[str] = []
        self._build()

    def _build(self) -> None:
        try:
            self.ln = cp.RawKernel(_LN_SRC, "ln_fwd", options=("--std=c++11",))
            d = cp.zeros((1, 768), dtype=cp.float32)
            w = cp.ones(768, dtype=cp.float32)
            self.ln((1,), (256,), (d, w, w, d, np.int32(768), np.float32(1e-5), np.int32(1)),
                    shared_mem=(256 >> 5) * 2 * 4)
            self.notes.append("fused layernorm: RawKernel 已启用")
        except Exception as e:  # noqa: BLE001
            self.ln = None
            self.notes.append(f"fused layernorm: 降级到 cupy 表达式（{type(e).__name__}: {e}）")
        try:
            self.geglu = cp.RawKernel(_GEGLU_SRC, "geglu_fwd", options=("--std=c++11",))
            g = cp.zeros((2, 64), dtype=cp.float32)
            o = cp.zeros((2, 32), dtype=cp.float32)
            self.geglu((1,), (64,), (g, o, np.int32(2), np.int32(32)))
            self.notes.append("fused GeGLU: RawKernel 已启用")
        except Exception as e:  # noqa: BLE001
            self.geglu = None
            self.notes.append(f"fused GeGLU: 降级到 cupy 表达式（{type(e).__name__}: {e}）")
        try:
            self.rope = cp.RawKernel(_ROPE_SRC, "rope_fwd", options=("--std=c++11",))
            self.notes.append("fused RoPE: RawKernel 已启用")
        except Exception as e:  # noqa: BLE001
            self.rope = None
            self.notes.append(f"fused RoPE: 降级到 cupy 表达式（{type(e).__name__}: {e}）")

    @property
    def ok(self) -> int:
        return sum(1 for k in (self.ln, self.geglu, self.rope) if k is not None)


# =============================================================================
# 2. GPU 后端模型
# =============================================================================

class CudaLayaModel(L.LayaModel):
    """LayaModel 的 CUDA 后端：接口完全一致，仅更换数组设备与计时方式。"""

    def __init__(self, st: "L.SafeTensors", cfg: Dict[str, Any],
                 dtype: str = "fp32", fused: Optional[FusedKernels] = None):
        self.st = st
        self.cfg = cfg if cfg is not None else dict(L.DEFAULT_CFG)
        for k, v in L.DEFAULT_CFG.items():
            self.cfg.setdefault(k, v)
        c = self.cfg
        self.H, self.L = c["hidden"], c["n_layers"]
        self.nh = c["n_heads"]
        self.dh = self.H // self.nh
        self.scale = self.dh ** -0.5

        self.dtype_name = dtype
        self.gemm_dtype = cp.float16 if dtype == "fp16" else cp.float32
        self.fused = fused or FusedKernels()

        def up(name: str, cast_gemm: bool = True):
            """host → device。GEMM 权重按档位转 fp16，其余保持 fp32。"""
            a = st.get(name, fp32=True)
            if cast_gemm and self.gemm_dtype == cp.float16:
                a = a.astype(np.float16)
            return cp.asarray(a)

        # 词表：fp16 常驻（256000×768 fp32 = 786 MB，没必要）
        self.tok = cp.asarray(st.get("encoder.embeddings.tok_embeddings.weight", fp32=False))
        self.emb_norm = up("encoder.embeddings.norm.weight", cast_gemm=False)
        self.final_norm = up("encoder.final_norm.weight", cast_gemm=False)

        self.layers: List[Dict[str, Any]] = []
        for i in range(self.L):
            p = f"encoder.layers.{i}."
            d: Dict[str, Any] = {
                "Wqkv": up(p + "attn.Wqkv.weight"),
                "Wo": up(p + "attn.Wo.weight"),
                "Wi": up(p + "mlp.Wi.weight"),
                "Wo2": up(p + "mlp.Wo.weight"),
            }
            # ModernBERT 指纹：layer 0 的 attn_norm 是 Identity（权重文件里不存在）
            an = st.get_optional(p + "attn_norm.weight", fp32=True)
            d["attn_norm"] = None if an is None else cp.asarray(an.astype(np.float32))
            d["mlp_norm"] = cp.asarray(st.get(p + "mlp_norm.weight", fp32=True))
            self.layers.append(d)
        self.identity_first_attn_norm = self.layers[0]["attn_norm"] is None

        self.head_layers: List[Dict[str, Any]] = []
        for i in range(len(st.matching(r"^head\.layers\.\d+\.norm1\.weight$"))):
            p = f"head.layers.{i}."
            self.head_layers.append({
                "in_w": up(p + "self_attn.in_proj_weight"),
                "in_b": up(p + "self_attn.in_proj_bias", cast_gemm=False),
                "out_w": up(p + "self_attn.out_proj.weight"),
                "out_b": up(p + "self_attn.out_proj.bias", cast_gemm=False),
                "l1_w": up(p + "linear1.weight"), "l1_b": up(p + "linear1.bias", cast_gemm=False),
                "l2_w": up(p + "linear2.weight"), "l2_b": up(p + "linear2.bias", cast_gemm=False),
                "n1_w": up(p + "norm1.weight", cast_gemm=False),
                "n1_b": up(p + "norm1.bias", cast_gemm=False),
                "n2_w": up(p + "norm2.weight", cast_gemm=False),
                "n2_b": up(p + "norm2.bias", cast_gemm=False),
            })

        import re as _re
        self.scorer_ops: List[Tuple[int, Dict[str, Any]]] = []
        idxs = sorted(int(m.group(1)) for m in
                      (_re.match(r"^scorer\.(\d+)\.weight$", k) for k in st.keys) if m)
        self.scorer_gap = False
        for pos, i in enumerate(idxs):
            if pos and i != idxs[pos - 1] + 1:
                self.scorer_gap = True
            self.scorer_ops.append((i, {"w": up(f"scorer.{i}.weight"),
                                        "b": up(f"scorer.{i}.bias", cast_gemm=False)}))
        self.scorer_out_dim = int(self.scorer_ops[-1][1]["w"].shape[0]) if self.scorer_ops else 0

        self.act_ops: List[Tuple[int, Dict[str, Any]]] = []
        aidx = sorted(int(m.group(1)) for m in
                      (_re.match(r"^act_head\.(\d+)\.weight$", k) for k in st.keys) if m)
        for i in aidx:
            self.act_ops.append((i, {"w": up(f"act_head.{i}.weight"),
                                     "b": up(f"act_head.{i}.bias", cast_gemm=False)}))
        self.act_in_dim = int(self.act_ops[0][1]["w"].shape[1]) if self.act_ops else 0

        self.type_emb = up("type_emb.weight", cast_gemm=False)
        self.temperature = np.asarray(st.get("temperature", fp32=False), dtype=np.float64)

        self._rope_theta_used = float(c["rope_theta"])
        self._inv_freq_host = 1.0 / (self._rope_theta_used **
                                     (np.arange(0, self.dh, 2, dtype=np.float64) / self.dh))
        self.inv_freq = cp.asarray(self._inv_freq_host)
        self._rope_cache: Dict[int, Tuple[Any, Any]] = {}
        self._band_cache: Dict[Tuple[int, int], Any] = {}
        self._feats_cache: Dict[int, Any] = {}
        self._sync_rope()

    # -- RoPE / band 缓存（同一 T 只算一次） --------------------------------
    def _sync_rope(self) -> None:
        theta = float(self.cfg["rope_theta"])
        if theta != self._rope_theta_used:
            self._rope_theta_used = theta
            self._inv_freq_host = 1.0 / (theta ** (np.arange(0, self.dh, 2, dtype=np.float64) / self.dh))
            self.inv_freq = cp.asarray(self._inv_freq_host)
            self._rope_cache.clear()

    def _rope_tables(self, T: int) -> Tuple[Any, Any]:
        got = self._rope_cache.get(T)
        if got is not None:
            return got
        pos = np.arange(T, dtype=np.float64)
        f = np.outer(pos, self._inv_freq_host)          # [T, dh/2]
        emb = np.concatenate([f, f], axis=-1)
        cos = cp.asarray(np.cos(emb).astype(np.float32))
        sin = cp.asarray(np.sin(emb).astype(np.float32))
        self._rope_cache[T] = (cos, sin)
        return self._rope_cache[T]

    def _band_mask(self, T: int, band: int) -> Any:
        key = (T, band)
        got = self._band_cache.get(key)
        if got is not None:
            return got
        idx = np.arange(T)
        far = np.abs(idx[:, None] - idx[None, :]) > band
        self._band_cache[key] = cp.asarray(far)
        return self._band_cache[key]

    # -- 基础块（全部 cupy） -------------------------------------------------
    def _ln(self, x, w, b=None, eps: float = 1e-5):
        T, D = x.shape
        if self.fused.ln is not None and x.dtype == cp.float32:
            y = cp.empty_like(x)
            blk = 256
            self.fused.ln((T,), (blk,),
                          (x, w, b if b is not None else w, y,
                           np.int32(D), np.float32(eps),
                           np.int32(1 if b is not None else 0)),
                          shared_mem=(blk >> 5) * 2 * 4)
            return y
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        y = xc / cp.sqrt(var + eps)
        return y * w if b is None else y * w + b

    def _act(self, x, kind: str = "gelu"):
        if kind == "relu":
            return cp.maximum(x, 0.0)
        if kind == "gelu_tanh":
            return 0.5 * x * (1.0 + cp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
        a = cp.abs(x) / 1.4142135623730951
        t = 1.0 / (1.0 + 0.3275911 * a)
        erf = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                      - 0.284496736) * t + 0.254829592) * t * cp.exp(-a * a)
        erf = cp.where(x < 0, -erf, erf)
        return 0.5 * x * (1.0 + erf)

    def _softmax(self, z, axis: int = -1):
        z = z - z.max(axis=axis, keepdims=True)
        e = cp.exp(z)
        return e / e.sum(axis=axis, keepdims=True)

    def _gemm(self, a, w):
        """a[M,K] @ w[N,K].T → [M,N]。fp16 档走 TensorCore，输出回到 fp32 累加。"""
        if self.gemm_dtype == cp.float16:
            out = cp.matmul(a.astype(cp.float16), w.T.astype(cp.float16))
            return out.astype(cp.float32)
        return cp.matmul(a, w.T)

    def _rope_apply(self, q, k, T: int):
        """q,k: [nh,T,dh] → RoPE 后。融合 kernel 把 q/k 拼一次处理。"""
        cos, sin = self._rope_tables(T)          # [T, dh]
        if self.fused.rope is not None:
            qk = cp.concatenate([q.reshape(-1), k.reshape(-1)])
            out = cp.empty_like(qk)
            half = self.dh // 2
            total = int(qk.shape[0])
            blk = 256
            self.fused.rope((((total + blk - 1) // blk),), (blk,),
                            (qk, cos, sin, out, np.int64(total),
                             np.int32(half), np.int32(T)))
            n = total // 2
            return out[:n].reshape(q.shape), out[n:].reshape(k.shape)
        half = self.dh // 2
        c = cos.T[:, None, :] if False else cos[:, None, :]        # [T,1,dh]
        s = sin[:, None, :]
        rot = lambda x: cp.concatenate([-x[..., half:], x[..., :half]], axis=-1)  # noqa: E731
        return (q * c + rot(q) * s), (k * c + rot(k) * s)

    def _attend(self, q, k, v, band: Optional[int]):
        # 单条 [nh,T,dh] / 批量 [B,nh,T,dh] —— 转置轴数不同，别写死
        kt = k.transpose(0, 1, 3, 2) if k.ndim == 4 else k.transpose(0, 2, 1)
        s = cp.matmul(q, kt) * self.scale
        T = q.shape[-2]
        if band is not None and T > band + 1:
            s = cp.where(self._band_mask(T, band)[None], cp.float32(-1e30), s)
        return cp.matmul(self._softmax(s, -1), v)

    def _encoder_layer(self, x, w: Dict[str, Any], band: Optional[int], offset: int):
        T = x.shape[0]
        h = x if w["attn_norm"] is None else self._ln(x, w["attn_norm"], None, self.cfg["norm_eps"])
        qkv = self._gemm(h, w["Wqkv"])                            # [T, 2304]
        qkv = qkv.reshape(T, 3, self.nh, self.dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(1, 0, 2).copy(); k = k.transpose(1, 0, 2).copy()
        v = v.transpose(1, 0, 2).copy()
        q, k = self._rope_apply(q, k, T)
        ctx = self._attend(q, k, v, band)
        ctx = ctx.transpose(1, 0, 2).reshape(T, self.H)
        x = x + self._gemm(ctx, w["Wo"])

        h = self._ln(x, w["mlp_norm"], None, self.cfg["norm_eps"])
        g = self._gemm(h, w["Wi"])                                # [T, 2304]
        inter = g.shape[-1] // 2
        if self.fused.geglu is not None:
            out = cp.empty((T, inter), dtype=cp.float32)
            blk = 256
            n = T * inter
            self.fused.geglu(((n + blk - 1) // blk,), (blk,),
                             (g, out, np.int32(T), np.int32(inter)))
        else:
            if self.cfg["mlp_gate"] == "interleaved":
                a, b = g[:, 0::2], g[:, 1::2]
            else:
                a, b = g[:, :inter], g[:, inter:]
            out = self._act(a, "gelu") * b
        x = x + self._gemm(out, w["Wo2"])
        return x

    def _head_layer(self, x, w: Dict[str, Any]):
        # pre-norm —— 与 laya_verify.py 的 _head_layer 保持一致（原因见那里）。
        # 原来的 post-norm 会让最后一步 LayerNorm 的 bias 主导输出，把 marker
        # 隐状态压成同一个向量，读出对选项不敏感。
        eps = self.cfg["norm_eps"]
        T = x.shape[0]
        h = self._ln(x, w["n1_w"], w["n1_b"], eps)
        qkv = self._gemm(h, w["in_w"]) + w["in_b"]
        q, k, v = cp.split(qkv, 3, axis=-1)
        r = lambda t: t.reshape(T, self.nh, self.dh).transpose(1, 0, 2).copy()  # noqa: E731
        ctx = self._attend(r(q), r(k), r(v), None)
        ctx = ctx.transpose(1, 0, 2).reshape(T, self.H)
        x = x + self._gemm(ctx, w["out_w"]) + w["out_b"]
        h = self._ln(x, w["n2_w"], w["n2_b"], eps)
        f = self._act(self._gemm(h, w["l1_w"]) + w["l1_b"], self.cfg["head_act"])
        return x + self._gemm(f, w["l2_w"]) + w["l2_b"]

    def _scorer(self, markers):
        h = markers
        prev = None
        for i, op in self.scorer_ops:
            if prev is not None and i != prev + 1:
                h = self._act(h, "gelu")
            if op["w"].ndim == 1:
                h = self._ln(h, op["w"], op["b"], self.cfg["norm_eps"])
            else:
                h = self._gemm(h, op["w"]) + op["b"]
            prev = i
        return h.reshape(-1) if h.ndim > 1 else h

    def _act_head(self, pooled, feats):
        h = cp.concatenate([pooled, feats]).astype(cp.float32)[None]
        prev = None
        for i, op in self.act_ops:
            if prev is not None and i != prev + 1:
                h = self._act(h, "gelu")
            h = self._gemm(h, op["w"]) + op["b"]
            prev = i
        return self._softmax(h.reshape(-1), -1)

    # -- 主前向（核心：纯 GPU，无任何 D2H / 同步，可被 CUDA Graph 捕获） ----
    def _forward_core(self, idx: Any, mp: Any, mt: Any, feats: Any):
        self._sync_rope()
        T = int(idx.shape[0])
        x = self.tok[idx].astype(cp.float32)
        x = self._ln(x, self.emb_norm, None, self.cfg["norm_eps"])

        stage = self.cfg["type_emb_stage"]
        if stage == "input":
            x[mp] = x[mp] + self.type_emb[mt]

        attn_mode = self.cfg["attention"]
        bands = []
        for i in range(self.L):
            if attn_mode == "alternating3":
                bands.append(None if i % 3 == 0 else self.cfg["local_radius"])
            else:
                bands.append(None)
        for i, w in enumerate(self.layers):
            x = self._encoder_layer(x, w, bands[i], offset=0)
        x = self._ln(x, self.final_norm, None, self.cfg["norm_eps"])

        if stage == "pre_head":
            x[mp] = x[mp] + self.type_emb[mt]

        for hl in self.head_layers:
            x = self._head_layer(x, hl)

        marker_h = x[mp]
        logits = self._scorer(marker_h)
        pooled = marker_h.mean(0)
        act = self._act_head(pooled, feats)
        return logits, marker_h, act, x

    def _feats_buffer(self, n_opt: int, type0: int) -> Any:
        """act_head 的 4 个标量特征。地址必须固定（否则 CUDA Graph 捕获的是悬空指针），
        所以按维度缓存 buffer，只更新内容。"""
        n = max(0, self.act_in_dim - self.H)
        key = n
        buf = self._feats_cache.get(key)
        if buf is None:
            buf = cp.zeros(n, dtype=cp.float32)
            self._feats_cache[key] = buf
        vals = np.zeros(n, dtype=np.float32)
        if n >= 3:
            vals[int(type0) % 3] = 1.0
        if n >= 4:
            vals[3] = math.log1p(max(1, n_opt)) / math.log(64.0)
        buf.set(vals)
        return buf

    def forward(self, ids: List[int], marker_pos: List[int],
                marker_type: List[int]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        T = len(ids)
        idx = cp.asarray(np.asarray(ids, dtype=np.int64))
        mp = cp.asarray(np.asarray(marker_pos, dtype=np.int64))
        mt = cp.asarray(np.asarray(marker_type, dtype=np.int64))
        feats = self._feats_buffer(len(marker_pos), int(np.asarray(marker_type)[0]))

        logits, marker_h, act, x = self._forward_core(idx, mp, mt, feats)

        # 关键：CUDA 是异步的，不 synchronize 测到的是 launch 时间
        cp.cuda.get_current_stream().synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0

        nan_flag = bool(cp.any(cp.isnan(x))) or bool(cp.any(cp.isinf(x)))
        health = {
            "has_nan": bool(nan_flag),
            "has_inf": False,
            "hidden_absmax": float(cp.abs(x).max()),
            "marker_absmax": float(cp.abs(marker_h).max()),
            "mode": "eager",
        }
        return {
            # logits / act 传回 host：下游（温度、softmax、报告）全是 numpy 逻辑，
            # 且这两个张量只有几十个元素，D2H 开销可忽略（<0.05 ms）。
            "logits": cp.asnumpy(logits).astype(np.float64),
            "marker_h": marker_h,
            "act": cp.asnumpy(act).astype(np.float64),
            "tokens": T,
            "elapsed_ms": elapsed,
            "hidden": x,
            "health": health,
        }

    # -- CUDA Graph 路径 -----------------------------------------------------
    # -- 批处理前向 ----------------------------------------------------------
    # 为什么 batch 才是这里的杠杆：单条的 op 数固定（约 500 个 cupy 调用），
    # 实测其中 ~70% 是 Python 调度开销而非算力。把 B 条样本塞进同一次前向，
    # op 数**不变**、GEMM 的 M 维变大 → 单条摊到的开销被除以 B。
    # 代价：需要 padding + attention mask（padding 位置不能当 key 被看到）。
    def _ln_any(self, x, w, b=None, eps: float = 1e-5):
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        return self._ln(x2, w, b, eps).reshape(orig)

    def _head_layer_b(self, x, w: Dict[str, Any], B: int, T: int):
        # pre-norm —— 与 _head_layer 一致。
        eps = self.cfg["norm_eps"]
        h = self._ln_any(x, w["n1_w"], w["n1_b"], eps)
        qkv = self._gemm(h, w["in_w"]) + w["in_b"]
        q, k, v = cp.split(qkv, 3, axis=-1)
        r = lambda t: cp.ascontiguousarray(  # noqa: E731
            t.reshape(B, T, self.nh, self.dh).transpose(0, 2, 1, 3))
        ctx = self._attend(r(q), r(k), r(v), None)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.H)
        x = x + self._gemm(ctx, w["out_w"]) + w["out_b"]
        h = self._ln_any(x, w["n2_w"], w["n2_b"], eps)
        f = self._act(self._gemm(h, w["l1_w"]) + w["l1_b"], self.cfg["head_act"])
        return x + self._gemm(f, w["l2_w"]) + w["l2_b"]

    def _forward_batch(self, ids_b: Any, pad_mask: Any, mp_flat: Any,
                       mt_flat: Any, feats_b: Any, marker_slices: List[Tuple[int, int]]):
        B, T = ids_b.shape
        eps = self.cfg["norm_eps"]
        x = self.tok[ids_b].astype(cp.float32)                     # [B,T,H]
        x = cp.where(pad_mask[..., None], cp.float32(0.0), x)      # padding 位置置 0
        x = self._ln_any(x, self.emb_norm, None, eps)

        def inject_type(x3):
            if self.cfg["type_emb_stage"] != "input":
                return x3
            flat = x3.reshape(B * T, self.H)
            flat[mp_flat] = flat[mp_flat] + self.type_emb[mt_flat]
            return flat.reshape(B, T, self.H)

        x = inject_type(x)

        pad_key = pad_mask[:, None, None, :]                       # [B,1,1,T]
        radius = self.cfg["local_radius"]
        idx = np.arange(T)
        band_np = np.abs(idx[:, None] - idx[None, :]) > radius
        band = cp.asarray(band_np)

        for i, w in enumerate(self.layers):
            is_full = (self.cfg["attention"] != "alternating3") or (i % 3 == 0)
            h = x if w["attn_norm"] is None else self._ln_any(x, w["attn_norm"], None, eps)
            qkv = self._gemm(h, w["Wqkv"]).reshape(B, T, 3, self.nh, self.dh)
            q = cp.ascontiguousarray(qkv[:, :, 0].transpose(0, 2, 1, 3))
            k = cp.ascontiguousarray(qkv[:, :, 1].transpose(0, 2, 1, 3))
            v = cp.ascontiguousarray(qkv[:, :, 2].transpose(0, 2, 1, 3))
            q, k = self._rope_apply(q, k, T)
            s = cp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale  # [B,nh,T,T]
            m = pad_key if is_full else (pad_key | band[None, None, :, :])
            s = cp.where(m, cp.float32(-1e30), s)
            ctx = cp.matmul(self._softmax(s, -1), v)
            ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.H)
            x = x + self._gemm(ctx, w["Wo"])

            h = self._ln_any(x, w["mlp_norm"], None, eps)
            g = self._gemm(h, w["Wi"])                              # [B,T,2*inter]
            inter = g.shape[-1] // 2
            if self.fused.geglu is not None:
                g2 = g.reshape(-1, 2 * inter)
                out = cp.empty((g2.shape[0], inter), dtype=cp.float32)
                blk, n = 256, g2.shape[0] * inter
                self.fused.geglu(((n + blk - 1) // blk,), (blk,),
                                 (g2, out, np.int32(g2.shape[0]), np.int32(inter)))
                out = out.reshape(B, T, inter)
            else:
                a, b = (g[:, :, :inter], g[:, :, inter:]) \
                    if self.cfg["mlp_gate"] != "interleaved" else (g[:, :, 0::2], g[:, :, 1::2])
                out = self._act(a, "gelu") * b
            x = x + self._gemm(out, w["Wo2"])

        x = self._ln_any(x, self.final_norm, None, eps)
        if self.cfg["type_emb_stage"] == "pre_head":
            x = inject_type(x)
        for hl in self.head_layers:
            x = self._head_layer_b(x, hl, B, T)

        flat = x.reshape(B * T, self.H)
        marker_h = flat[mp_flat]                                    # [M,H]
        logits = self._scorer(marker_h)
        pooled = cp.stack([marker_h[s:e].mean(0) for s, e in marker_slices])
        act = self._act_head_b(pooled, feats_b)
        return logits, act

    def _act_head_b(self, pooled, feats_b):
        h = cp.concatenate([pooled, feats_b], axis=1).astype(cp.float32)
        prev = None
        for i, op in self.act_ops:
            if prev is not None and i != prev + 1:
                h = self._act(h, "gelu")
            h = self._gemm(h, op["w"]) + op["b"]
            prev = i
        return self._softmax(h, -1)

    def forward_graph(self, ids: List[int], marker_pos: List[int],
                      marker_type: List[int], runner: "GraphRunner") -> Dict[str, Any]:
        t0 = time.perf_counter()
        logits_np, act_np, hidden_np, health = runner.run(ids, marker_pos, marker_type)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {"logits": logits_np, "marker_h": None, "act": act_np,
                "tokens": len(ids), "elapsed_ms": elapsed, "hidden": hidden_np,
                "health": health}


class GraphRunner:
    """把一次前向捕获成 CUDA Graph。

    为什么需要它：eager 模式下每个 cupy op 都要走一遍 Python → cupy → cuBLAS 的
    调度，22 层 × 约 20 个 op ≈ 440 次，实测这一项就占了 20 ms 里的 16 ms
    （纯 GEMM 只要 4.4 ms）。捕获后重放是**一次** launch，Python 开销归零。

    硬约束（违反会在 end_capture 时报错）：
      · 捕获期间不能有 D2H/同步 → 所以 _forward_core 里没有任何 asnumpy / float()
      · 捕获期间不能新分配显存 → 捕获前先在同一条 stream 上 warmup 一次，
        让 cupy 内存池把临时块都分配好
      · 输入输出 buffer 地址必须固定 → 预分配 ids/mp/mt/feats/logits/act，只更新内容
      · 形状必须固定 → 按 (T, n_markers) 分桶缓存
    """

    def __init__(self, model: CudaLayaModel):
        self.model = model
        self.graphs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.enabled = True
        self.last_error: Optional[str] = None

    def run(self, ids, marker_pos, marker_type) -> Tuple[Any, Any, Dict[str, Any]]:
        key = (len(ids), len(marker_pos))
        ent = self.graphs.get(key)
        if ent is None:
            ent = self._capture(ids, marker_pos, marker_type)
            if ent is None:                    # 捕获失败 → 交由上层降级
                raise RuntimeError(self.last_error or "graph capture failed")
            self.graphs[key] = ent
        ent["ids"].set(np.asarray(ids, dtype=np.int64))
        ent["mp"].set(np.asarray(marker_pos, dtype=np.int64))
        ent["mt"].set(np.asarray(marker_type, dtype=np.int64))
        feats = self.model._feats_buffer(len(marker_pos), int(marker_type[0]))
        ent["graph"].launch()
        cp.cuda.get_current_stream().synchronize()
        # 重放后才取 health 与 hidden：这两个必然要做 D2H，放在图外，
        # 否则会破坏捕获（捕获期间禁止同步）。开销 ~0.05 ms，相对收益可忽略。
        x = ent["x"]
        health = {"has_nan": bool(cp.any(cp.isnan(x))) or bool(cp.any(cp.isinf(x))),
                  "has_inf": False,
                  "hidden_absmax": float(cp.abs(x).max()),
                  "marker_absmax": float(cp.abs(ent["mh"]).max()),
                  "mode": "cuda-graph(replay)"}
        return (cp.asnumpy(ent["logits"]).astype(np.float64),
                cp.asnumpy(ent["act"]).astype(np.float64),
                cp.asnumpy(x), health)

    def _capture(self, ids, marker_pos, marker_type) -> Optional[Dict[str, Any]]:
        m = self.model
        ids_d = cp.asarray(np.asarray(ids, dtype=np.int64))
        mp_d = cp.asarray(np.asarray(marker_pos, dtype=np.int64))
        mt_d = cp.asarray(np.asarray(marker_type, dtype=np.int64))
        feats_d = m._feats_buffer(len(marker_pos), int(marker_type[0]))
        try:
            # 1) warmup：把内存池喂饱，避免捕获期间 cudaMalloc
            with self.stream:
                lg, mh, ac, xx = m._forward_core(ids_d, mp_d, mt_d, feats_d)
            self.stream.synchronize()
            # 2) capture
            with self.stream:
                self.stream.begin_capture()
                lg, mh, ac, xx = m._forward_core(ids_d, mp_d, mt_d, feats_d)
                g = self.stream.end_capture()
            # 3) 验证：重放一次，与 eager 结果比对
            ref = cp.asnumpy(m._forward_core(ids_d, mp_d, mt_d, feats_d)[0])
            g.launch()
            self.stream.synchronize()
            got = cp.asnumpy(lg)
            drift = float(np.abs(ref - got).max())
            if drift > 1e-4:
                self.last_error = f"graph replay 与 eager 偏差 {drift:.2e}，已禁用"
                self.enabled = False
                return None
            return {"graph": g, "ids": ids_d, "mp": mp_d, "mt": mt_d,
                    "logits": lg, "act": ac, "x": xx, "mh": mh,
                    "capture_drift": drift}
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            self.enabled = False
            return None


class CudaPredictor(L.Predictor):
    """复用原 Predictor 的 prompt 组装 / 温度 / 输出封装，只换 model。"""

    def __init__(self, st, tok, cfg: Dict[str, Any], dtype: str = "fp32",
                 fused: Optional[FusedKernels] = None, use_graph: bool = False):
        self.st = st
        self.tok = tok
        self.cfg = dict(L.DEFAULT_CFG)
        self.cfg.update(cfg or {})
        self.model = CudaLayaModel(st, self.cfg, dtype=dtype, fused=fused)
        assert self.model.cfg is self.cfg, "配置必须在 Predictor 与 Model 间共享"
        self.lock = threading.Lock()
        self.calls = 0
        # graph 用实例属性覆盖 model.forward：这样 L.Predictor.predict 完全不用改，
        # 也不用把它的 60 行后处理逻辑抄一遍。
        self.graph: Optional[GraphRunner] = GraphRunner(self.model) if use_graph else None
        if self.graph is not None:
            self._eager_forward = self.model.forward
            self.model.forward = self._route_forward  # type: ignore[method-assign]

    def _route_forward(self, ids, marker_pos, marker_type):
        if self.graph is not None and self.graph.enabled:
            try:
                return self.model.forward_graph(ids, marker_pos, marker_type, self.graph)
            except Exception as e:  # noqa: BLE001
                self.graph.enabled = False
                sys.stderr.write(f"[laya-cuda] CUDA Graph 已禁用，回退 eager："
                                 f"{self.graph.last_error or e}\n")
        return self._eager_forward(ids, marker_pos, marker_type)

    def predict_batch(self, states: Sequence[Any],
                      qsets: Sequence[Sequence["L.Question"]]) -> List[Dict[str, Any]]:
        """真 batch：一次前向处理 B 条样本。

        实现要点：
          · 各样本长度不同 → 右 padding 对齐，pad 位置在 attention 里被 mask 掉
            （key 侧屏蔽即可；query 侧的 garbage 不会被任何真实位置读到）
          · marker 用 (batch_id*T + pos) 的扁平索引 gather，一次取出全部
          · 后置处理（温度 / softmax / 输出封装）完全复用原 Predictor 的逻辑，
            因此单条与批量的输出结构逐字段一致
        返回与逐条 predict 同构的结果列表。
        """
        if not states:
            return []
        built = [self.build(s, list(qs)) for s, qs in zip(states, qsets)]
        T = max(len(b["ids"]) for b in built)
        B = len(built)
        pad_id = self.tok.specials.get("pad", 0)

        ids_b = np.full((B, T), pad_id, dtype=np.int64)
        pad_mask = np.ones((B, T), dtype=np.bool_)
        mp_flat, mt_flat, slices = [], [], []
        for bi, b in enumerate(built):
            n = len(b["ids"])
            ids_b[bi, :n] = b["ids"]
            pad_mask[bi, :n] = False
            start = len(mp_flat)
            mp_flat.extend(bi * T + p for p in b["marker_pos"])
            mt_flat.extend(b["marker_type"])
            slices.append((start, len(mp_flat)))

        type0 = int(built[0]["marker_type"][0]) if built[0]["marker_type"] else 0
        n_opt = max(1, len(built[0]["marker_pos"]))
        feats_one = np.zeros(max(0, self.model.act_in_dim - self.model.H), dtype=np.float32)
        if feats_one.size >= 3:
            feats_one[type0 % 3] = 1.0
        if feats_one.size >= 4:
            feats_one[3] = math.log1p(n_opt) / math.log(64.0)
        feats_b = cp.asarray(np.tile(feats_one, (B, 1)))

        with self.lock:
            logits, act = self.model._forward_batch(
                cp.asarray(ids_b), cp.asarray(pad_mask),
                cp.asarray(np.asarray(mp_flat, dtype=np.int64)),
                cp.asarray(np.asarray(mt_flat, dtype=np.int64)),
                feats_b, slices)
            cp.cuda.get_current_stream().synchronize()
            self.calls += 1
        logits_np = cp.asnumpy(logits).astype(np.float64)
        act_np = cp.asnumpy(act).astype(np.float64)

        out: List[Dict[str, Any]] = []
        temps, tmode = self.temp_vector()
        for bi, b in enumerate(built):
            qs = list(qsets[bi])
            s, e = slices[bi]
            lg_all = logits_np[s:e]
            answers: Dict[str, Any] = {}
            cursor = 0
            for q, span in zip(qs, b["spans"]):
                n = len(span["options"])
                lg = lg_all[cursor:cursor + n]
                cursor += n
                t = float(temps[L.PRIMITIVES[q.type]]) if tmode != "off" else 1.0
                t = t if abs(t) > 1e-6 else 1.0
                p = L.softmax(lg / t, -1)
                keys = [o["key"] for o in span["options"]]
                if q.type == "choice":
                    best = int(np.argmax(p))
                    answers[q.id] = {"type": "choice", "choice": keys[best],
                                     "confidence": round(float(p[best]), 6),
                                     "probs": {k: round(float(v), 6) for k, v in zip(keys, p)}}
                elif q.type == "score":
                    idxf = np.arange(n, dtype=np.float64)
                    answers[q.id] = {"type": "score",
                                     "score": round(float((p * idxf).sum()), 4),
                                     "max": float(n - 1), "argmax_level": int(np.argmax(p)),
                                     "probs": [round(float(v), 6) for v in p], "levels": keys}
                else:
                    yes = keys.index("yes") if "yes" in keys else int(np.argmax(p))
                    answers[q.id] = {"type": "noul", "noul": round(float(p[yes]), 6),
                                     "p_no": round(float(p[1 - yes]), 6)}
            src = L.render_state(states[bi])
            out.append({
                "answers": answers,
                "routing": L.route(src, ["multilingual"]),
                "act": {"act": round(float(act_np[bi][0]), 6),
                        "escalate": round(float(act_np[bi][-1]), 6)},
                "meta": {"tokens": len(b["ids"]), "truncated": b["truncated"],
                         "n_questions": len(qs), "n_markers": len(b["marker_pos"]),
                         "latency_ms": None, "temperature_mode": tmode,
                         "tokenizer": self.tok.mode, "batch": B},
                "prompt": {"preview": b["prompt_preview"], "spans": b["spans"]},
            })
        return out


# =============================================================================
# 3. 一致性 & 基准
# =============================================================================

def _answer_signature(res: Dict[str, Any]) -> Dict[str, Any]:
    """把预测结果压成可比较的签名：逐题 top-1（或数值）。"""
    sig = {}
    for k, v in res["answers"].items():
        if v["type"] == "choice":
            sig[k] = v["choice"]
        elif v["type"] == "score":
            sig[k] = round(v["score"], 3)
        else:
            sig[k] = round(v["noul"], 3)
    return sig


def compare(cpu_pred: "L.Predictor", gpu_pred: CudaPredictor,
            states: Sequence[Any], qsets: Sequence[Sequence["L.Question"]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for s, qs in zip(states, qsets):
        rc = cpu_pred.predict(s, list(qs))
        rg = gpu_pred.predict(s, list(qs))
        # 从 answers 里取不到原始 logits，故单独跑一次裸前向拿 logits 对比
        bc = cpu_pred.build(s, list(qs))
        bg = gpu_pred.build(s, list(qs))
        assert bc["ids"] == bg["ids"], "CPU/GPU prompt 必须逐 token 一致"
        oc = cpu_pred.model.forward(bc["ids"], bc["marker_pos"], bc["marker_type"])
        og = gpu_pred.model.forward(bg["ids"], bg["marker_pos"], bg["marker_type"])
        lc = np.asarray(oc["logits"], dtype=np.float64)
        lg = np.asarray(og["logits"], dtype=np.float64)
        ad = float(np.abs(lc - lg).max())
        scale = max(1e-9, float(np.abs(lc).max()))
        sc_cpu = _answer_signature(rc)
        sc_gpu = _answer_signature(rg)
        rows.append({
            "state": (L.render_state(s) or "")[:48],
            "n_markers": len(lc),
            "max_abs_diff": ad,
            "rel_diff": ad / scale,
            "logit_scale": scale,
            "same_top1": sc_cpu == sc_gpu,
            "cpu": sc_cpu, "gpu": sc_gpu,
            "cpu_probs": {k: v.get("probs") for k, v in rc["answers"].items()},
            "gpu_probs": {k: v.get("probs") for k, v in rg["answers"].items()},
        })
    worst = max(r["rel_diff"] for r in rows)
    ok = all(r["same_top1"] for r in rows) and worst < 1e-3
    return {"rows": rows, "worst_rel_diff": worst,
            "all_same_top1": all(r["same_top1"] for r in rows),
            "status": "PASS" if ok else "FAIL"}


def bench_once(pred, state, qs, n: int = 7, warm: int = 3) -> Dict[str, Any]:
    for _ in range(warm):
        pred.predict(state, qs)
    ts: List[float] = []
    for _ in range(n):
        t = time.perf_counter()
        r = pred.predict(state, qs)
        ts.append((time.perf_counter() - t) * 1000.0)
    return {"median_ms": round(statistics.median(ts), 3),
            "min_ms": round(min(ts), 3), "max_ms": round(max(ts), 3),
            "tokens": r["meta"]["tokens"], "n_markers": r["meta"]["n_markers"]}


# =============================================================================
# 4. CLI
# =============================================================================

def build(model_path: Optional[str] = None, dtype: str = "fp32",
          use_graph: bool = False, tokenizer: Optional[str] = None,
          config: Optional[str] = None, strict: bool = False
          ) -> Tuple[Any, Any, CudaPredictor, FusedKernels, Dict[str, Any]]:
    """解析 --model 并加载。返回 (st, tok, pred, fused, info)。

    和原版的区别只有三点：
      1. `model_path` 先过 `resolve_model()`（文件/目录/glob/省略后缀/相对路径）；
      2. 解析结果**重钉**到 `DEFAULT_MODEL` / `L.DEFAULT_MODEL`，让 route 元数据、
         config 与 tokenizer 的回退目录都指向真正加载的这个模型；
      3. 把"加载了谁、配置和分词器来自哪"作为生效清单打到 stderr，并随 info 返回。
    """
    path = resolve_model(model_path)

    global DEFAULT_MODEL
    DEFAULT_MODEL = path
    L.DEFAULT_MODEL = path          # route()["repo"] 取的就是它

    cfg_dirs, cfg_src = resolve_cfg_dirs(path, config, strict)
    tok_path, tok_src = resolve_tokenizer(path, tokenizer, strict)
    preflight(path)

    t0 = time.perf_counter()
    with cfg_dirs_override(cfg_dirs):
        L.apply_official_config(path)        # 官方 config.json 优先于反推默认值
    st = L.SafeTensors(path)
    if tok_path:
        try:
            tok = L.HFJsonTokenizer(tok_path)
        except Exception as e:  # noqa: BLE001
            tok = L.ByteFallbackTokenizer()
            tok.note += f" （已找到 {os.path.basename(tok_path)} 但解析失败：{e}）"
    else:
        tok = L.ByteFallbackTokenizer()
    fused = FusedKernels()
    t1 = time.perf_counter()
    pred = CudaPredictor(st, tok, L.DEFAULT_CFG, dtype=dtype, fused=fused,
                         use_graph=use_graph)
    cp.cuda.get_current_stream().synchronize()
    up = time.perf_counter() - t1
    dev = cp.cuda.Device()
    try:
        import cupy.cuda.runtime as rt
        gname = rt.getDeviceProperties(dev.id)["name"].decode() if isinstance(
            rt.getDeviceProperties(dev.id)["name"], bytes) else rt.getDeviceProperties(dev.id)["name"]
        mem = rt.memGetInfo()
    except Exception:  # noqa: BLE001
        gname, mem = "?", (0, 0)

    n_layers = len(st.matching(r"^encoder\.layers\.\d+\.attn\.Wqkv\.weight$"))
    print(f"[laya-cuda] GPU {gname}｜显存 {mem[1]/2**30:.1f} GB｜cupy {cp.__version__}"
          f"｜dtype={dtype}", file=sys.stderr)
    # ---- 生效清单：一次说清"到底加载了谁、配置与分词器来自哪 ----
    print(f"[laya-cuda] 模型 {path}", file=sys.stderr)
    print(f"[laya-cuda]   {st.file_bytes/1e6:.0f} MB｜{len(st.header)} 张量｜"
          f"{st.params()/1e6:.1f}M 参数｜encoder {n_layers} 层", file=sys.stderr)
    print(f"[laya-cuda] 配置 {cfg_src}｜已对齐 "
          f"{sum(1 for c in L.CFG_ALIGN if c['kind'] == 'applied')} 项｜未建模 "
          f"{sum(1 for c in L.CFG_ALIGN if c['kind'] == 'unmodeled')} 项｜"
          f"rope_theta={L.DEFAULT_CFG['rope_theta']:.0f}｜attention={L.DEFAULT_CFG['attention']}"
          + (f"｜层数={L.DEFAULT_CFG['n_layers']}" if L.DEFAULT_CFG['n_layers'] != n_layers else "")
          , file=sys.stderr)
    if L.DEFAULT_CFG["n_layers"] != n_layers:
        sys.stderr.write(
            f"[laya-cuda] [!] 配置声明 {L.DEFAULT_CFG['n_layers']} 层、权重实测 {n_layers} 层 ——\n"
            "  配置与权重可能不是同一份模型，请核对 --config / 模型目录下的 config.json。\n")
    print(f"[laya-cuda] 分词器 {tok_src}"
          + (f"（{os.path.basename(tok_path)}）" if tok_path else "")
          + f"｜{tok.mode} (faithful={tok.faithful})"
          + ("" if getattr(tok, "faithful", False) else " ⚠ 语义不可靠"), file=sys.stderr)
    fb = [n for n, s in (("config.json", cfg_src), ("tokenizer.json", tok_src)) if "回退" in s]
    if fb:
        sys.stderr.write(
            f"[laya-cuda] [!] 模型目录没有 {'/'.join(fb)} → 已回落到脚本目录那一份。\n"
            "  若这是**同架构的微调权重**（如 labs/finetune/artifacts/*.safetensors），这正是想要的；\n"
            "  若确实换了另一个模型，请加 --strict，或用 --config/--tokenizer 明确指定，\n"
            "  否则等于拿脚本目录那份配置/分词器去读这个权重（张冠李戴）。\n")
    print(f"[laya-cuda] 权重上传显存 {up*1000:.0f} ms；融合 kernel {fused.ok}/3 "
          + "；".join(fused.notes), file=sys.stderr)

    info: Dict[str, Any] = {
        "path": path, "dir": os.path.dirname(path),
        "size_mb": round(st.file_bytes / 1e6, 1), "tensors": len(st.header),
        "params_m": round(st.params() / 1e6, 1), "n_layers": n_layers,
        "config_source": cfg_src, "config_dirs": cfg_dirs,
        "tokenizer_source": tok_src, "tokenizer_path": tok_path,
        "strict": strict, "dtype": dtype, "graph": use_graph,
        "load_ms": round((t1 - t0) * 1000, 1), "upload_ms": round(up * 1000, 1),
        "fused_kernels": fused.notes, "device": str(gname),
    }
    return st, tok, pred, fused, info


def main():
    ap = argparse.ArgumentParser(description="Laya 决策模型 CUDA 加速版")
    ap.add_argument("--model", "-m", default=None,
                    help="权重路径：文件 / 目录（含 model.safetensors）/ glob；相对路径按 "
                         "CWD → 脚本目录 → $LAYA_MODEL_DIR 解析；可省略 .safetensors 后缀。"
                         "默认脚本目录下的 model.safetensors")
    ap.add_argument("--list", nargs="?", const="", default=None, metavar="MODEL",
                    help="列出搜索目录下可见的 *.safetensors 后退出（不加载）；"
                         "可附带一个路径用于标注当前解析目标")
    ap.add_argument("--tokenizer", default=None,
                    help="显式指定 tokenizer.json（默认按 模型目录 → $LAYA_TOKENIZER → 脚本目录）")
    ap.add_argument("--config", default=None,
                    help="显式指定 config.json / rl_agent_config.json 所在目录或文件")
    ap.add_argument("--strict", action="store_true",
                    help="配置与分词器只从模型目录（或 --config/--tokenizer）取，"
                         "不静默回落到脚本目录 —— 换模型时强烈建议开启")
    ap.add_argument("--info", action="store_true",
                    help="加载并打印生效清单（JSON：模型/配置/分词器出处）后退出")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--bench", action="store_true", help="CPU vs GPU 延迟 + 数值一致性")
    ap.add_argument("--selftest", action="store_true", help="用 GPU 后端跑原版验证报告")
    ap.add_argument("--batch", type=int, default=0, help="批量吞吐测试（样本数）")
    ap.add_argument("--graph", action="store_true", help="启用 CUDA Graph 重放（按形状缓存）")
    ap.add_argument("--port", type=int, default=0, help="起 HTTP 服务（复用原 index.html）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repeat", type=int, default=7)
    a = ap.parse_args()

    if a.list is not None:
        show_models(a.list or a.model)
        return

    st, tok, pred, fused, info = build(a.model, a.dtype, use_graph=a.graph,
                                       tokenizer=a.tokenizer, config=a.config,
                                       strict=a.strict)

    if a.info:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    if a.port:
        L.STATE.update({"st": st, "tok": tok, "pred": pred,
                        "model_path": info["path"], "model_info": info})
        srv = L.ThreadingHTTPServer((a.host, a.port), L.Handler)
        url = f"http://{a.host}:{a.port}/"
        print(f"[laya-cuda] 验证台(GPU) → {url}｜模型 {os.path.basename(info['path'])}",
              file=sys.stderr)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[laya-cuda] bye", file=sys.stderr)
        return

    if a.selftest:
        v = L.Verifier(st, tok, pred, info["path"])
        rep = v.run()
        for c in rep["checks"]:
            print(f"[{c['status']:7s}] {c['group']:14s} {c['name']}\n          {c['detail']}")
        print("\n" + json.dumps(rep["summary"], ensure_ascii=False))
        print(rep["verdict"])
        print(f"\n模型: {info['path']}｜配置 {info['config_source']}｜"
              f"分词器 {info['tokenizer_source']}")
        return

    # 默认：bench
    qsets = [[L.Question(**q) for q in L.PRESETS[0]["questions"]],
             [L.Question(**q) for q in L.PRESETS[2]["questions"]]]
    states = [L.PRESETS[0]["state"], L.PRESETS[2]["state"]]

    cpu_pred = L.Predictor(st, tok, L.DEFAULT_CFG)
    gpu_eager = pred if not a.graph else CudaPredictor(
        st, tok, L.DEFAULT_CFG, dtype=a.dtype, fused=fused, use_graph=False)
    print(f"\n=== 1) 数值一致性（{os.path.basename(info['path'])}，CPU fp32 vs GPU {a.dtype}）===")
    cmp_res = compare(cpu_pred, gpu_eager, states, qsets)
    for r in cmp_res["rows"]:
        print(f"  markers={r['n_markers']:2d}  max|Δlogit|={r['max_abs_diff']:.3e}  "
              f"相对差={r['rel_diff']:.3e}  top1一致={r['same_top1']}")
        if not r["same_top1"]:
            print(f"      CPU {r['cpu']}\n      GPU {r['gpu']}")
    print(f"  → {cmp_res['status']}（判据：逐题 top-1 一致 且 相对差 < 1e-3）")

    print("\n=== 2) 延迟对比（warm-up 后取 %d 次中位数）===" % a.repeat)
    out = {}
    for name, s, qs in (("模型卡示例(4问)", states[0], qsets[0]),
                        ("安全事件(3问)", states[1], qsets[1])):
        c = bench_once(cpu_pred, s, qs, n=a.repeat)
        g = bench_once(gpu_eager, s, qs, n=a.repeat)
        line = (f"  {name}: tokens={g['tokens']:3d}  CPU {c['median_ms']:8.2f} ms  →  "
                f"GPU-eager {g['median_ms']:7.2f} ms（×{c['median_ms']/max(1e-6,g['median_ms']):.1f}）")
        out[name] = {"cpu": c, "gpu_eager": g,
                     "speedup": round(c["median_ms"] / max(1e-6, g["median_ms"]), 1)}
        if a.graph and pred.graph is not None and pred.graph.enabled:
            gg = bench_once(pred, s, qs, n=a.repeat)
            out[name]["gpu_graph"] = gg
            out[name]["graph_speedup_vs_eager"] = round(
                g["median_ms"] / max(1e-6, gg["median_ms"]), 2)
            line += (f"  │  GPU-graph {gg['median_ms']:6.2f} ms"
                     f"（×{c['median_ms']/max(1e-6,gg['median_ms']):.1f} vs CPU，"
                     f"×{g['median_ms']/max(1e-6,gg['median_ms']):.2f} vs eager）")
        line += f"  [CPU min {c['min_ms']:.0f}｜GPU min {g['min_ms']:.2f}]"
        print(line)
    if a.graph and pred.graph is not None and pred.graph.enabled:
        for k, v in pred.graph.graphs.items():
            print(f"  graph 桶 (T, n_markers)={k} 捕获重放偏差 drift={v['capture_drift']:.2e}")
    elif a.graph:
        print(f"  [!] CUDA Graph 未生效：{pred.graph.last_error if pred.graph else 'n/a'}")

    if a.batch:
        print(f"\n=== 3) 批量吞吐（{a.batch} 条，真 batch：padding + attention mask）===")
        many = [states[0]] * a.batch
        manys = [qsets[0]] * a.batch
        # 先校验 batch 与逐条一致（同一输入，两种路径必须给出相同的 top-1）
        single = gpu_eager.predict(many[0], manys[0])
        batched = pred.predict_batch(many, manys)
        same = all(_answer_signature(single) == _answer_signature(b) for b in batched)
        worst_p = 0.0
        for b in batched:
            for k, v in b["answers"].items():
                pa = single["answers"][k].get("probs")
                pb = v.get("probs")
                if isinstance(pa, dict) and isinstance(pb, dict):
                    worst_p = max(worst_p, max(abs(pa[kk] - pb[kk]) for kk in pa))
        print(f"  一致性：batch vs 逐条 top-1 {'一致' if same else '不一致 ✗'}；"
              f"概率最大偏差 {worst_p:.2e}")

        def timed(fn, rep):
            for _ in range(max(1, rep // 3)):
                fn()
            ts = []
            for _ in range(rep):
                t = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t) * 1000.0)
            return statistics.median(ts)

        tg = timed(lambda: pred.predict_batch(many, manys), 7)
        tc = timed(lambda: [cpu_pred.predict(s, qs) for s, qs in zip(many, manys)], 3)
        tgs = timed(lambda: [gpu_eager.predict(s, qs) for s, qs in zip(many, manys)], 5)
        print(f"  CPU  逐条 {tc:.0f} ms（{tc/a.batch:.1f} ms/条）")
        print(f"  GPU  逐条 {tgs:.0f} ms（{tgs/a.batch:.2f} ms/条）")
        print(f"  GPU  batch {tg:.0f} ms（{tg/a.batch:.2f} ms/条）"
              f"  → vs CPU ×{tc/max(1e-6,tg):.1f}，vs GPU 逐条 ×{tgs/max(1e-6,tg):.1f}")

    print(f"\n模型: {info['path']}（{info['params_m']}M 参数 / {info['n_layers']} 层 / "
          f"{info['size_mb']} MB）")
    print(f"配置: {info['config_source']}｜分词器: {info['tokenizer_source']}")
    print("fused kernels: " + " | ".join(fused.notes))


if __name__ == "__main__":
    main()
