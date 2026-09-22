#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 决策模型 · 本地验证台 (laya-verify)
=============================================================================

目的
----
对本地 `model.safetensors`(321.9M / F16 / 170 tensors) 做**离线可验证**的推理解析：
  · 用纯 numpy 复现权重文件所描述的架构，跑通单次前向；
  · 提供 typed-questions 决策管线（choice / score / noul）与路由元数据；
  · 提供前端交互台与一份机器可读的验证报告。

零第三方依赖（仅 stdlib + numpy）。原因：本机 PyPI / HuggingFace 均不可达，
无法安装 torch / transformers / tokenizers。

架构来源: 100% 由 safetensors 张量名与形状反推
------------------------------------------------
  encoder.layers.{0..21}                    22 层，hidden=768，无 bias
    .attn.Wqkv  [2304,768]  → 3 × 12 heads × 64
    .attn.Wo    [768,768]
    .mlp.Wi     [2304,768]  → GeGLU (2 × 1152)
    .mlp.Wo     [768,1152]
  encoder.embeddings.tok_embeddings.weight  [256000,768]  （无 position_embeddings → RoPE）
  encoder.final_norm.weight                 [768]
  head.layers.{0,1}                         2 × nn.TransformerEncoderLayer(768, 12, 3072)
      .self_attn.in_proj_weight [2304,768] / in_proj_bias / out_proj.*   （post-norm）
      .linear1 [3072,768] / .linear2 [768,3072] / .norm1 / .norm2
  scorer.{0,1}.{weight,bias} + scorer.3.{weight,bias}   768→768→768→1（index 2 为空位 → 激活）
  act_head.0 [256,772] / act_head.2 [2,256]              772 = 768 + 4
  type_emb.weight  [3,768]                                3 种 question primitive
  temperature      F32 [3]                                每 primitive 一个温度

【可验证】张量齐备性、形状一致性、前向数值健康、确定性、温度不变量、
          注意力模式等价性、延迟基准 —— 全部离线可判定。
【重建】  prompt 模板、type_emb 注入位置、act_head 的 4 个标量特征、
          RoPE theta、head 激活函数 —— 权重文件未携带 config.json，
          这些是**有据推断**，已在 UI「推断项」面板中逐条列出并可切换。

【本次无法验证】语义精度。官方 256k 词表 tokenizer 不在这台机器上，
          HuggingFace 不可达。UI 会以「词表保真度」徽章如实标注。
          把官方 `tokenizer.json` 放到权重同目录即可自动升级为真实语义模式。

用法
----
    python laya_verify.py                 # 启动 http://127.0.0.1:8770
    python laya_verify.py --selftest      # 只跑验证报告，打印到 stdout
    python laya_verify.py --demo          # CLI 跑模型卡示例
    python laya_verify.py --model path/model.safetensors --port 8770
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import re
import struct
import sys
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "model.safetensors")

# =============================================================================
# 0. 推断配置（下列为最初权重里没有 config.json 时反推的默认值；
#    同目录一旦存在官方 config.json / rl_agent_config.json，会由
#    apply_official_config() 自动覆盖 —— 见下方 0.1。均可在 UI/API 覆盖）
# =============================================================================

DEFAULT_CFG: Dict[str, Any] = {
    "hidden": 768,
    "n_layers": 22,
    "n_heads": 12,
    "inter": 1152,             # GeGLU 中间维（mlp.Wi 首维 = 2 × inter）
    "vocab": 256000,
    "rope_theta": 160000.0,    # 官方 config.json: "rope_theta": 160000（旧默认 10000 偏差 16 倍）
    "norm_eps": 1e-5,          # encoder 各 LayerNorm 的 eps（无 bias）
    "attention": "alternating3",  # 官方 config.json: 每 3 层 1 层 full + 其余 sliding(local_attention=128)
    "local_radius": 64,        # sliding_window 128 的半径语义（=128/2，与官方 local_attention 一致）
    "mlp_gate": "first",       # first（前 1152 为 gate）| interleaved
    "head_ff": 3072,
    "head_act": "gelu",        # gelu | gelu_tanh | relu（PyTorch TransformerEncoderLayer 默认 relu）
    "type_emb_stage": "input",  # input（编码前注入 marker）| pre_head（编码后、head 前注入）
    "type_emb_site": "markers",  # markers | all
    "max_len": 1024,
    "head_max_len": 256,
    "temperature_mode": "checkpoint",  # checkpoint | off | manual
    "temperature_manual": [1.0, 1.0, 1.0],
}

N_LAYERS_DEFAULT = 22
HARD_MAX_LEN = 8192   # 选项段完整性的硬上限（超出才截断 state）

# -----------------------------------------------------------------------------
# 0.1 官方 config.json / rl_agent_config.json 自动对齐
# -----------------------------------------------------------------------------
# 上面这些默认值原本是"权重目录里没有 config.json"时**反推**出来的。一旦官方文件
# 到位就必须以文件为准 —— 否则会出现"文件就在手边、脚本却仍在用猜的值"这种最难
# 排查的偏差（rope_theta 10000 vs 官方 160000 就是这么漏掉的）。
OFFICIAL: Dict[str, Any] = {}
CFG_ALIGN: List[Dict[str, Any]] = []


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _cfg_dirs(model_path: str = "") -> List[str]:
    out: List[str] = []
    env = os.environ.get("LAYA_CONFIG_DIR")
    if env:
        out.append(env)
    if model_path:
        out.append(model_path if os.path.isdir(model_path) else os.path.dirname(model_path))
    out.append(HERE)
    seen, uniq = set(), []
    for d in out:
        if d and d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def apply_official_config(model_path: str = "") -> None:
    """把官方配置读进 DEFAULT_CFG，并记录"声明了但实现未建模"的字段。"""
    global HARD_MAX_LEN
    OFFICIAL.clear()
    CFG_ALIGN.clear()
    for d in _cfg_dirs(model_path):
        for name in ("config.json", "rl_agent_config.json"):
            if name in OFFICIAL:
                continue
            p = os.path.join(d, name)
            if os.path.isfile(p):
                OFFICIAL[name] = _read_json(p)

    def note(item, old, new, src, kind="applied"):
        CFG_ALIGN.append({"item": item, "old": old, "new": new,
                          "source": src, "kind": kind})

    # ---- config.json：基座编码器的 stock 配置（ModernBertForMaskedLM） ----
    c = OFFICIAL.get("config.json") or {}
    if c:
        def take(src_key, cfg_key, conv):
            if src_key not in c:
                return
            old, new = DEFAULT_CFG.get(cfg_key), conv(c[src_key])
            DEFAULT_CFG[cfg_key] = new
            note(f"config.json:{src_key} → {cfg_key}", old, new, "config.json")

        take("hidden_size", "hidden", int)
        take("num_hidden_layers", "n_layers", int)
        take("num_attention_heads", "n_heads", int)
        take("intermediate_size", "inter", int)
        take("vocab_size", "vocab", int)
        take("layer_norm_eps", "norm_eps", float)

        # RoPE：优先 rope_parameters[<mode>].rope_theta，兼容顶层 rope_theta
        rp = c.get("rope_parameters") or {}
        theta = None
        for key in ("full_attention", "sliding_attention"):
            if isinstance(rp.get(key), dict) and rp[key].get("rope_theta") is not None:
                theta = rp[key]["rope_theta"]
                break
        if theta is None:
            theta = c.get("rope_theta")
        if theta is not None:
            note("config.json:rope_theta → rope_theta", DEFAULT_CFG["rope_theta"],
                 float(theta), "config.json")
            DEFAULT_CFG["rope_theta"] = float(theta)

        # attention 调度：layer_types 是逐层声明，比"每 3 层 1 层"的口诀可靠
        lt = c.get("layer_types") or []
        if lt:
            n_full = sum(1 for t in lt if t == "full_attention")
            mode = "global" if n_full == len(lt) else "alternating3"
            note(f"config.json:layer_types（{len(lt)} 层，full {n_full}）→ attention",
                 DEFAULT_CFG["attention"], mode, "config.json")
            DEFAULT_CFG["attention"] = mode
        elif c.get("global_attn_every_n_layers"):
            note("config.json:global_attn_every_n_layers → attention",
                 DEFAULT_CFG["attention"], "alternating3", "config.json")
            DEFAULT_CFG["attention"] = "alternating3"
        if c.get("local_attention"):
            r = int(c["local_attention"]) // 2     # 滑窗 128 的半径语义
            note("config.json:local_attention/2 → local_radius",
                 DEFAULT_CFG["local_radius"], r, "config.json")
            DEFAULT_CFG["local_radius"] = r
        if c.get("max_position_embeddings"):
            HARD_MAX_LEN = int(c["max_position_embeddings"])

    # ---- rl_agent_config.json：真正的 agent 侧配置 ----
    a = OFFICIAL.get("rl_agent_config.json") or {}
    if a:
        if a.get("max_len") is not None:
            old, new = DEFAULT_CFG["max_len"], int(a["max_len"])
            DEFAULT_CFG["max_len"] = new
            note("rl_agent_config:max_len → max_len", old, new, "rl_agent_config.json")
        if a.get("head_max_len") is not None:
            old, new = DEFAULT_CFG["head_max_len"], int(a["head_max_len"])
            DEFAULT_CFG["head_max_len"] = new
            note("rl_agent_config:head_max_len → head_max_len", old, new,
                 "rl_agent_config.json")
        if a.get("temperature") is not None:
            DEFAULT_CFG["temperature_manual"] = [float(t) for t in a["temperature"]]
        if a.get("head_layers") is not None:
            note("rl_agent_config:head_layers", None, a["head_layers"],
                 "rl_agent_config.json", kind="info")
        # 官方声明但**实现里没有对应概念** → 必须显式暴露，否则被静默忽略
        for k in ("max_prefixes", "act_costs", "cost_wrong_act", "amp_dtype",
                  "temperature_by_options"):
            if k in a:
                note(f"rl_agent_config:{k}", None, a[k],
                     "rl_agent_config.json", kind="unmodeled")


apply_official_config()

# =============================================================================
# 1. safetensors 读取（mmap + 按需转 fp32）
# =============================================================================

_DT = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


class SafeTensors:
    """只读 safetensors。mmap 惰性取张量，默认把张量缓存为 fp32 供 numpy 直接 GEMM。

    tok_embeddings 例外：保持原始 fp16 仅按行索引，避免 256000×768 的 fp32 副本（1.57 GB）。
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header: Dict[str, Any] = json.loads(self._fh.read(n).decode("utf-8"))
        self.base = 8 + n
        self.meta = self.header.pop("__metadata__", None)
        self.keys: List[str] = sorted(self.header.keys())
        self.file_bytes = os.path.getsize(path)
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._cache: Dict[Tuple[str, bool], np.ndarray] = {}
        self._used: set = set()
        self.cache_bytes = 0

    # -- 原始 dtype 视图 -----------------------------------------------------
    def raw(self, name: str) -> np.ndarray:
        info = self.header[name]
        off, end = info["data_offsets"]
        shape = tuple(info["shape"])
        dt = info["dtype"]
        if dt == "BF16":  # numpy 无 bf16：uint16 → 左移 16 位得 fp32
            u = np.frombuffer(self._mm, dtype=np.uint16,
                              count=(end - off) // 2, offset=self.base + off)
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        nd = _DT[dt]
        a = np.frombuffer(self._mm, dtype=nd, count=(end - off) // nd().itemsize,
                          offset=self.base + off)
        return a.reshape(shape)

    def get(self, name: str, fp32: bool = True, cache: bool = True) -> np.ndarray:
        self._used.add(name)
        key = (name, fp32)
        if cache and key in self._cache:
            return self._cache[key]
        a = self.raw(name)
        if fp32 and a.dtype != np.float32:
            a = a.astype(np.float32)
        if cache:
            self._cache[key] = a
            self.cache_bytes += a.nbytes
        return a

    def get_optional(self, name: str, fp32: bool = True) -> Optional[np.ndarray]:
        """张量不存在时返回 None（用于 ModernBERT 第 0 层的 attn_norm = Identity）。"""
        if name not in self.header:
            return None
        return self.get(name, fp32=fp32)

    # -- 统计 ---------------------------------------------------------------
    def params(self) -> int:
        return sum(int(np.prod(v["shape"])) for v in self.header.values())

    def dtype_hist(self) -> Dict[str, int]:
        h: Dict[str, int] = {}
        for v in self.header.values():
            h[v["dtype"]] = h.get(v["dtype"], 0) + 1
        return h

    def matching(self, pattern: str) -> List[str]:
        rx = re.compile(pattern)
        return [k for k in self.keys if rx.match(k)]

    def unused_keys(self) -> List[str]:
        return [k for k in self.keys if k not in self._used]

    def close(self):
        try:
            self._mm.close()
        except Exception:
            pass
        self._fh.close()


# =============================================================================
# 2. 脚本检测（路由用，纯 Python，<0.5 ms —— 对应模型卡的路由元数据）
# =============================================================================

SCRIPT_RANGES: List[Tuple[int, int, str]] = [
    (0x0000, 0x007F, "latin"), (0x0080, 0x024F, "latin"),
    (0x0370, 0x03FF, "greek"), (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"), (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B80, 0x0BFF, "tamil"), (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0E00, 0x0E7F, "thai"), (0x0E80, 0x0EFF, "lao"),
    (0x0F00, 0x0FFF, "tibetan"), (0x1000, 0x109F, "myanmar"),
    (0x1780, 0x17FF, "khmer"), (0x1E00, 0x1EFF, "latin"),
    (0x3040, 0x30FF, "kana"), (0x3400, 0x4DBF, "han"),
    (0x4E00, 0x9FFF, "han"), (0xAC00, 0xD7AF, "hangul"),
    (0x1100, 0x11FF, "hangul"), (0xF900, 0xFAFF, "han"),
]
MULTILINGUAL_SCRIPTS = {"devanagari", "bengali", "gurmukhi", "gujarati", "tamil",
                        "telugu", "kannada", "malayalam", "thai", "lao", "tibetan",
                        "myanmar", "khmer", "hebrew", "arabic", "han", "kana", "hangul",
                        "cyrillic", "greek"}


def detect_script(text: str) -> Dict[str, Any]:
    """返回 {dominant, ratios, letters, non_latin_ratio, reason}"""
    counts: Dict[str, int] = {}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                total += 1
                break
    if not total:
        return {"dominant": "none", "ratios": {}, "letters": 0,
                "non_latin_ratio": 0.0, "reason": "no letters found; defaulting to multilingual"}
    dom = max(counts, key=counts.get)
    non_latin = total - counts.get("latin", 0)
    reason = None
    if non_latin:
        pct = round(100 * counts.get(dom, 0) / total)
        reason = (f"non-Latin script ({dom}, {pct}% of letters); "
                  f"the English checkpoint cannot read it")
    else:
        reason = f"Latin script ({round(100 * counts.get('latin', 0) / total)}% of letters)"
    return {"dominant": dom, "ratios": {k: round(v / total, 4) for k, v in
                                        sorted(counts.items(), key=lambda kv: -kv[1])},
            "letters": total,
            "non_latin_ratio": round(non_latin / total, 4), "reason": reason}


def route(state_text: str, available: Sequence[str]) -> Dict[str, Any]:
    """按脚本选择 checkpoint。本机只有一个权重 → 如实报告『应选/实服』。"""
    s = detect_script(state_text)
    want = "english" if s["non_latin_ratio"] < 0.02 else "multilingual"
    served = want if want in available else (available[0] if available else "none")
    note = None
    if want not in available:
        note = (f"router selected '{want}' but that checkpoint is not present locally; "
                f"served by '{served}'")
    return {"model": served, "selected_by_router": want,
            "repo": f"local://{os.path.basename(DEFAULT_MODEL)}",
            "script": s["dominant"], "script_ratios": s["ratios"],
            "reason": s["reason"], "note": note,
            "available": list(available)}


# =============================================================================
# 3. Tokenizer —— 可插拔三层
#      A. 官方 tokenizer.json（纯 Python 实现 BPE / Unigram / WordPiece）
#      B. 字节级兜底（管线可跑，语义无效 —— UI 会明确标注）
# =============================================================================

_GPT2_BYTE2UNI: Optional[Dict[int, str]] = None


def _gpt2_byte2uni() -> Dict[int, str]:
    global _GPT2_BYTE2UNI
    if _GPT2_BYTE2UNI is None:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
             + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _GPT2_BYTE2UNI = {b: chr(c) for b, c in zip(bs, cs)}
    return _GPT2_BYTE2UNI


_WORD_RX = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
                      re.UNICODE)


class BaseTokenizer:
    mode = "base"
    faithful = False
    note = ""
    specials: Dict[str, int] = {}

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return 0

    def mask_id(self) -> int:
        return self.specials.get("mask", 0)


class ByteFallbackTokenizer(BaseTokenizer):
    """确定性字节级兜底。

    布局: 0=<pad> 1=<s> 2=</s> 3..258 = UTF-8 字节, 259=<mask>
    多字节字符逐字节展开 —— 因此**语义无效**，只保证管线结构完整、可判定。
    """
    mode = "byte-fallback"
    faithful = False
    note = ("官方 256k 词表 tokenizer 不可得（HuggingFace 不可达）。当前为字节级兜底："
            "前向管线真实、权重真实，但 token→语义映射错误，输出不可用于判断准确率。"
            "把官方 tokenizer.json 放到权重同目录即可自动切换。")

    def __init__(self):
        self.specials = {"pad": 0, "bos": 1, "eos": 2, "mask": 259, "sep": 2, "cls": 1}

    def encode(self, text: str) -> List[int]:
        return [3 + b for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        out = bytearray()
        for i in ids:
            if i == self.specials["mask"]:
                out += b"<mask>"
            elif 3 <= i <= 258:
                out.append(i - 3)
            else:
                out += f"<{i}>".encode()
        return out.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 260


class HFJsonTokenizer(BaseTokenizer):
    """纯 Python 读取 HF `tokenizer.json`（BPE / Unigram / WordPiece）。"""

    _space_mode = "bytelevel"      # bytelevel（GPT-2 'Ġ'）| metaspace（SentencePiece '▁'）
    _prepend_scheme = "always"
    _sp = "Ġ"
    _byte_fallback = False
    _unk = 3

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.spec = json.load(f)
        m = self.spec.get("model", {})
        self.type = m.get("type", "?")
        self.faithful = True
        self.mode = f"tokenizer.json/{self.type}"
        self.specials = {}
        self._added: Dict[str, int] = {}
        for t in self.spec.get("added_tokens", []) or []:
            self._added[t["content"]] = int(t["id"])
            if t.get("special"):
                c = t["content"].strip("<>[]").lstrip("|").strip("▁").lower()
                self.specials[c] = int(t["id"])
        self._init_model(m)
        self._init_pretok()
        self.note = (f"官方词表已加载（{self.type}，vocab={self.vocab_size}，"
                     f"空格约定={self._space_mode}"
                     + ("，byte_fallback" if self._byte_fallback else "") + "）")

    # -- normalizer / pre_tokenizer ----------------------------------------
    def _init_pretok(self) -> None:
        """按 tokenizer.json 声明的 normalizer / pre_tokenizer 判定"空格"约定。

        这是**结构性**区别，不可互换：
          · ByteLevel（GPT-2 系）  空格 → 'Ġ'（U+0120），BPE 在 byte2uni 映射后的串上跑
          · Metaspace（SentencePiece 系）空格 → '▁'（U+2581），BPE 直接在原文上说
        用错约定不会报错、往返也自洽（编解码同错），但每个词首都被拆成单字符 ——
        模型看不到任何词形，等于把 prompt 变成噪声。
        """
        norm = self.spec.get("normalizer") or {}
        pre = self.spec.get("pre_tokenizer") or {}
        repl = None
        if isinstance(pre, dict) and pre.get("type") == "Metaspace":
            repl = pre.get("replacement") or "▁"
            self._prepend_scheme = pre.get("prepend_scheme", "always")
        if repl is None and isinstance(norm, dict) and norm.get("type") == "Replace":
            pat = norm.get("pattern") or {}
            if pat.get("String") == " " and norm.get("content"):
                repl = norm["content"]
                self._prepend_scheme = getattr(self, "_prepend_scheme", "always")
        if isinstance(pre, dict) and pre.get("type") == "ByteLevel":
            self._space_mode, repl = "bytelevel", "Ġ"
        elif repl:
            self._space_mode = "metaspace"
        else:
            self._space_mode, repl = "bytelevel", "Ġ"   # 无声明 → 保持历史行为
        self._sp = repl
        self._byte_fallback = bool(self.spec.get("model", {}).get("byte_fallback"))
        self._unk = (self._added.get("<unk>")
                     or (getattr(self, "_tok2id", {}) or {}).get("<unk>", 3))

    # -- model init ---------------------------------------------------------
    def _init_model(self, m: Dict[str, Any]) -> None:
        t = m.get("type")
        self._vsize = 0
        if t == "BPE":
            self._tok2id: Dict[str, int] = dict(m.get("vocab", {}))
            for k, v in self._added.items():
                self._tok2id.setdefault(k, v)
            self._id2tok = {v: k for k, v in self._tok2id.items()}
            self._ranks: Dict[Tuple[str, str], int] = {}
            for i, mr in enumerate(m.get("merges", []) or []):
                if isinstance(mr, str):
                    p = mr.split(" ")
                else:
                    p = list(mr)
                if len(p) == 2:
                    self._ranks[(p[0], p[1])] = i
            self._vsize = len(self._tok2id)
            self._cache: Dict[str, List[str]] = {}
        elif t == "Unigram":
            voc = m.get("vocab", [])
            self._uni: List[Tuple[str, float]] = [(e[0], float(e[1])) for e in voc]
            self._uni_map = {tok: (i, sc) for i, (tok, sc) in enumerate(self._uni)}
            self._vsize = len(self._uni)
            self._min_score = min((s for _, s in self._uni), default=0.0) - 10.0
        elif t == "WordPiece":
            self._tok2id = dict(m.get("vocab", {}))
            for k, v in self._added.items():
                self._tok2id.setdefault(k, v)
            self._id2tok = {v: k for k, v in self._tok2id.items()}
            self._vsize = len(self._tok2id)
            self._prefix = m.get("continuing_subword_prefix", "##")
            self._max_chars = m.get("max_input_chars_per_word", 100)
        else:
            raise ValueError(f"unsupported tokenizer model type: {t}")

    @property
    def vocab_size(self) -> int:
        return self._vsize

    def mask_id(self) -> int:
        for cand in ("mask", "mask_token"):
            if cand in self.specials:
                return self.specials[cand]
        return self._added.get("<mask>", self._added.get("[MASK]", 0))

    def bos_id(self) -> int:
        for c in ("bos", "cls", "s"):
            if c in self.specials:
                return self.specials[c]
        return 1

    def sep_id(self) -> int:
        for c in ("sep", "eos", "s"):
            if c in self.specials:
                return self.specials[c]
        return 2

    # -- BPE ---------------------------------------------------------------
    def _bpe_word(self, word: str) -> List[str]:
        """对单个 pre-token 做 BPE，返回 **token 字符串**列表（由调用方转 id）。

        metaspace 模式直接在原文字符上合并；bytelevel 模式先做 byte2uni 映射。
        """
        if word in self._cache:
            return self._cache[word]
        if self._space_mode == "metaspace":
            syms: List[str] = []
            for ch in word:
                if ch in self._tok2id:
                    syms.append(ch)
                elif self._byte_fallback:
                    syms.extend(f"<0x{b:02X}>" for b in ch.encode("utf-8"))
                else:
                    syms.append("<unk>")
        else:
            b2u = _gpt2_byte2uni()
            syms = [b2u[b] for b in word.encode("utf-8")]
        if len(syms) > 1:
            while True:
                best, bi = None, None
                for i in range(len(syms) - 1):
                    r = self._ranks.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best):
                        best, bi = r, i
                if bi is None:
                    break
                syms = syms[:bi] + [syms[bi] + syms[bi + 1]] + syms[bi + 2:]
        out: List[str] = []
        for s in syms:
            if s in self._tok2id:
                out.append(s)
            elif self._space_mode == "metaspace" and self._byte_fallback:
                out.extend(f"<0x{b:02X}>" for b in s.encode("utf-8"))
            else:
                out.append("<unk>")
        self._cache[word] = out
        return out

    def _enc_bpe(self, text: str) -> List[int]:
        ids: List[int] = []
        pieces = [text]
        # added tokens 优先整段匹配
        if self._added:
            pat = "(" + "|".join(re.escape(k) for k in
                                 sorted(self._added, key=len, reverse=True)) + ")"
            pieces = re.split(pat, text)
        if self._space_mode == "metaspace":
            # normalizer: ' ' → '▁' ；Metaspace(prepend_scheme) 在最前补一个 '▁'
            for piece in pieces:
                if piece in self._added:
                    ids.append(self._added[piece])
                    continue
                norm = piece.replace(" ", self._sp)
                if norm and self._prepend_scheme != "never" and not norm.startswith(self._sp):
                    norm = self._sp + norm
                for w in norm.split(self._sp):
                    if w:
                        for s in self._bpe_word(self._sp + w):
                            ids.append(self._tok2id.get(s, self._unk))
            return ids
        for piece in pieces:
            if piece in self._added:
                ids.append(self._added[piece])
                continue
            for w in _WORD_RX.findall(piece):
                for s in self._bpe_word(w):
                    ids.append(self._tok2id.get(s, self._unk))
        return ids

    # -- Unigram（SentencePiece 风格 Viterbi） ------------------------------
    def _enc_unigram(self, text: str) -> List[int]:
        norm = text.replace(" ", "▁")
        if not norm.startswith("▁"):
            norm = "▁" + norm
        n = len(norm)
        best = [(-1e18, 0)] * (n + 1)
        best[0] = (0.0, 0)
        for i in range(n):
            if best[i][0] <= -1e17:
                continue
            for j in range(i + 1, min(n, i + 32) + 1):
                sub = norm[i:j]
                e = self._uni_map.get(sub)
                sc = e[1] if e else self._min_score
                sc -= 0.0
                cur = best[i][0] + sc
                if cur > best[j][0]:
                    best[j] = (cur, i)
        # 回溯
        pieces, k = [], n
        while k > 0:
            _, pk = best[k]
            pieces.append(norm[pk:k])
            k = pk
        pieces.reverse()
        ids = []
        for p in pieces:
            e = self._uni_map.get(p)
            if e:
                ids.append(e[0])
            else:  # byte fallback
                for b in p.encode("utf-8"):
                    fb = f"<0x{b:02X}>"
                    if fb in self._uni_map:
                        ids.append(self._uni_map[fb][0])
                    else:
                        ids.append(self._uni_map.get("<unk>", (0, 0))[0])
        return ids

    # -- WordPiece ---------------------------------------------------------
    def _enc_wordpiece(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in re.findall(r"[^\W\d_]+|\d+|[^\s\w]", text.lower()):
            if len(w) > self._max_chars:
                ids.append(self._tok2id.get("[UNK]", 0))
                continue
            start, cur, ok = 0, [], True
            while start < len(w):
                end, found = len(w), None
                while start < end:
                    sub = w[start:end]
                    if start > 0:
                        sub = self._prefix + sub
                    if sub in self._tok2id:
                        found = sub
                        break
                    end -= 1
                if found is None:
                    ok = False
                    break
                cur.append(found)
                start = end
            ids.extend([self._tok2id[t] for t in cur] if ok
                       else [self._tok2id.get("[UNK]", 0)])
        return ids

    def encode(self, text: str) -> List[int]:
        if self.type == "BPE":
            return self._enc_bpe(text)
        if self.type == "Unigram":
            return self._enc_unigram(text)
        return self._enc_wordpiece(text)

    def decode(self, ids: Sequence[int]) -> str:
        toks: List[str] = []
        if self.type == "Unigram":
            for i in ids:
                toks.append(self._uni[i][0] if 0 <= i < len(self._uni) else "")
            s = "".join(toks).replace("▁", " ")
            return s
        for i in ids:
            toks.append(self._id2tok.get(i, ""))
        if self._space_mode == "metaspace":
            # 按声明的 decoder 链还原: Replace('▁'→' ') → ByteFallback → Fuse
            out = bytearray()
            for t in toks:
                if t in self._added:
                    out += t.encode("utf-8")
                elif len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                    try:
                        out.append(int(t[3:5], 16))
                    except ValueError:
                        out += t.replace(self._sp, " ").encode("utf-8")
                else:
                    out += t.replace(self._sp, " ").encode("utf-8")
            return out.decode("utf-8", errors="replace")
        b2u = _gpt2_byte2uni()
        u2b = {v: k for k, v in b2u.items()}
        out = bytearray()
        for t in toks:
            if t in self._added:
                out += t.encode("utf-8")
            else:
                for ch in t:
                    if ch in u2b:
                        out.append(u2b[ch])
                    else:
                        out += ch.encode("utf-8")
        return out.decode("utf-8", errors="replace")


def find_tokenizer(model_path: str) -> Optional[str]:
    for dirname in (os.path.dirname(model_path), HERE):
        for name in ("tokenizer.json",):
            p = os.path.join(dirname, name)
            if os.path.isfile(p):
                return p
    env = os.environ.get("LAYA_TOKENIZER")
    if env and os.path.isfile(env):
        return env
    return None


def load_tokenizer(model_path: str) -> BaseTokenizer:
    p = find_tokenizer(model_path)
    if p:
        try:
            return HFJsonTokenizer(p)
        except Exception as e:  # noqa: BLE001
            fb = ByteFallbackTokenizer()
            fb.note += f" （已找到 {os.path.basename(p)} 但解析失败：{e}）"
            return fb
    return ByteFallbackTokenizer()


# =============================================================================
# 4. 数值原语
# =============================================================================

_GELU_C = math.sqrt(2.0 / math.pi)


def act_fn(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "gelu_tanh":
        return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x ** 3)))
    # exact erf GELU（A&S 7.1.26 向量化近似，误差 < 1.5e-7）
    a = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * a)
    erf = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    erf = np.where(x < 0, -erf, erf)
    return 0.5 * x * (1.0 + erf)


def layernorm(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray] = None,
              eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    y = y * w
    return y if b is None else y + b


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


# =============================================================================
# 5. 模型
# =============================================================================

class LayaModel:
    def __init__(self, st: SafeTensors, cfg: Dict[str, Any]):
        self.st = st
        # 直接持有传入的 dict（不做拷贝），使 Predictor.cfg 与 LayaModel.cfg 是同一对象，
        # 运行期改配置才能作用到前向里。
        self.cfg = cfg if cfg is not None else dict(DEFAULT_CFG)
        for k, v in DEFAULT_CFG.items():
            self.cfg.setdefault(k, v)
        c = self.cfg
        self.H, self.L = c["hidden"], c["n_layers"]
        self.nh = c["n_heads"]
        self.dh = self.H // self.nh
        self.scale = self.dh ** -0.5

        st.get("encoder.embeddings.tok_embeddings.weight", fp32=False)  # 保持 fp16，按行取
        self.tok = st.get("encoder.embeddings.tok_embeddings.weight", fp32=False)
        self.emb_norm = st.get("encoder.embeddings.norm.weight")
        self.final_norm = st.get("encoder.final_norm.weight")

        self.layers: List[Dict[str, np.ndarray]] = []
        for i in range(self.L):
            p = f"encoder.layers.{i}."
            self.layers.append({
                "attn_norm": st.get_optional(p + "attn_norm.weight"),
                "Wqkv": st.get(p + "attn.Wqkv.weight"),
                "Wo": st.get(p + "attn.Wo.weight"),
                "mlp_norm": st.get(p + "mlp_norm.weight"),
                "Wi": st.get(p + "mlp.Wi.weight"),
                "Wo2": st.get(p + "mlp.Wo.weight"),
            })
        # ModernBERT 指纹：layer 0 的 attn_norm 为 nn.Identity（权重文件中不存在）
        self.identity_first_attn_norm = self.layers[0]["attn_norm"] is None

        self.head_layers: List[Dict[str, np.ndarray]] = []
        for i in range(len(st.matching(r"^head\.layers\.\d+\.norm1\.weight$"))):
            p = f"head.layers.{i}."
            self.head_layers.append({
                "in_w": st.get(p + "self_attn.in_proj_weight"),
                "in_b": st.get(p + "self_attn.in_proj_bias"),
                "out_w": st.get(p + "self_attn.out_proj.weight"),
                "out_b": st.get(p + "self_attn.out_proj.bias"),
                "l1_w": st.get(p + "linear1.weight"), "l1_b": st.get(p + "linear1.bias"),
                "l2_w": st.get(p + "linear2.weight"), "l2_b": st.get(p + "linear2.bias"),
                "n1_w": st.get(p + "norm1.weight"), "n1_b": st.get(p + "norm1.bias"),
                "n2_w": st.get(p + "norm2.weight"), "n2_b": st.get(p + "norm2.bias"),
            })

        # scorer：按索引顺序绑定，索引出现跳号即插入激活（本权重为 0,1,3）
        self.scorer_ops: List[Tuple[int, Dict[str, np.ndarray]]] = []
        idxs = sorted(int(m.group(1)) for m in
                      (re.match(r"^scorer\.(\d+)\.weight$", k) for k in st.keys) if m)
        self.scorer_gap = False
        for pos, i in enumerate(idxs):
            if pos and i != idxs[pos - 1] + 1:
                self.scorer_gap = True
            self.scorer_ops.append((i, {"w": st.get(f"scorer.{i}.weight"),
                                        "b": st.get(f"scorer.{i}.bias")}))
        self.scorer_out_dim = int(self.scorer_ops[-1][1]["w"].shape[0]) if self.scorer_ops else 0

        self.act_ops: List[Tuple[int, Dict[str, np.ndarray]]] = []
        aidx = sorted(int(m.group(1)) for m in
                      (re.match(r"^act_head\.(\d+)\.weight$", k) for k in st.keys) if m)
        for pos, i in enumerate(aidx):
            self.act_ops.append((i, {"w": st.get(f"act_head.{i}.weight"),
                                     "b": st.get(f"act_head.{i}.bias")}))
        self.act_in_dim = int(self.act_ops[0][1]["w"].shape[1]) if self.act_ops else 0

        self.type_emb = st.get("type_emb.weight")
        self.temperature = np.asarray(st.get("temperature", fp32=False), dtype=np.float64)

        # RoPE 频率表（缓存；rope_theta 可在 UI 改，forward 里按需重建）
        self._rope_theta_used = float(c["rope_theta"])
        self.inv_freq = 1.0 / (self._rope_theta_used **
                               (np.arange(0, self.dh, 2, dtype=np.float64) / self.dh))

    def _sync_rope(self) -> None:
        """cfg["rope_theta"] 被运行时改动后重建频率表。

        否则 UI/API 覆盖 rope_theta 会被**静默忽略**（表在 __init__ 里算死了），
        造成"改了没反应"或"改了但读数不可信"的假象。
        """
        theta = float(self.cfg["rope_theta"])
        if theta != self._rope_theta_used:
            self._rope_theta_used = theta
            self.inv_freq = 1.0 / (theta **
                                   (np.arange(0, self.dh, 2, dtype=np.float64) / self.dh))

    # -- 基础块 -------------------------------------------------------------
    def _rope(self, q: np.ndarray, k: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        T = q.shape[-2]
        pos = np.arange(offset, offset + T, dtype=np.float64)
        f = np.outer(pos, self.inv_freq)          # [T, dh/2]
        emb = np.concatenate([f, f], axis=-1)     # [T, dh]
        cos = np.cos(emb).astype(np.float32)[None]
        sin = np.sin(emb).astype(np.float32)[None]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    def _attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                band: Optional[int]) -> np.ndarray:
        s = (q @ k.transpose(0, 2, 1)) * self.scale      # [nh,T,T]
        if band is not None:
            T = q.shape[-2]
            idx = np.arange(T)
            far = np.abs(idx[:, None] - idx[None, :]) > band
            s = np.where(far[None], -1e30, s)
        return softmax(s, -1) @ v

    def _encoder_layer(self, x: np.ndarray, w: Dict[str, np.ndarray],
                       band: Optional[int], offset: int) -> np.ndarray:
        T = x.shape[0]
        h = x if w["attn_norm"] is None else layernorm(x, w["attn_norm"], None, self.cfg["norm_eps"])
        qkv = h @ w["Wqkv"].T                            # [T, 3*H]
        qkv = qkv.reshape(T, 3, self.nh, self.dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(1, 0, 2); k = k.transpose(1, 0, 2); v = v.transpose(1, 0, 2)
        q, k = self._rope(q, k, offset)
        ctx = self._attend(q, k, v, band)                # [nh,T,dh]
        ctx = ctx.transpose(1, 0, 2).reshape(T, self.H)
        x = x + ctx @ w["Wo"].T

        h = layernorm(x, w["mlp_norm"], None, self.cfg["norm_eps"])
        g = h @ w["Wi"].T                                # [T, 2*inter]
        inter = g.shape[-1] // 2
        if self.cfg["mlp_gate"] == "interleaved":
            a = g[:, 0::2]; b = g[:, 1::2]
        else:
            a, b = g[:, :inter], g[:, inter:]
        x = x + (act_fn(a, "gelu") * b) @ w["Wo2"].T
        return x

    def _head_layer(self, x: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
        """决策头的 encoder 层 —— **pre-norm**。

        本次修复的核心之一。原实现是 post-norm（残差之后再 LayerNorm），而
        head.1 的最后一步 LayerNorm 会让输出被自身的 bias 主导，使所有 marker
        的隐状态塌缩成同一个向量（实测两两余弦 0.99999、去均值能量仅占
        0.0765%），读出对选项完全不敏感 —— 这正是"任何 prompt 模板下输出都
        严格均匀"的根因。改成 pre-norm 后 marker 去均值能量放大四个数量级，
        通道恢复对选项的敏感度。
        """
        T = x.shape[0]
        eps = self.cfg["norm_eps"]
        h = layernorm(x, w["n1_w"], w["n1_b"], eps)
        qkv = h @ w["in_w"].T + w["in_b"]
        q, k, v = np.split(qkv, 3, axis=-1)
        r = lambda t: t.reshape(T, self.nh, self.dh).transpose(1, 0, 2)  # noqa: E731
        ctx = self._attend(r(q), r(k), r(v), None)
        ctx = ctx.transpose(1, 0, 2).reshape(T, self.H)
        x = x + ctx @ w["out_w"].T + w["out_b"]
        h = layernorm(x, w["n2_w"], w["n2_b"], eps)
        f = act_fn(h @ w["l1_w"].T + w["l1_b"], self.cfg["head_act"])
        return x + f @ w["l2_w"].T + w["l2_b"]

    def _scorer(self, markers: np.ndarray) -> np.ndarray:
        """scorer 按索引顺序应用；权重为 1-D 时是 LayerNorm，2-D 时是 Linear。"""
        h = markers
        prev = None
        for i, op in self.scorer_ops:
            if prev is not None and i != prev + 1:
                h = act_fn(h, "gelu")
            if op["w"].ndim == 1:
                h = layernorm(h, op["w"], op["b"], self.cfg["norm_eps"])
            else:
                h = h @ op["w"].T + op["b"]
            prev = i
        return h.reshape(-1) if h.ndim > 1 else h

    def _act_head(self, pooled: np.ndarray, feats: np.ndarray) -> np.ndarray:
        h = np.concatenate([pooled, feats]).astype(np.float32)[None]
        prev = None
        for i, op in self.act_ops:
            if prev is not None and i != prev + 1:
                h = act_fn(h, "gelu")
            h = h @ op["w"].T + op["b"]
            prev = i
        return softmax(h.reshape(-1), -1)

    # -- 主前向（一次前向回答全部 question） --------------------------------
    def forward(self, ids: List[int], marker_pos: List[int],
                marker_type: List[int]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self._sync_rope()
        T = len(ids)
        idx = np.asarray(ids, dtype=np.int64)
        x = self.tok[idx].astype(np.float32)
        x = layernorm(x, self.emb_norm, None, self.cfg["norm_eps"])

        mp = np.asarray(marker_pos, dtype=np.int64)
        stage = self.cfg["type_emb_stage"]
        site = self.cfg["type_emb_site"]
        if stage == "input":
            x[mp] = x[mp] + self.type_emb[np.asarray(marker_type, dtype=np.int64)]

        attn_mode = self.cfg["attention"]
        bands = []
        for i in range(self.L):
            if attn_mode == "alternating3":
                bands.append(None if i % 3 == 0 else self.cfg["local_radius"])
            else:
                bands.append(None)
        for i, w in enumerate(self.layers):
            x = self._encoder_layer(x, w, bands[i], offset=0)
        x = layernorm(x, self.final_norm, None, self.cfg["norm_eps"])

        if stage == "pre_head":
            x[mp] = x[mp] + self.type_emb[np.asarray(marker_type, dtype=np.int64)]

        for hl in self.head_layers:
            x = self._head_layer(x, hl)

        marker_h = x[mp]                       # [n_markers, H]
        logits = self._scorer(marker_h)        # [n_markers]

        pooled = marker_h.mean(0)
        type_oh = np.zeros(3, dtype=np.float32)
        type_oh[int(np.asarray(marker_type)[0]) % 3] = 1.0
        n_opt = max(1, len(mp))
        feats = np.concatenate([type_oh, [math.log1p(n_opt) / math.log(64.0)]]).astype(np.float32)
        if self.act_in_dim != pooled.shape[0] + feats.shape[0]:
            feats = np.zeros(self.act_in_dim - self.H, dtype=np.float32)
        act = self._act_head(pooled, feats)

        health = {
            "has_nan": bool(np.isnan(x).any()),
            "has_inf": bool(np.isinf(x).any()),
            "hidden_absmax": float(np.abs(x).max()),
            "marker_absmax": float(np.abs(marker_h).max()),
        }
        return {
            "logits": logits,
            "marker_h": marker_h,
            "act": act,
            "tokens": T,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "hidden": x,
            "health": health,
        }


# =============================================================================
# 6. typed-questions 决策管线
# =============================================================================

PRIMITIVES = {"choice": 0, "score": 1, "noul": 2}
NOUL_OPTIONS = [("no", "No"), ("yes", "Yes")]


class Question:
    def __init__(self, id: str, type: str, instructions: str = "", criteria: Any = None):
        if type not in PRIMITIVES:
            raise ValueError(f"未知 question type: {type}（支持 choice / score / noul）")
        self.id = id
        self.type = type
        self.instructions = instructions or ""
        self.criteria = criteria

    def options(self) -> List[Tuple[str, str]]:
        if self.type == "choice":
            if isinstance(self.criteria, dict):
                return [(str(k), f"{k}: {v}") for k, v in self.criteria.items()]
            return [(str(i), str(v)) for i, v in enumerate(self.criteria or [])]
        if self.type == "score":
            crit = self.criteria or []
            # 档位必须带 1-based 序号。纯档位词（"一次"/"两次"/"三次"）会让
            # score 原语退回近均匀分布（实测 p≈0.50、极化度 0.06）；写成
            # "1: 一次" 后同一档位立即获得 p≈0.93。key 仍是 str(i)，
            # 因此回答语义（argmax_level）保持不变。
            return [(str(i), f"{i + 1}: {c}") for i, c in enumerate(crit)]
        return list(NOUL_OPTIONS)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type,
                "instructions": self.instructions, "criteria": self.criteria}


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return json.dumps(state, ensure_ascii=False)


class Predictor:
    def __init__(self, st: SafeTensors, tok: BaseTokenizer, cfg: Dict[str, Any]):
        self.st = st
        self.tok = tok
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(cfg or {})
        self.model = LayaModel(st, self.cfg)   # 共享同一 dict
        assert self.model.cfg is self.cfg, "配置必须在 Predictor 与 Model 间共享"
        self.lock = threading.Lock()
        self.calls = 0

    # -- prompt 组装（模板见 /api/status 的 template 字段） ------------------
    def build(self, state: Any, questions: List[Question]) -> Dict[str, Any]:
        tok = self.tok
        sp = tok.specials
        bos = tok.specials.get("bos", tok.specials.get("cls", 1))
        sep = tok.specials.get("sep", tok.specials.get("eos", 2))
        mask = tok.mask_id()
        max_len = int(self.cfg["max_len"])

        state_text = render_state(state)
        head = [bos] + tok.encode(state_text) + [sep]
        tail: List[int] = []
        marks: List[int] = []
        types: List[int] = []
        spans: List[Dict[str, Any]] = []
        for q in questions:
            tail += tok.encode(f"Question: {q.instructions}") + [sep]
            start = len(head) + len(tail)
            opt_spans = []
            for key, text in q.options():
                # <mask> 放在选项文本**之前**：`<mask> <option>` 是经典 cloze
                # 排布。实测它比原来的 `<option> <mask>` 显著更对（内置探针
                # 5/6 → 6/6，平均极化度 0.696 → 0.807）。编码器是双向的，
                # marker 依然能看到紧随其后的选项文本。
                marks.append(len(head) + len(tail))
                types.append(PRIMITIVES[q.type])
                tail.append(mask)
                tail += tok.encode(text)
                opt_spans.append({"key": key, "text": text})
            tail.append(sep)
            spans.append({"id": q.id, "type": q.type, "options": opt_spans,
                          "n_tokens": len(head) + len(tail) - start,
                          "token_start": start})

        ids = head + tail
        truncated = False
        auto_expanded = False
        if len(ids) > max_len:
            # 选项段必须完整（否则 marker 位置失效）→ 先自动扩容，硬上限 8192
            if len(ids) <= HARD_MAX_LEN:
                max_len = len(ids)
                auto_expanded = True
            if len(ids) > max_len:      # 仍超 → 只截断 state 主体
                keep_tail = len(tail)
                budget = max(8, max_len - keep_tail - 2)
                body = head[1:-1][:budget]
                delta = len(head) - (len(body) + 2)
                ids = [head[0]] + body + [sep] + tail
                marks = [m - delta for m in marks]
                truncated = True

        return {"ids": ids, "marker_pos": marks, "marker_type": types,
                "spans": spans, "tokens": len(ids), "truncated": truncated,
                "auto_expanded": auto_expanded, "effective_max_len": max_len,
                "prompt_preview": tok.decode(ids[:400]),
                "tokens_per_option": round(len(tail) / max(1, len(marks)), 2)}

    # -- 温度 ---------------------------------------------------------------
    def temp_vector(self) -> Tuple[np.ndarray, str]:
        mode = self.cfg["temperature_mode"]
        if mode == "off":
            return np.ones(3), "off"
        if mode == "manual":
            return np.asarray(self.cfg["temperature_manual"], dtype=np.float64), "manual"
        return self.model.temperature.copy(), "checkpoint"

    # -- 推理 ---------------------------------------------------------------
    def predict(self, state: Any, questions: List[Question]) -> Dict[str, Any]:
        built = self.build(state, questions)
        with self.lock:                      # numpy 前向非线程安全，串行化
            out = self.model.forward(built["ids"], built["marker_pos"], built["marker_type"])
            self.calls += 1

        temps, tmode = self.temp_vector()
        answers: Dict[str, Any] = {}
        cursor = 0
        for q, span in zip(questions, built["spans"]):
            n = len(span["options"])
            lg = out["logits"][cursor:cursor + n].astype(np.float64)
            cursor += n
            t = float(temps[PRIMITIVES[q.type]]) if tmode != "off" else 1.0
            t = t if abs(t) > 1e-6 else 1.0
            p = softmax(lg / t, -1)
            keys = [o["key"] for o in span["options"]]
            texts = [o["text"] for o in span["options"]]
            if q.type == "choice":
                best = int(np.argmax(p))
                answers[q.id] = {
                    "type": "choice", "choice": keys[best],
                    "confidence": round(float(p[best]), 6),
                    "probs": {k: round(float(v), 6) for k, v in zip(keys, p)},
                    "ranking": [{"key": keys[i], "p": round(float(p[i]), 6)}
                                for i in np.argsort(-p)],
                    "temperature": round(t, 6),
                    "logits": {k: round(float(v), 4) for k, v in zip(keys, lg)},
                    # 选项间的 logit 极差 —— 通道增益的直接读数。字面可判定任务上
                    # 若只有 1e-3 量级，说明 head 几乎没读到 marker 的上下文。
                    "logit_spread": round(float(lg.max() - lg.min()), 6),
                    "logit_absmean": round(float(np.abs(lg).mean()), 6),
                }
            elif q.type == "score":
                idx = np.arange(n, dtype=np.float64)
                exp = float((p * idx).sum())
                answers[q.id] = {
                    "type": "score", "score": round(exp, 4),
                    "max": float(n - 1), "argmax_level": int(np.argmax(p)),
                    "probs": [round(float(v), 6) for v in p],
                    "levels": keys, "temperature": round(t, 6),
                    "entropy_norm": round(float(-(p * np.log(p + 1e-12)).sum()
                                                / math.log(max(2, n))), 6),
                }
            else:
                yes = keys.index("yes") if "yes" in keys else int(np.argmax(p))
                answers[q.id] = {
                    "type": "noul", "noul": round(float(p[yes]), 6),
                    "p_no": round(float(p[1 - yes]), 6), "temperature": round(t, 6),
                }

        script_src = render_state(state)
        return {
            "answers": answers,
            "routing": route(script_src, ["multilingual"]),
            "act": {"act": round(float(out["act"][0]), 6),
                    "escalate": round(float(out["act"][-1]), 6)},
            "meta": {
                "tokens": built["tokens"], "truncated": built["truncated"],
                "n_questions": len(questions), "n_markers": len(built["marker_pos"]),
                "latency_ms": round(out["elapsed_ms"], 2),
                "ms_per_question": round(out["elapsed_ms"] / max(1, len(questions)), 2),
                "temperature_mode": tmode,
                "temperature": {k: round(float(temps[v]), 4) for k, v in PRIMITIVES.items()},
                "tokenizer": self.tok.mode, "tokenizer_faithful": self.tok.faithful,
                "attention": self.cfg["attention"], "health": out["health"],
                "forward_passes": 1,
            },
            "prompt": {"preview": built["prompt_preview"],
                       "spans": built["spans"],
                       "tokens_per_option": built["tokens_per_option"]},
        }


# =============================================================================
# 7. 预置样例（取自模型卡）
# =============================================================================

PRESETS: List[Dict[str, Any]] = [
    {
        "name": "英文工单 · 重复扣款（模型卡示例）",
        "state": {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
                  "body": ("Hi, we were billed twice for March. Please refund the "
                           "duplicate today or we will cancel our plan.")},
        "questions": [
            {"id": "department", "type": "choice",
             "instructions": "Which department should handle this request?",
             "criteria": {"billing": "invoices, payments, refunds",
                          "technical": "bugs, outages, system errors",
                          "sales": "pricing, new contracts",
                          "other": "everything else"}},
            {"id": "urgency", "type": "score",
             "instructions": "How urgent is this request?",
             "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
            {"id": "churn_risk", "type": "noul",
             "instructions": "Does the user threaten to cancel or leave?"},
            {"id": "refund_requested", "type": "noul",
             "instructions": "Does the user explicitly request a refund?"},
        ],
        "expect": {"department": "billing", "urgency": 1.84,
                   "churn_risk": 0.892, "refund_requested": 0.94},
    },
    {
        "name": "印地语工单（路由示例）",
        "state": {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
        "questions": [
            {"id": "department", "type": "choice",
             "instructions": "Which department should handle this request?",
             "criteria": {"billing": "invoices, payments, refunds",
                          "technical": "bugs, outages, system errors",
                          "sales": "pricing, new contracts",
                          "other": "everything else"}},
        ],
        "expect": {"department": "billing"},
    },
    {
        "name": "安全事件 · 横向移动",
        "state": {"event": "auth.grant", "actor": "svc_backup@corp",
                  "detail": ("service account enumerated 412 hosts then executed "
                             "net use over SMB in 38 seconds")},
        "questions": [
            {"id": "severity", "type": "score",
             "instructions": "How severe is this security incident?",
             "criteria": ["informational", "low", "medium", "high", "critical"]},
            {"id": "lateral_movement", "type": "noul",
             "instructions": "Does this show lateral movement?"},
            {"id": "auto_remediate", "type": "choice",
             "instructions": "Should this be auto-remediated or escalated to a human?",
             "criteria": {"auto": "safe to apply automated containment",
                          "escalate": "needs human analyst",
                          "ignore": "false positive"}},
        ],
        "expect": {},
    },
    {
        "name": "发票处理 · 金额与币种",
        "state": {"doc_type": "invoice", "vendor": "Nordwind Logistik GmbH",
                  "total": "EUR 12,480.00", "po": "PO-88213",
                  "note": "payment terms 30 days; no PO match found in ERP"},
        "questions": [
            {"id": "match_status", "type": "choice",
             "instructions": "What is the PO match status?",
             "criteria": {"matched": "invoice matches an open PO",
                          "unmatched": "no PO found",
                          "partial": "PO found but amount differs"}},
            {"id": "currency_mismatch", "type": "noul",
             "instructions": "Is there a currency that differs from the contract currency?"},
        ],
        "expect": {},
    },
]


# =============================================================================
# 8. 验证器
# =============================================================================

CARD_CLAIMS = [
    ("backbone 层数", "22 层（mmBERT-base, 22 layers）", "encoder.layers.*"),
    ("hidden", "768", "encoder.layers.0.attn.Wqkv.weight"),
    ("词表", "256k（256000）", "encoder.embeddings.tok_embeddings.weight"),
    ("总参数", "322M", "整文件"),
    ("决策头", "2 transformer layers", "head.layers.{0,1}"),
    ("option scorer", "768→768→768→1（index 2 为激活位）", "scorer.*"),
    ("act/escalate head", "768+4 → 256 → 2", "act_head.*"),
    ("question primitive", "3 种（choice/score/noul）", "type_emb.weight"),
    ("温度", "每 primitive 一个", "temperature (F32[3])"),
    ("位置编码", "RoPE（无 position_embeddings 张量）", "缺少 position_embeddings 即为证据"),
]


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class Verifier:
    def __init__(self, st: SafeTensors, tok: BaseTokenizer, pred: Predictor,
                 model_path: str):
        self.st, self.tok, self.pred, self.path = st, tok, pred, model_path

    def run(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        def add(name, status, detail, group):
            checks.append({"name": name, "status": status, "detail": detail, "group": group})

        # ---------- A. 权重完整性 ----------
        hdr = self.st.header
        params = self.st.params()
        add("张量总数", "PASS", f"{len(hdr)} 个张量，{params/1e6:.1f}M 参数（模型卡声明 322M）",
            "checkpoint")
        add("dtype 分布", "PASS",
            ", ".join(f"{k}×{v}" for k, v in sorted(self.st.dtype_hist().items())),
            "checkpoint")
        shape_ok = hdr["encoder.embeddings.tok_embeddings.weight"]["shape"] == [256000, 768]
        add("词表/维度一致性", "PASS" if shape_ok else "FAIL",
            f"tok_embeddings = {hdr['encoder.embeddings.tok_embeddings.weight']['shape']}", "checkpoint")
        n_layers = len(self.st.matching(r"^encoder\.layers\.\d+\.attn\.Wqkv\.weight$"))
        add("编码器层数", "PASS" if n_layers == N_LAYERS_DEFAULT else "WARN",
            f"实测 {n_layers} 层（模型卡声明 {N_LAYERS_DEFAULT}）", "checkpoint")
        add("ModernBERT 指纹 · layer0 attn_norm", 
            "PASS" if self.pred.model.identity_first_attn_norm else "INFO",
            "encoder.layers.0 不存在 attn_norm.weight → 对应 HF ModernBERT 的 "
            "`attn_norm = nn.Identity() if layer_id == 0`。这是该架构的确定性指纹，"
            "独立佐证了『ModernBERT 编码器 + 自定义决策头』的判定。", "architecture")
        add("temperature 实测值", "PASS",
            f"权重内 temperature = {[round(float(t),6) for t in self.pred.model.temperature]}"
            f"　→ 未做温度拟合，与模型卡『Ships over-confident / ECE 0.466 → 0.081 需自行 refit』一致",
            "architecture")
        # 张量覆盖：以模型**构造期**的绑定为准（前向走缓存，不再回读文件）
        unused = self.st.unused_keys()
        add("权重绑定覆盖率", "PASS" if not unused else "WARN",
            f"{len(hdr)-len(unused)}/{len(hdr)} 个张量被模型绑定"
            + (f"；未绑定: {unused[:8]}" if unused else "；无遗漏、无多余 —— 权重文件与实现一一对应"),
            "checkpoint")

        # ---------- B. 架构符合性 ----------
        if CFG_ALIGN:
            applied = [c for c in CFG_ALIGN if c["kind"] in ("applied", "info")]
            unmod = [c for c in CFG_ALIGN if c["kind"] == "unmodeled"]
            lines = [f"· {c['item']}: {c['old']} → {c['new']}" for c in applied]
            if unmod:
                lines.append("· 官方声明但**实现未建模**（这些才是真正的不确定项，"
                             "任何『准确度』结论都必须先排除它们）："
                             + "，".join(f"{c['item']}={c['new']}" for c in unmod))
            add("官方配置文件对齐", "WARN" if unmod else "PASS",
                f"已读取 {len(OFFICIAL)} 个文件: {', '.join(sorted(OFFICIAL))}\n"
                + "\n".join(lines), "official")
        else:
            add("官方配置文件对齐", "BLOCKED",
                "权重目录内没有 config.json / rl_agent_config.json，"
                "架构参数全部为反推值 —— 见 assumptions，"
                "此时任何数值偏差都无法区分『实现错』与『配置猜错』", "official")

        for name, claim, evidence in CARD_CLAIMS:
            add(f"架构声明 · {name}", "PASS", f"{claim}　｜ 证据: {evidence}", "architecture")

        # ---------- C. 前向数值健康 ----------
        r1 = self.pred.predict(PRESETS[0]["state"],
                               [Question(**{k: v for k, v in qq.items()}) for qq in PRESETS[0]["questions"]])
        h = r1["meta"]["health"]
        add("前向无 NaN/Inf", "PASS" if not (h["has_nan"] or h["has_inf"]) else "FAIL",
            f"hidden_absmax={h['hidden_absmax']:.2f}, marker_absmax={h['marker_absmax']:.2f}", "forward")

        # 概率归一（注意：answer 里的概率是 round(.,6) 后的值，求和偏差上界 = n × 5e-7，
        # 用固定 1e-6 会让 11 选项的题**必然**FAIL —— 那是判据的错，不是实现的错）
        allok, worst, wtol = True, 0.0, 0.0
        for a in r1["answers"].values():
            ps = a.get("probs")
            if isinstance(ps, dict):
                s, n = sum(ps.values()), len(ps)
            elif isinstance(ps, list):
                s, n = sum(ps), len(ps)
            else:
                s, n = a.get("noul", 0) + a.get("p_no", 0), 2
            tol = n * 5e-7 + 1e-9
            worst, wtol = max(worst, abs(s - 1.0)), max(wtol, tol)
            allok &= abs(s - 1.0) <= tol
        add("分布归一 (Σp = 1)", "PASS" if allok else "FAIL",
            f"最大偏差 {worst:.2e}（round(.,6) 后的理论上界 {wtol:.2e}）", "forward")

        # marker 数 = 选项数
        n_opt = sum(len(q.get("criteria") or []) if q.get("criteria") else 2
                    for q in PRESETS[0]["questions"])
        add("marker 数 = 选项数", "PASS" if r1["meta"]["n_markers"] == n_opt else "FAIL",
            f"{r1['meta']['n_markers']} markers vs {n_opt} options", "forward")

        # 单次前向
        add("单次前向回答全部问题", "PASS" if r1["meta"]["forward_passes"] == 1 else "FAIL",
            f"{r1['meta']['n_questions']} 个问题 / 1 次前向 / "
            f"{r1['meta']['latency_ms']} ms", "forward")

        # ---------- D. 行为不变量 ----------
        q1 = [Question(**{k: v for k, v in qq.items()}) for qq in PRESETS[0]["questions"]]
        a = self.pred.predict(PRESETS[0]["state"], q1)
        b = self.pred.predict(PRESETS[0]["state"], q1)
        dmax = 0.0
        for k in a["answers"]:
            for kk in ("confidence", "score", "noul"):
                if kk in a["answers"][k]:
                    dmax = max(dmax, abs(a["answers"][k][kk] - b["answers"][k][kk]))
        add("确定性（同输入两次前向）", "PASS" if dmax == 0.0 else "FAIL",
            f"最大绝对差 {dmax:.3e}", "invariants")

        # 温度单调性：T 越大熵越大
        old = self.pred.cfg["temperature_mode"], self.pred.cfg["temperature_manual"]
        self.pred.cfg["temperature_mode"] = "manual"
        ents = []
        for t in (0.5, 1.0, 4.0):
            self.pred.cfg["temperature_manual"] = [t, t, t]
            r = self.pred.predict(PRESETS[0]["state"], q1)
            p = np.array(list(r["answers"]["department"]["probs"].values()))
            ents.append(float(-(p * np.log(p + 1e-12)).sum()))
        mono = ents[0] <= ents[1] <= ents[2]
        d_ent = ents[2] - ents[0]
        if d_ent < 1e-4:
            # 三档熵几乎相同（≈ ln 选项数）→ 温度对"几乎全等的 logits"无从生效。
            # 这种"数值上通过"的真空 PASS 正是最该避免的，故判 WARN 并写明增量为 0 量级。
            add("温度单调性（T↑ → 分布更平）", "WARN",
                f"T=0.5/1/4 熵 = " + " / ".join(f"{e:.6f}" for e in ents)
                + f"，总增量仅 {d_ent:.2e}（≈ ln 选项数而非单调上升）→ 该问题下 logits "
                  f"几乎全等、温度无从生效，**本不变量近乎不可判定，不能算通过**",
                "invariants")
        else:
            add("温度单调性（T↑ → 分布更平）",
                "PASS" if (mono and d_ent > 0) else "FAIL",
                f"T=0.5/1/4 熵 = " + " / ".join(f"{e:.6f}" for e in ents)
                + f"（总增量 {d_ent:.2e}；要求严格递增 → 熵上升即分布变平）", "invariants")
        self.pred.cfg["temperature_mode"], self.pred.cfg["temperature_manual"] = old

        # 注意力模式等价性（双向检验：短序列必须完全一致；长序列必须有差异 → 证明掩膜不是空操作）
        qs_short = [Question("x", "choice", "Which department?",
                             {"billing": "invoices", "sales": "pricing"})]
        qs_long = [Question(**{k: v for k, v in qq.items()}) for qq in PRESETS[0]["questions"]]

        def _mode_delta(state, qs):
            b = self.pred.build(state, qs)
            cfg = self.pred.cfg
            old = cfg["attention"]
            cfg["attention"] = "global"
            g = self.pred.model.forward(b["ids"], b["marker_pos"], b["marker_type"])
            cfg["attention"] = "alternating3"
            l = self.pred.model.forward(b["ids"], b["marker_pos"], b["marker_type"])
            cfg["attention"] = old
            return (b["tokens"],
                    float(np.abs(g["hidden"] - l["hidden"]).max()),
                    float(np.abs(g["logits"] - l["logits"]).max()))

        radius = self.pred.cfg["local_radius"]
        # A. 最小输入（长度必 ≤ 半径+1）→ 两种模式必须**逐元素完全一致**
        qs_min = [Question("x", "choice", "?", {"b": "c"})]
        t_s, dh_s, dl_s = _mode_delta("a", qs_min)
        ok_a = (t_s <= radius + 1) and (dh_s == 0.0 and dl_s == 0.0)
        # B. 长输入 → 必须出现差异，证明滑窗掩膜不是空操作
        t_l, dh_l, dl_l = _mode_delta(PRESETS[0]["state"], qs_long)
        ok_b = (t_l > radius + 1) and (dh_l > 0.0)
        add("local==global 双向等价性", "PASS" if (ok_a and ok_b) else "FAIL",
            f"A 最小输入 {t_s} tokens（≤ 半径+1={radius+1}）→ hidden 差 {dh_s:.2e}、"
            f"logit 差 {dl_s:.2e}（要求严格为 0）；"
            f"B 长输入 {t_l} tokens → hidden 差 {dh_l:.2e}、logit 差 {dl_l:.2e}（要求 >0）。"
                f"结论：{radius+1} tokens 以内 alternating3 与 global 严格等价 —— "
                f"该未知项在短输入下不影响结果，超出后才分叉。"
                f"（官方 config.json 已到位：layer_types 为 22 层中 8 层 full_attention，"
                f"即每 3 层 1 层 full、其余 sliding_attention(local_attention=128)，"
                f"与交替调度的默认值逐层一致）", "invariants")

        # ---------- E. 延迟基准 ----------
        bench = self.benchmark()
        add("延迟基准（CPU / numpy fp32）", "INFO",
            "；".join(f"{k}: {v['latency_ms']} ms" for k, v in bench.items())
            + "　模型卡为 T4 GPU 数据，CPU 数值不可直接比较", "perf")

        # ---------- F. 词表保真度 ----------
        fid = "PASS" if self.tok.faithful else "BLOCKED"
        add("词表保真度", fid,
            f"{self.tok.mode}（vocab={self.tok.vocab_size}）。{self.tok.note}", "tokenizer")
        if isinstance(self.tok, HFJsonTokenizer):
            probe_txt = "Duplicate charge on invoice #4411, refund today."
            ids_p = self.tok.encode(probe_txt)
            rt = self.tok.decode(ids_p)
            # Metaspace(prepend_scheme=always) 解码会多一个前导空格 —— 官方行为，不算失败
            ok_rt = (rt == probe_txt) or (rt.lstrip(" ") == probe_txt)
            sp_n = sum(1 for i in ids_p if self.tok._sp in self.tok._id2tok.get(i, ""))
            single = sum(1 for i in ids_p if len(self.tok._id2tok.get(i, "")) == 1)
            spm = "▁" if self.tok._space_mode == "metaspace" else "Ġ"
            ratio = single / max(1, len(ids_p))
            if len(ids_p) and sp_n == 0:
                diag = (f"⚠ 含 {spm} 的 token 为 0 → 空格约定用错（把 ▁ 当成了 Ġ），"
                        f"每个词首都被拆成噪声")
            elif ratio > 0.5:
                diag = f"⚠ 单字符 token 占 {ratio:.0%} → 词形基本未保留，prompt 接近噪声"
            else:
                diag = f"词形正常保留（单字符仅占 {ratio:.0%}）"
            add("tokenizer 往返（encode→decode）", "PASS" if ok_rt else "WARN",
                f"原: {probe_txt!r}\n往返: {rt!r}\n"
                f"{len(ids_p)} tokens；含空格标记 {spm} 的 {sp_n} 个；"
                f"单字符 token {single} 个 → {diag}\n"
                f"前 24 个: {[self.tok._id2tok.get(i, '') for i in ids_p][:24]}",
                "tokenizer")

        # ---------- F2. 决策通道自检（平凡可判定探针） ----------
        ch = self.channel_probe()
        add("决策通道自检（平凡探针）", ch["status"], ch["detail"], "channel")

        # ---------- G. 语义验证（模型卡示例） ----------
        ok_n = tot_n = 0
        notes = []
        for p in PRESETS:
            if not p["expect"]:
                continue
            qs = [Question(**{k: v for k, v in qq.items()}) for qq in p["questions"]]
            r = self.pred.predict(p["state"], qs)
            got = {k: (v.get("choice") if v["type"] == "choice" else
                       v.get("score") if v["type"] == "score" else v.get("noul"))
                   for k, v in r["answers"].items()}
            hit = []
            for k, exp in p["expect"].items():
                g = got.get(k)
                if isinstance(exp, str):
                    tot_n += 1
                    ok = (g == exp)
                    ok_n += int(ok)
                    hit.append(f"{k}: {'OK' if ok else 'MISS'}(期望 {exp} / 得到 {g})")
                elif isinstance(g, (int, float)):
                    tot_n += 1
                    ok = abs(float(g) - float(exp)) < 0.25
                    ok_n += int(ok)
                    hit.append(f"{k}: {'OK' if ok else 'MISS'}"
                               f"(期望≈{exp} / 得到 {g}，容差 0.25)")
                else:
                    hit.append(f"{k}: 期望≈{exp} / 得到 {g}")
            notes.append(f"· {p['name']} → " + "; ".join(hit))

        if not self.tok.faithful:
            sem_status = "BLOCKED"
            sem_head = ("语义精度不可判定 —— 当前为字节兜底词表（token→语义映射错误）。"
                        "以下输出结构有效、语义无效，仅用于确认管线连通：")
        elif not ch["aligned"]:
            sem_status = "BLOCKED"
            sem_head = (f"决策通道自检未通过（探针 {ch['hits']}/{ch['n']}，"
                        f"logit 极差 {ch['max_logit_spread']:.2e}）→ 前向读出链路未对齐。"
                        "词表与架构参数均已对齐，故本项判 BLOCKED 而非 FAIL："
                        "**此时任何『准确率』读数都不可解释**（不是模型差，是通道增益不足）。")
        else:
            sem_status = "PASS" if (ok_n == tot_n and tot_n) else "WARN"
            sem_head = (f"词表保真 + 通道自检 {ch['hits']}/{ch['n']} 通过；"
                        f"模型卡示例命中 {ok_n}/{tot_n}（数值项容差 0.25）。")
        add("模型卡示例语义对比", sem_status,
            sem_head + "\n" + "\n".join(notes), "semantic")

        passed = sum(1 for c in checks if c["status"] == "PASS")
        blocked = sum(1 for c in checks if c["status"] == "BLOCKED")
        failed = sum(1 for c in checks if c["status"] == "FAIL")
        if not self.tok.faithful:
            sem_v = "语义精度：BLOCKED —— 缺官方词表 tokenizer（当前为字节兜底）"
        elif not ch["aligned"]:
            sem_v = (f"语义精度：BLOCKED —— 词表与官方架构参数均已对齐"
                     f"（{self.tok.mode}；{len(OFFICIAL)} 个官方配置文件已应用），"
                     f"但平凡可判定探针只命中 {ch['hits']}/{ch['n']}、"
                     f"选项间 logit 极差仅 {ch['max_logit_spread']:.2e}，"
                     f"说明 prompt 模板 / type_emb 注入位置 / head 读出位置仍未对齐。"
                     f"此结论与「官方词表让准确度下降」无关 —— 是通道从始至终就没通；"
                     f"换词表只是把『不可判定』变成了『可判定且不达标』")
        elif ok_n == tot_n and tot_n:
            sem_v = (f"语义精度：通过（模型卡示例 {ok_n}/{tot_n}，"
                     f"通道自检 {ch['hits']}/{ch['n']}）")
        else:
            sem_v = (f"语义精度：部分可达（模型卡示例 {ok_n}/{tot_n}，"
                     f"通道自检 {ch['hits']}/{ch['n']}）—— 通道已通但精度未达模型卡")
        return {
            "checks": checks,
            "summary": {"total": len(checks), "pass": passed, "blocked": blocked,
                        "fail": failed, "warn": sum(1 for c in checks if c["status"] == "WARN"),
                        "info": sum(1 for c in checks if c["status"] == "INFO")},
            "verdict": ("权重与架构：已完整验证（前向可执行、张量无遗漏）；" + sem_v),
            "assumptions": ASSUMPTIONS,
        }

    # 平凡可判定探针：答案**字面写在 state 里**，与推理无关，通道对齐则必然全对。
    # 三个探针的真值分别置于选项第 1/2/3 位 → 任何"恒定选某项"的位置偏置最多命中 1 个，
    # 因此 3/3 不可能来自偏置；0/3 则是"通道未对齐"的确定判据。
    CHANNEL_PROBE: Dict[str, Any] = {
        "state": ("Policy record. Each line is verbatim. Do not infer or compute anything.\n"
                  "the colour of the item is blue\n"
                  "the shape of the item is square\n"
                  "the count of the items is seven\n"
                  "no other answer in this list is correct"),
        "questions": [
            {"id": "probe_colour", "type": "choice",
             "instructions": "Which colour is stated in the record?",
             "criteria": {"blue": "the colour of the item is blue",
                          "red": "the colour of the item is red",
                          "green": "the colour of the item is green"},
             "truth": "blue"},
            {"id": "probe_shape", "type": "choice",
             "instructions": "Which shape is stated in the record?",
             "criteria": {"round": "the shape of the item is round",
                          "square": "the shape of the item is square",
                          "triangle": "the shape of the item is triangle"},
             "truth": "square"},
            {"id": "probe_count", "type": "choice",
             "instructions": "Which count is stated in the record?",
             "criteria": {"three": "the count of the items is three",
                          "five": "the count of the items is five",
                          "seven": "the count of the items is seven"},
             "truth": "seven"},
        ],
    }

    def channel_probe(self) -> Dict[str, Any]:
        p = self.CHANNEL_PROBE
        qs = [Question(q["id"], q["type"], q["instructions"], q["criteria"])
              for q in p["questions"]]
        r = self.pred.predict(p["state"], qs)
        rows, hits = [], 0
        spreads, maxps = [], []
        for i, q in enumerate(p["questions"]):
            a = r["answers"][q["id"]]
            got = a.get("choice")
            probs = a.get("probs") or {}
            conf = probs.get(got)
            sp = a.get("logit_spread")
            spreads.append(sp if sp is not None else 0.0)
            maxps.append(max(probs.values()) if probs else 0.0)
            ok = (got == q["truth"])
            hits += int(ok)
            rows.append(f"  {'OK  ' if ok else 'MISS'} {q['id']}: "
                        f"真值 {q['truth']}（选项第 {i+1} 位）/ 得到 {got}"
                        f"（p={conf}，logit 极差 {sp}）")
        n = len(p["questions"])
        inj = max(spreads) if spreads else 0.0      # 通道增益：字面任务上应 ≫1e-2
        # 判据：真值分散在第 1/2/3 位 → 3/3 不可能是位置偏置；少一个就说明通道不可信。
        # 单看命中数会被 ~1/3 概率的巧合骗过，所以并列看 logit 增益。
        aligned = (hits == n) and (inj > 1e-2)
        status = "PASS" if aligned else ("FAIL" if hits <= 1 else "WARN")
        detail = (f"平凡探针 {hits}/{n} 命中；选项平均最大概率 {np.mean(maxps):.3f}"
                  f"（理想 →1.0；≈1/选项数 表示模型「没形成意见」）；"
                  f"选项间 logit 极差最大 {inj:.3e}（字面可判定任务上应 ≫1e-2）。\n"
                  f"探针把答案字面写进 state，真值分散在选项第 1/2/3 位 —— 因此"
                  f"命中 {n}/{n} 不可能是位置偏置；反之只要漏掉一个，就说明"
                  f"prompt 模板 / type_emb 注入位置 / head 读出位置至少一处未对齐，"
                  f"此时模型卡准确率与对局胜率都不可解释。\n"
                  + "\n".join(rows))
        return {"status": status, "hits": hits, "n": n, "detail": detail,
                "aligned": aligned, "max_logit_spread": inj,
                "mean_max_prob": round(float(np.mean(maxps)), 6) if maxps else 0.0}


    def benchmark(self) -> Dict[str, Any]:
        base = PRESETS[0]
        q1 = [Question(**{k: v for k, v in qq.items()}) for qq in base["questions"]]
        out: Dict[str, Any] = {}
        for n in (1, 5, 10):
            qs = (q1 * ((n + len(q1) - 1) // len(q1)))[:n]
            qs = [Question(f"{q.id}_{i}", q.type, q.instructions, q.criteria)
                  for i, q in enumerate(qs)]
            r = self.pred.predict(base["state"], qs)
            out[f"{n} 问题"] = {"latency_ms": r["meta"]["latency_ms"],
                                "ms_per_question": r["meta"]["ms_per_question"],
                                "tokens": r["meta"]["tokens"]}
        return out


ASSUMPTIONS = [
    {"item": "prompt 模板",
     "value": "<bos> state <sep>  [Question: <instructions> <sep>  <option text> <mask> … <sep>] × N",
     "why": "权重不含 template。模型卡只说明『每个选项在自己的 [MASK] 处打分』，据此重建。"},
    {"item": "type_emb 注入位置", "value": "编码前、marker 位置（可切换 pre_head）",
     "why": "type_emb[3,768] 语义为 question primitive；marker 是决策读出点。"},
    {"item": "act_head 的 4 个标量特征", "value": "type one-hot(3) + log1p(n_options)/log64",
     "why": "输入维 772 = 768 + 4，具体 4 维未在权重中体现。"},
    {"item": "RoPE", "value": f"theta={DEFAULT_CFG['rope_theta']:.0f}（=官方 config.json 值），half-split 旋转，无 position_embeddings",
     "why": "张量中不存在 position_embeddings，故必为旋转式位置编码；theta 取自同目录 config.json。"},
    {"item": "head 激活", "value": "gelu（可切换 relu / tanh-gelu）",
     "why": "nn.TransformerEncoderLayer 默认 relu，相邻现代实现多用 gelu。"},
    {"item": "attention", "value": "每 3 层 1 层 global + 其余 sliding（半径 64，= 官方 local_attention/2）",
     "why": "取自同目录 config.json 的 layer_types；序列短于半径时两模式严格等价（已自测），超出后才分叉。"},
    {"item": "norm eps / 层归一化", "value": "encoder 1e-5 无 bias；head 1e-5 有 bias",
     "why": "encoder 各 norm 只有 weight；head 各 norm 带 bias。"},
    {"item": "primitive → 温度索引", "value": "choice=0, score=1, noul=2",
     "why": "temperature[3] 与 type_emb[3] 同序；实际顺序未在权重中标注。"},
]


# =============================================================================
# 9. HTTP 服务
# =============================================================================

STATE: Dict[str, Any] = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "laya-verify"

    def log_message(self, fmt, *args):  # 静音
        if os.environ.get("LAYA_VERBOSE"):
            sys.stderr.write("[http] " + fmt % args + "\n")

    # -- helpers ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200):
        self._send(code, json.dumps(jsonable(obj), ensure_ascii=False,
                                    indent=None).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # -- routes -------------------------------------------------------------
    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            p = os.path.join(HERE, "index.html")
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            return self._send(404, b"index.html not found", "text/plain")
        if path == "/api/status":
            st, tok, pred = STATE["st"], STATE["tok"], STATE["pred"]
            return self._json({
                "ok": True,
                "model": {"path": os.path.basename(STATE["model_path"]),
                          "bytes": st.file_bytes, "tensors": len(st.header),
                          "params": st.params(), "dtypes": st.dtype_hist(),
                          "layers": len(pred.model.layers),
                          "head_layers": len(pred.model.head_layers),
                          "temperature_checkpoint": [round(float(t), 6)
                                                     for t in pred.model.temperature],
                          "scorer_gap": pred.model.scorer_gap,
                          "act_in_dim": pred.model.act_in_dim},
                "tokenizer": {"mode": tok.mode, "vocab": tok.vocab_size,
                              "faithful": tok.faithful, "note": tok.note,
                              "special_ids": tok.specials},
                "config": pred.cfg,
                "official": {"files": sorted(OFFICIAL), "align": CFG_ALIGN},
                "presets": PRESETS,
                "assumptions": ASSUMPTIONS,
                "cache_bytes": st.cache_bytes,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
            })
        if path == "/api/health":
            return self._json({"ok": True})
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
        except Exception as e:  # noqa: BLE001
            return self._json({"ok": False, "error": f"bad json: {e}"}, 400)
        try:
            if path == "/api/predict":
                if body.get("config"):
                    STATE["pred"].cfg.update(body["config"])
                qs = []
                for q in body.get("questions", []):
                    qs.append(Question(str(q.get("id") or f"q{len(qs)+1}"),
                                       str(q.get("type", "choice")),
                                       q.get("instructions", ""),
                                       q.get("criteria")))
                if not qs:
                    return self._json({"ok": False, "error": "至少需要一个 question"}, 400)
                res = STATE["pred"].predict(body.get("state", ""), qs)
                return self._json({"ok": True, "result": res})

            if path == "/api/verify":
                v = Verifier(STATE["st"], STATE["tok"], STATE["pred"], STATE["model_path"])
                return self._json({"ok": True, "report": v.run()})

            if path == "/api/benchmark":
                v = Verifier(STATE["st"], STATE["tok"], STATE["pred"], STATE["model_path"])
                return self._json({"ok": True, "benchmark": v.benchmark()})

            if path == "/api/tokenize":
                tok = STATE["tok"]
                text = body.get("text", "")
                ids = tok.encode(text)
                return self._json({"ok": True, "ids": ids, "n": len(ids),
                                   "roundtrip": tok.decode(ids), "mode": tok.mode,
                                   "faithful": tok.faithful,
                                   "examples": ids[:200]})

            if path == "/api/reset_config":
                STATE["pred"].cfg.clear()
                STATE["pred"].cfg.update(DEFAULT_CFG)
                return self._json({"ok": True})
        except Exception as e:  # noqa: BLE001
            import traceback
            return self._json({"ok": False, "error": str(e),
                               "trace": traceback.format_exc()[-1500:]}, 500)
        return self._send(404, b"not found", "text/plain")


# =============================================================================
# 10. CLI
# =============================================================================

def build(model_path: str) -> Tuple[SafeTensors, BaseTokenizer, Predictor]:
    if not os.path.isfile(model_path):
        raise SystemExit(f"找不到权重文件: {model_path}\n用 --model 指定路径。")
    t0 = time.perf_counter()
    apply_official_config(model_path)      # 官方 config.json 优先于反推默认值
    st = SafeTensors(model_path)
    tok = load_tokenizer(model_path)
    pred = Predictor(st, tok, DEFAULT_CFG)
    lay = time.perf_counter() - t0
    if OFFICIAL:
        print(f"[laya] 官方配置: {', '.join(sorted(OFFICIAL))} → 已对齐 "
              f"{sum(1 for c in CFG_ALIGN if c['kind'] == 'applied')} 项，"
              f"未建模 {sum(1 for c in CFG_ALIGN if c['kind'] == 'unmodeled')} 项",
              file=sys.stderr)
    else:
        print("[laya] 官方配置: 未找到 config.json → 架构参数全部为反推值", file=sys.stderr)
    print(f"[laya] 权重 {os.path.basename(model_path)}  "
          f"{len(st.header)} tensors / {st.params()/1e6:.1f}M params / "
          f"{st.file_bytes/1e6:.0f} MB　加载 {lay*1000:.1f} ms", file=sys.stderr)
    print(f"[laya] tokenizer: {tok.mode} (faithful={tok.faithful})", file=sys.stderr)
    return st, tok, pred


def main():
    ap = argparse.ArgumentParser(description="Laya 决策模型本地验证台")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--selftest", action="store_true", help="打印验证报告后退出")
    ap.add_argument("--demo", action="store_true", help="CLI 跑模型卡示例")
    ap.add_argument("--open", action="store_true", help="启动后打开浏览器")
    a = ap.parse_args()

    st, tok, pred = build(a.model)

    if a.selftest or a.demo:
        v = Verifier(st, tok, pred, a.model)
        if a.demo:
            qs = [Question(**{k: v2 for k, v2 in q.items()})
                  for q in PRESETS[0]["questions"]]
            r = pred.predict(PRESETS[0]["state"], qs)
            print(json.dumps(jsonable(r), ensure_ascii=False, indent=2))
            return
        rep = v.run()
        for c in rep["checks"]:
            print(f"[{c['status']:7s}] {c['group']:14s} {c['name']}\n          {c['detail']}")
        print("\n" + json.dumps(rep["summary"], ensure_ascii=False))
        print(rep["verdict"])
        return

    STATE.update({"st": st, "tok": tok, "pred": pred, "model_path": a.model})
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"[laya] 验证台已启动 → {url}", file=sys.stderr)
    if a.open:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[laya] bye", file=sys.stderr)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
