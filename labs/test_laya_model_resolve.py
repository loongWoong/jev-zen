# -*- coding: utf-8 -*-
"""隔离测试：laya_verify_cuda.py 的 `--model` 解析层。

不加载真实模型、不起 GPU 前向 —— 用临时目录里的微型合法 safetensors 覆盖
「路径形态 / 歧义与候选报错 / 加载前体检 / 配置与分词器出处 / 全局锚点回钉」五组用例。

跑法（需 laya-cuda venv：脚本 import cupy，所以不能用普通 python）：
    C:/Users/Administrator/.workbuddy/binaries/python/envs/laya-cuda/Scripts/python.exe \
        labs/test_laya_model_resolve.py
退出码 0 = 全绿。
"""
import json
import os
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import laya_verify_cuda as C  # noqa: E402

OK, FAIL = "PASS", "FAIL"
rows = []


def check(name, cond, detail=""):
    rows.append((OK if cond else FAIL, name, detail))
    print(f"[{OK if cond else FAIL:4s}] {name}  {detail}")


def write_st(path, keys):
    hdr, data, off = {}, b"", 0
    for k, shape in keys.items():
        n = 1
        for s in shape:
            n *= s
        hdr[k] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n * 4]}
        data += b"\x00" * (n * 4)
        off += n * 4
    with open(path, "wb") as f:
        h = json.dumps(hdr).encode()
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        f.write(data)


FULL = {"encoder.embeddings.tok_embeddings.weight": [4, 3],
        "encoder.layers.0.attn.Wqkv.weight": [2, 2],
        "scorer.0.weight": [2, 2], "scorer.0.bias": [2]}
HEAD_ONLY = {"scorer.0.weight": [2, 2], "scorer.0.bias": [2]}

tmp = tempfile.mkdtemp(prefix="laya_res_")
dA = os.path.join(tmp, "models", "maze")
os.makedirs(dA)
good = os.path.join(dA, "model.safetensors")
write_st(good, FULL)
with open(os.path.join(dA, "config.json"), "w", encoding="utf-8") as f:
    json.dump({"hidden_size": 768, "num_hidden_layers": 1, "rope_parameters":
               {"full_attention": {"rope_theta": 12345}}}, f)
with open(os.path.join(dA, "tokenizer.json"), "w", encoding="utf-8") as f:
    f.write("{}")

dB = os.path.join(tmp, "two")           # 目录内两个权重 → 必须报歧义
os.makedirs(dB)
write_st(os.path.join(dB, "a.safetensors"), FULL)
write_st(os.path.join(dB, "b.safetensors"), FULL)

dC = os.path.join(tmp, "headonly")      # 增量/仅头部权重 → 必须拦下
os.makedirs(dC)
write_st(os.path.join(dC, "model.safetensors"), HEAD_ONLY)

bogus = os.path.join(tmp, "bogus.safetensors")
with open(bogus, "w", encoding="utf-8") as f:
    f.write("i am not a safetensors file")

# ---- 1. 路径形式 ----------------------------------------------------------
check("文件路径", C.resolve_model(good) == C._norm(good), good.split(os.sep)[-1])
check("目录", C.resolve_model(dA) == C._norm(good))
check("省略后缀", C.resolve_model(os.path.join(dA, "model")) == C._norm(good))
check("glob（唯一命中）", C.resolve_model(os.path.join(dA, "model*.safetensors"))
      == C._norm(good))
check("空值走默认", C.resolve_model(None) == C._norm(C.DEFAULT_MODEL),
      os.path.basename(C.DEFAULT_MODEL))

try:
    C.resolve_model(dB)
    check("目录内多权重报歧义", False)
except SystemExit as e:
    check("目录内多权重报歧义", "不唯一" in str(e), str(e).splitlines()[0])

try:
    C.resolve_model(os.path.join(tmp, "nope.safetensors"))
    check("路径不存在报错", False)
except SystemExit as e:
    msg = str(e)
    check("路径不存在报错 + 带候选", "找不到权重文件" in msg and "model.safetensors" in msg,
          msg.splitlines()[0])

try:
    C.resolve_model(os.path.join(tmp, "**", "*.safetensors"))
    check("glob 多命中报歧义", False)
except SystemExit as e:
    check("glob 多命中报歧义", "命中" in str(e), str(e).splitlines()[0])

# ---- 2. 加载前体检 --------------------------------------------------------
try:
    info = C.preflight(good)
    check("体检放行完整权重", info["n_layers"] == 1 and info["has_scorer"] is True, str(info))
except SystemExit as e:
    check("体检放行完整权重", False, str(e))

try:
    C.preflight(os.path.join(dC, "model.safetensors"))
    check("拦下仅头部/增量权重", False)
except SystemExit as e:
    check("拦下仅头部/增量权重", "增量" in str(e), str(e).splitlines()[1].strip())

try:
    C.preflight(bogus)
    check("拦下非 safetensors", False)
except SystemExit as e:
    check("拦下非 safetensors", "不是合法" in str(e), str(e).splitlines()[0])

# ---- 3. 配置 / 分词器出处 -------------------------------------------------
dirs, src = C.resolve_cfg_dirs(good, None, strict=False)
check("config 默认含脚本目录回退", dirs[0] == os.path.dirname(good) and C._norm(C.HERE) in dirs, src)
dirs2, src2 = C.resolve_cfg_dirs(good, None, strict=True)
check("--strict 去掉脚本目录回退", C._norm(C.HERE) not in dirs2, src2)
dirs3, src3 = C.resolve_cfg_dirs(good, tmp, strict=False)
check("--config 独占", dirs3 == [C._norm(tmp)], src3)

tok, tsrc = C.resolve_tokenizer(good, None, strict=False)
check("tokenizer 取模型目录", tok == C._norm(os.path.join(dA, "tokenizer.json")), tsrc)
tok2, tsrc2 = C.resolve_tokenizer(os.path.join(dC, "model.safetensors"), None, strict=True)
check("--strict 下无 tokenizer 时兜底", tok2 is None, tsrc2)

# ---- 4. 配置生效（重钉 + 目录优先级） -------------------------------------
os.environ.pop("LAYA_CONFIG_DIR", None)
with C.cfg_dirs_override(dirs2):
    C.L.apply_official_config(good)
check("模型目录 config.json 生效", C.L.DEFAULT_CFG["n_layers"] == 1
      and C.L.DEFAULT_CFG["rope_theta"] == 12345.0,
      f"n_layers={C.L.DEFAULT_CFG['n_layers']} rope_theta={C.L.DEFAULT_CFG['rope_theta']:.0f}")
check("_cfg_dirs 已还原", C.L._cfg_dirs(good) == [os.path.dirname(good), C._norm(C.HERE)]
      or C._norm(C.HERE) in C.L._cfg_dirs(good), str(C.L._cfg_dirs(good)))

# ---- 5. 全局锚点 ----------------------------------------------------------
fake = os.path.join(tmp, "anchor.safetensors")
write_st(fake, FULL)
C.DEFAULT_MODEL = fake
C.L.DEFAULT_MODEL = fake
r = C.L.route("hello world", ["multilingual"])
check("route() repo 跟随模型", os.path.basename(fake) in r["repo"], r["repo"])

bad = [x for x in rows if x[0] == FAIL]
print(f"\n结果：{len(rows) - len(bad)}/{len(rows)} 通过" + ("" if not bad else " ✗"))
sys.exit(1 if bad else 0)
