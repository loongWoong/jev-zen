#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · 控制端后端（control_server.py）

让 labs/finetune/dashboard.html 成为真正的控制面板：
  - 模型加载   ：在「服务端」选择/加载 .safetensors，build + 通道探针 + 延迟 + 标签准确率
  - 微调       ：一键跑 convert_maze.py → finetune_run.py（numpy head-only），流式日志 + loss 曲线
  - 运行监控   ：对齐门状态、延迟、choice top-1 准确率、训练进度实时轮询

运行（用带 numpy 的 envs/default）：
  python control_server.py --port 8772
浏览器打开 http://127.0.0.1:8772/ 即为 live 控制面板。

说明：模型文件在服务端文件系统，所谓「模型加载」是选服务端路径（下拉 + 手动输入），
不支持浏览器上传 643MB 权重。
"""
from __future__ import annotations
import os
import sys
import json
import time
import threading
import subprocess
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))        # 工作树根，含 laya_verify.py
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from laya_verify import build, Verifier              # noqa: E402
import benchmark                                       # noqa: E402

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ARTIFACTS = os.path.join(HERE, "artifacts")
DATASET = os.path.join(ARTIFACTS, "maze_dataset.jsonl")
FINETUNED = os.path.join(ARTIFACTS, "model_maze_ft.safetensors")
SEARCH_DIRS = [
    os.path.join("C:/Users/Administrator/Downloads/jev-zen"),
    ARTIFACTS,
    HERE,
]

# 静态模型事实（架构反推，与 dashboard 内 MODEL 一致）
MODEL_FACTS = {
    "params": "321.9M", "tensors": 170, "sizeMB": 644, "layers": 22,
    "hidden": 768, "heads": 12,
    "tokenizer": "tokenizer.json / BPE (faithful=True)",
    "rope": "theta=160000",
    "attn": "alternating3 (每3层1全局+sliding r=64)",
    "norm": "encoder 1e-5 无bias / head 1e-5 有bias",
}

# ---------------------------------------------------------------------------
# 全局状态（线程安全）
# ---------------------------------------------------------------------------
LOCK = threading.Lock()
FT_STOP = threading.Event()
FT_PROC = {"proc": None}

STATE = {
    "backend": "control_server",
    "model": {
        "path": None, "loaded": False, "loading": False, "error": None,
        "facts": MODEL_FACTS,
        "channel": None,        # {status, aligned, hits, n, max_logit_spread, mean_max_prob}
        "latency": None,        # {"1 问题": {...}, ...}
        "label_accuracy": None, # {n, choice_top1_acc}
    },
    "finetune": {
        "running": False, "stage": "idle",      # idle|convert|train|done|error|stopped
        "started_at": None, "finished_at": None,
        "epochs": 0, "loss_curve": [],          # [[epoch, loss], ...]
        "result": None,                         # {before_acc, after_acc, delta, ...}
        "error": None, "auto_loaded": False,
    },
}

LOG = deque(maxlen=800)        # 全量事件日志
FT_LOG = deque(maxlen=800)     # 微调日志（含 convert/train 输出）

MODEL_OBJ = {"st": None, "tok": None, "pred": None, "ver": None}


def ts():
    return time.strftime("%H:%M:%S")


def log_event(msg, tag="sys"):
    line = f"[{ts()}][{tag}] {msg}"
    LOG.append(line)
    FT_LOG.append(line)


def native(x):
    """numpy / 其他非 JSON 原生类型 → 原生类型。"""
    if isinstance(x, dict):
        return {k: native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return x
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return str(x)


# ---------------------------------------------------------------------------
# 模型加载 / 对齐 / 评测（服务端进程内）
# ---------------------------------------------------------------------------
def run_load_model(path):
    with LOCK:
        STATE["model"].update(loaded=False, loading=True, error=None, path=path)
    log_event(f"加载模型: {path}")
    t0 = time.perf_counter()
    try:
        st, tok, pred = build(path)
        ver = Verifier(st, tok, pred, path)
        # 把静态事实里可量化的部分用真实值覆盖
        facts = dict(MODEL_FACTS)
        try:
            facts["params"] = f"{st.params()/1e6:.1f}M"
            facts["tensors"] = len(st.header)
            facts["sizeMB"] = int(st.file_bytes / 1e6)
        except Exception:
            pass
        with LOCK:
            MODEL_OBJ["st"], MODEL_OBJ["tok"] = st, tok
            MODEL_OBJ["pred"], MODEL_OBJ["ver"] = pred, ver
        # 一次性评测：通道 + 延迟 + 标签准确率
        rep = benchmark.run(path, DATASET if os.path.isfile(DATASET) else None, "live")
        m = rep.get("metrics", {})
        dt = time.perf_counter() - t0
        with LOCK:
            STATE["model"].update(
                loaded=True, loading=False, error=None,
                facts=facts,
                channel=m.get("channel"),
                latency=m.get("latency"),
                label_accuracy=m.get("label_accuracy"),
            )
        log_event(f"模型就绪（{dt:.1f}s）对齐={m.get('channel',{}).get('status')} "
                  f"acc={m.get('label_accuracy',{}).get('choice_top1_acc')}")
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATE["model"].update(loaded=False, loading=False, error=str(e))
        log_event(f"加载失败: {e}", "ERR")


def run_align():
    with LOCK:
        if not MODEL_OBJ["ver"]:
            return
        ver = MODEL_OBJ["ver"]
    try:
        ch = ver.channel_probe()
        with LOCK:
            STATE["model"]["channel"] = native(ch)
        log_event(f"重新对齐：{ch['status']} hits={ch['hits']}/{ch['n']} "
                  f"spread={ch['max_logit_spread']:.2e}")
    except Exception as e:  # noqa: BLE001
        log_event(f"对齐失败: {e}", "ERR")


def run_benchmark():
    with LOCK:
        if not STATE["model"].get("path"):
            return
        path = STATE["model"]["path"]
    try:
        rep = benchmark.run(path, DATASET if os.path.isfile(DATASET) else None, "live")
        m = rep.get("metrics", {})
        with LOCK:
            STATE["model"].update(
                channel=m.get("channel"),
                latency=m.get("latency"),
                label_accuracy=m.get("label_accuracy"),
            )
        log_event("重新评测完成")
    except Exception as e:  # noqa: BLE001
        log_event(f"评测失败: {e}", "ERR")


# ---------------------------------------------------------------------------
# 微调流水线（子进程流式）
# ---------------------------------------------------------------------------
def _stream(proc, tag):
    """读子进程 stdout，逐行写入 FT_LOG；返回全部行。"""
    lines = []
    for raw in proc.stdout:
        s = raw.rstrip("\n")
        if not s:
            continue
        FT_LOG.append(f"[{ts()}][{tag}] {s}")
        lines.append(s)
        # loss 曲线
        if tag == "train":
            import re
            mo = re.search(r"epoch\s+(\d+)/\d+\s+loss=([\d.]+)", s)
            if mo:
                with LOCK:
                    STATE["finetune"]["loss_curve"].append(
                        [int(mo.group(1)), float(mo.group(2))])
        if FT_STOP.is_set():
            break
    return lines


def run_finetune(epochs, lr):
    import re
    with LOCK:
        STATE["finetune"].update(running=True, stage="convert", error=None,
                                 result=None, loss_curve=[], started_at=ts(),
                                 finished_at=None, auto_loaded=False)
    FT_STOP.clear()
    FT_LOG.clear()
    log_event(f"启动微调流水线 epochs={epochs} lr={lr}")

    py = sys.executable
    # 1) 转换数据集
    with LOCK:
        STATE["finetune"]["stage"] = "convert"
    try:
        p1 = subprocess.Popen(
            [py, os.path.join(HERE, "convert_maze.py")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=HERE, text=True, bufsize=1, encoding="utf-8", errors="replace")
        FT_PROC["proc"] = p1
        _stream(p1, "convert")
        p1.wait()
        if FT_STOP.is_set():
            raise InterruptedError("用户停止")
        if p1.returncode != 0:
            raise RuntimeError(f"convert_maze.py 退出码 {p1.returncode}")
        log_event("数据集转换完成")
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATE["finetune"].update(running=False, stage="error", error=str(e))
        log_event(f"转换阶段失败: {e}", "ERR")
        return

    # 2) 头训练
    with LOCK:
        STATE["finetune"]["stage"] = "train"
    try:
        p2 = subprocess.Popen(
            [py, os.path.join(HERE, "finetune_run.py"),
             "--epochs", str(epochs), "--lr", str(lr), "--log-every", "5"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=HERE, text=True, bufsize=1, encoding="utf-8", errors="replace")
        FT_PROC["proc"] = p2
        lines = _stream(p2, "train")
        p2.wait()
        if FT_STOP.is_set():
            raise InterruptedError("用户停止")
        if p2.returncode != 0:
            raise RuntimeError(f"finetune_run.py 退出码 {p2.returncode}")

        # 解析最终结果
        before = after = delta = None
        for s in lines:
            m = re.search(r"训练前:\s*([\d.]+)", s)
            if m:
                before = float(m.group(1))
            m = re.search(r"训练后:\s*([\d.]+)", s)
            if m:
                after = float(m.group(1))
            m = re.search(r"Δ\s*:\s*([-\d.]+)", s)
            if m:
                delta = float(m.group(1))
        with LOCK:
            STATE["finetune"].update(
                running=False, stage="done", finished_at=ts(),
                result={"before_acc": before, "after_acc": after,
                        "delta": delta,
                        "note": "head-only（scorer）训练，权重已落盘至 model_maze_ft.safetensors，并自动重载刷新监控"})
        log_event(f"微调完成：{before} → {after}（Δ {delta}）")
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATE["finetune"].update(running=False, stage="error", error=str(e))
        log_event(f"训练阶段失败: {e}", "ERR")
        return

    # 3) 自动用微调后模型刷新监控（重新加载已落盘的权重，得到真实 after 状态）
    if os.path.isfile(FINETUNED):
        try:
            log_event("刷新运行监控（重载微调后模型，复算对齐/延迟/标签准确率）…")
            run_load_model(FINETUNED)
            with LOCK:
                STATE["finetune"]["auto_loaded"] = True
        except Exception as e:  # noqa: BLE001
            log_event(f"刷新监控失败（不影响训练结果）: {e}", "ERR")
    else:
        log_event("未找到落盘模型，跳过监控刷新", "WARN")


def refresh_acc_after():
    """保留占位（历史实现），当前由 run_load_model(FINETUNED) 替代。"""
    pass


# ---------------------------------------------------------------------------
# HTTP 路由
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "laya-control"

    def log_message(self, fmt, *args):
        if os.environ.get("LAYA_VERBOSE"):
            sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(native(obj), ensure_ascii=False),
                   "application/json; charset=utf-8")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "dashboard.html not found", "text/plain")
            return
        if path == "/api/state":
            with LOCK:
                snap = json.loads(json.dumps(native(STATE)))
                log_tail = list(LOG)[-200:]
                ft_tail = list(FT_LOG)[-300:]
            snap["log"] = log_tail
            snap["ft_log"] = ft_tail
            self._json(snap)
            return
        if path == "/api/models":
            self._json(list_models())
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/load_model":
            d = self._body()
            path = d.get("path")
            if not path or not os.path.isfile(path):
                self._json({"ok": False, "error": f"文件不存在: {path}"}, 400)
                return
            threading.Thread(target=run_load_model, args=(path,), daemon=True).start()
            self._json({"ok": True, "loading": True})
            return
        if route == "/api/align":
            threading.Thread(target=run_align, daemon=True).start()
            self._json({"ok": True})
            return
        if route == "/api/benchmark":
            threading.Thread(target=run_benchmark, daemon=True).start()
            self._json({"ok": True})
            return
        if route == "/api/finetune":
            with LOCK:
                if STATE["finetune"]["running"]:
                    self._json({"ok": False, "error": "已在运行中"}, 409)
                    return
            d = self._body()
            epochs = int(d.get("epochs", 80))
            lr = float(d.get("lr", 0.02))
            threading.Thread(target=run_finetune, args=(epochs, lr), daemon=True).start()
            self._json({"ok": True, "started": True})
            return
        if route == "/api/finetune/stop":
            FT_STOP.set()
            p = FT_PROC.get("proc")
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
            with LOCK:
                STATE["finetune"]["stage"] = "stopped"
            self._json({"ok": True, "stopped": True})
            return
        self._send(404, "not found", "text/plain")


def list_models():
    out = []
    seen = set()
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors") and fn not in seen:
                seen.add(fn)
                out.append({"name": fn, "path": os.path.join(d, fn)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Laya 微调控制端后端")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[control] 控制端后端已启动 → http://{args.host}:{args.port}/")
    print(f"[control] 权重搜索目录: {SEARCH_DIRS}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[control] 已停止")


if __name__ == "__main__":
    main()
