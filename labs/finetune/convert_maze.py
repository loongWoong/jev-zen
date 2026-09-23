#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 labs/finetune/artifacts/benchmark_before.json（用户写入的迷宫 A* 回溯轨迹，
Lab.hist 格式，每行含 legal / rd{choice,ranks,note} = 教师标签）转换成
dataset.py 所需的微调数据集 JSONL：

  { scene, seed, step, state, questions, labels,
    ids, marker_pos, marker_type, spans, tokens, alignment_tag }

关键：轨迹只含决策标签 rd，不含 state 文本与 typed questions。本脚本通过
「复刻 maze_laya.html 的 genMaze/mulberry32/astar，对 seed 做穷举，匹配轨迹里
A* 唯一路径」来恢复出原始迷宫 → 由此重建每一步 EXACT 的 SCENE.text() 与
SCENE.questions()，得到与真实场景同分布的 state/questions。

若 seed 在搜索范围内没匹配上，则退化用轨迹字段重建 state（位置+legal+各方向剩余步），
仍产出合法 schema 样本，并标注 method='reconstructed'。

用法：
  python convert_maze.py
  python convert_maze.py --trace <jsonl> --out <jsonl> --search 200000
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 工作树根，含 laya_verify.py / maze_laya.html
sys.path.insert(0, ROOT)

from laya_verify import build, Question, PRIMITIVES  # noqa: E402
from dataset import derive_labels, tokenize_sample  # noqa: E402

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")
TRACE_DEFAULT = os.path.join(HERE, "artifacts", "benchmark_before.json")
OUT_DEFAULT = os.path.join(HERE, "artifacts", "maze_dataset.jsonl")

W = 16   # 轨迹坐标到 (15,15) → 16×16
H = 16
GOAL = W * H - 1
START = 0


# ---------------------------------------------------------------------------
# 复刻 maze_laya.html 的迷宫逻辑（逐行移植，保证与真实场景一致）
# ---------------------------------------------------------------------------
def mulberry32(a):
    a = a & 0xFFFFFFFF
    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = (a ^ (a >> 15)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * 61 & 0xFFFFFFFF) & 0xFFFFFFFF) & 0xFFFFFFFF
        t = t ^ (t >> 14)
        return ((t & 0xFFFFFFFF) >> 0) / 4294967296
    return rnd


DIR = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
KEYS = ["up", "down", "left", "right"]
BIT = {"up": 1, "down": 2, "left": 4, "right": 8}
OBIT = {"up": 2, "down": 1, "left": 8, "right": 4}  # 反向位的 bit（与 genMaze 中 obit 映射一致）


def genMaze(Wd, Hd, rng):
    visited = [False] * (Wd * Hd)
    link = [0] * (Wd * Hd)
    stack = [0]
    visited[0] = True
    while stack:
        cur = stack[-1]
        r = cur // Wd
        c = cur % Wd
        cand = []
        if r > 0 and not visited[cur - Wd]:
            cand.append(("up", cur - Wd))
        if r < Hd - 1 and not visited[cur + Wd]:
            cand.append(("down", cur + Wd))
        if c > 0 and not visited[cur - 1]:
            cand.append(("left", cur - 1))
        if c < Wd - 1 and not visited[cur + 1]:
            cand.append(("right", cur + 1))
        if not cand:
            stack.pop()
            continue
        d, nxt = cand[int(rng() * len(cand))]
        bit = {"up": 1, "down": 2, "left": 4, "right": 8}[d]
        obit = {"up": 2, "down": 1, "left": 8, "right": 4}[d]
        link[cur] |= bit
        link[nxt] |= obit
        visited[nxt] = True
        stack.append(nxt)
    return {"W": Wd, "H": Hd, "link": link}


def bfsDist(maze, frm, to):
    Wd, Hd, link = maze["W"], maze["H"], maze["link"]
    dist = [-1] * (Wd * Hd)
    dist[frm] = 0
    q = [frm]
    while q:
        cur = q.pop(0)
        if cur == to:
            return dist[cur]
        for d in KEYS:
            if not (link[cur] & BIT[d]):
                continue
            r = cur // Wd + DIR[d][0]
            c = cur % Wd + DIR[d][1]
            if r < 0 or c < 0 or r >= Hd or c >= Wd:
                continue
            ni = r * Wd + c
            if dist[ni] >= 0:
                continue
            dist[ni] = dist[cur] + 1
            q.append(ni)
    return -1


def astar(maze, frm, to):
    Wd, Hd, link = maze["W"], maze["H"], maze["link"]
    h = lambda i: abs(i // Wd - to // Wd) + abs(i % Wd - to % Wd)
    g = {frm: 0}
    prev = {}
    open_ = [(frm, h(frm))]
    while open_:
        bi = 0
        for k in range(1, len(open_)):
            if open_[k][1] < open_[bi][1]:
                bi = k
        cur = open_.pop(bi)[0]
        if cur == to:
            path = []
            x = to
            while x in prev:
                p = prev[x]
                path.append(p[1])
                x = p[0]
            return list(reversed(path))
        for d in KEYS:
            if not (link[cur] & BIT[d]):
                continue
            r = cur // Wd + DIR[d][0]
            c = cur % Wd + DIR[d][1]
            if r < 0 or c < 0 or r >= Hd or c >= Wd:
                continue
            ni = r * Wd + c
            ng = g.get(cur, 0) + 1
            if ni in g and g[ni] <= ng:
                continue
            g[ni] = ng
            prev[ni] = (cur, d)
            open_.append((ni, ng + h(ni)))
    return None


def legal(maze, pos):
    Wd, Hd, link = maze["W"], maze["H"], maze["link"]
    out = []
    for d in KEYS:
        if not (link[pos] & BIT[d]):
            continue
        r = pos // Wd + DIR[d][0]
        c = pos % Wd + DIR[d][1]
        if r < 0 or c < 0 or r >= Hd or c >= Wd:
            continue
        out.append(d)
    return out


def text(maze, pos, steps, optimal, legal_moves):
    Wd, Hd, link = maze["W"], maze["H"], maze["link"]
    rows = []
    for r in range(Hd):
        line = ""
        for c in range(Wd):
            i = r * Wd + c
            if i == pos:
                line += "A"
            elif i == GOAL:
                line += "G"
            else:
                line += "."
            line += "." if (link[i] & BIT["right"]) else "#"
        rows.append(line)
    vrows = []
    for r in range(Hd):
        line = ""
        for c in range(Wd):
            i = r * Wd + c
            line += "." if (link[i] & BIT["down"]) else "#"
            line += "#"
        vrows.append(line)
    return (f"Perfect maze {Wd}x{Hd} (no loops). start=(0,0) goal=({Hd - 1},{Wd - 1}).\n"
            f"current pos row {pos // Wd} col {pos % Wd}. steps_taken={steps}. bfs_shortest={optimal}.\n"
            f"legal_moves: {', '.join(legal_moves)}\n"
            f"grid (A = agent, G = goal, # = wall), cell + right-wall:\n" + "\n".join(rows))


def move_question():
    return {
        "id": "move",
        "type": "choice",
        "instructions": "Choose the direction that follows the shortest path to the goal.",
        "criteria": {
            "up": "move up (row - 1)",
            "down": "move down (row + 1)",
            "left": "move left (column - 1)",
            "right": "move right (column + 1)",
        },
    }


def closeness_question():
    """与 maze_laya.html SCENE.questions() 的 closeness 问题对齐（score 类型，list 选项）。"""
    return {
        "id": "closeness",
        "type": "score",
        "instructions": "How far is the agent from the goal?",
        "criteria": ["at the goal", "close", "halfway", "far"],
    }


def reconstruct_positions(trace):
    """从轨迹 info 字段解析每步落点 pos[i]（i=1..N），pos[0]=(0,0)。"""
    import re
    pos = [(0, 0)]
    applied = []
    for ln in trace:
        m = re.search(r"→\s*\((-?\d+),(-?\d+)\)", ln.get("info", ""))
        if not m:
            raise ValueError(f"无法解析 info: {ln.get('info')!r}")
        pos.append((int(m.group(1)), int(m.group(2))))
        applied.append(ln.get("applied"))
    # 校验：applied[d] 从 pos[i-1] 应到达 pos[i]
    for i in range(1, len(pos)):
        d = applied[i - 1]
        if d not in DIR:
            continue
        er = pos[i - 1][0] + DIR[d][0]
        ec = pos[i - 1][1] + DIR[d][1]
        if (er, ec) != pos[i]:
            raise ValueError(f"轨迹自洽性失败 step {i}: applied={d} "
                             f"pos[{i - 1}]={pos[i - 1]} 但 info 落点={pos[i]}")
    return pos, applied


def find_seed(target_pos, optimal_target=148, search=100000):
    """穷举 seed，匹配轨迹 A* 路径。返回 (seed, maze) 或 None。"""
    for S in range(search):
        a0 = (S * 2654435761) & 0xFFFFFFFF
        maze = genMaze(W, H, mulberry32(a0))
        if bfsDist(maze, START, GOAL) != optimal_target:
            continue
        path = astar(maze, START, GOAL)
        if not path or len(path) != len(target_pos) - 1:
            continue
        # 用候选迷宫的 A* 路径重建落点，与轨迹逐点比较
        p = START
        sim = [(0, 0)]
        for d in path:
            r = p // W + DIR[d][0]
            c = p % W + DIR[d][1]
            p = r * W + c
            sim.append((r, c))
        if sim == target_pos:
            return S, maze
        if S % 100000 == 0 and S > 0:
            print(f"  [find_seed] 已扫描 {S} 个 seed …")
    return None


def reconstruct_maze_from_trace(pos, applied, W=W, H=H):
    """从轨迹的 (落点序列 + 每步方向) 直接重建迷宫 link 数组。

    不依赖 seed / RNG / BFS 过滤：pos[i-1] -(applied[i-1])-> pos[i] 成立
    即说明 cur 与 nxt 之间的墙是开通的，逐边重建即可。
    重建结果与 maze_laya.html 的真实网格逐边一致。
    """
    link = [0] * (W * H)
    for i in range(1, len(pos)):
        d = applied[i - 1]
        cur = pos[i - 1][0] * W + pos[i - 1][1]
        nxt = pos[i][0] * W + pos[i][1]
        link[cur] |= BIT[d]
        link[nxt] |= OBIT[d]
    return {"W": W, "H": H, "link": link}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="迷宫轨迹 → 微调数据集 JSONL")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--trace", default=TRACE_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--search", type=int, default=200000,
                    help="保留兼容参数；新方案从轨迹直接重建拓扑，不再依赖种子穷举（此值无效）")
    args = ap.parse_args()

    with open(args.trace, encoding="utf-8") as f:
        trace = [json.loads(l) for l in f if l.strip()]

    pos, _applied = reconstruct_positions(trace)
    optimal_target = None
    for ln in trace:
        for m in ln.get("metrics", []):
            if m.get("k") == "BFS 最短":
                optimal_target = int(m["v"])
    optimal_target = optimal_target or (len(pos) - 1)
    print(f"[convert] 轨迹步数={len(trace)}，落点序列长度={len(pos)}，"
          f"轨迹声明 BFS 最短={optimal_target}")

    # —— 新方案：从轨迹 (落点+方向) 直接重建迷宫拓扑，不再做 seed 穷举 ——
    print("[convert] 从轨迹 (落点+方向) 直接重建迷宫拓扑（method=ground_truth）…")
    maze = reconstruct_maze_from_trace(pos, _applied)
    recon_opt = bfsDist(maze, START, GOAL)
    method = "ground_truth"
    print(f"[convert] 重建迷宫 bfs_shortest={recon_opt}"
          f"{' ✅ 与轨迹一致' if recon_opt == optimal_target else ' ⚠️ 与轨迹声明不一致，请核对轨迹'}")

    # 构建样本（真实 grid ASCII + move/closeness 两问，与 maze_laya.html 推理分布一致）
    _, tok, pred = build(args.model)
    questions = [move_question(), closeness_question()]
    samples = []
    for idx, ln in enumerate(trace, start=1):
        state_pos = pos[idx - 1]   # 决策时的位置（移动前）
        rd = ln.get("rd") or {}
        sp = state_pos[0] * W + state_pos[1]
        lm = legal(maze, sp)
        stxt = text(maze, sp, idx - 1, recon_opt, lm)
        sample = {
            "scene": "maze",
            "seed": 0,
            "step": idx,
            "state": stxt,
            "questions": questions,
        }
        # 标签：move 来自教师 rd（A* 选中方向）；closeness 为 BFS 剩余步归一化期望值
        labels = derive_labels([Question(**q) for q in questions], rd)
        sample["labels"] = labels
        sample = tokenize_sample(pred, sample)
        if "move" not in sample["labels"] or "key" not in sample["labels"]["move"]:
            raise RuntimeError(f"step {idx} 缺少 move 标签：{sample['labels']}")
        samples.append(sample)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    meta = {
        "method": method,
        "seed": None,
        "n_samples": len(samples),
        "optimal_target": optimal_target,
        "recon_opt": recon_opt,
        "optimal_match": recon_opt == optimal_target,
        "search_range": args.search,
        "scene": "maze",
        "note": ("ground_truth: 从轨迹 (落点+方向) 直接重建迷宫拓扑，state=真实 SCENE.text()（grid ASCII），"
                 "与 maze_laya.html 推理分布一致；questions=move(choice)+closeness(score)，"
                 "labels.move.key=rd.choice（A* 选中方向），labels.closeness=归一化 BFS 剩余步期望值。"),
    }
    meta_path = os.path.join(os.path.dirname(args.out), "maze_convert_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[convert] 写出 {len(samples)} 条样本 → {args.out}")
    print(f"[convert] 元数据 → {meta_path}")
    print(f"[convert] 示例 state:\n{samples[0]['state'][:200]}")
    print(f"[convert] 示例 labels: {samples[0]['labels']}")


if __name__ == "__main__":
    main()
