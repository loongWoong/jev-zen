#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · 评测台（benchmark.py）

对齐门（前置，必须 PASS 否则下游指标不可信）：Verifier.channel_probe()
指标：
  - 对齐状态 / logit 极差（通道增益）
  - 延迟：Verifier.benchmark（1/5/10 问题，ms/问题）
  - 标签准确率：测试集 top-1 vs rule 教师标签（微调目标代理）
  - 任务迁移 / 泛化：见 FRAMEWORK.md §5（需接 labs 场景 rollout，本脚手架提供 label-acc 代理）

输出：labs/finetune/artifacts/benchmark_<tag>.json
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 工作树根，含 laya_verify.py
sys.path.insert(0, ROOT)

from laya_verify import build, Verifier, Question  # noqa: E402

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")

# 迷宫 move 选项顺序（与 maze_laya.html SCENE.questions 的 criteria 一致）
MOVE_ORDER = ["up", "down", "left", "right"]
MOVE_DIR = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _maze_step(maze, pos, d):
    """与 maze_laya.html SCENE.step 同语义：返回移动后的 1D 单元索引。"""
    W = maze["W"]
    r = pos // W + MOVE_DIR[d][0]
    c = pos % W + MOVE_DIR[d][1]
    return r * W + c


def eval_episodes(model_path, n_mazes=12, seed_base=1000, max_steps=None):
    """真实 grid 整局对弈评测（Model-only 模式，与 maze_laya.html decide() 一致）。

    直接回答用户关心的「卡关更早/更晚」：测可解率、平均步数、死胡同率、首次错步深度。
    模型选择若非法则回退 BFS 最优（与 maze_laya.html 的 model 分支一致）。
    """
    from convert_maze import (genMaze, mulberry32, legal as maze_legal,
                              bfsDist, text as maze_text, START, GOAL, W, H)
    st, tok, pred = build(model_path)
    solved, steps_list, deadends, first_wrong_list = [], [], [], []
    for k in range(n_mazes):
        maze = genMaze(W, H, mulberry32((seed_base + k) * 2654435761 & 0xFFFFFFFF))
        pos = START
        optimal = bfsDist(maze, START, GOAL)
        steps = 0
        max_steps = (W * W * 4) if max_steps is None else max_steps
        fw = None
        dead = False
        while pos != GOAL and steps < max_steps:
            lm = maze_legal(maze, pos)
            if not lm:
                dead = True
                break
            stxt = maze_text(maze, pos, steps, optimal, lm)
            qs = [Question("move", "choice",
                           "Choose the direction that follows the shortest path to the goal.",
                           {"up": "move up (row - 1)", "down": "move down (row + 1)",
                            "left": "move left (column - 1)", "right": "move right (column + 1)"})]
            built = pred.build(stxt, qs)
            out = pred.model.forward(built["ids"], built["marker_pos"], built["marker_type"])
            mv = out["logits"][:4]
            choice = MOVE_ORDER[int(np.argmax(mv))]
            # 教师最优（BFS 距离最小的方向）
            best = min(lm, key=lambda d: bfsDist(maze, _maze_step(maze, pos, d), GOAL))
            if fw is None and choice in lm and choice != best:
                fw = steps
            applied = choice if choice in lm else best   # 非法则回退最优（同 maze_laya.html）
            pos = _maze_step(maze, pos, applied)
            steps += 1
        solved.append(1 if pos == GOAL else 0)
        steps_list.append(steps)
        deadends.append(1 if dead else 0)
        first_wrong_list.append(fw if fw is not None else steps)
    n = len(solved)
    return {
        "n": n,
        "solved_rate": round(sum(solved) / n, 3),
        "avg_steps": round(sum(steps_list) / n, 1),
        "deadend_rate": round(sum(deadends) / n, 3),
        "avg_first_wrong_depth": round(sum(first_wrong_list) / n, 1),
    }


def eval_label_accuracy(pred, jsonl_path, limit=200):
    """测试集 top-1 选项 vs rule 教师标签。读 tokenized 样本（ids/marker_pos/marker_type）。"""
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return None
    n = 0
    hit_choice = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if limit and n >= limit:
                break
            smp = json.loads(line)
            ids = smp.get("ids")
            if ids is None:
                continue
            qs = [Question(**q) for q in smp["questions"]]
            out = pred.model.forward(ids, smp["marker_pos"], smp["marker_type"])
            logits = out["logits"]
            cursor = 0
            for q in qs:
                n_opt = len(q.options())
                lg = logits[cursor:cursor + n_opt]
                cursor += n_opt
                gold = (smp.get("labels") or {}).get(q.id, {}).get("key")
                if gold is None:
                    continue
                pred_key = q.options()[int(__import__("numpy").argmax(lg))][0]
                n += 1
                if pred_key == gold:
                    hit_choice += 1
    return {"n": n, "choice_top1_acc": (hit_choice / n) if n else None}


def run(model_path, dataset=None, tag="before"):
    st, tok, pred = build(model_path)
    ver = Verifier(st, tok, pred, model_path)

    report = {"tag": tag, "model": model_path, "metrics": {}}

    # 1) 对齐门
    ch = ver.channel_probe()
    report["metrics"]["channel"] = {
        "status": ch["status"], "aligned": ch["aligned"],
        "hits": ch["hits"], "n": ch["n"],
        "max_logit_spread": ch["max_logit_spread"],
        "mean_max_prob": ch["mean_max_prob"],
    }

    # 2) 延迟
    try:
        bm = ver.benchmark()
        report["metrics"]["latency"] = bm
    except Exception as e:  # noqa: BLE001
        report["metrics"]["latency"] = {"error": str(e)}

    # 3) 标签准确率（若有测试集）
    if dataset:
        la = eval_label_accuracy(pred, dataset)
        if la:
            report["metrics"]["label_accuracy"] = la

    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    out = os.path.join(HERE, "artifacts", f"benchmark_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"=== benchmark[{tag}] ===")
    print(f"对齐: {ch['status']} (aligned={ch['aligned']}) hits={ch['hits']}/{ch['n']} "
          f"spread={ch['max_logit_spread']:.2e}")
    if dataset and report["metrics"].get("label_accuracy"):
        acc = report["metrics"]["label_accuracy"]["choice_top1_acc"]
        print(f"标签准确率(choice top1): {acc:.3f}  (n={report['metrics']['label_accuracy']['n']})")
    print(f"报告: {out}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Laya 评测台")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default=os.path.join(HERE, "artifacts", "samples.jsonl"))
    ap.add_argument("--tag", default="before")
    ap.add_argument("--episodes", type=int, default=0,
                    help="真实 grid 整局对弈评测的迷宫数（0=关闭，改用频道门+标签准确率）")
    ap.add_argument("--max-steps", type=int, default=80,
                    help="每局最大步数上限（控制评测耗时；不影响跨模型公平性）")
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()
    if args.episodes and args.episodes > 0:
        rep = eval_episodes(args.model, n_mazes=args.episodes,
                            seed_base=args.seed_base, max_steps=args.max_steps)
        out = os.path.join(HERE, "artifacts", f"episodes_{args.tag}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "episodes": rep}, f, ensure_ascii=False, indent=2)
        print(f"=== 真实 grid 整局评测 [{args.tag}] （n={rep['n']}）===")
        print(f"可解率 solved_rate      : {rep['solved_rate']}")
        print(f"平均步数 avg_steps      : {rep['avg_steps']}")
        print(f"死胡同率 deadend_rate   : {rep['deadend_rate']}")
        print(f"首次错步深度 first_wrong: {rep['avg_first_wrong_depth']}")
        print(f"报告: {out}")
        return
    run(args.model, args.dataset if os.path.isfile(args.dataset) else None, args.tag)


if __name__ == "__main__":
    main()
