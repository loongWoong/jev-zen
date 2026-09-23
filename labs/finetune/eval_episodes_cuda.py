import os, sys, json, time, argparse
import numpy as np

ROOT = "C:/Users/Administrator/Downloads/jev-zen"
FT = os.path.join(ROOT, "labs", "finetune")
sys.path.insert(0, ROOT)
sys.path.insert(0, FT)

from laya_verify_cuda import build
from laya_verify import Question
from convert_maze import (genMaze, mulberry32, legal as maze_legal,
                          bfsDist, text as maze_text, BIT, START, GOAL, W, H)

MOVE_ORDER = ["up", "down", "left", "right"]
MOVE_DIR = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _maze_step(maze, pos, d):
    r = pos // W + MOVE_DIR[d][0]
    c = pos % W + MOVE_DIR[d][1]
    return r * W + c


def _is_leaf(maze, pos):
    # 完美迷宫里开墙数 == 1 的格子就是死胡同（只能原路返回）
    cnt = 0
    v = maze["link"][pos]
    for b in (1, 2, 4, 8):
        if v & b:
            cnt += 1
    return cnt == 1


def eval_episodes(model_path, n_mazes=3, seed_base=1000, max_steps=120):
    st, tok, pred, fused, info = build(model_path)
    solved, steps_list, deadends = [], [], []
    first_wrong_list, leaf_list, revisit_list = [], [], []
    spread_sum, spread_n = 0.0, 0
    for k in range(n_mazes):
        maze = genMaze(W, H, mulberry32((seed_base + k) * 2654435761 & 0xFFFFFFFF))
        pos = START
        optimal = bfsDist(maze, START, GOAL)
        steps = 0
        fw = None
        fl = None
        fr = None
        visited = set([pos])
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
            spread_sum += float(np.max(mv) - np.min(mv)); spread_n += 1
            best = min(lm, key=lambda d: bfsDist(maze, _maze_step(maze, pos, d), GOAL))
            if fw is None and choice in lm and choice != best:
                fw = steps
            applied = choice if choice in lm else best
            nxt = _maze_step(maze, pos, applied)
            if fl is None and _is_leaf(maze, nxt):
                fl = steps + 1
            if fr is None and nxt in visited:
                fr = steps + 1
            visited.add(nxt)
            pos = nxt
            steps += 1
        solved.append(1 if pos == GOAL else 0)
        steps_list.append(steps)
        deadends.append(1 if dead else 0)
        first_wrong_list.append(fw if fw is not None else steps)
        leaf_list.append(fl if fl is not None else steps)
        revisit_list.append(fr if fr is not None else steps)
        print(f"  [{os.path.basename(model_path)}] maze {k}: "
              f"{'SOLVED' if pos==GOAL else ('DEAD' if dead else 'TIMEOUT')} "
              f"steps={steps} first_wrong={fw} first_leaf={fl} first_revisit={fr}",
              file=sys.stderr, flush=True)
    n = len(solved)
    return {
        "model": model_path,
        "n": n,
        "optimal_ref": optimal,
        "solved_rate": round(sum(solved) / n, 3),
        "avg_steps": round(sum(steps_list) / n, 1),
        "deadend_rate": round(sum(deadends) / n, 3),
        "avg_first_wrong_depth": round(sum(first_wrong_list) / n, 1),
        "avg_first_leaf_depth": round(sum(leaf_list) / n, 1),
        "avg_first_revisit_depth": round(sum(revisit_list) / n, 1),
        "mean_move_logit_spread": round(spread_sum / spread_n, 3) if spread_n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--outdir", default=os.path.join(FT, "artifacts"))
    args = ap.parse_args()
    out = os.path.join(args.outdir, f"episodes_cuda_{args.tag}.json")
    t0 = time.time()
    res = eval_episodes(args.model, n_mazes=args.n,
                        seed_base=args.seed_base, max_steps=args.max_steps)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
