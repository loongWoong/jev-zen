/* =============================================================================
 * 场景：迷宫 Maze
 * 生成：recursive backtracker 生成「完美迷宫」（无环，任意两点间路径唯一），
 *       这样 BFS 的最短步数就是唯一基准，可以直接验证 A* / DFS 是否最优
 * 算法：BFS / A*（曼哈顿）/ DFS（右手法则式深搜，作为非最优对照）
 * 目标：实际步数 == BFS 最短路（仅 BFS / A* 应达成）
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const DIR = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
const KEYS = ['up', 'down', 'left', 'right'];
const DIRCN = { up: '上', down: '下', left: '左', right: '右' };

/* 完美迷宫：cells 为 H×W 的通行格，walls 记录相邻格之间是否打通 */
function genMaze(W, H, rng) {
  const visited = new Array(W * H).fill(false);
  const link = new Array(W * H).fill(0);   // 位标记：1=up 2=down 4=left 8=right
  const stack = [0];
  visited[0] = true;
  while (stack.length) {
    const cur = stack[stack.length - 1];
    const r = Math.floor(cur / W), c = cur % W;
    const cand = [];
    if (r > 0 && !visited[cur - W]) cand.push(['up', cur - W]);
    if (r < H - 1 && !visited[cur + W]) cand.push(['down', cur + W]);
    if (c > 0 && !visited[cur - 1]) cand.push(['left', cur - 1]);
    if (c < W - 1 && !visited[cur + 1]) cand.push(['right', cur + 1]);
    if (!cand.length) { stack.pop(); continue; }
    const [d, nxt] = cand[Math.floor(rng() * cand.length)];
    const bit = { up: 1, down: 2, left: 4, right: 8 }[d];
    const obit = { up: 2, down: 1, left: 8, right: 4 }[d];
    link[cur] |= bit; link[nxt] |= obit;
    visited[nxt] = true;
    stack.push(nxt);
  }
  return { W, H, link };
}

const BIT = { up: 1, down: 2, left: 4, right: 8 };

function bfsDist(maze, from, to) {
  const { W, H, link } = maze;
  const dist = new Array(W * H).fill(-1);
  dist[from] = 0;
  const q = [from];
  while (q.length) {
    const cur = q.shift();
    if (cur === to) return dist[cur];
    for (const d of KEYS) {
      if (!(link[cur] & BIT[d])) continue;
      const r = Math.floor(cur / W) + DIR[d][0], c = cur % W + DIR[d][1];
      if (r < 0 || c < 0 || r >= H || c >= W) continue;
      const ni = r * W + c;
      if (dist[ni] >= 0) continue;
      dist[ni] = dist[cur] + 1;
      q.push(ni);
    }
  }
  return -1;
}

/* A*：曼哈顿启发式，与 BFS 同样应给出最短路 */
function astar(maze, from, to) {
  const { W, H, link } = maze;
  const h = i => Math.abs(Math.floor(i / W) - Math.floor(to / W)) + Math.abs(i % W - to % W);
  const g = new Map([[from, 0]]);
  const prev = new Map();
  const open = [{ i: from, f: h(from) }];
  while (open.length) {
    let bi = 0;
    for (let k = 1; k < open.length; k++) if (open[k].f < open[bi].f) bi = k;
    const cur = open.splice(bi, 1)[0];
    if (cur.i === to) {
      const path = [];
      let x = to;
      while (prev.has(x)) { const p = prev.get(x); path.push(p.d); x = p.i; }
      return path.reverse();
    }
    for (const d of KEYS) {
      if (!(link[cur.i] & BIT[d])) continue;
      const r = Math.floor(cur.i / W) + DIR[d][0], c = cur.i % W + DIR[d][1];
      if (r < 0 || c < 0 || r >= H || c >= W) continue;
      const ni = r * W + c;
      const ng = (g.get(cur.i) || 0) + 1;
      if (g.has(ni) && g.get(ni) <= ng) continue;
      g.set(ni, ng);
      prev.set(ni, { i: cur.i, d });
      open.push({ i: ni, f: ng + h(ni) });
    }
  }
  return null;
}

/* DFS：找到一条路即可，不保证最短（对照用） */
function dfsPath(maze, from, to) {
  const { W, H, link } = maze;
  const seen = new Set();
  const path = [];
  const go = (i) => {
    if (i === to) return true;
    seen.add(i);
    for (const d of KEYS) {
      if (!(link[i] & BIT[d])) continue;
      const r = Math.floor(i / W) + DIR[d][0], c = i % W + DIR[d][1];
      if (r < 0 || c < 0 || r >= H || c >= W) continue;
      const ni = r * W + c;
      if (seen.has(ni)) continue;
      path.push(d);
      if (go(ni)) return true;
      path.pop();
    }
    return false;
  };
  return go(from) ? path : null;
}

const SCENE = {
  id: 'maze',
  name: '迷宫',
  sub: '完美迷宫 · BFS / A* / DFS 三种寻路对照',
  goal: '目标：实际步数 == BFS 最短路',
  hint: '迷宫由 recursive backtracker 生成，是无环的完美迷宫 —— 两点间路径唯一，'
      + '所以 BFS 的步数就是唯一基准：BFS 与 A* 都应等于它，DFS 会绕远（对照）。',

  controls: [
    { id: 'algo', label: '寻路算法', type: 'select', value: 'bfs', options: [
      { v: 'bfs', t: 'BFS（最短路）' }, { v: 'astar', t: 'A*（曼哈顿，最短路）' },
      { v: 'dfs', t: 'DFS（不保证最短，对照）' }] },
    { id: 'size', label: '迷宫规模', type: 'select', value: '10', options: [
      { v: '6', t: '6×6' }, { v: '10', t: '10×10' }, { v: '16', t: '16×16' }] }
  ],

  init(seed, sopt) {
    const W = parseInt(sopt.size || '10', 10), H = W;
    const rng = mulberry32((seed * 2654435761) >>> 0);
    const maze = genMaze(W, H, rng);
    const start = 0, goal = W * H - 1;
    const s = { maze, start, goal, pos: start, steps: 0, rng,
                algo: sopt.algo || 'bfs', optimal: bfsDist(maze, start, goal) };
    return s;
  },

  legal(s) {
    const { W, H, link } = s.maze;
    const out = [];
    for (const d of KEYS) {
      if (!(link[s.pos] & BIT[d])) continue;
      const r = Math.floor(s.pos / W) + DIR[d][0], c = s.pos % W + DIR[d][1];
      if (r < 0 || c < 0 || r >= H || c >= W) continue;
      out.push(d);
    }
    return out;
  },

  step(s, d) {
    const { W } = s.maze;
    const r = Math.floor(s.pos / W) + DIR[d][0], c = s.pos % W + DIR[d][1];
    s.pos = r * W + c;
    s.steps++;
    return { info: `${DIRCN[d]} → (${r},${c}) · 已走 ${s.steps}/${s.optimal}（BFS 最短 ${s.optimal}）` };
  },

  text(s, legal) {
    const { W, H, link } = s.maze;
    const rows = [];
    for (let r = 0; r < H; r++) {
      let line = '';
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        if (i === s.pos) line += 'A';
        else if (i === s.goal) line += 'G';
        else line += '.';
        line += (link[i] & BIT.right) ? '.' : '#';   // 右侧是否通
      }
      rows.push(line);
    }
    const vrows = [];
    for (let r = 0; r < H; r++) {
      let line = '';
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        line += (link[i] & BIT.down) ? '.' : '#';
        line += '#';
      }
      vrows.push(line);
    }
    return `Perfect maze ${W}x${H} (no loops). start=(0,0) goal=(${H - 1},${W - 1}).\n`
      + `current pos row ${Math.floor(s.pos / W)} col ${s.pos % W}. steps_taken=${s.steps}. `
      + `bfs_shortest=${s.optimal}.\n`
      + `legal_moves: ${legal.join(', ')}\n`
      + `grid (A = agent, G = goal, # = wall), cell + right-wall:\n` + rows.join('\n');
  },

  questions(s, legal) {
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the direction that follows the shortest path to the goal.',
        criteria: {
          up: 'move up (row - 1)', down: 'move down (row + 1)',
          left: 'move left (column - 1)', right: 'move right (column + 1)'
        } },
      { id: 'closeness', type: 'score',
        instructions: 'How far is the agent from the goal?',
        criteria: ['at the goal', 'close', 'halfway', 'far'] }
    ];
  },

  rule(s, legal, sopt) {
    const algo = sopt.algo || 'bfs';
    let next = null, note = '';
    if (algo === 'dfs') {
      const p = dfsPath(s.maze, s.pos, s.goal);
      next = p && p.length ? p[0] : null;
      note = `DFS：找到一条路就走（剩余 ${p ? p.length : '?'} 步），不保证最短。`;
    } else if (algo === 'astar') {
      const p = astar(s.maze, s.pos, s.goal);
      next = p && p.length ? p[0] : null;
      note = `A*（曼哈顿启发）：剩余 ${p ? p.length : '?'} 步，应等于 BFS 最短。`;
    } else {
      // BFS：选「到终点 BFS 距离最小」的邻居
      let bd = Infinity;
      for (const d of legal) {
        const { W } = s.maze;
        const r = Math.floor(s.pos / W) + DIR[d][0], c = s.pos % W + DIR[d][1];
        const dd = bfsDist(s.maze, r * W + c, s.goal);
        if (dd >= 0 && dd < bd) { bd = dd; next = d; }
      }
      note = `BFS：选到终点距离最小的方向（剩余 ${bd} 步）。`;
    }
    if (!next || legal.indexOf(next) < 0) next = legal[0];
    const ranks = legal.map(d => {
      const { W } = s.maze;
      const r = Math.floor(s.pos / W) + DIR[d][0], c = s.pos % W + DIR[d][1];
      const dd = bfsDist(s.maze, r * W + c, s.goal);
      return { key: d, label: DIRCN[d], score: dd >= 0 ? -dd : -999,
               detail: `走后到终点还剩 ${dd < 0 ? '∞' : dd} 步`, risky: dd < 0 };
    });
    ranks.sort((a, b) => b.score - a.score);
    return { choice: next, ranks, note };
  },

  metrics(s, sopt) {
    const { W } = s.maze;
    const { W: _w } = s.maze;
    const remain = bfsDist(s.maze, s.pos, s.goal);
    return [
      { k: '已走', v: s.steps },
      { k: 'BFS 最短', v: s.optimal },
      { k: '剩余', v: remain < 0 ? '—' : remain },
      { k: '步数/最短', v: `${(s.steps / Math.max(1, s.optimal)).toFixed(2)}×`,
        tone: s.steps <= s.optimal ? 'ok' : 'warn' },
      { k: '算法', v: ({ bfs: 'BFS', astar: 'A*', dfs: 'DFS' })[sopt.algo || 'bfs'] },
      { k: '状态', v: s.pos === s.goal ? '到达' : '寻路中',
        tone: s.pos === s.goal ? 'ok' : '' }
    ];
  },

  render(s, el) {
    const { W, H, link } = s.maze;
    // 用「格 + 右墙 + 下墙」的网格渲染，边长 2W-1
    const n = 2 * W - 1;
    let h = `<div style="display:grid;grid-template-columns:repeat(${n},1fr);gap:1px;`
          + `background:#cdd5df;padding:3px;border-radius:6px;max-width:340px;margin:0 auto">`;
    const cellBg = i => (i === s.pos ? '#2f6fed' : (i === s.goal ? '#0d9488' : '#fff'));
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        h += `<div style="background:${cellBg(i)};aspect-ratio:1/1;border-radius:1px"></div>`;
        if (c < W - 1) {
          h += `<div style="background:${(link[i] & BIT.right) ? '#fff' : '#5b6b7c'};aspect-ratio:1/1"></div>`;
        }
      }
      if (r < H - 1) {
        for (let c = 0; c < W; c++) {
          const i = r * W + c;
          h += `<div style="background:${(link[i] & BIT.down) ? '#fff' : '#5b6b7c'};aspect-ratio:1/1"></div>`;
          if (c < W - 1) h += `<div style="background:#5b6b7c;aspect-ratio:1/1"></div>`;
        }
      }
    }
    h += `</div><div class="hint" style="text-align:center">蓝 = 当前位置 · 绿 = 终点 · 深灰 = 墙</div>`;
    el.innerHTML = h;
  },

  done(s) { return s.pos === s.goal; },
  goalOk(s, sopt) {
    const algo = sopt.algo || 'bfs';
    return s.pos === s.goal && (algo === 'dfs' ? true : s.steps === s.optimal);
  },
  maxSteps(sopt) { const W = parseInt(sopt.size || '10', 10); return W * W * 4; }
};
