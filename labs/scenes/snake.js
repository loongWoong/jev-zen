/* =============================================================================
 * 场景：贪吃蛇 Snake
 * 算法：BFS 寻食 + 洪水填充安全检查（safe） / 哈密顿回路（hamilton）
 *      / 回路 + 安全短路（hybrid）
 * 目标：10×10 棋盘上长度 ≥ 50；hamilton 模式可填满 100 格
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const DIRS = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
const OPP = { up: 'down', down: 'up', left: 'right', right: 'left' };
const DIRKEYS = ['up', 'down', 'left', 'right'];
const DIRCN = { up: '上', down: '下', left: '左', right: '右' };

/* 偶数边长方格棋盘上的哈密顿回路：
 *   第 0 列自上而下 → 其余列蛇形回扫，终点 (0,1) 与起点 (0,0) 相邻 */
function hamiltonCycle(N) {
  const cyc = [];
  for (let r = 0; r < N; r++) cyc.push([r, 0]);
  for (let r = N - 1; r >= 0; r--) {
    if ((N - 1 - r) % 2 === 0) { for (let c = 1; c < N; c++) cyc.push([r, c]); }
    else { for (let c = N - 1; c >= 1; c--) cyc.push([r, c]); }
  }
  return cyc;
}

/* safe 策略权重：坐标下降扫出、再由 48 局全新 seed 终裁
 *   A free60/hug5/fd25  60.7±3.2  →  B free60/hug25/fd15 64.5±3.3（采用）
 *   D free120/hug5/fd25 51.7±3.1  →  C 初始默认          29.7±1.9
 * 教训：12 局的扫描结论在样本外会掉一半（66 → 37），必须 ≥ 48 局才能定案。 */
const W_SAFE = { gate: 5000, free: 60, hug: 25, eat: 400, fd: 15, td: 0, spaceGate: 2000 };

/* 从 start 出发在 occ 之外的空格上做 BFS，返回可达空格数 */
function floodFree(N, start, occ) {
  const seen = new Set([start]);
  const q = [start]; let n = 0;
  while (q.length) {
    const cur = q.pop();
    const r = Math.floor(cur / N), c = cur % N;
    for (const k of DIRKEYS) {
      const nr = r + DIRS[k][0], nc = c + DIRS[k][1];
      if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
      const id = nr * N + nc;
      if (seen.has(id) || occ.has(id)) continue;
      seen.add(id); q.push(id); n++;
    }
  }
  return n;
}

/* BFS 最短距离，不可达返回 -1 */
function bfsDist(N, from, to, occ) {
  if (from === to) return 0;
  const seen = new Set([from]);
  let layer = [from], d = 0;
  while (layer.length) {
    d++; const nx = [];
    for (const cur of layer) {
      const r = Math.floor(cur / N), c = cur % N;
      for (const k of DIRKEYS) {
        const nr = r + DIRS[k][0], nc = c + DIRS[k][1];
        if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
        const id = nr * N + nc;
        if (seen.has(id) || occ.has(id)) continue;
        if (id === to) return d;
        seen.add(id); nx.push(id);
      }
    }
    layer = nx;
  }
  return -1;
}

const SCENE = {
  id: 'snake',
  name: '贪吃蛇',
  sub: '10×10 · BFS 寻食 + 洪水填充安全检查 / 哈密顿回路',
  goal: '目标：长度 ≥ 50（hamilton 模式可填满 100 格）',
  hint: 'safe = BFS 找食物 + 洪水填充判断「走完还能不能回头」；hamilton = 沿预生成哈密顿回路行进，'
      + '理论上不会自锁、可填满全盘。模型只做对照留痕。',

  controls: [
    { id: 'policy', label: '算法策略', type: 'select', value: 'safe', options: [
      { v: 'safe', t: 'safe · BFS + 洪水安全检查' },
      { v: 'hamilton', t: 'hamilton · 哈密顿回路' },
      { v: 'hybrid', t: 'hybrid · 回路 + 安全短路' }] },
    { id: 'size', label: '棋盘边长', type: 'select', value: '10', options: [
      { v: '6', t: '6×6' }, { v: '8', t: '8×8' }, { v: '10', t: '10×10' }, { v: '12', t: '12×12' }] },
    { id: 'target', label: '目标长度', type: 'number', value: 50, min: 5, step: 5 }
  ],

  init(seed, sopt) {
    const N = parseInt(sopt.size || '10', 10);
    if (N % 2 !== 0) throw new Error('哈密顿回路要求偶数边长');
    const rng = mulberry32((seed * 2654435761) >>> 0);
    const cyc = hamiltonCycle(N);
    const cycIdx = new Int32Array(N * N);
    cyc.forEach((p, i) => { cycIdx[p[0] * N + p[1]] = i; });
    const useCyc = sopt.policy === 'hamilton' || sopt.policy === 'hybrid';
    const snake = useCyc
      ? [cyc[2].slice(), cyc[1].slice(), cyc[0].slice()]
      : (() => { const r = N >> 1, c = N >> 1; return [[r, c], [r, c - 1], [r, c - 2]]; })();
    const s = { N, snake, cyc, cycIdx, food: null, dir: 'right',
                alive: true, steps: 0, eaten: 0, rng };
    this._spawn(s);
    return s;
  },

  _spawn(s) {
    const N = s.N;
    const occ = new Set(s.snake.map(p => p[0] * N + p[1]));
    const free = [];
    for (let i = 0; i < N * N; i++) if (!occ.has(i)) free.push(i);
    if (!free.length) { s.alive = false; return; }
    const id = free[Math.floor(s.rng() * free.length)];
    s.food = [Math.floor(id / N), id % N];
  },

  legal(s) {
    const N = s.N, [hr, hc] = s.snake[0];
    const res = [];
    for (const k of DIRKEYS) {
      if (s.snake.length > 1 && k === OPP[s.dir]) continue;
      const nr = hr + DIRS[k][0], nc = hc + DIRS[k][1];
      if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
      const eat = nr === s.food[0] && nc === s.food[1];
      const body = eat ? s.snake : s.snake.slice(0, -1);
      if (body.some(p => p[0] === nr && p[1] === nc)) continue;
      res.push(k);
    }
    return res;
  },

  step(s, key) {
    const N = s.N, [hr, hc] = s.snake[0], d = DIRS[key];
    const nr = hr + d[0], nc = hc + d[1];
    s.dir = key; s.steps++;
    const eat = nr === s.food[0] && nc === s.food[1];
    const body = eat ? s.snake : s.snake.slice(0, -1);
    if (body.some(p => p[0] === nr && p[1] === nc)) {
      s.alive = false; return { info: `撞到自己 (${nr},${nc})` };
    }
    s.snake.unshift([nr, nc]);
    if (!eat) s.snake.pop();
    else { s.eaten++; this._spawn(s); }
    if (s.snake.length >= N * N) s.alive = false;   // 填满即终局
    return { info: `${DIRCN[key]} → (${nr},${nc})${eat ? ' · 吃到食物' : ''} · 长度 ${s.snake.length}` };
  },

  text(s, legal) {
    const N = s.N;
    const occ = new Map();
    s.snake.forEach((p, i) => occ.set(p[0] * N + p[1], i === 0 ? 'H' : 's'));
    const lines = [];
    for (let r = 0; r < N; r++) {
      let line = '';
      for (let c = 0; c < N; c++) {
        const id = r * N + c;
        if (s.food[0] === r && s.food[1] === c) line += 'F';
        else line += occ.get(id) || '.';
      }
      lines.push(line);
    }
    const [hr, hc] = s.snake[0];
    return `Snake on a ${N}x${N} grid. step=${s.steps} length=${s.snake.length} eaten=${s.eaten}.\n`
      + `head=(${hr},${hc}) food=(${s.food[0]},${s.food[1]}) tail=(${s.snake[s.snake.length - 1][0]},${s.snake[s.snake.length - 1][1]}).\n`
      + `legal_moves: ${legal.join(', ')}.\n`
      + `grid (. empty, H head, s body, F food), rows top to bottom:\n`
      + lines.join('\n');
  },

  questions(s, legal) {
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the next move for the snake: survive first, then grow.',
        criteria: {
          up: 'move one cell up (row - 1)',
          down: 'move one cell down (row + 1)',
          left: 'move one cell left (column - 1)',
          right: 'move one cell right (column + 1)'
        } },
      { id: 'room', type: 'score',
        instructions: 'How much free room does the snake have after this step?',
        criteria: ['no room, about to die', 'very tight', 'playable', 'roomy'] }
    ];
  },

  rule(s, legal, sopt) {
    const policy = sopt.policy || 'safe';
    if (policy === 'hamilton' || policy === 'hybrid') return this._ruleCyc(s, legal, policy);
    return this._ruleSafe(s, legal);
  },

  /* --- BFS 寻食 + 洪水填充安全检查 --- */
  _ruleSafe(s, legal) {
    const N = s.N, len = s.snake.length;
    const foodId = s.food[0] * N + s.food[1];
    const ranks = [];
    for (const k of legal) {
      const [hr, hc] = s.snake[0], d = DIRS[k];
      const nr = hr + d[0], nc = hc + d[1];
      const eat = nr === s.food[0] && nc === s.food[1];
      const body = eat ? s.snake : s.snake.slice(0, -1);
      if (body.some(p => p[0] === nr && p[1] === nc)) {
        ranks.push({ key: k, label: DIRCN[k], score: -1e6, detail: '致死', risky: true }); continue;
      }
      const nsnake = [[nr, nc]].concat(body);
      const occ2 = new Set(nsnake.map(p => p[0] * N + p[1]));
      const headId = nr * N + nc;
      const free = floodFree(N, headId, occ2);
      const fd = bfsDist(N, headId, foodId, occ2);
      const tail = nsnake[nsnake.length - 1];
      const occ3 = new Set(nsnake.slice(0, -1).map(p => p[0] * N + p[1]));
      const td = bfsDist(N, headId, tail[0] * N + tail[1], occ3);
      const canTail = td >= 0;
      // 贴边奖励：把身体压向墙壁/自身，减少碎洞
      let hug = 0;
      for (const kk of DIRKEYS) {
        const ar = nr + DIRS[kk][0], ac = nc + DIRS[kk][1];
        if (ar < 0 || ac < 0 || ar >= N || ac >= N) hug++;
        else if (occ2.has(ar * N + ac)) hug++;
      }
      // 硬门：走完必须还能追到尾巴；其次空间大、离食物近、贴边压身。
      // 权重由 12 局坐标下降扫出（均值 29 → 66），spaceGate 惩罚「剩余空间已装不下自己」。
      const W = W_SAFE;
      const score = (canTail ? W.gate : 0) + free * W.free + hug * W.hug + (eat ? W.eat : 0)
                  - (fd < 0 ? 0 : fd * W.fd) - (canTail ? td * W.td : 0)
                  - (free < nsnake.length ? W.spaceGate : 0);
      ranks.push({
        key: k, label: DIRCN[k], score: score, risky: !canTail,
        detail: `可达空格 ${free} · 到食物 ${fd < 0 ? '∞' : fd} · 能否追尾 ${canTail ? '通(' + td + ')' : '断'} · 贴边 ${hug}${eat ? ' · 吃到' : ''}`
      });
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: ranks[0].key, ranks, note: 'safe 策略：先保证「走完还能回头」，再逼近食物。' };
  },

  /* --- 沿哈密顿回路 / 短路 --- */
  _ruleCyc(s, legal, policy) {
    const N = s.N, total = N * N, len = s.snake.length;
    const [hr, hc] = s.snake[0];
    const hi = s.cycIdx[hr * N + hc];
    const fi = s.cycIdx[s.food[0] * N + s.food[1]];
    const gap = (fi - hi + total) % total;      // 沿回路到食物还有几步
    const ranks = [];
    for (const k of legal) {
      const d = DIRS[k];
      const nr = hr + d[0], nc = hc + d[1];
      const eat = nr === s.food[0] && nc === s.food[1];
      const body = eat ? s.snake : s.snake.slice(0, -1);
      if (body.some(p => p[0] === nr && p[1] === nc)) {
        ranks.push({ key: k, label: DIRCN[k], score: -1e6, detail: '致死', risky: true }); continue;
      }
      const adv = (s.cycIdx[nr * N + nc] - hi + total) % total;
      if (adv === 0 || adv > gap) {
        ranks.push({ key: k, label: DIRCN[k], score: -1e5,
                     detail: adv === 0 ? '回路倒退' : '越过食物', risky: true }); continue;
      }
      let score = 1000 + adv, detail = `回路前进 ${adv} 步`;
      if (adv > 1) {
        // 短路 = 沿回路跳过 adv-1 格。跳过的格子会成为「洞」，必须留足一圈的余量
        // 让蛇身重新连成一段，否则头绕回时会撞进自己身体里。
        const maxSkip = Math.max(0, Math.floor((total - len) / 6));
        if (policy !== 'hybrid') { score = -1e5; detail = '非回路步进'; }
        else if (adv - 1 > maxSkip) { score = -1e4; detail += ` · 跳过量超限(≤${maxSkip})`; }
        else {
          const nsnake = [[nr, nc]].concat(body);
          const occ2 = new Set(nsnake.map(p => p[0] * N + p[1]));
          const free = floodFree(N, nr * N + nc, occ2);
          const need = total - nsnake.length;    // 想填满还需这么多格
          if (free >= need) { score = 1000 + adv * 10; detail += ` · 短路(空 ${free}≥需 ${need})`; }
          else { score = -1e4; detail += ` · 短路会锁死(空 ${free}<需 ${need})`; }
        }
      }
      ranks.push({ key: k, label: DIRCN[k], score, detail, risky: score < 0 });
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: ranks[0].key, ranks,
             note: policy === 'hybrid'
               ? 'hybrid：沿回路行进，仅在「跳过之后仍能填满剩余格子」时才抄近道。'
               : 'hamilton：严格沿哈密顿回路，不会自锁，最终填满全盘。' };
  },

  metrics(s, sopt) {
    const target = parseInt(sopt.target, 10) || 50;
    const len = s.snake.length, total = s.N * s.N;
    return [
      { k: '步数', v: s.steps },
      { k: '长度', v: `${len}/${total}`, tone: len >= target ? 'ok' : '' },
      { k: '吃到', v: s.eaten },
      { k: '空格', v: total - len },
      { k: '目标进度', v: `${Math.min(100, Math.round(len / target * 100))}%`,
        tone: len >= target ? 'ok' : 'warn', sm: `≥ ${target}` },
      { k: '状态', v: s.alive ? '存活' : '终局', tone: s.alive ? 'ok' : 'bad' }
    ];
  },

  render(s, el) {
    const N = s.N, cell = 100 / N;
    const m = new Map();
    s.snake.forEach((p, i) => m.set(p[0] * N + p[1], i === 0 ? 'H' : 's'));
    let h = `<div style="display:grid;grid-template-columns:repeat(${N},1fr);gap:2px;`
          + `background:#e4e7ec;padding:3px;border-radius:8px;aspect-ratio:1/1">`;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const id = r * N + c;
      const isF = s.food[0] === r && s.food[1] === c;
      const t = m.get(id);
      let bg = '#f7f8fa', bd = '';
      if (t === 'H') { bg = '#1d2430'; }
      else if (t === 's') { bg = '#7d8b9c'; }
      else if (isF) { bg = '#e8590c'; }
      h += `<div style="background:${bg};border-radius:3px"></div>`;
    }
    h += `</div><div class="hint">深灰=蛇头 · 浅灰=蛇身 · 橙=食物 · 当前方向 ${DIRCN[s.dir] || '—'}</div>`;
    el.innerHTML = h;
  },

  done(s) { return !s.alive || s.snake.length >= s.N * s.N; },
  goalOk(s, sopt) { return s.snake.length >= (parseInt(sopt.target, 10) || 50); },
  /* 注意：蛇每吃一个食物平均要走 5~10 步，长到 50 需 300~500 步、填满 100 格需
   * 2000+ 步。上限给太小会把「没死」误判成「没达成」。 */
  maxSteps(sopt) {
    const N = parseInt(sopt.size || '10', 10);
    return Math.min(4000, Math.max(1000, N * N * 24));
  }
};
