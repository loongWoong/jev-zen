/* =============================================================================
 * 场景：15 拼图 15-Puzzle
 * 算法：IDA*（曼哈顿距离 + 线性冲突），开局求一次最优解并缓存，偏离后自动重解
 * 目标：20 步打乱后复原
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const NB = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };

/* 曼哈顿距离 + 线性冲突（同一行/列中两块的目标顺序相反 → 至少要绕开一次，+2） */
function heur(b, N) {
  let h = 0;
  for (let i = 0; i < N * N; i++) {
    const v = b[i];
    if (!v) continue;
    const tr = Math.floor((v - 1) / N), tc = (v - 1) % N;
    const r = Math.floor(i / N), c = i % N;
    h += Math.abs(r - tr) + Math.abs(c - tc);
  }
  for (let r = 0; r < N; r++) {
    for (let c1 = 0; c1 < N; c1++) for (let c2 = c1 + 1; c2 < N; c2++) {
      const a = b[r * N + c1], d = b[r * N + c2];
      if (!a || !d) continue;
      if (Math.floor((a - 1) / N) === r && Math.floor((d - 1) / N) === r
          && (a - 1) % N > (d - 1) % N) h += 2;
    }
  }
  for (let c = 0; c < N; c++) {
    for (let r1 = 0; r1 < N; r1++) for (let r2 = r1 + 1; r2 < N; r2++) {
      const a = b[r1 * N + c], d = b[r2 * N + c];
      if (!a || !d) continue;
      if ((a - 1) % N === c && (d - 1) % N === c
          && Math.floor((a - 1) / N) > Math.floor((d - 1) / N)) h += 2;
    }
  }
  return h;
}

/* IDA*：就地修改 board 并回溯，返回应滑动的方块编号序列 */
function idaStar(board, N, nodeLimit) {
  const n = N * N;
  let blank = board.indexOf(0);
  const path = [];
  let nodes = 0;
  const limit = nodeLimit || 400000;

  const dfs = (g, bound, prevBlank) => {
    const h = heur(board, N);
    const f = g + h;
    if (f > bound) return f;
    if (h === 0) return -1;
    let min = Infinity;
    const r = Math.floor(blank / N), c = blank % N;
    for (const d of ['up', 'down', 'left', 'right']) {
      const nr = r + NB[d][0], nc = c + NB[d][1];
      if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
      const ni = nr * N + nc;
      if (ni === prevBlank) continue;
      const tile = board[ni];
      board[blank] = tile; board[ni] = 0;
      const ob = blank; blank = ni;
      path.push(tile);
      const t = dfs(g + 1, bound, ob);
      if (t === -1) return -1;
      if (t < min) min = t;
      path.pop();
      blank = ob; board[ob] = 0; board[ni] = tile;
      if (++nodes > limit) return Infinity;
    }
    return min;
  };

  let bound = heur(board, N);
  for (let iter = 0; iter < 60; iter++) {
    const t = dfs(0, bound, -1);
    if (t === -1) return { ok: true, path: path.slice(), nodes };
    if (t === Infinity) return { ok: false, nodes, reason: '节点超限' };
    bound = t;
    if (bound > 90) return { ok: false, nodes, reason: '解过长' };
  }
  return { ok: false, nodes, reason: '迭代超限' };
}

const SCENE = {
  id: 'puzzle15',
  name: '15 拼图',
  sub: '4×4 · IDA* + 曼哈顿距离与线性冲突',
  goal: '目标：20 步打乱后复原',
  hint: 'IDA* 以「曼哈顿距离 + 线性冲突」为下界做迭代加深，保证给出的是最优步数解。'
      + '开局求一次并缓存计划；若被模型带偏，会自动从当前局面重解。',

  controls: [
    { id: 'scramble', label: '打乱步数', type: 'number', value: 20, min: 1, max: 60, step: 1 },
    { id: 'size', label: '棋盘', type: 'select', value: '4', options: [
      { v: '3', t: '3×3（8 拼图）' }, { v: '4', t: '4×4（15 拼图）' }] },
    { id: 'nodeLimit', label: '搜索节点上限', type: 'select', value: '400000', options: [
      { v: '100000', t: '10 万（快）' }, { v: '400000', t: '40 万' }, { v: '2000000', t: '200 万（慢）' }] }
  ],

  init(seed, sopt) {
    const N = parseInt(sopt.size || '4', 10);
    const n = N * N;
    const rng = mulberry32((seed * 2654435761) >>> 0);
    const board = new Array(n);
    for (let i = 0; i < n - 1; i++) board[i] = i + 1;
    board[n - 1] = 0;
    let blank = n - 1, prev = -1;
    const k = parseInt(sopt.scramble, 10) || 20;
    let done = 0;
    for (let i = 0; i < k * 4 && done < k; i++) {
      const r = Math.floor(blank / N), c = blank % N;
      const cand = [];
      for (const d of ['up', 'down', 'left', 'right']) {
        const nr = r + NB[d][0], nc = c + NB[d][1];
        if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
        const ni = nr * N + nc;
        if (ni === prev) continue;
        cand.push(ni);
      }
      if (!cand.length) break;
      const pick = cand[Math.floor(rng() * cand.length)];
      board[blank] = board[pick]; board[pick] = 0;
      prev = blank; blank = pick; done++;
    }
    const s = { N, board, blank, steps: 0, scramble: done, plan: null, info: null };
    this._plan(s, sopt);
    return s;
  },

  _plan(s, sopt) {
    const r = idaStar(s.board.slice(), s.N, parseInt(sopt.nodeLimit || '400000', 10));
    s.info = r;
    s.plan = r.ok ? r.path.slice() : null;
    return r;
  },

  legal(s) {
    const N = s.N, r = Math.floor(s.blank / N), c = s.blank % N;
    const out = [];
    for (const d of ['up', 'down', 'left', 'right']) {
      const nr = r + NB[d][0], nc = c + NB[d][1];
      if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
      out.push('t' + s.board[nr * N + nc]);
    }
    return out;
  },

  step(s, key) {
    const tile = parseInt(key.slice(1), 10);
    const idx = s.board.indexOf(tile);
    if (idx < 0) return { info: 'bad key' };
    const okAdj = Math.abs(Math.floor(idx / s.N) - Math.floor(s.blank / s.N))
                 + Math.abs(idx % s.N - s.blank % s.N) === 1;
    if (!okAdj) { return { info: '该块不与空格相邻' }; }
    s.board[s.blank] = tile; s.board[idx] = 0;
    s.blank = idx; s.steps++;
    if (s.plan && s.plan.length) s.plan.shift();
    return { info: `滑动 ${tile} · 错位数 ${this._wrong(s)}` };
  },

  _wrong(s) {
    let w = 0;
    for (let i = 0; i < s.N * s.N - 1; i++) if (s.board[i] !== i + 1) w++;
    return w;
  },

  text(s, legal) {
    const N = s.N, rows = [];
    for (let r = 0; r < N; r++) {
      const cells = [];
      for (let c = 0; c < N; c++) {
        const v = s.board[r * N + c];
        cells.push(v === 0 ? ' .' : String(v).padStart(2));
      }
      rows.push(cells.join(' '));
    }
    return `${N}x${N} sliding puzzle. goal: 1..${N * N - 1} in row-major order, blank last.\n`
      + `steps=${s.steps} misplaced=${this._wrong(s)} manhattan=${heur(s.board, N)}.\n`
      + `legal_tiles (slidable into the blank): ${legal.join(' ')}\n`
      + `board:\n` + rows.join('\n');
  },

  questions(s, legal) {
    const crit = {};
    for (const k of legal.slice(0, 4)) crit[k] = `slide tile ${k.slice(1)} into the blank`;
    return [
      { id: 'move', type: 'choice',
        instructions: 'Which tile should be slid into the blank to solve the puzzle in fewest moves?',
        criteria: crit },
      { id: 'closeness', type: 'score',
        instructions: 'How close is the board to solved?',
        criteria: ['far', 'halfway', 'nearly solved', 'solved'] }
    ];
  },

  rule(s, legal, sopt) {
    if (!s.plan || !s.plan.length || legal.indexOf('t' + s.plan[0]) < 0) this._plan(s, sopt);
    const tile = s.plan && s.plan.length ? s.plan[0] : null;
    const key = tile ? 't' + tile : null;
    const ranks = legal.map(k => {
      const isNext = k === key;
      return { key: k, label: '块 ' + k.slice(1), score: isNext ? 1000 : 0,
               detail: isNext ? `IDA* 最优解第 1 步（共 ${s.plan.length} 步）` : '偏离最优解' };
    });
    if (!key) {
      // 无解：退化为「让曼哈顿距离下降最多」
      let best = null;
      for (const k of legal) {
        const t = parseInt(k.slice(1), 10);
        const idx = s.board.indexOf(t);
        const bak = s.board.slice(), bb = s.blank;
        s.board[s.blank] = t; s.board[idx] = 0; s.blank = idx;
        const hv = heur(s.board, s.N);
        s.board = bak; s.blank = bb;
        ranks.find(r => r.key === k).score = -hv;
        ranks.find(r => r.key === k).detail = `退化为贪心：走后曼哈顿 ${hv}`;
        if (!best || hv < best.hv) best = { k, hv };
      }
      ranks.sort((a, b) => b.score - a.score);
      return { choice: ranks[0].key, ranks,
               note: `IDA* 未解出（${s.info ? s.info.reason : '?'}, ${s.info ? s.info.nodes : 0} 节点），退化为贪心。` };
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: key, ranks,
             note: `IDA* 最优解 ${s.plan.length} 步（搜索 ${s.info.nodes} 节点）。` };
  },

  metrics(s, sopt) {
    const w = this._wrong(s);
    return [
      { k: '步数', v: s.steps },
      { k: '错位数', v: `${w}/${s.N * s.N - 1}`, tone: w === 0 ? 'ok' : '' },
      { k: '曼哈顿', v: heur(s.board, s.N) },
      { k: '最优解', v: s.plan ? s.plan.length : '未解出', tone: s.plan ? '' : 'bad' },
      { k: '打乱步数', v: s.scramble },
      { k: '状态', v: w === 0 ? '已复原' : '复原中', tone: w === 0 ? 'ok' : 'warn' }
    ];
  },

  render(s, el) {
    const N = s.N;
    let h = `<div style="display:grid;grid-template-columns:repeat(${N},1fr);gap:4px;`
          + `background:#dfe4ea;padding:4px;border-radius:8px;max-width:320px;margin:0 auto">`;
    for (let i = 0; i < N * N; i++) {
      const v = s.board[i];
      const correct = v !== 0 && v === i + 1;
      h += `<div style="background:${v ? (correct ? '#e6f6f3' : '#fff') : '#eef1f4'};`
         + `color:${correct ? '#0b6a60' : '#1d2430'};aspect-ratio:1/1;border-radius:5px;`
         + `display:flex;align-items:center;justify-content:center;font-family:var(--mono);`
         + `font-weight:700;font-size:15px;border:1px solid ${correct ? '#c9e7e1' : '#e4e7ec'}">${v || ''}</div>`;
    }
    h += `</div><div class="hint" style="text-align:center">绿底 = 已在正确位置 · 空格为灰块</div>`;
    el.innerHTML = h;
  },

  done(s) { return this._wrong(s) === 0; },
  goalOk(s) { return this._wrong(s) === 0; },
  maxSteps(sopt) { return 200; }
};
