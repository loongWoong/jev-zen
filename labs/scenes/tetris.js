/* =============================================================================
 * 场景：俄罗斯方块 Tetris
 * 算法：枚举「旋转 × 落点列」全部合法落子，用 Dellacherie 六特征评分
 *       （落高 / 消行 / 行变换 / 列变换 / 洞 / 井），可选 2 层前瞻
 * 目标：消 100 行（标准计分下同时逼近 10 万分）
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const SHAPES = {
  I: ['....', 'XXXX', '....', '....'],
  O: ['XX', 'XX'],
  T: ['.X.', 'XXX', '...'],
  S: ['.XX', 'XX.', '...'],
  Z: ['XX.', '.XX', '...'],
  J: ['X..', 'XXX', '...'],
  L: ['..X', 'XXX', '...']
};
const PTYPES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
const PCOLOR = { I: '#31c4be', O: '#f2c14e', T: '#a06cd5', S: '#3fae5a',
                 Z: '#e05252', J: '#4a7fd4', L: '#e08b3c' };

function parseShape(rows) {
  const cells = [];
  rows.forEach((line, r) => {
    for (let c = 0; c < line.length; c++) if (line[c] === 'X') cells.push([r, c]);
  });
  return cells;
}
function rot90(cells, size) { return cells.map(([r, c]) => [c, size - 1 - r]); }
function norm(cells) {
  const mr = Math.min(...cells.map(p => p[0])), mc = Math.min(...cells.map(p => p[1]));
  return cells.map(([r, c]) => [r - mr, c - mc]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}
const ROTS = {};
for (const t of PTYPES) {
  let cur = norm(parseShape(SHAPES[t]));
  const size = SHAPES[t].length;
  const list = [], seen = new Set();
  for (let i = 0; i < 4; i++) {
    const key = JSON.stringify(cur);
    if (!seen.has(key)) { seen.add(key); list.push(cur); }
    cur = norm(rot90(cur, size));
  }
  ROTS[t] = list;
}

const SCENE = {
  id: 'tetris',
  name: '俄罗斯方块',
  sub: '10×20 · Dellacherie 六特征启发式 · 可选 2 层前瞻',
  goal: '目标：消 100 行',
  hint: '每步枚举全部「旋转 × 落点列」，用落高/消行/行变换/列变换/洞/井六特征打分。'
      + '模型面对的是同样的候选落点集合。',

  controls: [
    { id: 'evaluator', label: '评分函数', type: 'select', value: 'dellacherie', options: [
      { v: 'dellacherie', t: 'Dellacherie · 六特征' },
      { v: 'holes', t: '只看洞与高度' },
      { v: 'random', t: '随机（对照基线）' }] },
    { id: 'lookahead', label: '前瞻层数', type: 'select', value: '1', options: [
      { v: '1', t: '1 层' }, { v: '2', t: '2 层（含 next）' }] },
    { id: 'target', label: '目标行数', type: 'number', value: 100, min: 10, step: 10 }
  ],

  init(seed, sopt) {
    const N = 10, M = 20;
    const s = { N, M, grid: [], score: 0, lines: 0, steps: 0, alive: true,
                bag: [], type: null, nextType: null, plan: null,
                rng: mulberry32((seed * 2654435761) >>> 0) };
    for (let r = 0; r < M; r++) s.grid.push(new Array(N).fill(0));
    this._fillBag(s);
    s.type = this._take(s);
    s.nextType = this._take(s);
    if (!this._fits(s, s.type, 0, 0, this._startCol(s, s.type, 0))) s.alive = false;
    return s;
  },

  _fillBag(s) {
    const b = PTYPES.slice();
    for (let i = b.length - 1; i > 0; i--) {
      const j = Math.floor(s.rng() * (i + 1));
      [b[i], b[j]] = [b[j], b[i]];
    }
    s.bag = s.bag.concat(b);
  },
  _take(s) { if (s.bag.length < 2) this._fillBag(s); return s.bag.shift(); },
  _startCol(s, type, rot) { return Math.floor((s.N - this._width(ROTS[type][rot])) / 2); },
  _width(cells) { return Math.max(...cells.map(p => p[1])) + 1; },

  _fits(s, type, rot, r0, c0) {
    for (const [dr, dc] of ROTS[type][rot]) {
      const r = r0 + dr, c = c0 + dc;
      if (r < 0 || r >= s.M || c < 0 || c >= s.N) return false;
      if (r >= 0 && s.grid[r][c]) return false;
    }
    return true;
  },

  /* 枚举全部合法 (rot, col)，返回 [{key,rot,col,row}] */
  _moves(s, type) {
    const out = [];
    const nRot = ROTS[type].length;
    for (let rot = 0; rot < nRot; rot++) {
      const w = this._width(ROTS[type][rot]);
      for (let col = 0; col + w <= s.N; col++) {
        if (!this._fits(s, type, rot, 0, col)) continue;
        let row = 0;
        while (this._fits(s, type, rot, row + 1, col)) row++;
        out.push({ key: `r${rot}c${col}`, rot, col, row });
      }
    }
    return out;
  },

  _place(s, type, rot, row, col) {
    const g = s.grid.map(r => r.slice());
    for (const [dr, dc] of ROTS[type][rot]) g[row + dr][col + dc] = type;
    return g;
  },

  _clear(g, M, N) {
    let n = 0;
    const out = [];
    for (let r = 0; r < M; r++) {
      if (g[r].every(v => v)) n++;
      else out.push(g[r]);
    }
    while (out.length < M) out.unshift(new Array(N).fill(0));
    return { g: out, cleared: n };
  },

  /* ---- Dellacherie 六特征 ---- */
  _feat(g, M, N, landingRow, landingH) {
    let holes = 0, rowT = 0, colT = 0, wells = 0;
    const heights = new Array(N).fill(0);
    for (let c = 0; c < N; c++) {
      let top = M;
      for (let r = 0; r < M; r++) if (g[r][c]) { top = r; break; }
      heights[c] = M - top;
    }
    const filled = (r, c) => (r < 0 || r >= M || c < 0 || c >= N) ? true : !!g[r][c];
    for (let r = 0; r < M; r++) {
      for (let c = 0; c < N; c++) {
        const v = !!g[r][c];
        // 行变换：左右边界视为实
        if (c === 0 && !v) rowT++;
        if (c > 0 && v !== !!g[r][c - 1]) rowT++;
        if (c === N - 1 && !v) rowT++;
        // 列变换：上下边界视为空
        if (r === 0 && v) colT++;
        if (r > 0 && v !== !!g[r - 1][c]) colT++;
        if (r === M - 1 && v) colT++;
        if (!v) {
          // 洞：上方有实格
          for (let rr = r - 1; rr >= 0; rr--) if (g[rr][c]) { holes++; break; }
          // 井：左右皆实
          if (filled(r, c - 1) && filled(r, c + 1)) {
            let d = 1;
            while (r + d < M && !g[r + d][c]) d++;
            wells += d;
          }
        }
      }
    }
    const maxH = Math.max(...heights);
    const aggH = heights.reduce((a, b) => a + b, 0);
    return { holes, rowT, colT, wells, maxH, aggH, landingH };
  },

  _score(g, M, N, cleared, landingH, evaluator) {
    const f = this._feat(g, M, N, 0, landingH);
    if (evaluator === 'holes') return -(f.holes * 12 + f.aggH * 2 + f.maxH) + cleared * 20;
    if (evaluator === 'random') return Math.random();
    return -4.500158825082766 * landingH
         + 3.4181268101392694 * cleared
         - 3.2178882868487753 * f.rowT
         - 9.348695305445199 * f.colT
         - 7.899265427351652 * f.holes
         - 3.3855972247263626 * f.wells;
  },

  /* 对某个落点做完整评估（含消行与计分） */
  _evalMove(s, type, mv, evaluator) {
    const g1 = this._place(s, type, mv.rot, mv.row, mv.col);
    const cl = this._clear(g1, s.M, s.N);
    const cells = ROTS[type][mv.rot];
    const lowRow = mv.row + Math.max(...cells.map(p => p[0]));
    const landingH = s.M - lowRow;
    return { g: cl.g, cleared: cl.cleared, landingH,
             score: this._score(cl.g, s.M, s.N, cl.cleared, landingH, evaluator) };
  },

  legal(s) { return this._moves(s, s.type).map(m => m.key); },

  step(s, key) {
    const m = /^r(\d+)c(\d+)$/.exec(key);
    if (!m) return { info: 'bad key' };
    const rot = +m[1], col = +m[2];
    let row = 0;
    while (this._fits(s, s.type, rot, row + 1, col)) row++;
    if (!this._fits(s, s.type, rot, row, col)) { s.alive = false; return { info: '非法落点' }; }
    const cl = this._clear(this._place(s, s.type, rot, row, col), s.M, s.N);
    s.grid = cl.g;
    s.steps++;
    if (cl.cleared) {
      s.lines += cl.cleared;
      const tbl = [0, 100, 300, 500, 800];
      s.score += tbl[cl.cleared] * (1 + Math.floor(s.lines / 10));
    }
    s.type = s.nextType;
    s.nextType = this._take(s);
    s.plan = null;
    if (!this._moves(s, s.type).length) s.alive = false;
    return { info: `${s.type === null ? '' : ''}落下 ${key}${cl.cleared ? ' · 消 ' + cl.cleared + ' 行' : ''} · 累计 ${s.lines} 行 / ${s.score} 分` };
  },

  text(s, legal) {
    const g = s.grid.map(row => row.map(v => v ? '#' : '.').join(''));
    return `Tetris board ${s.N} wide x ${s.M} high. piece=${s.type} next=${s.nextType}.\n`
      + `lines_cleared=${s.lines} score=${s.score} steps=${s.steps}.\n`
      + `board (# filled, . empty), rows top to bottom:\n` + g.join('\n')
      + `\ncandidate_placements: ${legal.slice(0, 24).join(' ')}`;
  },

  questions(s, legal) {
    const cand = legal.slice(0, 16);
    const crit = {};
    for (const k of cand) {
      const m = /^r(\d+)c(\d+)$/.exec(k);
      crit[k] = `rotation ${m[1]}, drop into column ${m[2]}`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the placement (rotation and column) that keeps the stack low and clears lines.',
        criteria: crit },
      { id: 'risk', type: 'score',
        instructions: 'How dangerous is the current stack?',
        criteria: ['flat and safe', 'slightly uneven', 'rough with holes', 'about to top out'] }
    ];
  },

  rule(s, legal, sopt) {
    const ev = sopt.evaluator || 'dellacherie';
    const moves = this._moves(s, s.type).filter(m => legal.indexOf(m.key) >= 0);
    if (!moves.length) return { choice: null, ranks: [], note: '无可落点' };
    const la = (sopt.lookahead === '2');
    const scored = moves.map(mv => {
      const e = this._evalMove(s, s.type, mv, ev);
      let sc = e.score;
      if (la) {
        // 2 层：对 next 方块取最佳回应，只取「最好」避免组合爆炸
        let best2 = -Infinity;
        const mv2 = this._moves({ N: s.N, M: s.M, grid: e.g }, s.nextType);
        for (const m2 of mv2) {
          const e2 = this._evalMove({ N: s.N, M: s.M, grid: e.g }, s.nextType, m2, ev);
          if (e2.score > best2) best2 = e2.score;
        }
        if (isFinite(best2)) sc += best2 * 0.5;
      }
      return { mv, e, sc };
    });
    scored.sort((a, b) => b.sc - a.sc);
    const top = scored[0];
    s.plan = { rot: top.mv.rot, col: top.mv.col, row: top.mv.row, type: s.type };
    const ranks = scored.slice(0, 10).map(x => ({
      key: x.mv.key, label: x.mv.key, score: x.sc, digits: 1,
      detail: `落高 ${x.e.landingH}${x.e.cleared ? ' · 消 ' + x.e.cleared + ' 行' : ''}`
    }));
    return { choice: top.mv.key, ranks,
             note: `${ev}${la ? ' + 2 层前瞻' : ''}：共 ${moves.length} 个合法落点，取分最高者。` };
  },

  metrics(s, sopt) {
    const target = parseInt(sopt.target, 10) || 100;
    const f = this._feat(s.grid, s.M, s.N, 0, 0);
    return [
      { k: '消行', v: `${s.lines}/${target}`, tone: s.lines >= target ? 'ok' : '' },
      { k: '分数', v: s.score },
      { k: '落块', v: s.steps },
      { k: '堆叠高度', v: f.maxH, tone: f.maxH >= 16 ? 'bad' : f.maxH >= 12 ? 'warn' : 'ok' },
      { k: '洞', v: f.holes, tone: f.holes ? 'warn' : 'ok' },
      { k: '状态', v: s.alive ? '进行中' : '顶出', tone: s.alive ? 'ok' : 'bad' }
    ];
  },

  render(s, el) {
    const N = s.N, M = s.M;
    const g = s.grid.map(r => r.slice());
    if (s.plan) {
      for (const [dr, dc] of ROTS[s.plan.type][s.plan.rot]) {
        const r = s.plan.row + dr, c = s.plan.col + dc;
        if (r >= 0 && r < M && c >= 0 && c < N) g[r][c] = 'ghost';
      }
    }
    let h = `<div style="display:grid;grid-template-columns:repeat(${N},1fr);gap:1px;`
          + `background:#dfe4ea;padding:2px;border-radius:6px;max-width:300px;margin:0 auto">`;
    for (let r = 0; r < M; r++) for (let c = 0; c < N; c++) {
      const v = g[r][c];
      const bg = v === 'ghost' ? '#c9d3e0' : (v ? PCOLOR[v] : '#f7f8fa');
      h += `<div style="background:${bg};aspect-ratio:1/1;border-radius:2px"></div>`;
    }
    h += `</div><div class="hint" style="text-align:center">当前 ${s.type || '—'} · next ${s.nextType || '—'}`
       + ` · 灰影 = 本步落点预览</div>`;
    el.innerHTML = h;
  },

  done(s) { return !s.alive; },
  goalOk(s, sopt) { return s.lines >= (parseInt(sopt.target, 10) || 100); },
  maxSteps(sopt) { return 3000; }
};
