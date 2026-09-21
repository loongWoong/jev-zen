/* =============================================================================
 * 场景：Lights Out（5×5）
 * 算法：GF(2) 上的高斯消元。点击 j 会翻转 j 及其上下左右，于是
 *       A·x = b (mod 2)，解出 x 即为「每个格子点或不点」
 * 目标：随机可解局全部熄灭
 * 注意：5×5 的影响矩阵秩为 23，零空间非平凡 —— 局面必须由「从全灭开始随机点击」
 *       生成，否则可能无解（此时求解器会明确报无解）
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* 点击 j 影响到的格子集合（位掩码）。
 * 用 BigInt 而不是 32 位整数：6×6 = 36 格会超出 Number 的位运算宽度（实测 6×6 全错）。 */
function effectMask(N, j) {
  const r = Math.floor(j / N), c = j % N;
  let m = 1n << BigInt(j);
  if (r > 0) m |= 1n << BigInt(j - N);
  if (r < N - 1) m |= 1n << BigInt(j + N);
  if (c > 0) m |= 1n << BigInt(j - 1);
  if (c < N - 1) m |= 1n << BigInt(j + 1);
  return m;
}

/* 解 A·x = b (mod 2)。litMask 为 BigInt；返回 BigInt 数组或 null（无解） */
function solveGF2(N, litMask) {
  const n = N * N;
  const M = [];
  for (let i = 0; i < n; i++) {
    let a = 0n;
    for (let j = 0; j < n; j++) if ((effectMask(N, j) >> BigInt(i)) & 1n) a |= 1n << BigInt(j);
    M.push({ a, b: Number((BigInt(litMask) >> BigInt(i)) & 1n) });
  }
  const where = new Array(n).fill(-1);
  let row = 0;
  for (let col = 0; col < n && row < n; col++) {
    let sel = -1;
    for (let i = row; i < n; i++) if ((M[i].a >> BigInt(col)) & 1n) { sel = i; break; }
    if (sel < 0) continue;
    const t = M[row]; M[row] = M[sel]; M[sel] = t;
    where[col] = row;
    for (let i = 0; i < n; i++) {
      if (i !== row && ((M[i].a >> BigInt(col)) & 1n)) { M[i].a ^= M[row].a; M[i].b ^= M[row].b; }
    }
    row++;
  }
  for (let i = row; i < n; i++) if (M[i].b) return null;   // 0 = 1，无解
  const x = new Array(n).fill(0);
  for (let col = 0; col < n; col++) if (where[col] >= 0) x[col] = M[where[col]].b;
  return x;
}

const SCENE = {
  id: 'lightsout',
  name: 'Lights Out',
  sub: '5×5 · GF(2) 高斯消元精确求解',
  goal: '目标：随机可解局全部熄灭',
  hint: '把「点哪些格」写成 GF(2) 上的线性方程组 A·x = b 直接解出来，'
      + '不是搜索而是精确求解 —— 可解局面必在 ⌈解集大小⌉ 步内熄灭。'
      + '局面由「从全灭随机点击若干次」生成，保证有解。',

  controls: [
    { id: 'size', label: '棋盘', type: 'select', value: '5', options: [
      { v: '4', t: '4×4' }, { v: '5', t: '5×5' }, { v: '6', t: '6×6' }] },
    { id: 'scramble', label: '随机点击次数', type: 'number', value: 7, min: 1, max: 30, step: 1 },
    { id: 'greedy', label: '对照：贪心（无消元）', type: 'checkbox', value: false, hint: '用笨办法对照' }
  ],

  init(seed, sopt) {
    const N = parseInt(sopt.size || '5', 10);
    const n = N * N;
    const rng = mulberry32((seed * 2654435761) >>> 0);
    let mask = 0n;
    const k = parseInt(sopt.scramble, 10) || 7;
    for (let i = 0; i < k; i++) {
      const j = Math.floor(rng() * n);
      mask ^= effectMask(N, j);
    }
    const s = { N, n, mask, steps: 0, plan: null, solvable: null, rng, scramble: k };
    this._plan(s, sopt);
    return s;
  },

  _plan(s, sopt) {
    if (sopt.greedy) { s.plan = null; s.solvable = 'greedy'; return; }
    const x = solveGF2(s.N, s.mask);
    s.solvable = x ? true : false;
    s.plan = [];
    if (x) for (let j = 0; j < s.n; j++) if (x[j]) s.plan.push(j);
    return x;
  },

  _lit(s, i) { return Number((s.mask >> BigInt(i)) & 1n); },

  legal(s) {
    const out = [];
    for (let i = 0; i < s.n; i++) out.push('c' + i);
    return out;
  },

  step(s, key) {
    const i = parseInt(key.slice(1), 10);
    s.mask ^= effectMask(s.N, i);
    s.steps++;
    const p = s.plan ? s.plan.indexOf(i) : -1;
    if (p >= 0) s.plan.splice(p, 1);
    return { info: `点击 (${Math.floor(i / s.N)},${i % s.N}) · 剩余亮灯 ${this._count(s)}` };
  },

  _count(s) {
    let c = 0;
    for (let i = 0; i < s.n; i++) if (this._lit(s, i)) c++;
    return c;
  },

  text(s, legal) {
    const N = s.N, rows = [];
    for (let r = 0; r < N; r++) {
      let line = '';
      for (let c = 0; c < N; c++) line += this._lit(s, r * N + c) ? '1' : '0';
      rows.push(line);
    }
    return `Lights Out ${N}x${N}. lit=${this._count(s)}/${s.n} steps=${s.steps}.\n`
      + `Clicking a cell toggles itself and its four neighbours. Goal: all 0.\n`
      + `board (1 = lit, 0 = dark), rows top to bottom:\n` + rows.join('\n')
      + `\ncell index = row*${N} + col`;
  },

  questions(s, legal) {
    const crit = {};
    const N = s.N;
    const cand = legal.slice(0, 16);
    for (const k of cand) {
      const i = parseInt(k.slice(1), 10);
      crit[k] = `click cell row ${Math.floor(i / N)}, col ${i % N}`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Which cell should be clicked to turn all lights off?',
        criteria: crit },
      { id: 'progress', type: 'score',
        instructions: 'How many lights are still on?',
        criteria: ['none (solved)', 'a few', 'about half', 'most'] }
    ];
  },

  rule(s, legal, sopt) {
    const N = s.N;
    const lit = this._count(s);
    if (lit === 0) return { choice: null, ranks: [], note: '已全灭' };
    if (sopt.greedy) {
      // 对照用的笨办法：选「翻转后亮灯数最少」的格子（会卡住，用来对比消元的价值）
      let best = null;
      const ranks = [];
      for (const k of legal) {
        const i = parseInt(k.slice(1), 10);
        const after = this._count(s) - 2 * this._lit(s, i) + (5 - this._edge(N, i));
        const delta = -after;
        ranks.push({ key: k, label: `(${Math.floor(i / N)},${i % N})`, score: delta,
                     detail: `贪心：翻转后亮灯 ${after}` });
        if (!best || after < best.after) best = { k, after };
      }
      ranks.sort((a, b) => b.score - a.score);
      return { choice: best.k, ranks: ranks.slice(0, 10), note: '贪心对照：只看一步，无法保证解出。' };
    }
    if (!s.plan || !s.plan.length) {
      const x = this._plan(s, sopt);
      if (!x) {
        const ranks = legal.slice(0, 6).map(k => ({ key: k, label: k, score: 0, detail: '无解，随便点' }));
        return { choice: legal[0], ranks, note: 'GF(2) 消元判定：该局面无解（不在可解子空间内）。' };
      }
    }
    const pick = s.plan[0];
    const key = 'c' + pick;
    const ranks = [{ key, label: `(${Math.floor(pick / N)},${pick % N})`,
                     score: 1000, detail: `高斯消元解集第 1 个（共 ${s.plan.length} 个待点）` }];
    // 再列几个「不在解集内」的对照选项
    let extra = 0;
    for (const k of legal) {
      if (extra >= 5) break;
      const i = parseInt(k.slice(1), 10);
      if (s.plan.indexOf(i) >= 0) continue;
      ranks.push({ key: k, label: `(${Math.floor(i / N)},${i % N})`, score: 0,
                   detail: '不在解集内（点了会破坏解）' });
      extra++;
    }
    return { choice: key, ranks,
             note: `GF(2) 消元：共 ${s.plan.length} 个格子需点击（解集大小固定，顺序任意）。` };
  },

  _edge(N, i) {
    const r = Math.floor(i / N), c = i % N;
    let k = 0;
    if (r === 0) k++; if (r === N - 1) k++;
    if (c === 0) k++; if (c === N - 1) k++;
    return k;
  },

  metrics(s, sopt) {
    const lit = this._count(s);
    return [
      { k: '亮灯', v: `${lit}/${s.n}`, tone: lit === 0 ? 'ok' : '' },
      { k: '步数', v: s.steps },
      { k: '待点击', v: s.plan ? s.plan.length : (s.solvable === false ? '无解' : '—'),
        tone: s.solvable === false ? 'bad' : '' },
      { k: '可解性', v: s.solvable === false ? '无解' : s.solvable === 'greedy' ? '贪心模式' : '可解',
        tone: s.solvable === false ? 'bad' : 'ok' },
      { k: '完成度', v: `${Math.round((1 - lit / s.n) * 100)}%` },
      { k: '状态', v: lit === 0 ? '全灭' : '进行中', tone: lit === 0 ? 'ok' : 'warn' }
    ];
  },

  render(s, el) {
    const N = s.N;
    let h = `<div style="display:grid;grid-template-columns:repeat(${N},1fr);gap:4px;`
          + `background:#dfe4ea;padding:4px;border-radius:8px;max-width:320px;margin:0 auto">`;
    for (let i = 0; i < s.n; i++) {
      const on = this._lit(s, i);
      const inPlan = s.plan && s.plan.indexOf(i) >= 0;
      h += `<div style="background:${on ? '#f2c14e' : '#3d4757'};aspect-ratio:1/1;border-radius:5px;`
         + `display:flex;align-items:center;justify-content:center;font-family:var(--mono);`
         + `font-size:11px;color:#fff;${inPlan ? 'outline:2px solid #2f6fed;outline-offset:1px' : ''}">`
         + `${inPlan ? '·' : ''}</div>`;
    }
    h += `</div><div class="hint" style="text-align:center">黄 = 亮 · 深 = 灭 · 蓝框 = 消元解集待点击</div>`;
    el.innerHTML = h;
  },

  done(s) { return this._count(s) === 0; },
  goalOk(s) { return this._count(s) === 0; },
  maxSteps(sopt) { const N = parseInt(sopt.size || '5', 10); return N * N; }
};
