/* =============================================================================
 * 场景：扫雷 Minesweeper（初级 9×9 / 10 雷）
 * 算法：基础单格规则 → 约束分量枚举（CSP）→ 按组合数加权算每格雷概率
 *       → 取概率最低格；无安全格时也必须猜，则猜概率最低（优先角/边）
 * 目标：批量胜率（初级经典 solver 约 85%~92%）
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const LF = [0];
for (let i = 1; i <= 400; i++) LF[i] = LF[i - 1] + Math.log(i);
const logC = (n, k) => (k < 0 || k > n || n < 0) ? -Infinity : LF[n] - LF[k] - LF[n - k];

const NB8 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

const SCENE = {
  id: 'minesweeper',
  name: '扫雷',
  sub: '9×9 / 10 雷 · 约束满足 CSP + 组合数加权概率推理',
  goal: '目标：批量胜率 ≥ 85%',
  hint: '先用单格规则推出「必安全 / 必是雷」；推不动时把边界拆成独立约束分量，'
      + '枚举每个分量的全部可行解并按 C(剩余空位, 剩余雷) 加权，得到每格是雷的概率。',

  controls: [
    { id: 'solver', label: '求解器', type: 'select', value: 'csp', options: [
      { v: 'csp', t: 'CSP + 概率推理' },
      { v: 'basic', t: '仅单格规则 + 猜' },
      { v: 'random', t: '随机（对照基线）' }] },
    { id: 'firstSafe', label: '首点保护', type: 'checkbox', value: true, hint: '首点必不中雷' },
    { id: 'size', label: '棋盘', type: 'select', value: '9', options: [
      { v: '9', t: '初级 9×9 · 10 雷' }, { v: '16', t: '中级 16×16 · 40 雷' }] }
  ],

  init(seed, sopt) {
    const N = parseInt(sopt.size || '9', 10);
    const M = N === 9 ? 10 : 40;
    const total = N * N;
    const s = { N, M, total, mine: new Uint8Array(total), count: new Int8Array(total),
                open: new Uint8Array(total), flag: new Uint8Array(total),
                opened: 0, steps: 0, alive: true, won: false, first: true,
                firstSafe: sopt.firstSafe !== false,
                rng: mulberry32((seed * 2654435761) >>> 0) };
    // 布雷
    const idx = Array.from({ length: total }, (_, i) => i);
    for (let i = total - 1; i > 0; i--) {
      const j = Math.floor(s.rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let i = 0; i < M; i++) s.mine[idx[i]] = 1;
    for (let i = 0; i < total; i++) {
      if (s.mine[i]) { s.count[i] = -1; continue; }
      let n = 0;
      const r = Math.floor(i / N), c = i % N;
      for (const [dr, dc] of NB8) {
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
        if (s.mine[nr * N + nc]) n++;
      }
      s.count[i] = n;
    }
    return s;
  },

  _nbrs(s, i) {
    const N = s.N, r = Math.floor(i / N), c = i % N, out = [];
    for (const [dr, dc] of NB8) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
      out.push(nr * N + nc);
    }
    return out;
  },

  /* 单格规则：返回 { safe:Set, mine:Set } */
  _basic(s) {
    const safe = new Set(), mine = new Set();
    for (let i = 0; i < s.total; i++) {
      if (!s.open[i] || s.count[i] <= 0) continue;
      const nb = this._nbrs(s, i);
      const unk = [], flg = [];
      for (const j of nb) {
        if (s.open[j]) continue;
        if (s.flag[j]) flg.push(j); else unk.push(j);
      }
      if (!unk.length) continue;
      const left = s.count[i] - flg.length;
      if (left === 0) unk.forEach(j => safe.add(j));
      else if (left === unk.length) unk.forEach(j => mine.add(j));
    }
    return { safe, mine };
  },

  /* CSP：边界拆成连通分量，枚举可行解并按组合数加权 */
  _csp(s) {
    const N = s.N;
    const cons = [];                       // {cells:[idx], mines:n}
    const borderSet = new Set();
    for (let i = 0; i < s.total; i++) {
      if (!s.open[i] || s.count[i] <= 0) continue;
      const nb = this._nbrs(s, i);
      const unk = nb.filter(j => !s.open[j] && !s.flag[j]);
      if (!unk.length) continue;
      const left = s.count[i] - nb.filter(j => s.flag[j]).length;
      if (left < 0 || left > unk.length) continue;
      cons.push({ cells: unk, mines: left });
      unk.forEach(j => borderSet.add(j));
    }
    const border = [...borderSet];
    if (!border.length) return null;

    // 按共享格子连通分量划分
    const compOf = new Map();
    const comps = [];
    for (const c of cons) {
      const hit = new Set();
      for (const j of c.cells) if (compOf.has(j)) hit.add(compOf.get(j));
      let target;
      if (!hit.size) { target = comps.length; comps.push({ cells: new Set(), cons: [] }); }
      else {
        target = Math.min(...hit);
        for (const h of hit) {
          if (h === target) continue;
          for (const j of comps[h].cells) { comps[target].cells.add(j); compOf.set(j, target); }
          comps[target].cons = comps[target].cons.concat(comps[h].cons);
          comps[h].cons = [];
        }
      }
      comps[target].cons.push(c);
      c.cells.forEach(j => { comps[target].cells.add(j); compOf.set(j, target); });
    }
    const live = comps.filter(c => c.cons.length);

    const unkAll = [];
    for (let i = 0; i < s.total; i++) if (!s.open[i] && !s.flag[i]) unkAll.push(i);
    const flagged = [...s.flag].reduce((a, b) => a + b, 0);
    const remMines = s.M - flagged;
    const outside = unkAll.length - border.length;   // 不受任何约束的格子数

    const prob = new Map();
    border.forEach(j => prob.set(j, 0));
    let wSum = 0, expBorder = 0;

    for (const comp of live) {
      const cells = [...comp.cells];
      const k = cells.length;
      if (k === 0) continue;
      if (k > 20) {   // 分量太大，退化为「每格 = 该分量雷数期望 / 格数」
        let lo = 0, hi = k;
        for (const c of comp.cons) { lo = Math.max(lo, c.mines); hi = Math.min(hi, k - (c.cells.length - c.mines)); }
        const p = ((lo + hi) / 2) / k;
        cells.forEach(j => prob.set(j, p));
        expBorder += (lo + hi) / 2;
        continue;
      }
      const tally = new Float64Array(k);
      let compSum = 0, expM = 0;
      for (let mask = 0; mask < (1 << k); mask++) {
        let ok = true;
        for (const c of comp.cons) {
          let n = 0;
          for (const j of c.cells) if (mask & (1 << cells.indexOf(j))) n++;
          if (n !== c.mines) { ok = false; break; }
        }
        if (!ok) continue;
        let m = 0;
        for (let b = 0; b < k; b++) if (mask & (1 << b)) m++;
        const other = remMines - m;
        const w = (other < 0 || other > outside) ? 0 : Math.exp(logC(outside, other));
        if (w <= 0) continue;
        for (let b = 0; b < k; b++) if (mask & (1 << b)) tally[b] += w;
        compSum += w; expM += w * m;
      }
      if (compSum <= 0) { cells.forEach(j => prob.set(j, remMines / Math.max(1, unkAll.length))); continue; }
      for (let b = 0; b < k; b++) prob.set(cells[b], tally[b] / compSum);
      expBorder += expM / compSum;
      wSum += compSum;
    }

    // 不受约束的格子：均分剩余雷
    const pOut = outside > 0
      ? Math.max(0, Math.min(1, (remMines - expBorder) / outside))
      : 1;
    const out = { border: new Map(prob), outside: pOut, outsideCells: unkAll.filter(j => !borderSet.has(j)) };
    return out;
  },

  /* 角 > 边 > 内部：同为最低概率时优先猜角，角落更容易滚出 0 格连锁 */
  _corner(s, j) {
    const N = s.N, r = Math.floor(j / N), c = j % N;
    const er = (r === 0 || r === N - 1), ec = (c === 0 || c === N - 1);
    return (er && ec) ? 3 : (er || ec) ? 2 : 1;
  },

  /* 候选动作集（限制规模，便于模型选择） */
  legal(s) {
    const keys = [];
    const { safe, mine } = this._basic(s);
    for (const j of mine) if (!s.flag[j]) keys.push('f' + j);
    for (const j of safe) if (!s.open[j] && !s.flag[j]) keys.push('o' + j);
    const csp = this._csp(s);
    const pool = new Set(keys);
    if (csp) {
      const arr = [...csp.border.entries()].sort((a, b) => a[1] - b[1]).slice(0, 10);
      for (const [j] of arr) pool.add('o' + j);
      for (const j of csp.outsideCells.slice(0, 3)) pool.add('o' + j);
    }
    if (!pool.size) {
      const unk = [];
      for (let i = 0; i < s.total; i++) if (!s.open[i] && !s.flag[i]) unk.push(i);
      unk.slice(0, 16).forEach(j => pool.add('o' + j));
    }
    return [...pool];
  },

  step(s, key) {
    const act = key[0], j = parseInt(key.slice(1), 10);
    s.steps++;
    if (act === 'f') { s.flag[j] = 1; return { info: `标记 (${Math.floor(j / s.N)},${j % s.N}) 为雷` }; }
    // 首点保护：把踩到的雷挪到别处
    if (s.mine[j] && s.first && s.firstSafe) {
      let moved = false;
      for (let i = 0; i < s.total && !moved; i++) {
        if (!s.mine[i] && i !== j) { s.mine[i] = 1; s.mine[j] = 0; moved = true; }
      }
      for (let i = 0; i < s.total; i++) {
        if (s.mine[i]) { s.count[i] = -1; continue; }
        let n = 0;
        for (const k of this._nbrs(s, i)) if (s.mine[k]) n++;
        s.count[i] = n;
      }
    }
    s.first = false;
    if (s.mine[j]) { s.alive = false; return { info: `踩雷 (${Math.floor(j / s.N)},${j % s.N})` }; }
    // 展开（0 格递归）
    const stack = [j];
    while (stack.length) {
      const cur = stack.pop();
      if (s.open[cur] || s.flag[cur]) continue;
      s.open[cur] = 1; s.opened++;
      if (s.count[cur] === 0) for (const k of this._nbrs(s, cur)) if (!s.open[k] && !s.flag[k]) stack.push(k);
    }
    if (s.opened >= s.total - s.M) s.won = true;
    return { info: `点开 (${Math.floor(j / s.N)},${j % s.N}) · 已开 ${s.opened}/${s.total - s.M}` };
  },

  text(s, legal) {
    const N = s.N, rows = [];
    for (let r = 0; r < N; r++) {
      let line = '';
      for (let c = 0; c < N; c++) {
        const i = r * N + c;
        if (s.flag[i]) line += 'F';
        else if (!s.open[i]) line += '.';
        else line += s.count[i] === 0 ? '0' : String(s.count[i]);
      }
      rows.push(line);
    }
    const rem = s.M - [...s.flag].reduce((a, b) => a + b, 0);
    return `Minesweeper ${N}x${N} with ${s.M} mines. opened=${s.opened}/${s.total - s.M} `
      + `flags_placed=${s.M - rem} mines_left=${rem}.\n`
      + `board (. unknown, F flagged, 0-8 adjacent mine count):\n` + rows.join('\n')
      + `\ncandidate_moves: ${legal.slice(0, 14).join(' ')} (o=open, f=flag, number = row*${N}+col)`;
  },

  questions(s, legal) {
    const cand = legal.slice(0, 14);
    const crit = {};
    for (const k of cand) {
      const j = parseInt(k.slice(1), 10);
      crit[k] = `${k[0] === 'o' ? 'open' : 'flag as mine'} cell row ${Math.floor(j / s.N)}, column ${j % s.N}`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the safest next move: open a cell or flag a certain mine.',
        criteria: crit },
      { id: 'safety', type: 'noul',
        instructions: 'Is there a move that is guaranteed safe from the visible numbers?' }
    ];
  },

  rule(s, legal, sopt) {
    const mode = sopt.solver || 'csp';
    const N = s.N;
    const keyOf = j => `r${Math.floor(j / N)}c${j % N}`;
    const { safe, mine } = this._basic(s);
    const ranks = [];

    if (mode === 'random') {
      const unk = [];
      for (let i = 0; i < s.total; i++) if (!s.open[i] && !s.flag[i]) unk.push(i);
      const pick = unk[Math.floor(Math.random() * unk.length)];
      return { choice: 'o' + pick, ranks: [{ key: 'o' + pick, label: keyOf(pick), score: 0, detail: '随机' }],
               note: '随机基线' };
    }

    // 1) 确定的雷 → 优先标记（标记能解锁后续推理）
    for (const j of mine) {
      if (!s.flag[j]) {
        ranks.push({ key: 'f' + j, label: 'flag ' + keyOf(j), score: 1000, detail: '单格规则：必是雷' });
      }
    }
    // 2) 确定的安全格
    for (const j of safe) {
      if (!s.open[j] && !s.flag[j]) {
        ranks.push({ key: 'o' + j, label: 'open ' + keyOf(j), score: 900, detail: '单格规则：必安全' });
      }
    }
    // 3) CSP 概率
    let note = `单格规则推出 ${safe.size} 个安全格 / ${mine.size} 个雷。`;
    if (mode === 'csp' && !ranks.length) {
      const csp = this._csp(s);
      if (csp) {
      // 边界格：概率升序，同概率优先角/边
      const arr = [...csp.border.entries()].sort((a, b) =>
        Math.abs(a[1] - b[1]) < 1e-6 ? this._corner(s, b[0]) - this._corner(s, a[0]) : a[1] - b[1]);
      for (const [j, p] of arr.slice(0, 8)) {
        ranks.push({ key: 'o' + j, label: 'open ' + keyOf(j), score: -p * 100, digits: 1,
                     detail: `边界格 · 是雷概率 ${(p * 100).toFixed(1)}%`, risky: p > 0.4 });
      }
      // 约束外（盲猜）格子：同样按角 > 边 > 内部
      const out = csp.outsideCells.slice().sort((a, b) => this._corner(s, b) - this._corner(s, a));
      for (const j of out.slice(0, 3)) {
        ranks.push({ key: 'o' + j, label: 'open ' + keyOf(j), score: -csp.outside * 100 + this._corner(s, j) * 0.5,
                     digits: 1, detail: `约束外${this._corner(s, j) === 3 ? '(角)' : this._corner(s, j) === 2 ? '(边)' : ''} · 是雷概率 ${(csp.outside * 100).toFixed(1)}%` });
      }
      note = `推不出确定结论：CSP 枚举 ${arr.length} 个边界格，最低雷概率 `
           + `${(arr[0] ? arr[0][1] * 100 : 0).toFixed(1)}%（约束外 ${(csp.outside * 100).toFixed(1)}%，盲猜优先角/边）。`;
      }
    }
    // 4) 兜底：随便开一个未知格
    if (!ranks.length) {
      const unk = [];
      for (let i = 0; i < s.total; i++) if (!s.open[i] && !s.flag[i]) unk.push(i);
      if (!unk.length) return { choice: null, ranks: [], note: '无未知格' };
      // 盲猜：先角，再边，再内部
      unk.sort((a, b) => this._corner(s, b) - this._corner(s, a));
      const best = this._corner(s, unk[0]);
      const pool = unk.filter(j => this._corner(s, j) === best);
      const pick = pool[Math.floor(Math.random() * pool.length)];
      ranks.push({ key: 'o' + pick, label: 'open ' + keyOf(pick), score: -50,
                   detail: `无信息，盲猜${best === 3 ? '角落' : best === 2 ? '边格' : '内部格'}` });
      note = `开局或边界为空：盲猜${best === 3 ? '角落' : best === 2 ? '边格' : '内部格'}（角/边更容易滚出 0 格连锁）。`;
    }
    ranks.sort((a, b) => b.score - a.score);
    const choice = ranks[0].key;
    if (legal.indexOf(choice) < 0) {
      // 规则首选不在候选集内时补进去
      ranks.unshift({ key: choice, label: choice, score: ranks[0].score, detail: ranks[0].detail });
    }
    return { choice, ranks: ranks.slice(0, 10), note };
  },

  metrics(s, sopt) {
    const flagged = [...s.flag].reduce((a, b) => a + b, 0);
    return [
      { k: '已开', v: `${s.opened}/${s.total - s.M}` },
      { k: '标记雷', v: `${flagged}/${s.M}` },
      { k: '步数', v: s.steps },
      { k: '剩余雷', v: s.M - flagged },
      { k: '结果', v: s.won ? '胜' : (s.alive ? '进行中' : '踩雷'),
        tone: s.won ? 'ok' : (s.alive ? '' : 'bad') },
      { k: '进度', v: `${Math.round(s.opened / (s.total - s.M) * 100)}%` }
    ];
  },

  render(s, el) {
    const N = s.N;
    let h = `<div style="display:grid;grid-template-columns:repeat(${N},1fr);gap:2px;`
          + `background:#dfe4ea;padding:3px;border-radius:6px;max-width:340px;margin:0 auto">`;
    const NC = ['', '#2f6fed', '#0d9488', '#d64545', '#7c3aed', '#c2740a', '#31c4be', '#1d2430', '#667085'];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const i = r * N + c;
      let bg = '#eef1f4', fg = '#1d2430', txt = '';
      if (s.flag[i]) { bg = '#fdecec'; txt = '⚑'; fg = '#d64545'; }
      else if (!s.open[i]) { bg = '#cdd5df'; }
      else if (s.count[i] > 0) { bg = '#fff'; txt = String(s.count[i]); fg = NC[s.count[i]]; }
      else { bg = '#fff'; }
      h += `<div style="background:${bg};color:${fg};aspect-ratio:1/1;border-radius:2px;`
         + `display:flex;align-items:center;justify-content:center;font-family:var(--mono);`
         + `font-weight:700;font-size:13px">${txt}</div>`;
    }
    h += `</div><div class="hint" style="text-align:center">⚑ = 已标记雷 · 数字 = 周围雷数 · 空白 = 未开</div>`;
    el.innerHTML = h;
  },

  done(s) { return !s.alive || s.won; },
  goalOk(s) { return s.won; },
  maxSteps(sopt) { const N = parseInt(sopt.size || '9', 10); return N * N; }
};
