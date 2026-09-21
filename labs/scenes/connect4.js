/* =============================================================================
 * 场景：Connect 4（四子棋）
 * 算法：Minimax + alpha-beta 剪枝，深度可调；评估函数统计所有 4 连窗口的
 *       己方/对方子数并给中心列加权
 * 目标：对随机对手胜率 ≥ 95%
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function makeBoard(W, H) { return new Int8Array(W * H); }
function dropRow(b, W, H, col) {
  for (let r = H - 1; r >= 0; r--) if (!b[r * W + col]) return r;
  return -1;
}
function winAt(b, W, H, r, c, mark) {
  const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (const [dr, dc] of dirs) {
    let n = 1;
    for (const sgn of [1, -1]) {
      let rr = r + dr * sgn, cc = c + dc * sgn;
      while (rr >= 0 && rr < H && cc >= 0 && cc < W && b[rr * W + cc] === mark) {
        n++; rr += dr * sgn; cc += dc * sgn;
      }
    }
    if (n >= 4) return true;
  }
  return false;
}

/* 评估：所有 4 连窗口打分 + 中心列加权 */
function evaluate(b, W, H, me) {
  const opp = 3 - me;
  let score = 0;
  const score4 = (cnt, ocnt) => {
    if (cnt && ocnt) return 0;
    if (cnt === 4) return 100000;
    if (cnt === 3) return 120;
    if (cnt === 2) return 12;
    if (cnt === 1) return 1;
    if (ocnt === 4) return -100000;
    if (ocnt === 3) return -150;
    if (ocnt === 2) return -15;
    if (ocnt === 1) return -1;
    return 0;
  };
  const scan = (cells) => {
    for (let i = 0; i + 3 < cells.length; i++) {
      let cnt = 0, ocnt = 0;
      for (let k = 0; k < 4; k++) {
        const v = cells[i + k];
        if (v === me) cnt++; else if (v === opp) ocnt++;
      }
      score += score4(cnt, ocnt);
    }
  };
  for (let r = 0; r < H; r++) {
    const row = [];
    for (let c = 0; c < W; c++) row.push(b[r * W + c]);
    scan(row);
  }
  for (let c = 0; c < W; c++) {
    const col = [];
    for (let r = 0; r < H; r++) col.push(b[r * W + c]);
    scan(col);
  }
  for (let r = 0; r < H - 3; r++) for (let c = 0; c < W - 3; c++) {
    scan([b[r * W + c], b[(r + 1) * W + c + 1], b[(r + 2) * W + c + 2], b[(r + 3) * W + c + 3]]);
  }
  for (let r = 3; r < H; r++) for (let c = 0; c < W - 3; c++) {
    scan([b[r * W + c], b[(r - 1) * W + c + 1], b[(r - 2) * W + c + 2], b[(r - 3) * W + c + 3]]);
  }
  const mid = (W - 1) / 2;
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const v = b[r * W + c];
    if (!v) continue;
    score += (v === me ? 1 : -1) * Math.round((2 - Math.abs(c - mid)) * 2);
  }
  return score;
}

/* Minimax + alpha-beta */
function search(b, W, H, depth, alpha, beta, maxing, me) {
  // 终局检测（只查最后落子不现实，这里查全盘四连）
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const v = b[r * W + c];
    if (!v) continue;
    if (winAt(b, W, H, r, c, v)) return v === me ? 1000000 - (10 - depth) : -1000000 + (10 - depth);
  }
  if (depth === 0) return evaluate(b, W, H, me);
  let moved = false;
  for (let col = 0; col < W; col++) {
    const row = dropRow(b, W, H, col);
    if (row < 0) continue;
    moved = true;
    b[row * W + col] = maxing ? me : 3 - me;
    const v = search(b, W, H, depth - 1, alpha, beta, !maxing, me);
    b[row * W + col] = 0;
    if (maxing) { if (v > alpha) alpha = v; if (alpha >= beta) return alpha; }
    else { if (v < beta) beta = v; if (alpha >= beta) return beta; }
  }
  if (!moved) return 0;   // 棋盘满
  return maxing ? alpha : beta;
}

const SCENE = {
  id: 'connect4',
  name: 'Connect 4',
  sub: '7×6 · Minimax + alpha-beta 剪枝',
  goal: '目标：对随机对手胜率 ≥ 95%',
  hint: 'Minimax 配 alpha-beta 剪枝，评估函数统计所有 4 连窗口的己方/对方子数，'
      + '并对中心列加权（中心列参与的四连最多）。对手为随机落子。',

  controls: [
    { id: 'depth', label: '搜索深度', type: 'select', value: '5', options: [
      { v: '3', t: '3 层（快）' }, { v: '5', t: '5 层' }, { v: '7', t: '7 层（慢）' }] },
    { id: 'aiFirst', label: 'AI 先手', type: 'select', value: '1', options: [
      { v: '1', t: 'AI 先手' }, { v: '0', t: 'AI 后手' }] }
  ],

  init(seed, sopt) {
    const W = 7, H = 6;
    const s = { W, H, b: makeBoard(W, H), steps: 0, result: null,
                ai: sopt.aiFirst === '0' ? 2 : 1,
                rng: mulberry32((seed * 2654435761) >>> 0) };
    if (s.ai === 2) {
      const cols = [];
      for (let c = 0; c < W; c++) if (dropRow(s.b, W, H, c) >= 0) cols.push(c);
      const c = cols[Math.floor(s.rng() * cols.length)];
      s.b[dropRow(s.b, W, H, c) * W + c] = 1;
    }
    return s;
  },

  legal(s) {
    if (s.result) return [];
    const out = [];
    for (let c = 0; c < s.W; c++) if (dropRow(s.b, s.W, s.H, c) >= 0) out.push('col' + c);
    return out;
  },

  _place(s, col, mark) {
    const row = dropRow(s.b, s.W, s.H, col);
    if (row < 0) return -1;
    s.b[row * s.W + col] = mark;
    return row;
  },

  _check(s, row, col, mark) {
    if (winAt(s.b, s.W, s.H, row, col, mark)) {
      s.result = mark === s.ai ? 'win' : 'lose';
      return true;
    }
    if (this.legal(s).length === 0) { s.result = 'draw'; return true; }
    return false;
  },

  step(s, key) {
    const col = parseInt(key.slice(3), 10);
    const row = this._place(s, col, s.ai);
    if (row < 0) return { info: '该列已满' };
    s.steps++;
    let info = `AI 落第 ${col + 1} 列`;
    if (this._check(s, row, col, s.ai)) return { info: info + ` · ${s.result}` };
    // 随机对手
    const cols = this.legal(s);
    if (cols.length) {
      const oc = parseInt(cols[Math.floor(s.rng() * cols.length)].slice(3), 10);
      const orow = this._place(s, oc, 3 - s.ai);
      info += ` · 对手 ${oc + 1} 列`;
      if (orow >= 0) this._check(s, orow, oc, 3 - s.ai);
    }
    if (s.result) info += ` · ${s.result}`;
    return { info };
  },

  text(s, legal) {
    const rows = [];
    for (let r = 0; r < s.H; r++) {
      let line = '';
      for (let c = 0; c < s.W; c++) {
        const v = s.b[r * s.W + c];
        line += v === 0 ? '.' : (v === s.ai ? 'X' : 'O');
      }
      rows.push(line);
    }
    return `Connect Four ${s.W} wide x ${s.H} high. AI plays ${s.ai === 1 ? 'X (first)' : 'O (second)'}.\n`
      + `columns are 0-indexed from left; a piece falls to the lowest empty row.\n`
      + `playable_columns: ${legal.map(k => k.slice(3)).join(',')}\n`
      + `board (X = AI, O = opponent, . = empty), rows top to bottom:\n` + rows.join('\n')
      + `\nGoal: get four in a row (horizontal, vertical or diagonal).`;
  },

  questions(s, legal) {
    const crit = {};
    for (const k of legal) {
      const c = parseInt(k.slice(3), 10);
      crit[k] = `drop a piece into column ${c}`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Which column should the AI drop its piece into to win?',
        criteria: crit },
      { id: 'edge', type: 'score',
        instructions: 'How strong is the AI position?',
        criteria: ['losing', 'equal', 'advantage', 'winning'] }
    ];
  },

  rule(s, legal, sopt) {
    const depth = parseInt(sopt.depth || '5', 10);
    const me = s.ai;
    const ranks = [];
    for (const k of legal) {
      const c = parseInt(k.slice(3), 10);
      const row = dropRow(s.b, s.W, s.H, c);
      s.b[row * s.W + c] = me;
      let v;
      if (winAt(s.b, s.W, s.H, row, c, me)) v = 999999;
      else v = search(s.b, s.W, s.H, depth - 1, -Infinity, Infinity, false, me);
      s.b[row * s.W + c] = 0;
      ranks.push({ key: k, label: `第 ${c + 1} 列`, score: v, digits: 0,
                   detail: v >= 999999 ? '立即四连获胜' : `α-β 估值 ${v}`,
                   risky: v < -500 });
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: ranks[0].key, ranks,
             note: `Minimax + α-β，深度 ${depth}：最优列估值 ${ranks[0].score}。` };
  },

  metrics(s, sopt) {
    let mine = 0, theirs = 0;
    for (let i = 0; i < s.b.length; i++) {
      if (s.b[i] === s.ai) mine++;
      else if (s.b[i]) theirs++;
    }
    return [
      { k: '回合', v: s.steps },
      { k: '结果', v: s.result === 'win' ? '胜' : s.result === 'draw' ? '和' : s.result === 'lose' ? '负' : '进行中',
        tone: s.result === 'win' ? 'ok' : s.result === 'lose' ? 'bad' : '' },
      { k: 'AI 子', v: mine },
      { k: '对手子', v: theirs },
      { k: '估值', v: evaluate(s.b, s.W, s.H, s.ai), digits: 0,
        tone: evaluate(s.b, s.W, s.H, s.ai) > 0 ? 'ok' : '' },
      { k: '可落列', v: this.legal(s).length }
    ];
  },

  render(s, el) {
    let h = `<div style="display:grid;grid-template-columns:repeat(${s.W},1fr);gap:4px;`
          + `background:#2f6fed;padding:5px;border-radius:8px;max-width:340px;margin:0 auto">`;
    for (let r = 0; r < s.H; r++) for (let c = 0; c < s.W; c++) {
      const v = s.b[r * s.W + c];
      const bg = v === 0 ? '#f7f8fa' : (v === s.ai ? '#31c4be' : '#e8590c');
      h += `<div style="background:${bg};aspect-ratio:1/1;border-radius:50%;`
         + `box-shadow:inset 0 -2px 3px rgba(0,0,0,.12)"></div>`;
    }
    h += `</div><div class="hint" style="text-align:center">青 = AI(${s.ai === 1 ? '先' : '后'}) · 橙 = 随机对手</div>`;
    el.innerHTML = h;
  },

  done(s) { return !!s.result; },
  goalOk(s) { return s.result === 'win'; },
  maxSteps() { return 42; }
};
