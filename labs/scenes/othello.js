/* =============================================================================
 * 场景：黑白棋 Othello / Reversi
 * 算法：Minimax + alpha-beta，评估 = 位置权重 + 行动力差 + 角与稳定子
 * 目标：对随机对手胜率 ≥ 90%
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const N = 8;
const DIR8 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
/* 角最值钱、角旁的格子最危险 —— 经典位置权重表 */
const WEIGHT = [
  120, -20, 20, 5, 5, 20, -20, 120,
  -20, -40, -5, -5, -5, -5, -40, -20,
  20, -5, 15, 3, 3, 15, -5, 20,
  5, -5, 3, 3, 3, 3, -5, 5,
  5, -5, 3, 3, 3, 3, -5, 5,
  20, -5, 15, 3, 3, 15, -5, 20,
  -20, -40, -5, -5, -5, -5, -40, -20,
  120, -20, 20, 5, 5, 20, -20, 120
];

function initBoard() {
  const b = new Int8Array(64);
  b[3 * 8 + 3] = 2; b[4 * 8 + 4] = 2;
  b[3 * 8 + 4] = 1; b[4 * 8 + 3] = 1;
  return b;
}

/* 落子后能翻转的格子；无子可翻则返回空 */
function flipsFor(b, idx, mark) {
  if (b[idx]) return [];
  const r = Math.floor(idx / 8), c = idx % 8;
  const out = [];
  for (const [dr, dc] of DIR8) {
    const line = [];
    let rr = r + dr, cc = c + dc;
    while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && b[rr * 8 + cc] === 3 - mark) {
      line.push(rr * 8 + cc); rr += dr; cc += dc;
    }
    if (line.length && rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && b[rr * 8 + cc] === mark) {
      for (const i of line) out.push(i);
    }
  }
  return out;
}

function legalMoves(b, mark) {
  const out = [];
  for (let i = 0; i < 64; i++) if (flipsFor(b, i, mark).length) out.push(i);
  return out;
}

function applyMove(b, idx, mark) {
  const f = flipsFor(b, idx, mark);
  if (!f.length) return false;
  b[idx] = mark;
  for (const i of f) b[i] = mark;
  return true;
}

function evaluate(b, me) {
  const opp = 3 - me;
  let sc = 0, mine = 0, theirs = 0;
  for (let i = 0; i < 64; i++) {
    if (b[i] === me) { sc += WEIGHT[i]; mine++; }
    else if (b[i] === opp) { sc -= WEIGHT[i]; theirs++; }
  }
  const myMob = legalMoves(b, me).length, opMob = legalMoves(b, opp).length;
  sc += (myMob - opMob) * 12;                       // 行动力
  if (mine + theirs >= 50) sc += (mine - theirs) * 8; // 终局盘子数更重要
  return sc;
}

function search(b, depth, alpha, beta, maxing, me, passed) {
  const myMoves = legalMoves(b, me).length;
  const opMoves = legalMoves(b, 3 - me).length;
  if (!myMoves && !opMoves) {
    let mine = 0, theirs = 0;
    for (let i = 0; i < 64; i++) { if (b[i] === me) mine++; else if (b[i]) theirs++; }
    return mine > theirs ? 100000 : mine < theirs ? -100000 : 0;
  }
  if (depth === 0) return evaluate(b, me);
  const turn = maxing ? me : 3 - me;
  const moves = legalMoves(b, turn);
  if (!moves.length) {
    // 该方无子可下 → 直接轮空，不消耗深度（避免深度被 pass 吃掉）
    if (passed) return evaluate(b, me);
    return search(b, depth, alpha, beta, !maxing, me, true);
  }
  let best = maxing ? -Infinity : Infinity;
  for (const mv of moves) {
    const nb = b.slice();
    applyMove(nb, mv, turn);
    const v = search(nb, depth - 1, alpha, beta, !maxing, me, false);
    if (maxing) { if (v > best) best = v; if (best > alpha) alpha = best; }
    else { if (v < best) best = v; if (best < beta) beta = best; }
    if (alpha >= beta) break;
  }
  return best;
}

const SCENE = {
  id: 'othello',
  name: '黑白棋',
  sub: '8×8 · Minimax + α-β · 位置权重 / 行动力 / 终局子数',
  goal: '目标：对随机对手胜率 ≥ 90%',
  hint: '评估 = 位置权重表（角 120、角旁 -40）+ 行动力差 ×12；'
      + '终局（≥50 子后）子数差权重升到 ×8。对手为随机合法落子。',

  controls: [
    { id: 'depth', label: '搜索深度', type: 'select', value: '4', options: [
      { v: '2', t: '2 层（快）' }, { v: '4', t: '4 层' }, { v: '6', t: '6 层（慢）' }] },
    { id: 'opponent', label: '对手', type: 'select', value: 'random', options: [
      { v: 'random', t: '随机对手' }, { v: 'greedy', t: '贪心（每次翻最多）' }] }
  ],

  init(seed, sopt) {
    const s = { b: initBoard(), ai: 1, turn: 1, steps: 0, result: null,
                rng: mulberry32((seed * 2654435761) >>> 0), opp: sopt.opponent || 'random',
                passed: false };
    return s;
  },

  legal(s) {
    if (s.result) return [];
    return legalMoves(s.b, s.ai).map(i => 'c' + i);
  },

  _count(s) {
    let a = 0, b = 0;
    for (let i = 0; i < 64; i++) { if (s.b[i] === s.ai) a++; else if (s.b[i]) b++; }
    return { me: a, opp: b, empty: 64 - a - b };
  },

  _finish(s) {
    if (!legalMoves(s.b, 1).length && !legalMoves(s.b, 2).length) {
      const c = this._count(s);
      s.result = c.me > c.opp ? 'win' : c.me < c.opp ? 'lose' : 'draw';
    }
  },

  _oppMove(s) {
    const moves = legalMoves(s.b, 3 - s.ai);
    if (!moves.length) return -1;
    if (s.opp === 'greedy') {
      let best = -1, bn = -1;
      for (const mv of moves) {
        const n = flipsFor(s.b, mv, 3 - s.ai).length;
        if (n > bn) { bn = n; best = mv; }
      }
      return best;
    }
    return moves[Math.floor(s.rng() * moves.length)];
  },

  step(s, key) {
    const i = parseInt(key.slice(1), 10);
    if (!applyMove(s.b, i, s.ai)) return { info: '非法落子' };
    s.steps++;
    let info = `AI 落 ${i}（翻 ${flipsFor(s.b, i, s.ai).length}）`;
    this._finish(s);
    if (s.result) return { info: info + ` · ${s.result}` };
    let guard = 0;
    let om = this._oppMove(s);
    while (om < 0 && guard++ < 2) {      // 对手无子可下 → AI 继续
      const mine = legalMoves(s.b, s.ai);
      if (!mine.length) break;
      const pick = mine[Math.floor(s.rng() * mine.length)];
      applyMove(s.b, pick, s.ai);
      info += ` · 对手停一手`;
      this._finish(s);
      if (s.result) return { info: info + ` · ${s.result}` };
      om = this._oppMove(s);
    }
    if (om >= 0) { applyMove(s.b, om, 3 - s.ai); info += ` · 对手 ${om}`; }
    this._finish(s);
    if (s.result) info += ` · ${s.result}`;
    return { info };
  },

  text(s, legal) {
    const rows = [];
    for (let r = 0; r < 8; r++) {
      let line = '';
      for (let c = 0; c < 8; c++) {
        const v = s.b[r * 8 + c];
        line += v === 0 ? '.' : (v === s.ai ? 'X' : 'O');
      }
      rows.push(line);
    }
    const cnt = this._count(s);
    return `Othello 8x8. AI plays X, opponent plays O.\n`
      + `discs: X=${cnt.me} O=${cnt.opp} empty=${cnt.empty}. move=${s.steps}.\n`
      + `legal_moves for X: ${legal.map(k => k.slice(1)).join(',')}\n`
      + `board:\n` + rows.join('\n')
      + `\ncell index = row*8 + col. Corners are worth the most.`;
  },

  questions(s, legal) {
    const crit = {};
    for (const k of legal.slice(0, 16)) {
      const i = parseInt(k.slice(1), 10);
      crit[k] = `place a disc at row ${Math.floor(i / 8)}, column ${i % 8}`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the move that maximizes the AI final disc count.',
        criteria: crit },
      { id: 'edge', type: 'score',
        instructions: 'How favorable is the AI position?',
        criteria: ['losing', 'slightly behind', 'slightly ahead', 'winning'] }
    ];
  },

  rule(s, legal, sopt) {
    const depth = parseInt(sopt.depth || '4', 10);
    const me = s.ai;
    const ranks = [];
    for (const k of legal.slice(0, 16)) {
      const i = parseInt(k.slice(1), 10);
      const nb = s.b.slice();
      applyMove(nb, i, me);
      const v = search(nb, depth - 1, -Infinity, Infinity, false, me, false);
      ranks.push({ key: k, label: `(${Math.floor(i / 8)},${i % 8})`, score: v, digits: 0,
                   detail: `翻 ${flipsFor(s.b, i, me).length} 子 · α-β 估值 ${v}` });
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: ranks[0].key, ranks,
             note: `Minimax + α-β 深度 ${depth}：最优估值 ${ranks[0].score}（共 ${legal.length} 个合法点）。` };
  },

  metrics(s, sopt) {
    const c = this._count(s);
    return [
      { k: '回合', v: s.steps },
      { k: 'AI 子', v: c.me, tone: c.me > c.opp ? 'ok' : '' },
      { k: '对手子', v: c.opp, tone: c.opp > c.me ? 'bad' : '' },
      { k: '子差', v: c.me - c.opp, tone: c.me > c.opp ? 'ok' : c.me < c.opp ? 'bad' : '' },
      { k: '空格', v: c.empty },
      { k: '结果', v: s.result === 'win' ? '胜' : s.result === 'draw' ? '和' : s.result === 'lose' ? '负' : '进行中',
        tone: s.result === 'win' ? 'ok' : s.result === 'lose' ? 'bad' : '' }
    ];
  },

  render(s, el) {
    let h = `<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:3px;`
          + `background:#1f6b4f;padding:4px;border-radius:6px;max-width:340px;margin:0 auto">`;
    for (let i = 0; i < 64; i++) {
      const v = s.b[i];
      const bg = v === 0 ? '#2e8b63' : (v === s.ai ? '#101820' : '#f5f6f8');
      h += `<div style="background:${bg};aspect-ratio:1/1;border-radius:50%;`
         + `box-shadow:inset 0 -2px 3px rgba(0,0,0,.15)"></div>`;
    }
    h += `</div><div class="hint" style="text-align:center">黑 = AI · 白 = 随机对手</div>`;
    el.innerHTML = h;
  },

  done(s) { return !!s.result; },
  goalOk(s) { return s.result === 'win'; },
  maxSteps() { return 64; }
};
