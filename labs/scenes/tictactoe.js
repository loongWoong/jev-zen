/* =============================================================================
 * 场景：井字棋 Tic-Tac-Toe
 * 算法：完整 Minimax（9! 极小，直接搜到底；评分带深度，优先快赢、慢输）
 * 目标：对随机对手 100% 不败
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

function winnerOf(b) {
  for (const L of LINES) {
    const [x, y, z] = L;
    if (b[x] && b[x] === b[y] && b[y] === b[z]) return b[x];
  }
  return 0;
}
function isFull(b) { return b.every(v => v); }

/* Minimax：返回「me 视角」的分数，深度参与评分以便优先快赢 */
function mm(b, turn, me, depth) {
  const w = winnerOf(b);
  if (w) return w === me ? 10 - depth : depth - 10;
  if (isFull(b)) return 0;
  let best = turn === me ? -Infinity : Infinity;
  for (let i = 0; i < 9; i++) {
    if (b[i]) continue;
    b[i] = turn;
    const v = mm(b, 3 - turn, me, depth + 1);
    b[i] = 0;
    best = turn === me ? Math.max(best, v) : Math.min(best, v);
  }
  return best;
}

const SCENE = {
  id: 'tictactoe',
  name: '井字棋',
  sub: '3×3 · 完整 Minimax（搜到底，评分带深度）',
  goal: '目标：对随机对手 100% 不败',
  hint: '棋盘只有 9! 个状态，直接搜到终局；评分用 10-depth，因此同样能赢时选最快的、'
      + '必输时拖到最慢。对手可选随机（应 100% 不败）或同为 Minimax（应 100% 和棋）。',

  controls: [
    { id: 'opponent', label: '对手', type: 'select', value: 'random', options: [
      { v: 'random', t: '随机对手' }, { v: 'minimax', t: 'Minimax 对手（完美）' }] },
    { id: 'aiFirst', label: 'AI 先手', type: 'select', value: '1', options: [
      { v: '1', t: 'AI 先手（X）' }, { v: '0', t: 'AI 后手（O）' }] }
  ],

  init(seed, sopt) {
    const s = { b: new Array(9).fill(0), steps: 0, result: null,
                ai: sopt.aiFirst === '0' ? 2 : 1, turn: 1,
                rng: mulberry32((seed * 2654435761) >>> 0), opp: sopt.opponent || 'random' };
    // AI 后手时，对手先走一步
    if (s.ai === 2) {
      const i = s.opp === 'minimax' ? this._bestMove(s.b, 1) : Math.floor(s.rng() * 9);
      s.b[i] = 1;
    }
    return s;
  },

  _bestMove(b, mark) {
    let bs = -Infinity, bi = -1;
    for (let i = 0; i < 9; i++) {
      if (b[i]) continue;
      b[i] = mark;
      const v = mm(b, 3 - mark, mark, 1);
      b[i] = 0;
      if (v > bs) { bs = v; bi = i; }
    }
    return bi;
  },

  legal(s) {
    if (s.result) return [];
    const out = [];
    for (let i = 0; i < 9; i++) if (!s.b[i]) out.push('c' + i);
    return out;
  },

  _finish(s) {
    const w = winnerOf(s.b);
    if (w) s.result = w === s.ai ? 'win' : 'lose';
    else if (isFull(s.b)) s.result = 'draw';
  },

  step(s, key) {
    const i = parseInt(key.slice(1), 10);
    if (s.b[i]) return { info: '该格已占' };
    s.b[i] = s.ai;
    s.steps++;
    this._finish(s);
    let info = `AI 落子 ${i}`;
    if (s.result) return { info: info + ` · ${s.result}` };
    // 对手应手
    const oppMark = 3 - s.ai;
    let oi;
    if (s.opp === 'minimax') oi = this._bestMove(s.b, oppMark);
    else {
      const empty = [];
      for (let k = 0; k < 9; k++) if (!s.b[k]) empty.push(k);
      oi = empty[Math.floor(s.rng() * empty.length)];
    }
    if (oi >= 0) { s.b[oi] = oppMark; this._finish(s); info += ` · 对手 ${oi}`; }
    if (s.result) info += ` · ${s.result}`;
    return { info };
  },

  text(s, legal) {
    const rows = [];
    for (let r = 0; r < 3; r++) {
      const cells = [];
      for (let c = 0; c < 3; c++) {
        const v = s.b[r * 3 + c];
        cells.push(v === 0 ? '.' : (v === s.ai ? 'X' : 'O'));
      }
      rows.push(cells.join(' '));
    }
    return `Tic-tac-toe 3x3. AI plays ${s.ai === 1 ? 'X (first)' : 'O (second)'}, `
      + `opponent plays ${s.ai === 1 ? 'O' : 'X'} and moves ${s.opp}.\n`
      + `empty_cells: ${legal.join(' ')}\nboard:\n` + rows.join('\n')
      + `\ncell index = row*3 + col. Goal: never lose.`;
  },

  questions(s, legal) {
    const crit = {};
    for (const k of legal) {
      const i = parseInt(k.slice(1), 10);
      crit[k] = `place mark at row ${Math.floor(i / 3)}, column ${i % 3}`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the move that guarantees the AI never loses.',
        criteria: crit },
      { id: 'threat', type: 'noul',
        instructions: 'Does the opponent currently have an immediate winning threat?' }
    ];
  },

  rule(s, legal, sopt) {
    const ranks = [];
    const me = s.ai;
    for (const k of legal) {
      const i = parseInt(k.slice(1), 10);
      s.b[i] = me;
      const v = mm(s.b, 3 - me, me, 1);
      s.b[i] = 0;
      const tag = v >= 9 ? '立即获胜' : v > 0 ? '可赢' : v === 0 ? '和棋' : v <= -9 ? '立即落败' : '劣势';
      ranks.push({ key: k, label: `(${Math.floor(i / 3)},${i % 3})`, score: v, digits: 0,
                   detail: `${tag}（minimax ${v}）`, risky: v < 0 });
    }
    ranks.sort((a, b) => b.score - a.score);
    const v0 = ranks[0].score;
    return { choice: ranks[0].key, ranks,
             note: `Minimax 搜到终局：最佳分 ${v0}（${v0 >= 9 ? '必胜' : v0 === 0 ? '和棋' : v0 > 0 ? '可赢' : '劣势'}）。` };
  },

  metrics(s, sopt) {
    const me = s.ai;
    return [
      { k: '回合', v: s.steps },
      { k: '结果', v: s.result === 'win' ? '胜' : s.result === 'draw' ? '和' : s.result === 'lose' ? '负' : '进行中',
        tone: s.result === 'win' ? 'ok' : s.result === 'lose' ? 'bad' : '' },
      { k: '执子', v: me === 1 ? 'X 先手' : 'O 后手' },
      { k: '对手', v: s.opp === 'random' ? '随机' : 'Minimax' },
      { k: '空格', v: s.b.filter(v => !v).length },
      { k: '不败', v: s.result ? (s.result === 'lose' ? '✗ 已负' : '✓') : '—',
        tone: s.result === 'lose' ? 'bad' : s.result ? 'ok' : '' }
    ];
  },

  render(s, el) {
    let h = `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:5px;`
          + `background:#dfe4ea;padding:5px;border-radius:8px;max-width:220px;margin:0 auto">`;
    for (let i = 0; i < 9; i++) {
      const v = s.b[i];
      const col = v === 0 ? '#fff' : (v === s.ai ? '#0d9488' : '#d64545');
      h += `<div style="background:${col};color:${v ? '#fff' : '#98a2b3'};aspect-ratio:1/1;`
         + `border-radius:6px;display:flex;align-items:center;justify-content:center;`
         + `font-family:var(--mono);font-weight:700;font-size:22px">${v === 0 ? '' : (v === s.ai ? 'X' : 'O')}</div>`;
    }
    h += `</div><div class="hint" style="text-align:center">绿 = AI(${s.ai === 1 ? 'X' : 'O'}) · 红 = 对手</div>`;
    el.innerHTML = h;
  },

  done(s) { return !!s.result; },
  goalOk(s) { return !!s.result && s.result !== 'lose'; },
  maxSteps() { return 9; }
};
