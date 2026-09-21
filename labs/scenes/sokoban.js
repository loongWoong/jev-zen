/* =============================================================================
 * 场景：推箱子 Sokoban（自建 20 关，由 A* 求解器验证可解）
 * 算法：A*（曼哈顿下界 + 角落死锁剪枝），每步求解一次并缓存计划
 * 目标：20 关全部解出
 * 字符：# 墙  @ 玩家  $ 箱子  . 目标  * 箱在目标  + 玩家在目标  空格
 * ===========================================================================*/

/* 自建关卡：每一关都由本文件的 A* 求解器验证过可解（箱数必等于目标数），
 * 按最优解步数递增排列：1 步 → 13 步。挑选时已筛掉推入死角必然无解的布局。 */
const LEVELS = [
  ['#####', '#@$.#', '#####'],                                            //  1 步
  ['######', '#@$ .#', '######'],                                         //  2 步
  ['######', '#.$  #', '#  @ #', '######'],                               //  2 步
  ['######', '#    #', '# $. #', '# @  #', '######'],                     //  3 步
  ['######', '#    #', '# .$ #', '#  @ #', '######'],                     //  3 步
  ['#######', '#. $  #', '#  @  #', '#######'],                           //  4 步
  ['#######', '#@ $  .#', '#######'],                                     //  4 步
  ['########', '#      #', '#  .$  #', '#  @   #', '########'],           //  4 步
  ['#######', '#  .   #', '#  $   #', '# .$   #', '#  @   #', '#######'], //  4 步
  ['########', '#@  $  .#', '########'],                                  //  5 步
  ['########', '#   .  #', '#  $   #', '# @    #', '########'],           //  5 步
  ['########', '#  .   #', '#   $  #', '#  @   #', '########'],           //  5 步
  ['########', '#  . . #', '#  $$  #', '#  @   #', '########'],           //  5 步
  ['#######', '#  . #', '# $@  #', '#     #', '#######'],                 //  6 步
  ['########', '# . .  #', '# $$   #', '#  @   #', '########'],           //  6 步
  ['#######', '#  ###', '# $@ #', '# .  #', '#######'],                   //  7 步
  ['#######', '#.  #  ', '#  $   #', '# @    #', '#######'],              //  8 步
  ['#######', '#  .  #', '# $   #', '#   $ #', '# . @ #', '#######'],     // 11 步
  ['########', '#  . . #', '# $$   #', '#   @  #', '########'],           // 12 步
  ['#######', '#  .  #', '# $$@ #', '#  .  #', '#######']                 // 13 步
];

const DIRS4 = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
const DIRCN = { up: '上', down: '下', left: '左', right: '右' };

function parseLevel(rows) {
  const H = rows.length, W = Math.max(...rows.map(r => r.length));
  const wall = [], goal = [], box = [];
  let player = -1;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      // 行长短不齐时，缺的部分按墙处理，避免「走出棋盘」的意外通路
      const ch = c < rows[r].length ? rows[r][c] : '#';
      const i = r * W + c;
      wall.push(ch === '#');
      goal.push(ch === '.' || ch === '*' || ch === '+');
      box.push(ch === '$' || ch === '*');
      if (ch === '@' || ch === '+') player = i;
    }
  }
  return { W, H, wall, goal, box0: box, player, nBox: box.filter(Boolean).length,
           nGoal: goal.filter(Boolean).length, rows };
}

/* A* 求解：状态 = 玩家位置 + 箱子位置集合 */
function solveLevel(L, nodeLimit) {
  const W = L.W, H = L.H;
  const isWall = i => L.wall[i];
  const rc = i => [Math.floor(i / W), i % W];
  const mv = (i, d) => {
    const [r, c] = rc(i);
    const nr = r + DIRS4[d][0], nc = c + DIRS4[d][1];
    if (nr < 0 || nc < 0 || nr >= H || nc >= W) return -1;
    return nr * W + nc;
  };
  const goals = [];
  for (let i = 0; i < L.goal.length; i++) if (L.goal[i]) goals.push(i);
  const h = (boxes) => {
    let s = 0;
    for (const b of boxes) {
      const [br, bc] = rc(b);
      let m = Infinity;
      for (const g of goals) {
        const [gr, gc] = rc(g);
        m = Math.min(m, Math.abs(br - gr) + Math.abs(bc - gc));
      }
      s += m;
    }
    return s;
  };
  const deadCorner = (i) => {
    if (L.goal[i]) return false;
    const upW = mv(i, 'up') < 0 || isWall(mv(i, 'up'));
    const dnW = mv(i, 'down') < 0 || isWall(mv(i, 'down'));
    const lfW = mv(i, 'left') < 0 || isWall(mv(i, 'left'));
    const rtW = mv(i, 'right') < 0 || isWall(mv(i, 'right'));
    return (upW || dnW) && (lfW || rtW);
  };
  const start = { p: L.player, b: L.box0.map((v, i) => v ? i : -1).filter(i => i >= 0) };
  const key = st => st.p + '|' + st.b.join(',');
  const done = st => st.b.every(b => L.goal[b]);
  const open = [{ st: start, g: 0, f: h(start.b) }];
  const gS = new Map([[key(start), 0]]);
  const prev = new Map();
  let nodes = 0;
  const limit = nodeLimit || 200000;
  while (open.length && nodes < limit) {
    // 取 f 最小（数组规模小，线性扫描足够）
    let bi = 0;
    for (let i = 1; i < open.length; i++) if (open[i].f < open[bi].f) bi = i;
    const cur = open.splice(bi, 1)[0];
    nodes++;
    if (done(cur.st)) {
      const path = [];
      let k = key(cur.st);
      while (prev.has(k)) { const p = prev.get(k); path.push(p.d); k = p.pk; }
      return { ok: true, path: path.reverse(), nodes };
    }
    for (const d of ['up', 'down', 'left', 'right']) {
      const np = mv(cur.st.p, d);
      if (np < 0 || isWall(np)) continue;
      let nb = cur.st.b;
      const bi2 = nb.indexOf(np);
      if (bi2 >= 0) {
        const nn = mv(np, d);
        if (nn < 0 || isWall(nn) || nb.indexOf(nn) >= 0) continue;
        if (deadCorner(nn)) continue;
        nb = nb.slice(); nb[bi2] = nn; nb.sort((a, b) => a - b);
      }
      const ns = { p: np, b: nb };
      const kk = key(ns);
      const ng = cur.g + 1;
      if (gS.has(kk) && gS.get(kk) <= ng) continue;
      gS.set(kk, ng);
      prev.set(kk, { pk: key(cur.st), d });
      open.push({ st: ns, g: ng, f: ng + h(nb) });
    }
  }
  return { ok: false, nodes };
}

const SCENE = {
  id: 'sokoban',
  name: '推箱子',
  sub: '自建 20 关 · A* 状态空间搜索 + 角落死锁剪枝',
  goal: '目标：20 关全部解出',
  hint: '状态 =（玩家位置 + 全部箱子位置），A* 用「箱子到最近目标的曼哈顿距离之和」作下界，'
      + '并把推进死角的分支直接剪掉。每关开局求解一次得到计划，之后逐步执行。',

  controls: [
    { id: 'level', label: '关卡', type: 'number', value: 1, min: 1, max: 20, step: 1 },
    { id: 'nodeLimit', label: '搜索节点上限', type: 'select', value: '200000', options: [
      { v: '20000', t: '2 万（快）' }, { v: '200000', t: '20 万' }, { v: '800000', t: '80 万（慢）' }] }
  ],

  init(seed, sopt) {
    const lv = parseInt(sopt.level, 10) || 1;
    const idx = (lv - 1) % LEVELS.length;
    const L = parseLevel(LEVELS[idx]);
    const s = { L, idx, lv: idx + 1, player: L.player, boxes: [], steps: 0, pushes: 0,
                plan: null, solvedInfo: null };
    for (let i = 0; i < L.box0.length; i++) if (L.box0[i]) s.boxes.push(i);
    s.boxes.sort((a, b) => a - b);
    this._plan(s, sopt);
    return s;
  },

  _plan(s, sopt) {
    const L = s.L;
    const probe = { W: L.W, H: L.H, wall: L.wall, goal: L.goal, box0: null, player: s.player };
    const boxSet = new Array(L.W * L.H).fill(false);
    s.boxes.forEach(b => boxSet[b] = true);
    probe.box0 = boxSet;
    const r = solveLevel(probe, parseInt(sopt.nodeLimit || '200000', 10));
    s.solvedInfo = r;
    s.plan = r.ok ? r.path.slice() : null;
    return r;
  },

  _mv(s, i, d) {
    const L = s.L, W = L.W;
    const r = Math.floor(i / W) + DIRS4[d][0], c = i % W + DIRS4[d][1];
    if (r < 0 || c < 0 || r >= L.H || c >= W) return -1;
    return r * W + c;
  },

  legal(s) {
    const out = [];
    for (const d of ['up', 'down', 'left', 'right']) {
      const np = this._mv(s, s.player, d);
      if (np < 0 || s.L.wall[np]) continue;
      const bi = s.boxes.indexOf(np);
      if (bi >= 0) {
        const nn = this._mv(s, np, d);
        if (nn < 0 || s.L.wall[nn] || s.boxes.indexOf(nn) >= 0) continue;
      }
      out.push(d);
    }
    return out;
  },

  step(s, d) {
    const np = this._mv(s, s.player, d);
    const bi = s.boxes.indexOf(np);
    let pushed = false;
    if (bi >= 0) {
      const nn = this._mv(s, np, d);
      s.boxes[bi] = nn;
      s.boxes.sort((a, b) => a - b);
      pushed = true; s.pushes++;
    }
    s.player = np;
    s.steps++;
    if (s.plan && s.plan.length) s.plan.shift();
    return { info: `${DIRCN[d]}${pushed ? ' · 推动箱子' : ''} · 已就位 ${this._fitted(s)}/${s.boxes.length}` };
  },

  _fitted(s) { return s.boxes.filter(b => s.L.goal[b]).length; },

  text(s, legal) {
    const L = s.L, W = L.W;
    const grid = [];
    for (let r = 0; r < L.H; r++) {
      let line = '';
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        if (L.wall[i]) line += '#';
        else {
          const onGoal = L.goal[i], isBox = s.boxes.indexOf(i) >= 0, isP = s.player === i;
          line += isP ? (onGoal ? '+' : '@') : isBox ? (onGoal ? '*' : '$') : onGoal ? '.' : ' ';
        }
      }
      grid.push(line);
    }
    return `Sokoban level ${s.lv}. size ${W}x${L.H}. boxes=${s.boxes.length} `
      + `on_goal=${this._fitted(s)}. steps=${s.steps} pushes=${s.pushes}.\n`
      + `player at row ${Math.floor(s.player / W)}, col ${s.player % W}.\n`
      + `legal_moves: ${legal.join(', ')}.\n`
      + `board (# wall, @ player, $ box, . goal, * box on goal, + player on goal):\n`
      + grid.join('\n');
  },

  questions(s, legal) {
    return [
      { id: 'move', type: 'choice',
        instructions: 'Choose the move that pushes the boxes onto the goal squares.',
        criteria: {
          up: 'move or push up (row - 1)',
          down: 'move or push down (row + 1)',
          left: 'move or push left (column - 1)',
          right: 'move or push right (column + 1)'
        } },
      { id: 'progress', type: 'score',
        instructions: 'How close is this level to being solved?',
        criteria: ['no box on goal', 'some boxes placed', 'almost done', 'solved'] }
    ];
  },

  rule(s, legal, sopt) {
    const L = s.L, W = L.W;
    const ranks = [];
    // 计划缓存：命中则直接取下一步（仍校验合法性）
    if (s.plan && s.plan.length && legal.indexOf(s.plan[0]) >= 0) {
      for (const d of legal) {
        const isNext = d === s.plan[0];
        ranks.push({ key: d, label: DIRCN[d], score: isNext ? 1000 : 0,
                     detail: isNext ? `A* 计划第 1 步（共 ${s.plan.length} 步）` : '偏离计划' });
      }
      return { choice: s.plan[0], ranks,
               note: `A* 已求出 ${s.plan.length} 步解法（搜索 ${s.solvedInfo.nodes} 节点）。` };
    }
    // 计划失效（被模型带偏 / 未解出）→ 重新求解
    const r = this._plan(s, sopt);
    if (r.ok && r.path.length && legal.indexOf(r.path[0]) >= 0) {
      for (const d of legal) {
        const isNext = d === r.path[0];
        ranks.push({ key: d, label: DIRCN[d], score: isNext ? 1000 : 0,
                     detail: isNext ? `重解：A* 计划第 1 步（共 ${r.path.length} 步）` : '偏离计划' });
      }
      return { choice: r.path[0], ranks, note: `重新求解成功：${r.path.length} 步（${r.nodes} 节点）。` };
    }
    // 无解：退化为贪心（把箱子往最近目标推）
    let best = null;
    for (const d of legal) {
      const np = this._mv(s, s.player, d);
      const bi = s.boxes.indexOf(np);
      let sc = 0, detail = '空走';
      if (bi >= 0) {
        const nn = this._mv(s, np, d);
        const [br, bc] = [Math.floor(nn / W), nn % W];
        let m = Infinity;
        for (let g = 0; g < L.goal.length; g++) if (L.goal[g]) {
          m = Math.min(m, Math.abs(br - Math.floor(g / W)) + Math.abs(bc - g % W));
        }
        sc = 100 - m * 10 + (L.goal[nn] ? 50 : 0);
        detail = `推箱，推后到目标 ${m}${L.goal[nn] ? ' · 就位' : ''}`;
      }
      ranks.push({ key: d, label: DIRCN[d], score: sc, detail });
      if (!best || sc > best.sc) best = { d, sc };
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: ranks[0].key, ranks,
             note: `A* 在节点上限内未解出（${r.nodes} 节点），退化为贪心推箱。` };
  },

  metrics(s, sopt) {
    const fit = this._fitted(s);
    return [
      { k: '关卡', v: `${s.lv}/${LEVELS.length}` },
      { k: '已就位', v: `${fit}/${s.boxes.length}`, tone: fit === s.boxes.length ? 'ok' : '' },
      { k: '步数', v: s.steps },
      { k: '推动', v: s.pushes },
      { k: '计划剩余', v: s.plan ? s.plan.length : '无解',
        tone: s.plan ? '' : 'bad' },
      { k: '状态', v: fit === s.boxes.length ? '已解出' : '求解中',
        tone: fit === s.boxes.length ? 'ok' : 'warn' }
    ];
  },

  render(s, el) {
    const L = s.L, W = L.W;
    let h = `<div style="display:grid;grid-template-columns:repeat(${W},1fr);gap:2px;`
          + `background:#dfe4ea;padding:3px;border-radius:6px;max-width:340px;margin:0 auto">`;
    for (let r = 0; r < L.H; r++) for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const onGoal = L.goal[i], isBox = s.boxes.indexOf(i) >= 0, isP = s.player === i;
      let bg = '#fff', fg = '#1d2430', txt = '';
      if (L.wall[i]) bg = '#5b6b7c';
      else if (isP) { bg = '#2f6fed'; txt = '@'; fg = '#fff'; }
      else if (isBox) { bg = onGoal ? '#0d9488' : '#c2740a'; txt = '$'; fg = '#fff'; }
      else if (onGoal) { bg = '#e6f6f3'; txt = '·'; fg = '#0d9488'; }
      h += `<div style="background:${bg};color:${fg};aspect-ratio:1/1;border-radius:2px;`
         + `display:flex;align-items:center;justify-content:center;font-family:var(--mono);`
         + `font-weight:700;font-size:13px">${txt}</div>`;
    }
    h += `</div><div class="hint" style="text-align:center">蓝=@ 玩家 · 橙=$ 箱 · 绿=箱已就位 · 浅绿=目标</div>`;
    el.innerHTML = h;
  },

  done(s) { return this._fitted(s) === s.boxes.length; },
  goalOk(s) { return this._fitted(s) === s.boxes.length; },
  maxSteps(sopt) { return 300; }
};
