/* =============================================================================
 * 场景：打砖块 Breakout
 * 动作：left / right / stay（挡板移动）
 * 算法：预测球的落点 —— 沿当前速度推进并考虑左右墙反弹，算出球到达挡板高度时的
 *       横坐标，挡板朝该点移动（带死区，避免在目标附近抖动）
 * 目标：清空全部砖块
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const W = 420, H = 320;
const PADDLE_W = 78, PADDLE_H = 11, PADDLE_Y = H - 26, PADDLE_SPEED = 7.5;
const BALL_R = 6;
const COLS = 9, ROWS = 5, BRICK_W = 42, BRICK_H = 15, BRICK_GAP = 4, BRICK_TOP = 42;

const SCENE = {
  id: 'breakout',
  name: '打砖块',
  sub: '挡板控制 · 预测落点追踪（含墙反弹）',
  goal: '目标：清空全部砖块',
  hint: '挡板不是简单地跟着球跑，而是先沿球的当前速度做一次「含左右墙反弹」的落点预测，'
      + '再朝预测点移动；带死区以防在目标附近来回抖动。每步推进若干物理帧。',

  controls: [
    { id: 'fps', label: '每步模拟帧数', type: 'select', value: '3', options: [
      { v: '1', t: '1 帧（细）' }, { v: '3', t: '3 帧' }, { v: '6', t: '6 帧（粗）' }] },
    { id: 'speed', label: '球速', type: 'select', value: '3.4', options: [
      { v: '2.6', t: '慢' }, { v: '3.4', t: '中' }, { v: '4.4', t: '快' }] },
    { id: 'deadzone', label: '死区(px)', type: 'select', value: '4', options: [
      { v: '0', t: '0（无死区）' }, { v: '4', t: '4' }, { v: '12', t: '12' }] }
  ],

  init(seed, sopt) {
    const rng = mulberry32((seed * 2654435761) >>> 0);
    const speed = parseFloat(sopt.speed || '3.4');
    const bricks = [];
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      bricks.push({ x: 12 + c * (BRICK_W + BRICK_GAP), y: BRICK_TOP + r * (BRICK_H + BRICK_GAP),
                    w: BRICK_W, h: BRICK_H, alive: true, row: r });
    }
    const ang = (-0.72 + rng() * 0.44);   // 略微随机初始角度
    const s = { bricks, paddleX: (W - PADDLE_W) / 2, bx: W / 2, by: H - 60,
                vx: Math.sin(ang) * speed, vy: -Math.abs(Math.cos(ang) * speed),
                frame: 0, hits: 0, alive: true, rng, speed,
                fps: parseInt(sopt.fps || '3', 10),
                deadzone: parseInt(sopt.deadzone || '4', 10) };
    return s;
  },

  legal() { return ['left', 'right', 'stay']; },

  /* 预测球到达挡板高度时的横坐标。
   * 必须一路模拟到球真正下落到挡板高度 —— 球上升时直接返回当前 x 会让挡板
   * 在半个回合里乱跑（实测清空率 0/8、平均只打掉 31/45 块）。 */
  _predict(s) {
    let x = s.bx, y = s.by, vx = s.vx, vy = s.vy;
    let guard = 900;
    while (y < PADDLE_Y - BALL_R && guard-- > 0) {
      x += vx; y += vy;
      if (x < BALL_R) { x = BALL_R; vx = -vx; }
      else if (x > W - BALL_R) { x = W - BALL_R; vx = -vx; }
      if (y < BALL_R) { y = BALL_R; vy = -vy; }     // 撞顶反弹，继续预测
    }
    return x;
  },

  step(s, key) {
    if (key === 'left') s.paddleX -= PADDLE_SPEED * (s.fps || 3);
    else if (key === 'right') s.paddleX += PADDLE_SPEED * (s.fps || 3);
    s.paddleX = Math.max(0, Math.min(W - PADDLE_W, s.paddleX));

    const fps = s.fps || 3;
    let gained = 0;
    for (let f = 0; f < fps; f++) {
      s.frame++;
      s.bx += s.vx; s.by += s.vy;
      if (s.bx < BALL_R) { s.bx = BALL_R; s.vx = -s.vx; }
      if (s.bx > W - BALL_R) { s.bx = W - BALL_R; s.vx = -s.vx; }
      if (s.by < BALL_R) { s.by = BALL_R; s.vy = -s.vy; }
      // 挡板
      if (s.vy > 0 && s.by + BALL_R >= PADDLE_Y && s.by - BALL_R <= PADDLE_Y + PADDLE_H
          && s.bx >= s.paddleX - BALL_R && s.bx <= s.paddleX + PADDLE_W + BALL_R) {
        s.by = PADDLE_Y - BALL_R;
        const rel = ((s.bx - (s.paddleX + PADDLE_W / 2)) / (PADDLE_W / 2));  // -1..1
        const ang = rel * 0.95;
        const sp = Math.max(s.speed, Math.hypot(s.vx, s.vy));
        s.vx = Math.sin(ang) * sp;
        s.vy = -Math.abs(Math.cos(ang) * sp);
        s.hits++;
      }
      // 砖块
      for (const b of s.bricks) {
        if (!b.alive) continue;
        if (s.bx + BALL_R > b.x && s.bx - BALL_R < b.x + b.w
            && s.by + BALL_R > b.y && s.by - BALL_R < b.y + b.h) {
          b.alive = false; gained++;
          const ox = Math.min(s.bx + BALL_R - b.x, b.x + b.w - (s.bx - BALL_R));
          const oy = Math.min(s.by + BALL_R - b.y, b.y + b.h - (s.by - BALL_R));
          if (ox < oy) s.vx = -s.vx; else s.vy = -s.vy;
          break;
        }
      }
      if (s.by > H + BALL_R) { s.alive = false; break; }
    }
    return { info: `${key === 'left' ? '左移' : key === 'right' ? '右移' : '不动'}`
      + `${gained ? ` · 打掉 ${gained} 块` : ''} · 剩余 ${this._left(s)}/${s.bricks.length}` };
  },

  _left(s) { return s.bricks.filter(b => b.alive).length; },

  text(s, legal) {
    const px = this._predict(s);
    const alive = s.bricks.filter(b => b.alive);
    return `Breakout. canvas ${W}x${H}. paddle x=${Math.round(s.paddleX)} (width ${PADDLE_W}, `
      + `at y=${PADDLE_Y}), moves ${PADDLE_SPEED}px per frame.\n`
      + `ball at (${Math.round(s.bx)}, ${Math.round(s.by)}) with velocity `
      + `(${s.vx.toFixed(1)}, ${s.vy.toFixed(1)}).\n`
      + `predicted landing x when the ball reaches the paddle: ${Math.round(px)}.\n`
      + `bricks_left=${alive.length}/${s.bricks.length} paddle_hits=${s.hits}.\n`
      + `actions: move paddle left, right, or stay.`;
  },

  questions(s, legal) {
    return [
      { id: 'move', type: 'choice',
        instructions: 'Which way should the paddle move to intercept the ball?',
        criteria: { left: 'move the paddle left', right: 'move the paddle right',
                    stay: 'keep the paddle where it is' } },
      { id: 'urgency', type: 'score',
        instructions: 'How urgent is it to move the paddle?',
        criteria: ['already aligned', 'slight adjustment', 'needs to move', 'about to miss'] }
    ];
  },

  rule(s, legal, sopt) {
    const px = this._predict(s);
    const cx = s.paddleX + PADDLE_W / 2;
    const dz = parseInt(sopt.deadzone || '4', 10);
    const diff = px - cx;
    let want = 'stay';
    if (diff < -dz) want = 'left';
    else if (diff > dz) want = 'right';
    const ranks = ['left', 'stay', 'right'].map(k => {
      const delta = Math.abs(k === 'left' ? (px - (cx - PADDLE_SPEED * (s.fps || 3)))
                           : k === 'right' ? (px - (cx + PADDLE_SPEED * (s.fps || 3)))
                           : diff);
      return { key: k, label: k === 'left' ? '左移' : k === 'right' ? '右移' : '不动',
               score: -delta, digits: 1,
               detail: `移动后挡板中心与预测落点 ${px.toFixed(0)} 相差 ${delta.toFixed(0)}px` };
    });
    ranks.sort((a, b) => b.score - a.score);
    return { choice: want, ranks,
             note: `预测落点 x=${px.toFixed(0)}（含左右墙反弹），挡板中心 ${cx.toFixed(0)}，`
                 + `死区 ${dz}px → ${want === 'left' ? '左移' : want === 'right' ? '右移' : '保持'}。` };
  },

  metrics(s, sopt) {
    const left = this._left(s);
    return [
      { k: '剩余砖块', v: `${left}/${s.bricks.length}`, tone: left === 0 ? 'ok' : '' },
      { k: '已打掉', v: s.bricks.length - left },
      { k: '接球次数', v: s.hits },
      { k: '帧数', v: s.frame },
      { k: '预测落点', v: Math.round(this._predict(s)) },
      { k: '状态', v: !s.alive ? '漏球' : (left === 0 ? '清空' : '进行中'),
        tone: !s.alive ? 'bad' : left === 0 ? 'ok' : '' }
    ];
  },

  render(s, el) {
    if (!el.querySelector('canvas')) {
      el.innerHTML = `<canvas width="${W}" height="${H}" style="width:100%;max-width:420px;`
                   + `border-radius:8px;background:#12161f;display:block;margin:0 auto"></canvas>`;
    }
    const cv = el.querySelector('canvas');
    const g = cv.getContext('2d');
    g.fillStyle = '#12161f'; g.fillRect(0, 0, W, H);
    const COLORS = ['#e05252', '#e08b3c', '#f2c14e', '#3fae5a', '#31c4be'];
    for (const b of s.bricks) {
      if (!b.alive) continue;
      g.fillStyle = COLORS[b.row % COLORS.length];
      g.fillRect(b.x, b.y, b.w, b.h);
    }
    // 预测落点
    const px = this._predict(s);
    g.strokeStyle = 'rgba(255,255,255,.28)';
    g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(px, s.by); g.lineTo(px, PADDLE_Y); g.stroke();
    g.setLineDash([]);
    g.fillStyle = '#f5f6f8';
    g.fillRect(s.paddleX, PADDLE_Y, PADDLE_W, PADDLE_H);
    g.fillStyle = '#31c4be';
    g.beginPath(); g.arc(s.bx, s.by, BALL_R, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#98a2b3';
    g.font = '12px ui-monospace, monospace';
    g.fillText(`left ${this._left(s)}  hits ${s.hits}`, 10, 20);
  },

  done(s) { return !s.alive || this._left(s) === 0; },
  goalOk(s) { return this._left(s) === 0; },
  /* 45 块砖、每块平均要 50+ 个决策步（一块砖一次往返约 160 物理帧），
   * 上限给小了会把「没打完」误判成「漏球」。 */
  maxSteps() { return 4000; }
};
