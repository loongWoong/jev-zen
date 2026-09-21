/* =============================================================================
 * 场景：Flappy Bird
 * 动作：flap（扇翅膀）/ hold（不动）
 * 算法：向前模拟若干帧，比较「扇」与「不扇」两条轨迹 —— 存活优先，其次贴近
 *       下一个管道缺口中心（缺口中心略偏上，给下落留余量）
 * 目标：连续通过 100 个管道
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const W = 420, H = 600;
const BIRD_X = 90, BIRD_R = 11;
const G = 0.62, FLAP_V = -9.2, VMAX = 11;
/* 管道间距必须给足调整时间：速度 3px/帧，间距 150px 只有 50 帧，而相邻缺口的
 * 高度落差最大可达 460px —— 全速上升也来不及（实测卡在 9 个管道）。
 * 放大间距并同时限制相邻落差，局面才是可解的。 */
const PIPE_W = 62, PIPE_SPACING = 235, PIPE_SPEED = 3.0;
const GAP_H = 150, MAX_GAP_SHIFT = 130;

const SCENE = {
  id: 'flappy',
  name: 'Flappy Bird',
  sub: '规则控制 · 前向模拟评估「扇 / 不扇」',
  goal: '目标：通过 100 个管道',
  hint: '每个决策点把两条轨迹（扇 / 不扇）各向前模拟几十帧，先比存活性，'
      + '再比轨迹与下一个管道缺口中心的最小距离。每步推进若干物理帧。',

  controls: [
    { id: 'target', label: '目标管道数', type: 'number', value: 100, min: 5, step: 5 },
    { id: 'fps', label: '每步模拟帧数', type: 'select', value: '2', options: [
      { v: '1', t: '1 帧（最细）' }, { v: '2', t: '2 帧' }, { v: '6', t: '6 帧（粗）' }] },
    { id: 'lookahead', label: '前瞻帧数', type: 'select', value: '40', options: [
      { v: '20', t: '20 帧' }, { v: '40', t: '40 帧' }, { v: '80', t: '80 帧' }] }
  ],

  init(seed, sopt) {
    const rng = mulberry32((seed * 2654435761) >>> 0);
    const s = { y: H * 0.45, vy: 0, frame: 0, pipes: [], score: 0, alive: true,
                rng, fps: parseInt(sopt.fps || '6', 10), target: parseInt(sopt.target, 10) || 100 };
    for (let i = 0; i < 3; i++) this._pipe(s, W + i * PIPE_SPACING);
    return s;
  },

  _pipe(s, x) {
    const margin = 70;
    let gapY = margin + s.rng() * (H - 2 * margin);
    const prev = s.pipes.length ? s.pipes[s.pipes.length - 1].gapY : null;
    if (prev !== null) {
      const lo = Math.max(margin, prev - MAX_GAP_SHIFT);
      const hi = Math.min(H - margin, prev + MAX_GAP_SHIFT);
      gapY = lo + s.rng() * (hi - lo);
    }
    s.pipes.push({ x, gapY, passed: false });
  },

  _nextPipe(s) {
    for (const p of s.pipes) if (p.x + PIPE_W > BIRD_X - BIRD_R && !p.passed) return p;
    for (const p of s.pipes) if (p.x + PIPE_W > BIRD_X - BIRD_R) return p;
    return s.pipes[0];
  },

  /* 轻量物理推进（不写 state，用于前瞻模拟）。
   * 评分 = 存活优先，其次「轨迹末端与下一个管道缺口中心的偏差」——
   * 注意目标必须始终用下一个管道，不能只在管道靠近时才计入，否则远处时
   * 两个动作得分相同，鸟会一路下坠（实测通过 0 个管道）。 */
  _sim(y, vy, pipes, frames, flapAtStart) {
    let py = y, pvy = vy;
    const ps = pipes.map(p => ({ x: p.x, gapY: p.gapY, passed: p.passed }));
    if (flapAtStart) pvy = FLAP_V;
    let alive = true;
    for (let f = 0; f < frames; f++) {
      pvy = Math.max(-VMAX, Math.min(VMAX, pvy + G));
      py += pvy;
      for (const p of ps) p.x -= PIPE_SPEED;
      if (py < BIRD_R || py > H - BIRD_R) { alive = false; break; }
      for (const p of ps) {
        if (p.x < BIRD_X + BIRD_R && p.x + PIPE_W > BIRD_X - BIRD_R) {
          if (Math.abs(py - p.gapY) > GAP_H / 2 - BIRD_R) { alive = false; break; }
        }
      }
      if (!alive) break;
    }
    const np = pipes.find(p => !p.passed) || pipes[0];
    const aim = (np ? np.gapY : H / 2) - 10;   // 略高于缺口中心，给下落留余量
    return { alive, err: Math.abs(py - aim) + 0.35 * Math.abs(pvy) };
  },

  legal() { return ['hold', 'flap']; },

  step(s, key) {
    if (key === 'flap') s.vy = FLAP_V;
    const fps = s.fps || 6;
    for (let f = 0; f < fps; f++) {
      s.vy = Math.max(-VMAX, Math.min(VMAX, s.vy + G));
      s.y += s.vy;
      s.frame++;
      for (const p of s.pipes) p.x -= PIPE_SPEED;
      if (s.pipes.length && s.pipes[s.pipes.length - 1].x < W - PIPE_SPACING) this._pipe(s, W);
      while (s.pipes.length && s.pipes[0].x + PIPE_W < -20) s.pipes.shift();
      for (const p of s.pipes) {
        if (!p.passed && p.x + PIPE_W < BIRD_X - BIRD_R) { p.passed = true; s.score++; }
      }
      if (s.y < BIRD_R || s.y > H - BIRD_R) { s.alive = false; break; }
      for (const p of s.pipes) {
        if (p.x < BIRD_X + BIRD_R && p.x + PIPE_W > BIRD_X - BIRD_R) {
          if (Math.abs(s.y - p.gapY) > GAP_H / 2 - BIRD_R) { s.alive = false; break; }
        }
      }
      if (!s.alive) break;
    }
    return { info: `${key === 'flap' ? '扇' : '悬停'} · 通过 ${s.score}/${s.target} 管道 · y=${Math.round(s.y)}` };
  },

  text(s, legal) {
    const np = this._nextPipe(s);
    return `Flappy Bird. canvas ${W}x${H}. bird at x=${BIRD_X}, y=${Math.round(s.y)}, `
      + `vertical velocity=${s.vy.toFixed(1)} (positive = falling).\n`
      + `pipes_passed=${s.score} target=${s.target}.\n`
      + `next pipe: x=${Math.round(np.x)}, gap center y=${Math.round(np.gapY)}, gap height=${GAP_H}.\n`
      + `physics: gravity ${G} per frame, flap sets velocity to ${FLAP_V}, pipe speed ${PIPE_SPEED}.\n`
      + `actions: flap (jump up) or hold (keep falling).`;
  },

  questions(s, legal) {
    return [
      { id: 'move', type: 'choice',
        instructions: 'Should the bird flap now to pass through the next pipe gap?',
        criteria: { flap: 'flap wings, setting vertical velocity to a strong upward value',
                    hold: 'do nothing and keep falling under gravity' } },
      { id: 'danger', type: 'score',
        instructions: 'How close is the bird to crashing?',
        criteria: ['safe', 'slightly off', 'risky', 'about to crash'] }
    ];
  },

  rule(s, legal, sopt) {
    // 注意：不能用「扇一次后自由落体 N 帧」的模拟来直接选动作 ——
    // 实际上每 fps 帧就能再扇一次，模拟假设「之后不再操作」会低估连续扇翅膀
    // 撞顶的风险（实测鸟一路撞到顶）。改用阈值规则：预测若不扇就会掉到缺口
    // 中心以下，才扇。这样在顶部会自动停手、在底部会自动拉升。
    const np = s.pipes.find(p => !p.passed) || s.pipes[0];
    const aim = (np ? np.gapY : H / 2) - 12;
    // 扇一次之后还会凭惯性再上升约 68px 才到最高点（FLAP_V²/(2G)）。若这个最高点
    // 会越过缺口上沿，就不能扇 —— 否则会从缺口下面直接冲到缺口上面撞死（实测死因）。
    const apexIfFlap = s.y - (FLAP_V * FLAP_V) / (2 * G);
    const topLimit = (np ? np.gapY : H / 2) - (GAP_H / 2 - BIRD_R) + 10;
    const needUp = s.y + s.vy * 2 > aim;
    const canFlap = apexIfFlap > topLimit;
    const pick = (needUp && canFlap) ? 'flap' : 'hold';
    const la = parseInt(sopt.lookahead || '40', 10);
    const ranks = [];
    for (const k of ['flap', 'hold']) {
      const r = this._sim(s.y, s.vy, s.pipes, la, k === 'flap');
      const sc = r.alive ? -r.err : -1e6 + (k === 'flap' ? 1 : 0);
      ranks.push({ key: k, label: k === 'flap' ? '扇翅膀' : '不扇', score: sc, digits: 1,
                   detail: (r.alive ? `模拟存活 · 末端偏差 ${r.err.toFixed(0)}px`
                                    : `模拟中坠毁（前瞻 ${la} 帧）`)
                          + (k === pick ? ' · 选中' : '')
                          + (k === 'flap' && !canFlap ? ' · 会冲过缺口上沿，禁扇' : ''),
                   risky: !r.alive });
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: pick, ranks,
             note: `阈值规则：缺口中心 ${Math.round(np ? np.gapY : 0)}（瞄准 ${Math.round(aim)}，上沿 ${Math.round(topLimit)}），`
                 + `当前 y=${Math.round(s.y)} vy=${s.vy.toFixed(1)} → ${pick === 'flap' ? '扇' : '不扇'}`
                 + (needUp && !canFlap ? '（想扇但会冲过上沿，先继续下坠）' : '') + `。` };
  },

  metrics(s, sopt) {
    const np = this._nextPipe(s);
    return [
      { k: '通过管道', v: `${s.score}/${s.target}`, tone: s.score >= s.target ? 'ok' : '' },
      { k: '帧数', v: s.frame },
      { k: '高度 y', v: Math.round(s.y), tone: s.y > H - 60 ? 'bad' : '' },
      { k: '垂直速度', v: s.vy.toFixed(1) },
      { k: '缺口中心', v: Math.round(np.gapY) },
      { k: '状态', v: s.alive ? '飞行中' : '坠毁', tone: s.alive ? 'ok' : 'bad' }
    ];
  },

  render(s, el) {
    if (!el.querySelector('canvas')) {
      el.innerHTML = `<canvas width="${W}" height="${H}" style="width:100%;max-width:300px;`
                   + `border-radius:8px;background:#dff3fb;display:block;margin:0 auto"></canvas>`;
    }
    const cv = el.querySelector('canvas');
    const g = cv.getContext('2d');
    g.clearRect(0, 0, W, H);
    g.fillStyle = '#dff3fb'; g.fillRect(0, 0, W, H);
    // 管道
    g.fillStyle = '#5aa64b';
    for (const p of s.pipes) {
      const top = p.gapY - GAP_H / 2;
      g.fillRect(p.x, 0, PIPE_W, top);
      g.fillRect(p.x, p.gapY + GAP_H / 2, PIPE_W, H - (p.gapY + GAP_H / 2));
      g.fillStyle = '#3f8a34';
      g.fillRect(p.x - 4, top - 22, PIPE_W + 8, 22);
      g.fillRect(p.x - 4, p.gapY + GAP_H / 2, PIPE_W + 8, 22);
      g.fillStyle = '#5aa64b';
    }
    // 地面
    g.fillStyle = '#ded6b8'; g.fillRect(0, H - 24, W, 24);
    // 小鸟
    g.fillStyle = '#f2c14e';
    g.beginPath(); g.arc(BIRD_X, s.y, BIRD_R, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#e8590c';
    g.beginPath(); g.arc(BIRD_X + 8, s.y - 2, 4, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#1d2430';
    g.font = 'bold 20px ui-monospace, monospace';
    g.fillText(String(s.score), 14, 34);
    el.innerHTML = el.innerHTML;   // 保持结构
  },

  done(s) { return !s.alive; },
  goalOk(s, sopt) { return s.score >= (parseInt(sopt.target, 10) || 100); },
  maxSteps(sopt) {
    const t = parseInt(sopt.target, 10) || 100;
    const fps = Math.max(1, parseInt(sopt.fps || '2', 10));
    return Math.ceil(t * (PIPE_SPACING / PIPE_SPEED) / fps) + 600;
  }
};
