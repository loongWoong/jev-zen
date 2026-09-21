/* =============================================================================
 * 场景：交通灯控制 Traffic Light
 * 单路口，南北 / 东西两个相位。每步 = 1 个时间刻度
 * 动作：keep（保持当前相位）/ switch（立即切换）
 * 算法：自适应规则 —— 绿灯方向已放空且另一方向有车 → 切；
 *       绿灯时长达到上限且另一方向有车 → 切；最小绿灯时长保护（防频繁切换）
 * 对照：固定配时（每 8 刻度一切）
 * 目标：车辆平均等待 ≤ 目标刻度
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const MIN_GREEN = 3;
const MAX_GREEN = 12;
const THROUGHPUT = 2;      // 绿灯每刻度放行的车辆数

const SCENE = {
  id: 'traffic',
  name: '交通灯控制',
  sub: '单路口双相位 · 自适应规则 / 固定配时对照',
  goal: '目标：平均等待 ≤ 12 刻度',
  hint: '自适应规则会看「本方向是否已放空」和「另一方向是否积压」来决定切不切，'
      + '并用最小绿灯时长防止来回乱切；固定配时（每 8 刻度一切）作为对照基线。',

  controls: [
    { id: 'strategy', label: '控制策略', type: 'select', value: 'adaptive', options: [
      { v: 'adaptive', t: '自适应规则' }, { v: 'fixed', t: '固定配时（对照）' }] },
    { id: 'lambda', label: '车流强度(×100)', type: 'select', value: '30', options: [
      { v: '15', t: '稀疏' }, { v: '30', t: '中等' }, { v: '45', t: '拥堵' }] },
    { id: 'target', label: '平均等待上限', type: 'number', value: 4, min: 2, step: 1 }
  ],

  init(seed, sopt) {
    const lam = parseInt(sopt.lambda || '30', 10) / 100;
    const rng = mulberry32((seed * 2654435761) >>> 0);
    const arrivals = [];
    for (let t = 0; t < 300; t++) {
      for (const dir of ['ns', 'ew']) {
        let n = 0;
        while (rng() < lam) { n++; if (n > 3) break; }
        for (let k = 0; k < n; k++) arrivals.push({ t, dir });
      }
    }
    const s = { arrivals, ai: 0, t: 0, phase: 'ns', green: 0,
                q: { ns: 0, ew: 0 }, served: [], waitSum: 0, rng,
                strategy: sopt.strategy || 'adaptive',
                target: parseInt(sopt.target, 10) || 4, switches: 0 };
    return s;
  },

  legal() { return ['keep', 'switch']; },

  step(s, key) {
    /* 到达 */
    while (s.ai < s.arrivals.length && s.arrivals[s.ai].t === s.t) {
      const a = s.arrivals[s.ai];
      s.q[a.dir]++;
      s.waitingQ = s.waitingQ || [];
      s.waitingQ.push({ dir: a.dir, t0: s.t });
      s.ai++;
    }
    /* 相位 */
    if (key === 'switch' && s.green >= MIN_GREEN) { s.phase = s.phase === 'ns' ? 'ew' : 'ns'; s.green = 0; s.switches++; }
    else s.green++;
    /* 放行 */
    const n = Math.min(THROUGHPUT, s.q[s.phase]);
    for (let i = 0; i < n; i++) {
      const idx = (s.waitingQ || []).findIndex(v => v.dir === s.phase);
      if (idx < 0) break;
      const v = s.waitingQ.splice(idx, 1)[0];
      s.waitSum += s.t - v.t0;
      s.served.push(v);
      s.q[s.phase]--;
    }
    s.t++;
    return { info: `${key === 'switch' ? '切换' : '保持'} · ${s.phase === 'ns' ? '南北' : '东西'}绿灯`
      + ` · 排队 南北${s.q.ns} 东西${s.q.ew} · 平均等待 ${this._avg(s)}` };
  },

  _avg(s) { return s.served.length ? s.waitSum / s.served.length : 0; },

  text(s, legal) {
    return `Traffic light at a single intersection, two phases: ns (north-south) and ew (east-west).\n`
      + `tick=${s.t} green_phase=${s.phase} green_duration=${s.green} ticks.\n`
      + `queue ns=${s.q.ns} ew=${s.q.ew} (vehicles waiting).\n`
      + `served=${s.served.length} avg_wait=${this._avg(s).toFixed(2)} ticks.\n`
      + `each green tick releases up to ${THROUGHPUT} vehicles from the green direction.\n`
      + `actions: keep the current phase, or switch (minimum green ${MIN_GREEN} ticks).`;
  },

  questions(s, legal) {
    const other = s.phase === 'ns' ? s.q.ew : s.q.ns;
    return [
      { id: 'move', type: 'choice',
        instructions: 'Should the light keep the current green phase or switch now?',
        criteria: {
          keep: `keep ${s.phase} green (${s.q[s.phase]} waiting, green for ${s.green} ticks)`,
          switch: `switch to ${s.phase === 'ns' ? 'ew' : 'ns'} green (${other} waiting there)`
        } },
      { id: 'congestion', type: 'score',
        instructions: 'How congested is the intersection?',
        criteria: ['clear', 'light', 'busy', 'gridlocked'] }
    ];
  },

  rule(s, legal, sopt) {
    const st = sopt.strategy || 'adaptive';
    const cur = s.q[s.phase], other = s.q[s.phase === 'ns' ? 'ew' : 'ns'];
    let choice, note;
    if (st === 'fixed') {
      choice = (s.green >= 8) ? 'switch' : 'keep';
      note = `固定配时：每 8 刻度切换（当前绿灯已 ${s.green}）→ ${choice === 'switch' ? '切换' : '保持'}。`;
    } else {
      let want = 'keep';
      if (s.green >= MIN_GREEN) {
        if (cur === 0 && other > 0) want = 'switch';                 // 本方向放空
        else if (s.green >= MAX_GREEN && other > cur) want = 'switch'; // 超限且对方更堵
      }
      choice = want;
      note = `自适应：本方向 ${cur} 辆 / 另一方向 ${other} 辆，绿灯已 ${s.green} 刻度`
           + `（下限 ${MIN_GREEN}、上限 ${MAX_GREEN}）→ ${choice === 'switch' ? '切换' : '保持'}。`;
    }
    const ranks = [
      { key: 'keep', label: '保持', score: choice === 'keep' ? 100 : 0,
        detail: `继续放行 ${s.phase === 'ns' ? '南北' : '东西'}（${cur} 辆）` },
      { key: 'switch', label: '切换', score: choice === 'switch' ? 100 : 0,
        detail: `转到 ${s.phase === 'ns' ? '东西' : '南北'}（${other} 辆）`
               + (s.green < MIN_GREEN ? ` · 最小绿灯未满足(${s.green}/${MIN_GREEN})` : '') }
    ];
    return { choice, ranks, note };
  },

  metrics(s, sopt) {
    const a = this._avg(s);
    return [
      { k: '刻度', v: s.t },
      { k: '已放行', v: s.served.length },
      { k: '平均等待', v: a.toFixed(2), tone: a <= s.target ? 'ok' : 'warn', sm: `≤ ${s.target}` },
      { k: '排队总计', v: s.q.ns + s.q.ew,
        tone: (s.q.ns + s.q.ew) > 25 ? 'bad' : (s.q.ns + s.q.ew) > 12 ? 'warn' : 'ok' },
      { k: '切换次数', v: s.switches },
      { k: '当前相位', v: s.phase === 'ns' ? '南北' : '东西' }
    ];
  },

  render(s, el) {
    const other = s.phase === 'ns' ? 'ew' : 'ns';
    const box = (dir, label) => {
      const on = s.phase === dir;
      return `<div style="flex:1;text-align:center">
        <div style="font-size:12px;color:var(--muted);font-family:var(--mono)">${label}</div>
        <div style="width:34px;height:34px;border-radius:50%;margin:5px auto;
          background:${on ? '#3fae5a' : '#d64545'};box-shadow:0 0 0 ${on ? '4px' : '0'} ${on ? 'rgba(63,174,90,.2)' : 'transparent'}"></div>
        <div style="font-family:var(--mono);font-size:16px;font-weight:680">${s.q[dir]}</div>
        <div style="font-size:10.5px;color:var(--dim)">辆排队</div></div>`;
    };
    el.innerHTML = `<div style="display:flex;gap:14px;justify-content:center;align-items:flex-end;
      background:#12161f;padding:14px;border-radius:8px;max-width:300px;margin:0 auto">
      ${box('ns', '南北')}<div style="width:2px;height:70px;background:#5b6b7c"></div>${box('ew', '东西')}
      </div>
      <div class="hint" style="text-align:center">绿灯已持续 ${s.green} 刻度 · 最小 ${MIN_GREEN} / 上限 ${MAX_GREEN}
      · 每刻度放行 ${THROUGHPUT} 辆</div>`;
  },

  done(s) { return s.t >= 300; },
  goalOk(s, sopt) {
    const t = parseInt(sopt.target, 10) || 12;
    return s.served.length >= 30 && this._avg(s) <= t;
  },
  maxSteps() { return 300; }
};
