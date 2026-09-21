/* =============================================================================
 * 场景：电梯调度 Elevator
 * 每步 = 1 个时间刻度；决策 = 把「等待最久的那个外呼」派给哪台电梯
 * 算法：代价估计 —— 到达该楼层所需刻度 + 沿途停靠开销 + 方向一致性 + 负载惩罚
 * 对照：随机派梯
 * 目标：服务指定人数且平均等待 ≤ 目标值
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const SCENE = {
  id: 'elevator',
  name: '电梯调度',
  sub: '10 层 · 3 台电梯 · 代价估计派梯（贪心调度）',
  goal: '目标：平均等待 ≤ 12 刻度',
  hint: '每个刻度把「等待最久的外呼」派给一台电梯。代价 = 到达该层的刻度数 + 沿途停靠开销'
      + ' + 方向不一致惩罚 + 负载惩罚；随机派梯作为对照基线。',

  controls: [
    { id: 'elevators', label: '电梯数', type: 'select', value: '3', options: [
      { v: '1', t: '1 台' }, { v: '2', t: '2 台' }, { v: '3', t: '3 台' }, { v: '4', t: '4 台' }] },
    { id: 'lambda', label: '到达强度(×100)', type: 'select', value: '12', options: [
      { v: '6', t: '稀疏' }, { v: '12', t: '中等' }, { v: '22', t: '繁忙' }] },
    { id: 'target', label: '平均等待上限', type: 'number', value: 20, min: 6, step: 2 },
    { id: 'randomAssign', label: '对照：随机派梯', type: 'checkbox', value: false, hint: '用随机策略对照' }
  ],

  init(seed, sopt) {
    const F = 10;
    const nE = parseInt(sopt.elevators || '3', 10);
    const lam = (parseInt(sopt.lambda || '12', 10) / 100);
    const rng = mulberry32((seed * 2654435761) >>> 0);
    // 预生成到达序列，保证同一 seed 完全可复现
    const arrivals = [];
    for (let t = 0; t < 400; t++) {
      let n = 0;
      while (rng() < lam) { n++; if (n > 3) break; }
      for (let k = 0; k < n; k++) {
        const from = Math.floor(rng() * F);
        let to = Math.floor(rng() * F);
        while (to === from) to = Math.floor(rng() * F);
        arrivals.push({ t, from, to, dir: to > from ? 1 : -1, id: arrivals.length });
      }
    }
    const s = {
      F, nE, arrivals, ai: 0, t: 0,
      elev: Array.from({ length: nE }, () => ({ pos: 0, dir: 0, targets: [], load: 0, door: 0 })),
      waiting: [], onboard: [], served: [], waitSum: 0,
      rng, target: parseInt(sopt.target, 10) || 20
    };
    return s;
  },

  legal(s) {
    const out = [];
    for (let i = 0; i < s.nE; i++) out.push('e' + i);
    return out;
  },

  /* 把某个外呼派给电梯 e 的估计代价（越小越好） */
  _cost(s, e, w) {
    const el = s.elev[e];
    let dist = Math.abs(el.pos - w.from);
    let extra = 0;
    // 方向一致性：顺路（电梯方向与请求方向相同且在其前方）几乎不额外付出代价
    if (el.dir !== 0 && el.dir !== w.dir) extra += 6;
    else if (el.dir !== 0 && el.dir === w.dir) {
      const ahead = (w.from - el.pos) * el.dir;
      if (ahead < 0) extra += dist + 4;      // 已经开过头，得折返
      else extra += 0;
    }
    return dist + el.targets.length * 2.5 + extra + el.load * 1.5;
  },

  step(s, key) {
    const ei = parseInt(key.slice(1), 10);
    /* --- 1) 到达 --- */
    while (s.ai < s.arrivals.length && s.arrivals[s.ai].t === s.t) {
      s.waiting.push(Object.assign({}, s.arrivals[s.ai], { t0: s.t, assigned: -1 }));
      s.ai++;
    }
    /* --- 2) 派梯：把等待最久的未分配请求派给选中的电梯 --- */
    let info = '';
    const pend = s.waiting.filter(w => w.assigned < 0);
    if (pend.length) {
      const w = pend.reduce((a, b) => (a.t0 <= b.t0 ? a : b));
      w.assigned = ei;
      s.elev[ei].targets.push(w.from);
      info = `外呼 ${w.from}层→${w.to}层 派给 ${ei + 1}号`;
    } else info = '本刻度无待派请求';

    /* --- 3) 电梯物理推进 --- */
    for (const el of s.elev) {
      if (el.door > 0) { el.door--; continue; }
      if (el.targets.indexOf(el.pos) >= 0) {
        // 开门：下客 + 上客
        el.targets = el.targets.filter(f => f !== el.pos);
        s.onboard = s.onboard.filter(p => {
          if (p.to === el.pos) {
            s.served.push(p); s.waitSum += s.t - p.t0; el.load--; return false;
          }
          return true;
        });
        const here = s.waiting.filter(w => w.assigned === s.elev.indexOf(el)
                                        && w.from === el.pos && w.assigned >= 0);
        for (const w of here) {
          if (el.load < 6) { el.load++; s.onboard.push(w); el.targets.push(w.to); }
          w.assigned = -2;                       // 已上梯（标记，稍后清理）
        }
        s.waiting = s.waiting.filter(w => w.assigned !== -2);
        el.door = 1;
        continue;
      }
      if (!el.targets.length) { el.dir = 0; continue; }
      // SCAN：保持当前方向一路服务到底，前方没有目标了才折返。
      // 「每次都奔向最近目标」会让电梯在两层之间来回抖动，实测等待明显更长。
      if (el.dir === 0) {
        const near = el.targets.reduce((a, b) =>
          Math.abs(a - el.pos) <= Math.abs(b - el.pos) ? a : b);
        el.dir = near === el.pos ? 0 : (near > el.pos ? 1 : -1);
      }
      const ahead = el.targets.some(f => (f - el.pos) * el.dir > 0);
      if (!ahead) {
        const back = el.targets.some(f => (f - el.pos) * -el.dir > 0);
        el.dir = back ? -el.dir : 0;
      }
      if (el.dir === 0) continue;
      el.pos = Math.max(0, Math.min(s.F - 1, el.pos + el.dir));
    }
    s.t++;
    return { info: `${info} · 已服务 ${s.served.length} · 平均等待 ${this._avgWait(s)}` };
  },

  _avgWait(s) { return s.served.length ? s.waitSum / s.served.length : 0; },

  text(s, legal) {
    const el = s.elev.map((e, i) => `${i + 1}号: ${e.pos}层 ${e.dir > 0 ? '上行' : e.dir < 0 ? '下行' : '待机'}`
      + ` 载${e.load} 目标[${e.targets.join(',')}]`).join('\n');
    const pend = s.waiting.filter(w => w.assigned < 0);
    return `Elevator system: ${s.F} floors, ${s.nE} cars. time tick=${s.t}.\n`
      + `served=${s.served.length} avg_wait=${this._avgWait(s).toFixed(2)} ticks `
      + `still_waiting=${s.waiting.length}.\n`
      + `${el}\n`
      + `pending hall calls: ${pend.slice(0, 6).map(w => `${w.from}F->${w.to}F(wait ${s.t - w.t0})`).join(', ') || 'none'}\n`
      + `actions: assign the oldest pending call to car e0..e${s.nE - 1}.`;
  },

  questions(s, legal) {
    const crit = {};
    for (let i = 0; i < s.nE; i++) {
      const e = s.elev[i];
      crit['e' + i] = `assign to car ${i} (at floor ${e.pos}, ${e.targets.length} stops queued, load ${e.load})`;
    }
    return [
      { id: 'move', type: 'choice',
        instructions: 'Which elevator car should serve the longest-waiting hall call?',
        criteria: crit },
      { id: 'load', type: 'score',
        instructions: 'How busy is the system right now?',
        criteria: ['idle', 'light', 'moderate', 'overloaded'] }
    ];
  },

  rule(s, legal, sopt) {
    const pend = s.waiting.filter(w => w.assigned < 0);
    const ranks = [];
    if (!pend.length) {
      for (const k of legal) ranks.push({ key: k, label: k, score: 0, detail: '无待派请求' });
      return { choice: legal[0], ranks, note: '本刻度没有待派外呼。' };
    }
    const w = pend.reduce((a, b) => (a.t0 <= b.t0 ? a : b));
    if (sopt.randomAssign) {
      const k = legal[Math.floor(Math.random() * legal.length)];
      for (const kk of legal) ranks.push({ key: kk, label: kk, score: 0, detail: '随机派梯' });
      return { choice: k, ranks, note: '随机派梯（对照基线）。' };
    }
    for (const k of legal) {
      const ei = parseInt(k.slice(1), 10);
      const c = this._cost(s, ei, w);
      ranks.push({ key: k, label: `${ei + 1} 号`, score: -c, digits: 1,
                   detail: `距 ${Math.abs(s.elev[ei].pos - w.from)} 层 · 排队停靠 ${s.elev[ei].targets.length}`
                     + ` · 载 ${s.elev[ei].load} · 代价 ${c.toFixed(1)}` });
    }
    ranks.sort((a, b) => b.score - a.score);
    return { choice: ranks[0].key, ranks,
             note: `最久等待：${w.from}F→${w.to}F（已等 ${s.t - w.t0} 刻度），`
                 + `派给代价最低的 ${ranks[0].label}。` };
  },

  metrics(s, sopt) {
    const aw = this._avgWait(s);
    return [
      { k: '刻度', v: s.t },
      { k: '已服务', v: s.served.length },
      { k: '平均等待', v: aw.toFixed(2), tone: aw <= s.target ? 'ok' : 'warn', sm: `≤ ${s.target}` },
      { k: '最长等待', v: s.waiting.length ? Math.max(...s.waiting.map(w => s.t - w.t0)) : 0,
        tone: (s.waiting.length && Math.max(...s.waiting.map(w => s.t - w.t0)) > 30) ? 'bad' : '' },
      { k: '候梯中', v: s.waiting.length },
      { k: '电梯数', v: s.nE }
    ];
  },

  render(s, el) {
    const F = s.F;
    let h = `<div style="display:flex;gap:10px;justify-content:center">`;
    for (let e = 0; e < s.nE; e++) {
      const car = s.elev[e];
      h += `<div style="flex:1;max-width:74px"><div style="font-size:11px;text-align:center;`
         + `color:var(--muted);font-family:var(--mono)">${e + 1}号 载${car.load}</div>`
         + `<div style="display:flex;flex-direction:column-reverse;gap:2px;background:#eef1f4;`
         + `padding:2px;border-radius:5px">`;
      for (let f = 0; f < F; f++) {
        const isCar = car.pos === f;
        const isTarget = car.targets.indexOf(f) >= 0;
        const hasWait = s.waiting.some(w => w.from === f && w.assigned < 0);
        let bg = '#f7f8fa';
        if (isCar) bg = '#2f6fed';
        else if (isTarget) bg = '#a9c4f8';
        else if (hasWait) bg = '#f2c14e';
        h += `<div style="background:${bg};height:15px;border-radius:2px;position:relative">`
           + `<span style="position:absolute;left:3px;top:0;font-size:9px;color:#98a2b3;font-family:var(--mono)">${f}</span></div>`;
      }
      h += `</div></div>`;
    }
    h += `</div><div class="hint" style="text-align:center">蓝 = 轿厢 · 浅蓝 = 已派停靠层 · 黄 = 有人候梯</div>`;
    el.innerHTML = h;
  },

  done(s) { return s.t >= 400 || (s.ai >= s.arrivals.length && !s.waiting.length && !s.onboard.length); },
  goalOk(s, sopt) {
    const tgt = parseInt(sopt.target, 10) || 12;
    return s.served.length >= 20 && this._avgWait(s) <= tgt;
  },
  maxSteps() { return 400; }
};
