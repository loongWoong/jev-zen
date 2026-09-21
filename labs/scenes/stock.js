/* =============================================================================
 * 场景：股票买卖模拟 Stock Trading
 * 价格序列 = 几何布朗运动 + 若干段趋势（牛/熊/震荡），同一 seed 完全可复现
 * 动作：buy（全仓买入）/ sell（全部卖出）/ hold
 * 算法：双均线交叉（快 MA5 / 慢 MA20）+ 7% 移动止损 + 2% 手续费单边
 * 对照：买入并持有（buy & hold）
 * 目标：收益跑赢买入持有，且最大回撤 ≤ 目标
 * ===========================================================================*/

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
/* Box-Muller */
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const FEE = 0.002;          // 单边手续费
const STOP_LOSS = 0.07;     // 相对持仓峰值的移动止损

function genPrices(seed, n) {
  const rng = mulberry32((seed * 2654435761) >>> 0);
  const prices = [100];
  let drift = 0.0004;
  let seg = 0;
  for (let i = 1; i < n; i++) {
    if (i % 45 === 0) {           // 每 45 天换一段趋势
      seg = rng();
      drift = seg < 0.34 ? 0.0035 : seg < 0.67 ? -0.0030 : 0.0002;
    }
    const shock = gauss(rng) * 0.012;
    prices.push(Math.max(5, prices[i - 1] * Math.exp(drift + shock)));
  }
  return prices;
}

function ma(prices, i, n) {
  if (i + 1 < n) return null;
  let s = 0;
  for (let k = 0; k < n; k++) s += prices[i - k];
  return s / n;
}

const SCENE = {
  id: 'stock',
  name: '股票买卖模拟',
  sub: 'GBM + 分段趋势 · 双均线交叉 + 移动止损',
  goal: '目标：跑赢买入持有且回撤 ≤ 25%',
  hint: '快均线（5 日）上穿慢均线（20 日）= 金叉买入，下穿 = 死叉卖出；'
      + '叠加 7% 移动止损控制回撤。单边手续费 0.2%。对照基准是买入并持有。',

  controls: [
    { id: 'fast', label: '快均线', type: 'number', value: 5, min: 2, max: 30, step: 1 },
    { id: 'slow', label: '慢均线', type: 'number', value: 20, min: 5, max: 60, step: 1 },
    { id: 'stop', label: '移动止损 %', type: 'number', value: 7, min: 0, max: 30, step: 1 },
    { id: 'buyhold', label: '对照：买入持有', type: 'checkbox', value: false, hint: '只买一次然后一直持有' }
  ],

  init(seed, sopt) {
    const n = 260;
    const prices = genPrices(seed, n);
    const s = { prices, i: 0, n, cash: 100000, shares: 0,
                trades: [], peak: 0, maxDD: 0, equity: [100000],
                hold: false, stop: (parseInt(sopt.stop, 10) || 7) / 100,
                fast: parseInt(sopt.fast, 10) || 5, slow: parseInt(sopt.slow, 10) || 20,
                buyhold: !!sopt.buyhold, entry: 0 };
    return s;
  },

  legal() { return ['buy', 'sell', 'hold']; },

  /* 终局时 i 已经等于 n，直接取 prices[i] 会是 undefined → 净值变 NaN */
  _equity(s) { return s.cash + s.shares * s.prices[Math.min(s.i, s.n - 1)]; },

  step(s, key) {
    const px = s.prices[s.i];
    let info = '';
    if (key === 'buy' && s.shares === 0 && s.cash > 0) {
      const qty = s.cash * (1 - FEE) / px;
      s.shares = qty; s.cash = 0; s.entry = px; s.peak = px;
      s.trades.push({ i: s.i, act: 'buy', px });
      info = `买入 @${px.toFixed(2)}`;
    } else if (key === 'sell' && s.shares > 0) {
      s.cash = s.shares * px * (1 - FEE); s.shares = 0;
      s.trades.push({ i: s.i, act: 'sell', px });
      info = `卖出 @${px.toFixed(2)}`;
    } else info = key === 'hold' ? '持有不动' : '无操作';
    s.i++;
    if (s.i < s.n) {
      const eq = this._equity(s);
      s.equity.push(eq);
      s.peak = Math.max(s.peak, eq);
      s.maxDD = Math.max(s.maxDD, (s.peak - eq) / s.peak);
    }
    return { info: `${info} · 净值 ${this._equity(s).toFixed(0)}` };
  },

  _ret(s) { return this._equity(s) / 100000 - 1; },
  _bh(s) { return s.prices[Math.min(s.i, s.n - 1)] / s.prices[0] - 1; },

  text(s, legal) {
    const i = s.i;
    const win = s.prices.slice(Math.max(0, i - 24), i + 1).map(p => p.toFixed(1)).join(' ');
    const f = ma(s.prices, i, s.fast), sl = ma(s.prices, i, s.slow);
    return `Stock trading simulation. day ${i}/${s.n}. price=${s.prices[i].toFixed(2)}.\n`
      + `cash=${s.cash.toFixed(0)} shares=${s.shares.toFixed(2)} equity=${this._equity(s).toFixed(0)} `
      + `(start 100000).\n`
      + `MA${s.fast}=${f === null ? 'n/a' : f.toFixed(2)} MA${s.slow}=${sl === null ? 'n/a' : sl.toFixed(2)} `
      + `${f !== null && sl !== null ? (f > sl ? '(golden cross zone)' : '(dead cross zone)') : ''}.\n`
      + `return=${(this._ret(s) * 100).toFixed(2)}% buy_hold=${(this._bh(s) * 100).toFixed(2)}% `
      + `max_drawdown=${(s.maxDD * 100).toFixed(1)}% trades=${s.trades.length}.\n`
      + `recent prices (last ${Math.min(25, i + 1)}): ${win}\n`
      + `actions: buy (all in), sell (all out), hold. Fee ${(FEE * 100).toFixed(1)}% per side.`;
  },

  questions(s, legal) {
    return [
      { id: 'move', type: 'choice',
        instructions: 'Should the strategy buy, sell, or hold today?',
        criteria: { buy: 'invest all cash into the stock',
                    sell: 'sell the entire position to cash',
                    hold: 'do nothing today' } },
      { id: 'trend', type: 'score',
        instructions: 'What is the current trend?',
        criteria: ['strong downtrend', 'mild downtrend', 'mild uptrend', 'strong uptrend'] }
    ];
  },

  rule(s, legal, sopt) {
    const i = s.i, px = s.prices[i];
    const f = ma(s.prices, i, s.fast), sl = ma(s.prices, i, s.slow);
    let choice = 'hold', note;
    if (s.buyhold) {
      choice = s.shares === 0 && s.cash > 0 ? 'buy' : 'hold';
      note = '买入持有对照：第一天全仓买入，之后一直持有。';
    } else if (s.shares > 0) {
      const dd = (s.peak - this._equity(s)) / s.peak;
      if (dd >= s.stop) { choice = 'sell'; note = `移动止损触发：净值距峰值回撤 ${(dd * 100).toFixed(1)}% ≥ ${(s.stop * 100).toFixed(0)}%。`; }
      else if (f !== null && sl !== null && f < sl) { choice = 'sell'; note = `死叉：MA${s.fast}(${f.toFixed(2)}) 下穿 MA${s.slow}(${sl.toFixed(2)})。`; }
      else { choice = 'hold'; note = `持仓中：快均线仍在慢均线之上，回撤 ${(dd * 100).toFixed(1)}% 未触发止损。`; }
    } else {
      if (f !== null && sl !== null && f > sl) { choice = 'buy'; note = `金叉：MA${s.fast}(${f.toFixed(2)}) 上穿 MA${s.slow}(${sl.toFixed(2)})。`; }
      else {
        choice = 'hold';
        // 注意：MA_fast 先成型、MA_slow 还是 null 的窗口期要单独判断，
        // 否则会对 null 调 toFixed 直接崩掉
        note = (f === null || sl === null)
          ? `数据不足，等待 MA${s.slow} 成型（已有 ${s.i + 1} 个交易日）。`
          : `空仓等待金叉：MA${s.fast}(${f.toFixed(2)}) ≤ MA${s.slow}(${sl.toFixed(2)})。`;
      }
    }
    const ranks = ['buy', 'hold', 'sell'].map(k => ({
      key: k, label: k === 'buy' ? '买入' : k === 'sell' ? '卖出' : '持有',
      score: k === choice ? 100 : 0,
      detail: k === choice ? '规则选中'
        : (f !== null && sl !== null) ? (f > sl ? '金叉区' : '死叉区') : '均线未成型'
    }));
    return { choice, ranks, note };
  },

  metrics(s, sopt) {
    const r = this._ret(s), bh = this._bh(s);
    return [
      { k: '交易日', v: `${s.i}/${s.n}` },
      { k: '收益率', v: (r * 100).toFixed(2) + '%', tone: r > 0 ? 'ok' : r < 0 ? 'bad' : '' },
      { k: '买入持有', v: (bh * 100).toFixed(2) + '%', tone: bh > 0 ? 'ok' : 'bad' },
      { k: '超额', v: ((r - bh) * 100).toFixed(2) + '%',
        tone: r > bh ? 'ok' : 'bad', sm: r > bh ? '跑赢' : '跑输' },
      { k: '最大回撤', v: (s.maxDD * 100).toFixed(1) + '%',
        tone: s.maxDD > 0.25 ? 'bad' : s.maxDD > 0.15 ? 'warn' : 'ok', sm: '≤ 25%' },
      { k: '交易次数', v: s.trades.length }
    ];
  },

  render(s, el) {
    const n = s.n, h = 150, w = 400;
    const eq = s.equity;
    const lo = Math.min(...eq, ...s.prices.slice(0, s.i + 1) / 1 || [1]);
    const vals = eq.slice();
    const min = Math.min(...vals), max = Math.max(...vals);
    const pt = (v, i) => `${(i / (n - 1) * w).toFixed(1)},${(h - (v - min) / Math.max(1e-9, max - min) * h).toFixed(1)}`;
    const path = vals.map((v, i) => pt(v, i)).join(' ');
    const buyPts = s.trades.filter(t => t.act === 'buy').map(t => t.i);
    el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" style="width:100%;max-width:420px;display:block;margin:0 auto;
        background:#fafbfc;border:1px solid var(--line2);border-radius:6px">
      <polyline points="${path}" fill="none" stroke="#2f6fed" stroke-width="1.6"/>
      ${buyPts.map(i => `<line x1="${(i / (n - 1) * w).toFixed(1)}" y1="0" x2="${(i / (n - 1) * w).toFixed(1)}" y2="${h}" stroke="#3fae5a" stroke-width="1" stroke-dasharray="2 3"/>`).join('')}
      </svg>
      <div class="hint" style="text-align:center">蓝 = 净值曲线 · 绿虚线 = 买入点 · 当前 ${s.prices[Math.min(s.i, n - 1)].toFixed(2)}</div>`;
  },

  done(s) { return s.i >= s.n; },
  goalOk(s, sopt) {
    if (s.i < s.n) return false;
    const r = this._ret(s), bh = this._bh(s);
    return r > bh && s.maxDD <= 0.25;
  },
  maxSteps() { return 260; }
};
