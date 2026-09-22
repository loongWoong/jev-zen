/* =============================================================================
 * Laya 决策模型 · 实验底座 (labs/core.js)
 * 提供：API 客户端 / 决策合成 / 留痕 / 统计 / 单局动画 / 批量验证
 * 场景模块需实现 SCENE 接口（见文件末尾约定）
 * ===========================================================================*/
'use strict';

/* ------------------------------ 工具 ------------------------------ */
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const pct = v => (v * 100).toFixed(1) + '%';
const nf = (v, d) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(d === undefined ? 2 : d) : '—';
const nowMs = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/* -------------------------- Laya API 客户端 -------------------------- */
const Laya = {
  base: null,
  ok: false,
  async probe() {
    const cands = location.protocol === 'file:'
      ? ['http://127.0.0.1:8771']
      : ['', 'http://127.0.0.1:8771'];
    for (const c of cands) {
      try {
        const r = await fetch(c + '/api/health', { method: 'GET', cache: 'no-store' });
        if (!r.ok) continue;
        const j = await r.json();
        if (j && j.ok) { this.base = c; this.ok = true; return c; }
      } catch (e) { /* 换下一个候选 */ }
    }
    this.base = null; this.ok = false; return null;
  },
  async post(path, body) {
    const opt = { method: body === undefined ? 'GET' : 'POST',
                  headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opt.body = JSON.stringify(body);
    const r = await fetch(this.base + path, opt);
    const t = await r.text();
    let j;
    try { j = JSON.parse(t); } catch (e) { throw new Error('非 JSON 响应: ' + t.slice(0, 200)); }
    if (!j.ok) throw new Error(j.error || 'unknown error');
    return j;
  },
  /* state: string | object ; questions: [{id,type,instructions,criteria}] */
  async predict(state, questions, temp) {
    const j = await this.post('/api/predict', {
      state: state,
      questions: questions,
      config: { temperature_mode: 'manual',
                temperature_manual: [temp || 3, 1, 1] }
    });
    return j.result;
  }
};

/* ------------------------------ Lab 框架 ------------------------------ */
const Lab = {
  s: null,
  sopt: {},
  strategy: 'rule',
  temp: 3,
  delayMs: 120,
  seed: 1,
  running: false,
  busy: false,
  gen: 0,
  hist: [],
  last: null,
  lastRaw: null,
  freshIdx: -1,
  batch: null,
  ST: null,

  freshStat() {
    return { steps: 0, calls: 0, latSum: 0, latMax: 0, spreadSum: 0, spreadN: 0,
             confSum: 0, agree: 0, agreeTot: 0, illegal: 0, errs: 0,
             goalHit: false, episodes: 0, wins: 0, scoreSum: 0, stepSum: 0 };
  },

  /* ---------- 场景自定义控件 ---------- */
  buildControls() {
    const host = $('sceneCtl');
    host.innerHTML = '';
    this.sopt = {};
    const cts = SCENE.controls || [];
    for (const c of cts) {
      const d = document.createElement('div');
      if (c.type === 'select') {
        d.innerHTML = `<label class="lb">${esc(c.label)}</label>
          <select id="sc_${c.id}">${(c.options || []).map(o =>
            `<option value="${esc(o.v)}"${o.v === c.value ? ' selected' : ''}>${esc(o.t)}</option>`).join('')}</select>`;
      } else if (c.type === 'checkbox') {
        d.innerHTML = `<label class="lb">${esc(c.label)}</label>
          <label class="row" style="font-size:12.5px;color:var(--muted);margin:5px 0 0">
            <input type="checkbox" id="sc_${c.id}" ${c.value ? 'checked' : ''} /> ${esc(c.hint || '启用')}</label>`;
      } else if (c.type === 'range') {
        d.innerHTML = `<label class="lb">${esc(c.label)} <span class="mono" id="scv_${c.id}">${c.value}</span></label>
          <input type="range" id="sc_${c.id}" min="${c.min ?? 0}" max="${c.max ?? 100}"
                 step="${c.step ?? 1}" value="${c.value}" />`;
      } else {
        d.innerHTML = `<label class="lb">${esc(c.label)}</label>
          <input type="number" id="sc_${c.id}" value="${c.value}" min="${c.min ?? 0}" step="${c.step ?? 1}" />`;
      }
      host.appendChild(d);
      this.sopt[c.id] = c.value;
    }
    host.querySelectorAll('[id^="sc_"]').forEach(el => {
      const id = el.id.slice(3);
      const on = () => {
        this.sopt[id] = el.type === 'checkbox' ? el.checked
                      : (el.type === 'number' || el.type === 'range') ? parseFloat(el.value)
                      : el.value;
        const lb = $('scv_' + id); if (lb) lb.textContent = this.sopt[id];
        if (SCENE.onOptChange) SCENE.onOptChange(id, this.sopt[id], this);
      };
      el.addEventListener('change', on);
      el.addEventListener('input', on);
    });
  },

  /* ---------- 决策合成 ---------- */
  async decide(s, legal, strategy, temp, useModel) {
    const stateText = SCENE.text(s, legal, this.sopt);
    const qs = SCENE.questions ? (SCENE.questions(s, legal, this.sopt) || []) : [];
    const t0 = nowMs();
    const rd = SCENE.rule(s, legal, this.sopt);
    const ruleMs = nowMs() - t0;

    let m = null, raw = null, modelChoice = null, probs = null, conf = null, spread = null;
    const want = !!Laya.ok && qs.length > 0 &&
                 (strategy === 'model' || strategy === 'mix' || useModel);
    if (want) {
      try {
        raw = await Laya.predict(stateText, qs, temp);
        m = raw.answers || {};
        const key = qs[0].id;
        const a = m[key];
        if (a && a.type === 'choice') {
          modelChoice = a.choice; probs = a.probs || {};
          conf = a.confidence; spread = a.logit_spread;
        }
      } catch (e) {
        this.ST.errs++;
        throw e;
      }
      this.ST.calls++;
      if (raw && raw.meta) {
        this.ST.latSum += raw.meta.latency_ms;
        this.ST.latMax = Math.max(this.ST.latMax, raw.meta.latency_ms);
      }
      if (typeof spread === 'number') { this.ST.spreadSum += spread; this.ST.spreadN++; }
      if (typeof conf === 'number') this.ST.confSum += conf;
      if (modelChoice !== null) {
        this.ST.agreeTot++;
        if (modelChoice === rd.choice) this.ST.agree++;
      }
    }

    let applied = rd.choice, source = 'rule', fallback = false, band = null;
    if (strategy === 'model' && modelChoice !== null) {
      source = 'model';
      if (legal.indexOf(modelChoice) >= 0) applied = modelChoice;
      else { fallback = true; applied = rd.choice; this.ST.illegal++; }
    } else if (strategy === 'mix' && modelChoice !== null) {
      source = 'mix';
      // 算法给出候选窄带，模型只在带内破平 —— 模型选不到带外的危险动作
      const bandV = (typeof this.sopt.mixBand === 'number') ? this.sopt.mixBand : 0.02;
      band = bandV;
      let cand = (rd.ranks || []).filter(r => legal.indexOf(r.key) >= 0);
      if (cand.length) {
        const vs = cand.map(r => r.score);
        const hi = Math.max(...vs), lo = Math.min(...vs);
        const near = cand.filter(r => hi - r.score <= bandV * Math.max(1e-9, hi - lo));
        cand = near.length ? near : cand.slice(0, 1);
      } else cand = [rd.choice];
      let bp = -1;
      applied = cand[0].key !== undefined ? cand[0].key : cand[0];
      for (const r of cand) {
        const k = r.key !== undefined ? r.key : r;
        const p = probs[k] ?? 0;
        if (p > bp) { bp = p; applied = k; }
      }
    }
    if (legal.indexOf(applied) < 0) {
      fallback = true; applied = rd.choice; if (strategy === 'model') this.ST.illegal++;
    }
    if (legal.indexOf(applied) < 0) applied = legal[0];

    return { applied, source, fallback, band, rd, m, raw, probs, conf, spread,
             modelChoice, stateText, qs, ruleMs, state: stateText };
  },

  /* ---------- 单步（动画模式） ---------- */
  async tick() {
    if (this.busy) return false;
    const s = this.s;
    if (!s || SCENE.done(s)) return false;
    const legal = SCENE.legal(s);
    if (!legal.length) { this.finish('无可行动作'); return false; }
    this.busy = true;
    $('stepBtn').disabled = true;
    try {
      const d = await this.decide(s, legal, this.strategy, this.temp, $('callModel').checked);
      const before = SCENE.snapshot ? SCENE.snapshot(s) : null;
      const res = SCENE.step(s, d.applied, this.sopt) || {};
      this.ST.steps++;
      const h = {
        i: this.hist.length + 1,
        applied: d.applied, source: d.source, fallback: d.fallback,
        legal: legal.slice(),
        rd: { choice: d.rd.choice, ranks: d.rd.ranks || [], note: d.rd.note || '' },
        modelChoice: d.modelChoice, probs: d.probs, conf: d.conf, spread: d.spread,
        ruleMs: d.ruleMs,
        lat: d.raw && d.raw.meta ? d.raw.meta.latency_ms : null,
        tokens: d.raw && d.raw.meta ? d.raw.meta.tokens : null,
        info: res.info || '',
        metrics: SCENE.metrics ? SCENE.metrics(s, this.sopt) : [],
        ts: Date.now()
      };
      this.hist.push(h);
      this.freshIdx = h.i;
      this.last = d; this.lastRaw = d.raw;
      if (SCENE.goalOk && SCENE.goalOk(s, this.sopt)) { this.ST.goalHit = true; }
      this.renderAll();
      $('errBox').style.display = 'none';
      if (SCENE.done(s)) {
        this.finish(SCENE.goalOk && SCENE.goalOk(s, this.sopt) ? '目标达成' : '本局结束');
        return false;
      }
      return true;
    } catch (e) {
      this.ST.errs++;
      const box = $('errBox');
      box.style.display = 'block';
      box.textContent = '决策失败: ' + (e && e.message ? e.message : String(e));
      this.stop();
      return false;
    } finally {
      this.busy = false;
      $('stepBtn').disabled = false;
    }
  },

  async loop() {
    const my = ++this.gen;
    while (this.running && my === this.gen) {
      const ok = await this.tick();
      if (!ok) break;
      const d = parseInt($('delay').value, 10);
      if (d > 0) await sleep(d);
      if (this.ST.goalHit && SCENE.stopOnGoal !== false) { this.finish('目标达成'); break; }
    }
  },
  start() {
    if (this.running) return;
    if (!this.s || SCENE.done(this.s)) this.reset(false);
    this.running = true;
    $('runBtn').textContent = '暂停';
    $('runTag').textContent = 'running'; $('runTag').className = 'tag ok';
    this.loop();
  },
  stop() {
    this.running = false;
    $('runBtn').textContent = '自动运行';
    $('runTag').textContent = 'idle'; $('runTag').className = 'tag';
  },
  finish(reason) {
    this.stop();
    $('runTag').textContent = reason || 'done';
    $('runTag').className = 'tag ' + (this.ST.goalHit ? 'ok' : 'warn');
  },
  reset(rerender) {
    this.stop();
    this.ST = this.freshStat();
    this.hist = []; this.freshIdx = -1; this.last = null; this.lastRaw = null;
    this.strategy = $('strategy').value || 'rule';
    this.temp = parseFloat($('temp').value) || 3;
    this.seed = parseInt($('seed').value, 10) || 1;
    this.s = SCENE.init(this.seed, this.sopt);
    $('stratTag').textContent = this.strategy;
    if (rerender !== false) this.renderAll();
  },

  /* ---------- 批量验证（无渲染，纯逻辑跑多局） ---------- */
  async playEpisode(seed, strategy, temp, maxSteps) {
    const s = SCENE.init(seed, this.sopt);
    const acc = { steps: 0, calls: 0, lat: 0, agree: 0, agreeTot: 0, illegal: 0,
                  goal: false, score: 0, errs: 0 };
    let guard = maxSteps || (SCENE.maxSteps ? SCENE.maxSteps(this.sopt) : 500);
    while (!SCENE.done(s) && guard-- > 0) {
      const legal = SCENE.legal(s);
      if (!legal.length) break;
      let applied;
      const rd = SCENE.rule(s, legal, this.sopt);
      if (strategy === 'rule' || !Laya.ok) {
        applied = rd.choice;
      } else {
        let d;
        try { d = await this.decide(s, legal, strategy, temp, false); }
        catch (e) { acc.errs++; applied = rd.choice; d = null; }
        if (d) {
          applied = d.applied;
          if (strategy === 'model' && d.fallback) acc.illegal++;
          if (d.modelChoice !== null) {
            acc.agreeTot++; if (d.modelChoice === rd.choice) acc.agree++;
          }
          if (d.raw && d.raw.meta) { acc.calls++; acc.lat += d.raw.meta.latency_ms; }
        }
      }
      if (legal.indexOf(applied) < 0) applied = legal[0];
      SCENE.step(s, applied, this.sopt);
      acc.steps++;
      if (SCENE.goalOk && SCENE.goalOk(s, this.sopt)) { acc.goal = true; break; }
    }
    if (!acc.goal && SCENE.goalOk && SCENE.goalOk(s, this.sopt)) acc.goal = true;
    const m = SCENE.metrics ? SCENE.metrics(s, this.sopt) : [];
    const sc = m.find(x => /分|score/i.test(x.k));
    acc.score = sc ? (typeof sc.v === 'number' ? sc.v : 0) : 0;
    return acc;
  },

  async runBatch() {
    const n = parseInt($('batchN').value, 10) || 20;
    const btn = $('batchBtn');
    btn.disabled = true;
    const rows = [];
    const strategies = ['rule', 'mix', 'model'];
    for (const st of strategies) {
      if (st !== 'rule' && !Laya.ok) { rows.push(null); continue; }
      const acc = { st, n: 0, wins: 0, steps: 0, score: 0, calls: 0, lat: 0,
                    agree: 0, agreeTot: 0, illegal: 0, errs: 0 };
      for (let i = 0; i < n; i++) {
        const r = await this.playEpisode(this.seed + i, st, this.temp, null);
        acc.n++;
        if (r.goal) acc.wins++;
        acc.steps += r.steps; acc.score += r.score;
        acc.calls += r.calls; acc.lat += r.lat;
        acc.agree += r.agree; acc.agreeTot += r.agreeTot;
        acc.illegal += r.illegal; acc.errs += r.errs;
        if (i % 5 === 0) {
          $('batchOut').innerHTML = this.batchHtml(rows.concat([acc]), st, i + 1, n);
          await sleep(0);
        }
      }
      rows.push(acc);
      $('batchOut').innerHTML = this.batchHtml(rows, st, n, n);
    }
    this.batch = rows;
    $('batchOut').innerHTML = this.batchHtml(rows, 'done', n, n);
    btn.disabled = false;
  },

  batchHtml(rows, cur, i, n) {
    const th = `<div class="row" style="font-size:12px;color:var(--muted);margin-bottom:6px">
      批量验证 · <span class="mono">${esc(cur === 'done' ? '完成' : cur + ' ' + i + '/' + n)}</span></div>`;
    if (!rows.filter(Boolean).length) return th + '<div class="dim">尚无结果</div>';
    const tb = `<table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px">
      <thead><tr style="color:var(--muted);font-size:11px;text-align:right">
        <th style="text-align:left">策略</th><th>局数</th><th>达成</th><th>达成率</th>
        <th>均步</th><th>均分</th><th>模型调用</th><th>均延迟ms</th><th>与规则一致</th><th>非法回退</th>
      </tr></thead><tbody>`;
    const body = rows.filter(Boolean).map(a => {
      const rate = a.n ? a.wins / a.n : 0;
      const cls = rate >= 0.9 ? 'ok' : rate >= 0.5 ? 'warn' : 'bad';
      return `<tr style="text-align:right;border-top:1px solid var(--line2)">
        <td style="text-align:left">${esc(a.st)}</td>
        <td>${a.n}</td><td>${a.wins}</td>
        <td style="color:${cls === 'ok' ? 'var(--ok)' : cls === 'warn' ? 'var(--warn)' : 'var(--bad)'};font-weight:650">${pct(rate)}</td>
        <td>${nf(a.n ? a.steps / a.n : 0, 1)}</td>
        <td>${nf(a.n ? a.score / a.n : 0, 0)}</td>
        <td>${a.calls}</td>
        <td>${nf(a.calls ? a.lat / a.calls : 0, 1)}</td>
        <td>${a.agreeTot ? pct(a.agree / a.agreeTot) : '—'}</td>
        <td>${a.illegal}</td></tr>`;
    }).join('');
    return th + tb + body + '</tbody></table>';
  },

  /* ---------- 渲染 ---------- */
  renderAll() {
    this.renderStage();
    this.renderMetrics();
    this.renderChan();
    this.renderTrace();
    this.renderLast();
    this.renderHist();
    $('stepTag').textContent = this.ST.steps;
    $('goalTag').textContent = this.ST.goalHit ? '已达成' : '未达成';
    $('goalTag').style.color = this.ST.goalHit ? 'var(--ok)' : '';
    $('latTag').textContent = this.ST.calls ? nf(this.ST.latSum / this.ST.calls, 1) + 'ms' : '—';
  },
  renderStage() {
    if (SCENE.render) SCENE.render(this.s, $('stage'), this.sopt);
  },
  statHtml(list) {
    return (list || []).map(m => `<div class="st${m.tone ? ' ' + m.tone : ''}">
      <div class="n">${esc(m.v)}</div><div class="t">${esc(m.k)}</div>
      ${m.sm ? `<div class="sm">${esc(m.sm)}</div>` : ''}</div>`).join('');
  },
  renderMetrics() {
    const list = SCENE.metrics ? SCENE.metrics(this.s, this.sopt) : [];
    $('metrics').innerHTML = this.statHtml(list);
  },
  renderChan() {
    const S = this.ST;
    const agree = S.agreeTot ? S.agree / S.agreeTot : null;
    $('chan').innerHTML = this.statHtml([
      { k: '模型调用', v: S.calls, sm: S.calls ? '均 ' + nf(S.latSum / S.calls, 1) + 'ms' : '—' },
      { k: '算法·模型一致', v: agree === null ? '—' : pct(agree),
        tone: agree === null ? '' : agree >= 0.6 ? 'ok' : agree >= 0.35 ? 'warn' : 'bad',
        sm: S.agreeTot ? S.agree + '/' + S.agreeTot : '—' },
      { k: '模型非法回退', v: S.illegal, tone: S.illegal ? 'bad' : 'ok' },
      { k: 'logit 极差', v: S.spreadN ? nf(S.spreadSum / S.spreadN, 4) : '—',
        tone: (S.spreadN && S.spreadSum / S.spreadN < 0.01) ? 'warn' : '',
        sm: '通道增益读数' },
      { k: '平均置信度', v: S.calls ? nf(S.confSum / S.calls, 3) : '—' },
      { k: '错误', v: S.errs, tone: S.errs ? 'bad' : 'ok' }
    ]);
  },
  renderTrace() {
    const h = this.hist[this.hist.length - 1];
    if (!h) { $('trace').innerHTML = '<div class="dim">尚无决策</div>'; return; }
    const maxP = Math.max(1e-9, ...Object.values(h.probs || {}).map(v => Math.abs(v)));
    const bars = Object.keys(h.probs || {}).map(k => {
      const p = h.probs[k] ?? 0;
      const isPick = k === h.modelChoice;
      const isApplied = k === h.applied;
      const cls = isApplied ? 'pick' : (isPick ? '' : '');
      return `<div class="pbar ${cls}"><span class="k">${esc(k)}</span>
        <span class="t"><i style="width:${(Math.abs(p) / maxP * 100).toFixed(1)}%"></i></span>
        <span class="v">${nf(p, 3)}${isApplied ? ' ▸' : ''}</span></div>`;
    }).join('');
    const ranks = (h.rd.ranks || []).map(r => {
      const pick = r.key === h.rd.choice;
      return `<div class="rank ${pick ? 'pick' : ''} ${r.risky ? 'risky' : ''}">
        <span class="k">${esc(r.label || r.key)}</span>
        <span class="v">${esc(r.detail || '')}</span>
        <span class="s">${typeof r.score === 'number' ? nf(r.score, r.digits === undefined ? 1 : r.digits) : esc(r.score)}</span></div>`;
    }).join('');
    const agree = (h.modelChoice === null || h.modelChoice === undefined) ? null
                : (h.modelChoice === h.rd.choice);
    const badge = agree === null ? '<span class="tag">未调用模型</span>'
      : agree ? '<span class="tag ok">模型与算法一致</span>'
              : `<span class="tag warn">分歧 模型=${esc(h.modelChoice)} / 算法=${esc(h.rd.choice)}</span>`;
    $('trace').innerHTML =
      `<div class="row" style="margin-bottom:7px">
         <span class="mono">#${h.i}</span>
         <span class="tag">执行 ${esc(h.applied)}</span>
         <span class="tag">来源 ${esc(h.source)}</span>
         ${h.fallback ? '<span class="tag bad">回退</span>' : ''}
         ${badge}
         <span class="dim mono" style="margin-left:auto">算法 ${nf(h.ruleMs, 2)}ms${h.lat !== null ? ' · 模型 ' + nf(h.lat, 1) + 'ms' : ''}</span>
       </div>
       ${ranks ? `<div style="font-size:11.5px;color:var(--muted);margin:6px 0 2px">算法分解（高亮 = 算法首选）</div>${ranks}` : ''}
       ${bars ? `<div style="font-size:11.5px;color:var(--muted);margin:9px 0 2px">Laya choice 概率（▸ = 实际执行）</div>${bars}` : ''}
       ${h.rd.note ? `<div class="hint">${esc(h.rd.note)}</div>` : ''}
       ${h.info ? `<div class="hint">本步：${esc(h.info)}</div>` : ''}`;
  },
  renderLast() {
    const d = this.last;
    if (!d) { $('last').innerHTML = '<div class="dim">尚无请求</div>'; return; }
    $('last').innerHTML =
      `<details open><summary>喂给模型的 state（${(d.stateText || '').length} 字符）</summary>
        <pre>${esc(d.stateText)}</pre></details>
       <details><summary>questions（${d.qs.length}）</summary>
        <pre>${esc(JSON.stringify(d.qs, null, 2))}</pre></details>
       ${d.raw ? `<details><summary>完整响应 JSON</summary><pre>${esc(JSON.stringify(d.raw, null, 2))}</pre></details>` : ''}`;
  },
  renderHist() {
    if (!this.hist.length) { $('hist').innerHTML = '<div class="dim">尚无记录</div>'; return; }
    const items = this.hist.slice(-120).reverse().map(h => {
      const agree = (h.modelChoice === null || h.modelChoice === undefined) ? null
                  : (h.modelChoice === h.rd.choice);
      const b = agree === null ? '' : agree
        ? '<span class="tag ok">一致</span>' : '<span class="tag warn">分歧</span>';
      return `<div class="hitem${h.i === this.freshIdx ? ' fresh' : ''}">
        <div class="hrow">
          <span class="idx">#${h.i}</span>
          <span class="act">${esc(h.applied)}</span>
          <span class="dim mono">${esc(h.source)}</span>
          ${b}
          ${h.probs && h.probs[h.applied] !== undefined ? `<span class="dim mono">p=${nf(h.probs[h.applied], 3)}</span>` : ''}
          <span class="dim mono" style="margin-left:auto">${h.lat !== null ? nf(h.lat, 0) + 'ms' : ''}</span>
        </div>
        <div class="hrow dim mono" style="font-size:11.5px">${esc(h.info || '')}</div>
      </div>`;
    }).join('');
    $('hist').innerHTML = items;
  },

  /* ---------- 启动 ---------- */
  async boot() {
    this.ST = this.freshStat();
    $('sceneTag').textContent = SCENE.id;
    this.buildControls();
    this.s = SCENE.init(this.seed, this.sopt);

    $('runBtn').onclick = () => { this.running ? this.stop() : this.start(); };
    $('stepBtn').onclick = async () => {
      if (!this.s || SCENE.done(this.s)) this.reset(false);
      await this.tick();
    };
    $('resetBtn').onclick = () => this.reset(true);
    $('strategy').onchange = () => {
      this.strategy = $('strategy').value || 'rule';
      $('stratTag').textContent = this.strategy;
    };
    $('temp').onchange = () => { this.temp = parseFloat($('temp').value) || 3; };
    $('seed').onchange = () => { this.seed = parseInt($('seed').value, 10) || 1; };
    $('copyBtn').onclick = () => {
      const txt = this.hist.map(h => JSON.stringify(h)).join('\n');
      navigator.clipboard ? navigator.clipboard.writeText(txt) : console.log(txt);
      $('copyBtn').textContent = '已复制 ' + this.hist.length + ' 条';
      setTimeout(() => { $('copyBtn').textContent = '导出 JSONL'; }, 1600);
    };
    $('clearBtn').onclick = () => { this.hist = []; this.renderHist(); };
    if ($('batchBtn')) $('batchBtn').onclick = () => this.runBatch();

    const c = await Laya.probe();
    $('apiTag').textContent = c ? (c || location.origin) + '/api' : '未连接';
    $('apiDot').className = 'dot ' + (c ? 'ok' : 'bad');
    const b = $('banner');
    if (!c) {
      b.className = 'banner bad';
      b.innerHTML = '<span class="ic">●</span><div>未连到 Laya 后端。请先启动 ' +
        '<code>python laya_verify.py --serve</code>（8771 端口），否则只能跑纯算法模式。</div>';
    } else {
      b.className = 'banner';
      b.innerHTML = '<span class="ic">●</span><div><b>通道提示</b>：模型权重里不含 prompt 模板 / type_emb 注入位置等' +
        '重建项，决策头读出的是近似通道。默认 <code>rule</code> 策略由算法主判、模型仅留痕对照；' +
        '切到 <code>model</code> 可观察模型独立表现。</div>';
    }
    this.renderAll();
  }
};

/* =============================================================================
 * SCENE 接口约定（每个场景模块必须实现）
 *   id / name / sub / goal / hint
 *   controls: [{ id, label, type:'select'|'number'|'checkbox'|'range', value, options, ... }]
 *   init(seed, sopt)            -> state（纯数据，可 JSON 化）
 *   legal(state)                -> [key, ...] 合法动作
 *   step(state, key, sopt)      -> { info }  就地推进
 *   text(state, legal, sopt)    -> 喂模型的 state 文本
 *   questions(state, legal, sopt) -> [{ id, type:'choice'|'score'|'noul', instructions, criteria }]
 *   rule(state, legal, sopt)    -> { choice, ranks:[{key,label,score,detail,risky}], note }
 *   metrics(state, sopt)        -> [{ k, v, tone, sm }]
 *   render(state, el, sopt)
 *   done(state)                 -> bool
 *   goalOk(state, sopt)         -> bool  （自动验证目标）
 *   maxSteps(sopt)              -> number （批量验证的保护上限，可选）
 * ===========================================================================*/
