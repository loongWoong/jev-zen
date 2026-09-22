/* 全场景前端端到端测试（纯算法模式，不连后端）：
 * 用最小 DOM 桩加载生成的 HTML，跑完整 tick（含 metrics / render）和批量验证，
 * 断言 Lab.ST.errs === 0 且不抛异常。捕获用户报告的 "Cannot read properties of
 * undefined (reading 'target')" 这类因 core 调用场景钩子时漏传 sopt / 字段未兜底
 * 引起的隐藏错误。 */
import fs from 'fs';

const ctxProxy = new Proxy({}, {
  get: (t, p) => (p in t ? t[p] : () => {}),
  set: () => true
});
const mkEl = () => {
  const fakeCanvas = { getContext: () => ctxProxy, width: 0, height: 0 };
  const el = {
    innerHTML: '', textContent: '', className: '', value: '', checked: false,
    disabled: false, style: {}, children: [], onclick: null, onchange: null,
    appendChild() {}, addEventListener() {}, querySelectorAll() { return []; },
    querySelector() { return fakeCanvas; },
    getContext() { return ctxProxy; }
  };
  return el;
};
const store = new Map();
global.document = {
  getElementById(id) { if (!store.has(id)) store.set(id, mkEl()); return store.get(id); },
  createElement() { return mkEl(); }
};
global.location = { protocol: 'file:', origin: 'null' };
global.window = global;
global.fetch = async () => { throw new Error('no backend in test'); };

const names = fs.readdirSync('/Users/wanglongzhen/Downloads/jev')
  .filter(f => /_laya\.html$/.test(f))
  .map(f => f.replace(/_laya\.html$/, ''))
  .sort();

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) pass++; else { fail++; console.log('  FAIL: ' + m); } };

for (const name of names) {
  const html = fs.readFileSync(`/Users/wanglongzhen/Downloads/jev/${name}_laya.html`, 'utf8');
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (scripts.length < 2) { console.log(`${name}: 仅 ${scripts.length} 段 script`); fail++; continue; }
  let Laya, Lab, SCENE;
  try {
    const fn = new Function(scripts[0] + '\n' + scripts[1] + '\nreturn {Laya,Lab,SCENE};');
    ({ Laya, Lab, SCENE } = fn());
  } catch (e) { console.log(`${name}: 加载失败 ${e.message}`); fail++; continue; }
  Laya.ok = false;                       // 强制纯算法模式，不触碰后端
  Lab.sopt = {};
  for (const c of (SCENE.controls || [])) Lab.sopt[c.id] = c.value;
  Lab.strategy = 'rule';
  Lab.ST = Lab.freshStat();

  let err = null;
  try {
    Lab.s = SCENE.init(Lab.seed, Lab.sopt);
    // 动画模式：最多 40 步 tick（rule 不调模型，但会走 metrics/render）
    let steps = 0;
    while (steps < 40 && !SCENE.done(Lab.s)) {
      const cont = await Lab.tick();
      steps++;
      if (Lab.ST.errs > 0) break;
      if (cont === false && SCENE.done(Lab.s)) break;
      if (cont === false) break;
    }
    // 再跑一局批量（rule），验证 playEpisode 路径
    const acc = await Lab.playEpisode(2, 'rule', 3, 400);
    ok(acc.errs === 0, `${name} 批量 errs=${acc.errs}`);
  } catch (e) {
    err = e;
  }
  if (err) { console.log(`  FAIL: ${name} 抛异常: ${err.message}`); fail++; }
  else ok(Lab.ST.errs === 0, `${name} tick errs=${Lab.ST.errs}`);
}

console.log(`\n结果: ${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
