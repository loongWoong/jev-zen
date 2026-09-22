// Headless screenshotter: drives system Edge via playwright-core.
// Captures a consistent-size screenshot of every game HTML page.
const { chromium } = require('C:/Users/Administrator/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const fs = require('fs');
const path = require('path');

const ROOT = 'C:/Users/Administrator/WorkBuddy/Worktrees/jev-zen/main-7e20891f';
const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const OUT = path.join(ROOT, 'portal', 'screenshots');

// canonical list of pages (index first, then games)
const PAGES = [
  'index.html',
  'snake_laya.html',
  'tetris_laya.html',
  'minesweeper_laya.html',
  'sokoban_laya.html',
  'puzzle15_laya.html',
  'lightsout_laya.html',
  'tictactoe_laya.html',
  'connect4_laya.html',
  'othello_laya.html',
  'maze_laya.html',
  'flappy_laya.html',
  'breakout_laya.html',
  'elevator_laya.html',
  'traffic_laya.html',
  'stock_laya.html',
  '2048_laya.html',
];

const W = 1280, H = 820;

(async () => {
  const browser = await chromium.launch({
    executablePath: EDGE,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--disable-extensions'],
  });
  const log = [];
  for (const rel of PAGES) {
    const name = rel.replace(/[._]/g, '_').replace('.html', '');
    const fileUrl = 'file://' + path.join(ROOT, rel);
    const page = await browser.newPage({ viewport: { width: W, height: H } });
    const errors = [];
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', e => errors.push('PAGEERR: ' + e.message));
    try {
      await page.goto(fileUrl, { waitUntil: 'load', timeout: 20000 });
    } catch (e) {
      log.push(`[goto-fail] ${rel}: ${e.message}`);
    }
    // let canvas mount + a few frames render
    await page.waitForTimeout(1500);
    // click any visible start/overlay button
    try {
      const clicked = await page.evaluate(() => {
        const cands = [...document.querySelectorAll('button, .btn, [class*="start"], [class*="overlay"]')];
        const btn = cands.find(b => {
          const t = (b.textContent || '').trim();
          const r = b.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && /开始|开始游戏|点击开始|start|play/i.test(t);
        });
        if (btn) { btn.click(); return btn.textContent.trim(); }
        return null;
      });
      if (clicked) { log.push(`[start-click] ${rel}: "${clicked}"`); await page.waitForTimeout(900); }
    } catch (e) { /* ignore */ }
    // a second render wait (catches auto-play)
    await page.waitForTimeout(1100);
    const outPath = path.join(OUT, name + '.png');
    await page.screenshot({ path: outPath, type: 'png' });
    const stat = fs.statSync(outPath);
    log.push(`[ok] ${rel} -> ${name}.png (${stat.size} bytes)${errors.length ? ' ERR:' + errors.slice(0,3).join(' | ') : ''}`);
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, 'shoot.log'), log.join('\n') + '\n');
  console.log(log.join('\n'));
})().catch(e => { console.error('FATAL', e); process.exit(1); });
