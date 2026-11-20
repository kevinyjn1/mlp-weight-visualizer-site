// The seventh tab: the world-model pilot of the research repo (weld-monitor/experiments/usmw_world_model_pilot,
// 2026-11-18, exploratory), run live in the page. Phase 1: a transformer encoder learned, with no labels, to predict
// hidden 25 ms patches of a weld from the rest (a JEPA). Phase 2: its frozen 128-number summary of each weld goes into
// a logistic-regression head trained on k labeled welds per condition. Every heatmap below is computed here, in the
// browser, from the exported weights (js/wmnet.js), except the loss curve and the accuracies over 5 splits, which come
// from the pilot's logs. Data: tools/world_model.py.

import { SENSOR_COLOR, INK, MUTED, GRID, ACCENT, pct, esc, niceTicks } from './chartkit.js';
import { createModel } from './wmnet.js';

// the 3D view's colours: weights blue (+) and orange (-) on a navy plate, values green (+) and pink (-) on a dark green
// plate, probabilities amber
const PAL = {
  w: { pos: [134, 193, 255], neg: [255, 143, 80], zero: [44, 54, 92], plate: '#1b2240' },
  d: { pos: [116, 230, 149], neg: [255, 116, 179], zero: [37, 66, 49], plate: '#162a1d' },
  p: { pos: [255, 207, 92], neg: [255, 207, 92], zero: [69, 59, 34], plate: '#2b2413' },
};
const ARM = {
  A: { name: 'Wavelet features (528)', short: 'Wavelet', color: '#8a90a0', dash: '5 4' },
  B: { name: 'World model (trained encoder)', short: 'World model', color: ACCENT },
  C: { name: 'Same encoder, random weights', short: 'Random encoder', color: '#CC79A7' },
  D: { name: 'Frame statistics (48)', short: 'Frame statistics', color: '#009E73' },
};
const HIDDEN_LOOK = 'rgba(241, 242, 244, 0.86)'; // a hidden patch, drawn over the frames
const MASK_EDGE = '#7b5cff';
const DPR = () => Math.min(2, window.devicePixelRatio || 1);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
/** Sorted patch numbers -> [first, last] of each run of neighbours. */
const runs = (list) => list.reduce((out, p) => {
  if (out.length && out.at(-1)[1] === p - 1) out.at(-1)[1] = p;
  else out.push([p, p]);
  return out;
}, []);

function rgbOf(pal, t) {
  const P = PAL[pal];
  const a = Math.min(1, Math.abs(t)) ** 0.75;
  const end = t >= 0 ? P.pos : P.neg;
  return `rgb(${Math.round(P.zero[0] + (end[0] - P.zero[0]) * a)},${Math.round(P.zero[1] + (end[1] - P.zero[1]) * a)},${Math.round(P.zero[2] + (end[2] - P.zero[2]) * a)})`;
}

/** The |value| below which 99% of the values lie, as the 3D view scales its blocks. */
function scaleOf(values, q = 0.99) {
  const a = Float32Array.from(values, Math.abs).sort();
  return a[Math.min(a.length - 1, Math.floor(q * a.length))] || 1;
}

/** Size a canvas to its box and the screen; returns its 2D context in CSS pixels. */
function fit(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = DPR();
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) {
    cv.width = w;
    cv.height = h;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}

/** A grid of cells: value(r, c) or null for an empty cell. Cells snap to whole pixels, so there are no seams. */
function heat(ctx, x0, y0, w, h, rows, cols, value, pal, scale, alpha = null) {
  ctx.fillStyle = PAL[pal].plate;
  ctx.fillRect(x0, y0, w, h);
  for (let r = 0; r < rows; r++) {
    const a = alpha ? alpha(r) : 1;
    if (a <= 0) continue;
    ctx.globalAlpha = a;
    const ya = Math.round(y0 + (r * h) / rows);
    const yb = Math.round(y0 + ((r + 1) * h) / rows);
    for (let c = 0; c < cols; c++) {
      const v = value(r, c);
      if (v == null) continue;
      const xa = Math.round(x0 + (c * w) / cols);
      ctx.fillStyle = rgbOf(pal, v / scale);
      ctx.fillRect(xa, ya, Math.round(x0 + ((c + 1) * w) / cols) - xa, yb - ya);
    }
  }
  ctx.globalAlpha = 1;
}

function text(ctx, s, x, y, { size = 11, color = MUTED, align = 'left', weight = 400, base = 'alphabetic' } = {}) {
  ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = base;
  ctx.fillText(s, x, y);
}

export function createWm({ root, data, state, actions }) {
  const meta = data.wm;
  const M = meta.model;
  const NP = M.n_patches;
  const LAST = meta.steps.length - 1;
  const S = state.wm;
  S.snap ??= LAST;
  S.hidden ??= [8, 9, 10, 11, 12, 24, 25, 26, 27];
  const short = (c) => meta.classes[c].replace(' tool, ', ' / ').replace(' surface', '');
  const weldName = (w) => `${short(meta.welds[w].label)} · test weld ${meta.welds[w].test_index + 1}`;
  const stepName = (k) => `step ${meta.steps[k].toLocaleString()}`;

  let model = null;
  let loadState = 'idle'; // idle, loading, ready, failed
  let built = null; // the board on screen
  let raf = null;
  let lastT = 0;
  let anim = 0; // seconds since the current animation started
  const imgCache = new Map();

  root.innerHTML = `<div class="lr-card wm-card"><div class="wm-head"></div><div class="wm-board"></div></div><div class="tip" hidden></div>`;
  const headEl = root.querySelector('.wm-head');
  const boardEl = root.querySelector('.wm-board');
  const tipEl = root.querySelector('.tip');
  const $ = (id) => root.querySelector(`#${id}`);

  // ---------------------------------------------------------------- tooltips on canvases

  function hover(cv, describe) {
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect();
      const html = describe(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
      if (!html) {
        tipEl.hidden = true;
        return;
      }
      tipEl.innerHTML = html;
      tipEl.hidden = false;
      const box = root.getBoundingClientRect();
      const x = e.clientX - box.left;
      const y = e.clientY - box.top + root.scrollTop;
      const left = Math.min(Math.max(x + 14, 8), box.width - tipEl.offsetWidth - 8);
      const top = y + 18 + tipEl.offsetHeight > root.scrollTop + box.height - 8 ? Math.max(y - tipEl.offsetHeight - 12, root.scrollTop + 8) : y + 18;
      tipEl.style.transform = `translate(${left}px, ${top}px)`;
    });
    cv.addEventListener('pointerleave', () => {
      tipEl.hidden = true;
    });
  }

  // ---------------------------------------------------------------- loading (the weights are fetched on first use)

  async function load() {
    loadState = 'loading';
    render();
    try {
      const [weights, welds] = await Promise.all([data.read('wm_weights.bin'), data.read('wm_welds.bin')]);
      model = createModel(meta, weights, welds);
      const gaps = model.selfCheck();
      console.info('world model: page vs PyTorch, largest gaps', gaps);
      loadState = 'ready';
    } catch (error) {
      console.error(error);
      loadState = 'failed';
      boardEl.innerHTML = `<p class="wm-fail"><b>The world model could not be loaded.</b> ${esc(String(error.message ?? error))}</p>`;
      return;
    }
    render();
  }

  // ---------------------------------------------------------------- controls

  const seg = (name, items, value) => `<div class="lr-seg" role="group">${items.map(([v, label]) =>
    `<button type="button" data-act="${name}:${v}" aria-pressed="${String(v) === String(value)}">${label}</button>`).join('')}</div>`;
  const weldSelect = () => `<label class="lr-select"><span>Weld</span><select data-ctl="weld">${meta.welds.map((_, w) =>
    `<option value="${w}"${w === S.weld ? ' selected' : ''}>${esc(weldName(w))}</option>`).join('')}</select></label>`;
  const snapSlider = () => `<label class="lr-select wm-range"><span>Training</span><input type="range" data-ctl="snap" min="0"
    max="${LAST}" value="${S.snap}" aria-label="Training step"><output>${stepName(S.snap)}</output></label>`;
  const playButton = (label) => `<button class="play" type="button" data-act="play" aria-pressed="${S.playing}"
    aria-label="${S.playing ? 'Pause' : label}" title="${S.playing ? 'Pause' : label}">
    <svg class="i-play" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5v11l9.5-5.5z" fill="currentColor"/></svg>
    <svg class="i-pause" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z" fill="currentColor"/></svg></button>`;
  const head = (title, sub, controls) => {
    headEl.innerHTML = `<div class="lr-head"><div><h2>${title}</h2><p>${sub}</p></div><div class="lr-controls">${controls}</div></div>`;
  };

  root.addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el) return;
    const [name, value] = el.dataset.act.split(':');
    act(name, value);
  });
  root.addEventListener('input', (e) => {
    const ctl = e.target.closest('[data-ctl]');
    if (!ctl) return;
    if (ctl.dataset.ctl === 'snap') S.snap = +ctl.value;
    else if (ctl.dataset.ctl === 'weld') S.weld = +ctl.value;
    else if (ctl.dataset.ctl === 'test') S.test = +ctl.value;
    if (ctl.dataset.ctl !== 'weld') {
      const out = ctl.parentElement.querySelector('output');
      if (out) out.textContent = ctl.dataset.ctl === 'snap' ? stepName(S.snap) : `${S.test + 1} of ${model.nTest}`;
    }
    anim = 0;
    paint();
  });
  root.addEventListener('change', (e) => {
    if (e.target.closest('[data-ctl="weld"]')) {
      S.weld = +e.target.value;
      anim = 0;
      paint();
    }
  });

  function act(name, value) {
    if (name === 'play') {
      setPlaying(!S.playing);
      return;
    }
    if (name === 'kind') {
      S.kind = value;
      S.hidden = newMask(value, 0);
    } else if (name === 'newmask') S.hidden = newMask(S.kind, Math.random() * 1e9);
    else if (name === 'net') S.net = value;
    else if (name === 'layer') S.layer = +value;
    else if (name === 'head') S.head = +value;
    else if (name === 'k') S.k = +value;
    else if (name === 'restart') S.test = 0;
    anim = 0;
    render();
  }

  /** The pilot's two mask types (pretrain.sample_mask), drawn with a small seeded generator. */
  function newMask(kind, seed) {
    let s = (Math.floor(seed) % 2147483647) + 1;
    const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
    const st = meta.settings;
    if (kind === 'future') {
      const t = seed === 0 ? 20 : st.FUT_START_MIN + Math.floor(rand() * (st.FUT_START_MAX - st.FUT_START_MIN + 1));
      return [...Array(st.FUT_LEN).keys()].map((i) => t + i).filter((p) => p < NP);
    }
    if (seed === 0) return [8, 9, 10, 11, 12, 24, 25, 26, 27];
    const set = new Set();
    for (let b = 0; b < st.N_BLOCKS; b++) {
      const len = st.BLOCK_MIN + Math.floor(rand() * (st.BLOCK_MAX - st.BLOCK_MIN + 1));
      const start = Math.floor(rand() * (NP - len + 1));
      for (let p = start; p < start + len; p++) set.add(p);
    }
    return [...set].sort((a, b) => a - b);
  }

  // ---------------------------------------------------------------- the animation clock

  function setPlaying(on) {
    S.playing = on;
    const b = headEl.querySelector('[data-act="play"]');
    if (b) {
      b.setAttribute('aria-pressed', String(on));
      b.title = b.ariaLabel = on ? 'Pause' : 'Play';
    }
    cancelAnimationFrame(raf);
    raf = null;
    if (on) {
      lastT = performance.now();
      raf = requestAnimationFrame(frame);
    }
  }

  function frame(t) {
    const dt = Math.min(0.1, (t - lastT) / 1000);
    lastT = t;
    if (!S.playing || loadState !== 'ready') return;
    anim += dt;
    BOARDS[built]?.tick?.(dt);
    paint();
    raf = requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- shared drawings

  /** The weld's 24 x 1,000 frame picture at full resolution, made once and scaled when drawn. */
  function frameImage(w) {
    if (imgCache.has(w)) return imgCache.get(w);
    const F = model.weldFrames(w);
    const cv = document.createElement('canvas');
    cv.width = M.n_frames;
    cv.height = M.n_ch;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(M.n_frames, M.n_ch);
    for (let c = 0; c < M.n_ch; c++) {
      for (let f = 0; f < M.n_frames; f++) {
        const m = rgbOf('d', F[c * M.n_frames + f] / 3).match(/\d+/g);
        const o = 4 * (c * M.n_frames + f);
        img.data[o] = +m[0];
        img.data[o + 1] = +m[1];
        img.data[o + 2] = +m[2];
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    imgCache.set(w, cv);
    return cv;
  }

  /** Frames picture with sensor labels; options: hidden patches, a highlighted patch, a playhead (ms). */
  function drawFrames(ctx, x0, y0, w, h, weld, { hidden = null, focus = null, playhead = null, labels = true, reveal = 1 } = {}) {
    const L = labels ? 96 : 0;
    const pw = w - L;
    ctx.imageSmoothingEnabled = true;
    ctx.fillStyle = PAL.d.plate;
    ctx.fillRect(x0 + L, y0, pw, h);
    const img = frameImage(weld);
    const cut = Math.round(M.n_frames * reveal);
    if (cut > 0) ctx.drawImage(img, 0, 0, cut, M.n_ch, x0 + L, y0, (pw * cut) / M.n_frames, h);
    const rowH = h / M.n_ch;
    if (labels) {
      meta.sensors.forEach((s, i) => {
        text(ctx, s, x0, y0 + (i * 6 + 3) * rowH + 4, { size: 12, color: INK, weight: 650 });
        if (rowH >= 8) for (let v = 0; v < 6; v++) text(ctx, meta.values[v], x0 + L - 5, y0 + (i * 6 + v + 0.5) * rowH, { size: Math.min(9.5, rowH - 1), align: 'right', base: 'middle' });
      });
    }
    ctx.strokeStyle = 'rgba(241,242,244,0.9)';
    ctx.lineWidth = 1.5;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(x0 + L, y0 + i * 6 * rowH);
      ctx.lineTo(x0 + w, y0 + i * 6 * rowH);
      ctx.stroke();
    }
    const pp = pw / NP;
    ctx.strokeStyle = 'rgba(241,242,244,0.2)';
    ctx.lineWidth = 1;
    for (let p = 1; p < NP; p++) {
      ctx.beginPath();
      ctx.moveTo(x0 + L + p * pp, y0);
      ctx.lineTo(x0 + L + p * pp, y0 + h);
      ctx.stroke();
    }
    if (hidden) {
      for (const p of hidden) {
        ctx.fillStyle = HIDDEN_LOOK;
        ctx.fillRect(x0 + L + p * pp, y0, pp, h);
      }
    }
    if (focus != null) {
      ctx.strokeStyle = '#ffcf5c';
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + L + focus * pp + 1, y0 + 1, pp - 2, h - 2);
    }
    if (playhead != null) {
      const x = x0 + L + (playhead / M.n_frames) * pw;
      ctx.strokeStyle = '#ffcf5c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y0 - 4);
      ctx.lineTo(x, y0 + h + 4);
      ctx.stroke();
    }
    return { L, pp };
  }

  /** The 40 patches as a strip of buttons: dark when the encoder sees it, light and dashed when hidden. */
  function drawStrip(ctx, x0, y0, w, h, hidden, { L = 96, query = null } = {}) {
    const pp = (w - L) / NP;
    const set = new Set(hidden);
    text(ctx, 'patches', x0 + L - 5, y0 + h / 2, { size: 11, align: 'right', base: 'middle', color: INK, weight: 600 });
    for (let p = 0; p < NP; p++) {
      const x = x0 + L + p * pp;
      if (set.has(p)) {
        ctx.fillStyle = 'rgba(123, 92, 255, 0.12)';
        ctx.fillRect(x + 1, y0, pp - 2, h);
        ctx.setLineDash([3, 2]);
        ctx.strokeStyle = MASK_EDGE;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x + 1.5, y0 + 0.5, pp - 3, h - 1);
        ctx.setLineDash([]);
      } else {
        ctx.fillStyle = p === query ? '#ffcf5c' : '#3a4566';
        ctx.fillRect(x + 1, y0, pp - 2, h);
      }
      if ((p + 1) % 10 === 0 || p === 0) text(ctx, String(p + 1), x + pp / 2, y0 - 4, { size: 10, align: 'center' });
    }
    return pp;
  }

  /** Bars for one vector: up for positive, down for negative, around a middle line. */
  function bars(ctx, x0, y0, w, h, values, pal, scale, { split = null, reveal = 1 } = {}) {
    ctx.fillStyle = PAL[pal].plate;
    ctx.fillRect(x0, y0, w, h);
    const mid = y0 + h / 2;
    const bw = w / values.length;
    for (let i = 0; i < values.length; i++) {
      const v = Math.max(-1, Math.min(1, values[i] / scale)) * reveal;
      ctx.fillStyle = rgbOf(pal, v >= 0 ? 1 : -1);
      const bh = (Math.abs(v) * (h - 6)) / 2;
      ctx.fillRect(Math.round(x0 + i * bw + 0.5), v >= 0 ? mid - bh : mid, Math.max(1, Math.round(bw - 1)), Math.max(0.5, bh));
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, mid);
    ctx.lineTo(x0 + w, mid);
    ctx.stroke();
    if (split) {
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.beginPath();
      ctx.moveTo(x0 + split * bw, y0);
      ctx.lineTo(x0 + split * bw, y0 + h);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  const rowsAxis = (ctx, x, y0, h, n = NP) => {
    for (const p of [1, 10, 20, 30, 40].filter((v) => v <= n)) text(ctx, String(p), x, y0 + ((p - 0.5) * h) / n, { size: 10, align: 'right', base: 'middle' });
  };

  // ---------------------------------------------------------------- the boards (one per step)

  const BOARDS = {};

  // 1. a weld, as the encoder sees it
  BOARDS.signal = {
    build() {
      head('One weld, as the encoder sees it', `${esc(weldName(S.weld))}. Press play to sweep through the second.`,
        `${weldSelect()}${playButton('Play')}`);
      boardEl.innerHTML = `
        <div class="wm-mod"><div class="wm-mod-head"><b>Raw signals</b><span>1 s at 200 kHz. The band is each 1 ms frame's lowest to highest sample, scaled per sensor</span></div>
          <canvas id="wm-sig" class="wm-cv" style="height:176px"></canvas></div>
        <div class="wm-mod"><div class="wm-mod-head"><b>Frame values</b><span>6 numbers per sensor for every 1 ms frame, standardized: green above the pretraining welds' average, pink below</span></div>
          <canvas id="wm-frames" class="wm-cv" style="height:250px"></canvas></div>
        <div class="wm-row">
          <div class="wm-mod"><div class="wm-mod-head"><b id="wm-patch-title"></b><span>24 rows × 25 frames = 600 numbers</span></div>
            <canvas id="wm-patch" class="wm-cv" style="width:300px;height:250px"></canvas></div>
          <div class="wm-arrow"><span>× embedding matrix</span><small>600 → 64, plus the patch's position</small></div>
          <div class="wm-mod wm-grow"><div class="wm-mod-head"><b>Its input token: 64 numbers</b><span>what the encoder's first layer receives for this patch</span></div>
            <canvas id="wm-token" class="wm-cv" style="height:250px"></canvas></div>
        </div>`;
      hover($('wm-frames'), (x, y, w, h) => {
        const L = 96;
        if (x < L) return null;
        const f = Math.min(M.n_frames - 1, Math.floor(((x - L) / (w - L)) * M.n_frames));
        const c = Math.min(M.n_ch - 1, Math.floor((y / h) * M.n_ch));
        const v = model.weldFrames(S.weld)[c * M.n_frames + f];
        return `<b>${esc(meta.sensors[Math.floor(c / 6)])} ${esc(meta.values[c % 6])}</b><span>frame ${f + 1} (${f} ms), patch ${Math.floor(f / M.patch) + 1}</span><span>value ${v.toFixed(2)} (standardized)</span>`;
      });
      hover($('wm-patch'), (x, y, w, h) => {
        const L = 96;
        if (x < L || y < 18) return null;
        const t = Math.min(M.patch - 1, Math.floor(((x - L) / (w - L)) * M.patch));
        const c = Math.min(M.n_ch - 1, Math.floor(((y - 18) / (h - 18)) * M.n_ch));
        const p = patchAt();
        const v = model.weldFrames(S.weld)[c * M.n_frames + p * M.patch + t];
        return `<b>${esc(meta.sensors[Math.floor(c / 6)])} ${esc(meta.values[c % 6])}</b><span>frame ${p * M.patch + t + 1}</span><span>value ${v.toFixed(2)}</span>`;
      });
    },
    tick(dt) {
      S.playhead = (S.playhead ?? 0) + dt * 160;
      if (S.playhead > M.n_frames) S.playhead = 0;
    },
    paint() {
      const ph = S.playhead ?? 0;
      const p = patchAt();
      {
        const { ctx, w, h } = fit($('wm-sig'));
        const L = 96;
        const rowH = h / 4;
        meta.sensors.forEach((s, i) => {
          const env = model.weldEnv(S.weld, i);
          const y0 = i * rowH;
          ctx.fillStyle = i % 2 ? '#f6f7fa' : '#fbfbfd';
          ctx.fillRect(L, y0, w - L, rowH);
          text(ctx, s, 0, y0 + rowH / 2, { size: 12, color: INK, weight: 650, base: 'middle' });
          ctx.fillStyle = SENSOR_COLOR[s];
          const cols = Math.floor(w - L);
          const cut = S.playing ? Math.floor((ph / M.n_frames) * cols) : cols;
          for (let x = 0; x < cut; x++) {
            const f0 = Math.floor((x / cols) * M.n_frames);
            const f1 = Math.max(f0 + 1, Math.floor(((x + 1) / cols) * M.n_frames));
            let lo = Infinity;
            let hi = -Infinity;
            for (let f = f0; f < f1; f++) {
              lo = Math.min(lo, env[2 * f]);
              hi = Math.max(hi, env[2 * f + 1]);
            }
            const ya = y0 + rowH / 2 - (hi / 1.5) * (rowH / 2 - 3);
            const yb = y0 + rowH / 2 - (lo / 1.5) * (rowH / 2 - 3);
            ctx.fillRect(L + x, ya, 1, Math.max(1, yb - ya));
          }
        });
        const pp = (w - L) / NP;
        ctx.strokeStyle = 'rgba(27,32,48,0.12)';
        for (let q = 1; q < NP; q++) {
          ctx.beginPath();
          ctx.moveTo(L + q * pp, 0);
          ctx.lineTo(L + q * pp, h);
          ctx.stroke();
        }
        ctx.strokeStyle = '#d99a00';
        ctx.lineWidth = 2;
        ctx.strokeRect(L + p * pp + 1, 1, pp - 2, h - 2);
      }
      {
        const { ctx, w, h } = fit($('wm-frames'));
        drawFrames(ctx, 0, 0, w, h - 18, S.weld, { focus: p, playhead: S.playing ? ph : null, reveal: S.playing ? ph / M.n_frames : 1 });
        for (const t of [0, 250, 500, 750, 1000]) text(ctx, `${t} ms`, 96 + (t / 1000) * (w - 96), h - 3, { size: 10.5, align: t === 0 ? 'left' : t === 1000 ? 'right' : 'center' });
      }
      $('wm-patch-title').textContent = `Patch ${p + 1} (${p * 25}-${p * 25 + 25} ms)`;
      {
        const { ctx, w, h } = fit($('wm-patch'));
        const F = model.weldFrames(S.weld);
        const L = 96;
        heat(ctx, L, 18, w - L, h - 18, M.n_ch, M.patch, (r, c) => F[r * M.n_frames + p * M.patch + c], 'd', 3);
        meta.sensors.forEach((s, i) => text(ctx, s, L - 6, 18 + ((i * 6 + 3) * (h - 18)) / M.n_ch, { size: 11.5, color: INK, weight: 650, align: 'right', base: 'middle' }));
        text(ctx, 'frame 1', L, 11, { size: 10 });
        text(ctx, '25', w, 11, { size: 10, align: 'right' });
      }
      {
        const { ctx, w, h } = fit($('wm-token'));
        const e = model.encode(model.snapshot(LAST).target, S.weld);
        const tok = e.embedded.subarray(p * M.dim, (p + 1) * M.dim);
        bars(ctx, 0, 0, w, h - 18, tok, 'd', scaleOf(e.embedded));
        text(ctx, '1', 2, h - 4, { size: 10 });
        text(ctx, '64', w - 2, h - 4, { size: 10, align: 'right' });
        text(ctx, 'trained weights (step 4,000)', w / 2, h - 4, { size: 10, align: 'center' });
      }
    },
  };
  const patchAt = () => Math.min(NP - 1, Math.floor((S.playhead ?? 12 * M.patch + 1) / M.patch));

  // 2. the game
  BOARDS.game = {
    build() {
      head('Phase 1: hide patches, predict them', 'No labels. Click the patches to hide or show them, or press play for a new mask every few seconds.',
        `${weldSelect()}${seg('kind', [['future', 'Future mask'], ['blocks', 'Block mask']], S.kind ?? 'blocks')}
        <button class="secondary wm-btn" type="button" data-act="newmask">New mask</button>${playButton('Play')}`);
      boardEl.innerHTML = `
        <div class="wm-mod"><div class="wm-mod-head"><b>The weld, with the hidden patches blanked</b><span>the encoder sees only the dark patches</span></div>
          <canvas id="wm-game-in" class="wm-cv wm-click" style="height:196px"></canvas></div>
        <div class="wm-row wm-four">
          <div class="wm-mod"><div class="wm-mod-head"><b>1. Encoder output</b><span>visible patches only, 64 numbers each</span></div><canvas id="wm-g1" class="wm-cv" style="height:282px"></canvas></div>
          <div class="wm-mod"><div class="wm-mod-head"><b>2. Predictor's guess</b><span>for each hidden patch</span></div><canvas id="wm-g2" class="wm-cv" style="height:282px"></canvas></div>
          <div class="wm-mod"><div class="wm-mod-head"><b>3. The answer</b><span>target encoder on the whole weld</span></div><canvas id="wm-g3" class="wm-cv" style="height:282px"></canvas></div>
          <div class="wm-mod wm-narrow"><div class="wm-mod-head"><b>4. Error</b><span>|guess − answer|</span></div><canvas id="wm-g4" class="wm-cv" style="height:282px"></canvas></div>
        </div>
        <div class="wm-readout" id="wm-g-read"></div>`;
      const cv = $('wm-game-in');
      cv.addEventListener('click', (e) => {
        const r = cv.getBoundingClientRect();
        const x = e.clientX - r.left - 96;
        if (x < 0) return;
        const p = Math.min(NP - 1, Math.floor((x / (r.width - 96)) * NP));
        const set = new Set(S.hidden);
        if (set.has(p)) set.delete(p);
        else set.add(p);
        if (set.size >= NP) return;
        S.hidden = [...set].sort((a, b) => a - b);
        setPlaying(false);
        paint();
      });
      hover(cv, (x, y, w) => {
        if (x < 96) return null;
        const p = Math.min(NP - 1, Math.floor(((x - 96) / (w - 96)) * NP));
        return `<b>Patch ${p + 1}</b><span>${p * 25}-${p * 25 + 25} ms · ${S.hidden.includes(p) ? 'hidden: click to show' : 'visible: click to hide'}</span>`;
      });
      for (const id of ['wm-g1', 'wm-g2', 'wm-g3']) {
        hover($(id), (x, y, w, h) => {
          const top = 14;
          if (x < 26 || y < top) return null;
          const p = Math.min(NP - 1, Math.floor(((y - top) / (h - top)) * NP));
          const c = Math.min(M.dim - 1, Math.floor(((x - 26) / (w - 26)) * M.dim));
          const g = model.game(LAST, S.weld, S.hidden);
          const j = g.tgt.indexOf(p);
          const i = g.ctx.indexOf(p);
          const v = id === 'wm-g1' ? (i >= 0 ? g.seen.out[i * M.dim + c] : null) : j >= 0 ? (id === 'wm-g2' ? g.guess : g.answer)[j * M.dim + c] : null;
          return v == null ? `<b>Patch ${p + 1}</b><span>${id === 'wm-g1' ? 'hidden: the encoder does not see it' : 'visible: nothing to predict'}</span>` : `<b>Patch ${p + 1}, number ${c + 1}</b><span>${v.toFixed(3)}</span>`;
        });
      }
      hover($('wm-g4'), (x, y, w, h) => {
        const p = Math.min(NP - 1, Math.floor(((y - 14) / (h - 14)) * NP));
        const g = model.game(LAST, S.weld, S.hidden);
        const j = g.tgt.indexOf(p);
        return j < 0 ? null : `<b>Patch ${p + 1}</b><span>average |guess − answer| over its 64 numbers: ${g.err[j].toFixed(3)}</span>`;
      });
      if (S.kind == null) S.kind = 'blocks';
    },
    tick() {
      if (anim > 4.6) {
        S.hidden = newMask(S.kind, Math.random() * 1e9);
        anim = 0;
      }
    },
    paint() {
      const g = model.game(LAST, S.weld, S.hidden);
      const t = S.playing ? anim : 99;
      const pIn = clamp01(t / 0.5);
      const pEnc = clamp01((t - 0.5) / 1.1);
      const pGuess = clamp01((t - 1.7) / 0.6);
      const pAns = clamp01((t - 2.4) / 0.6);
      const pErr = clamp01((t - 3.1) / 0.6);
      {
        const { ctx, w, h } = fit($('wm-game-in'));
        drawStrip(ctx, 0, 14, w, 22, S.hidden);
        drawFrames(ctx, 0, 44, w, h - 44, S.weld, { hidden: pIn > 0 ? S.hidden : [] });
      }
      const sc = 2.5;
      const panel = (id, value, alpha) => {
        const { ctx, w, h } = fit($(id));
        const top = 14;
        heat(ctx, 26, top, w - 26, h - top, NP, M.dim, value, 'd', sc, alpha);
        rowsAxis(ctx, 20, top, h - top);
        text(ctx, 'patch', 0, 9, { size: 10 });
        text(ctx, '64 numbers →', w, 9, { size: 10, align: 'right' });
        ctx.strokeStyle = MASK_EDGE;
        ctx.lineWidth = 1.5;
        for (const [a, b] of runs(g.tgt)) ctx.strokeRect(26.75, top + (a * (h - top)) / NP + 0.75, w - 27.5, ((b - a + 1) * (h - top)) / NP - 1.5);
        return { ctx, w, h, top };
      };
      const ci = new Map(g.ctx.map((p, i) => [p, i]));
      const ti = new Map(g.tgt.map((p, j) => [p, j]));
      panel('wm-g1', (r, c) => (ci.has(r) ? g.seen.out[ci.get(r) * M.dim + c] : null), (r) => clamp01(pEnc * NP - r));
      panel('wm-g2', (r, c) => (ti.has(r) ? g.guess[ti.get(r) * M.dim + c] : null), () => pGuess);
      panel('wm-g3', (r, c) => (ti.has(r) ? g.answer[ti.get(r) * M.dim + c] : null), () => pAns);
      {
        const { ctx, w, h } = fit($('wm-g4'));
        const top = 14;
        const rh = (h - top) / NP;
        ctx.fillStyle = '#f6f7fa';
        ctx.fillRect(0, top, w, h - top);
        const max = 1.2;
        g.tgt.forEach((p, j) => {
          const v = g.err[j] * pErr;
          ctx.fillStyle = '#e0533d';
          ctx.fillRect(0, top + p * rh + 1, (Math.min(v, max) / max) * (w - 30), Math.max(1, rh - 2));
          if (pErr === 1 && rh >= 11) text(ctx, g.err[j].toFixed(2), (Math.min(v, max) / max) * (w - 30) + 3, top + (p + 0.5) * rh, { size: 9.5, base: 'middle', color: INK });
        });
        text(ctx, '0', 0, 9, { size: 10 });
        text(ctx, '1.2', w - 30, 9, { size: 10, align: 'right' });
      }
      $('wm-g-read').innerHTML = g.tgt.length
        ? `<div><b>${pErr === 1 ? g.loss.toFixed(2) : '…'}</b><span>average error on the ${g.tgt.length} hidden patches (trained network, step 4,000)</span></div>
           <div><b>${g.zeros.toFixed(2)}</b><span>the error of a guess that knows nothing (all zeros): the answers are standardized, so this is about 0.8</span></div>
           <div><b>${(model.game(0, S.weld, S.hidden).loss).toFixed(2)}</b><span>the same patches at step 0, before any training</span></div>`
        : '<div><span>Hide at least one patch to give the predictor something to do.</span></div>';
    },
  };

  // 3. attention
  BOARDS.attention = {
    build() {
      head('Inside the encoder: which patches each patch looks at',
        'Every patch gathers information from the other patches, in each of 4 layers and 4 heads. Click a row or a patch.',
        `${weldSelect()}${seg('net', [['B', 'Trained'], ['C', 'Random start']], S.net)}
        ${seg('layer', [0, 1, 2, 3].map((l) => [l, `Layer ${l + 1}`]), S.layer)}
        ${seg('head', [[-1, 'All heads'], [0, '1'], [1, '2'], [2, '3'], [3, '4']], S.head)}${playButton('Play')}`);
      boardEl.innerHTML = `
        <div class="wm-mod"><div class="wm-mod-head"><b id="wm-arc-title"></b><span>line width and darkness: how much of its information comes from each patch</span></div>
          <canvas id="wm-arcs" class="wm-cv wm-click" style="height:180px"></canvas></div>
        <div class="wm-row">
          <div class="wm-mod" style="flex: 0 0 340px"><div class="wm-mod-head"><b>Attention weights</b><span>row: the patch that looks; column: the patch it looks at; each row adds up to 1</span></div>
            <canvas id="wm-attn" class="wm-cv wm-click" style="width:340px;height:340px"></canvas></div>
          <div class="wm-mod wm-grow"><div class="wm-mod-head"><b>The 40 tokens, layer by layer</b><span>each block: 40 patches (rows) × 64 numbers</span></div>
            <canvas id="wm-stack" class="wm-cv" style="height:340px"></canvas></div>
        </div>`;
      const pickRow = (y, h, top) => Math.min(NP - 1, Math.max(0, Math.floor(((y - top) / (h - top)) * NP)));
      $('wm-attn').addEventListener('click', (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        S.query = pickRow(e.clientY - r.top, r.height, 16);
        setPlaying(false);
        paint();
      });
      $('wm-arcs').addEventListener('click', (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - r.left - 96;
        if (x < 0) return;
        S.query = Math.min(NP - 1, Math.floor((x / (r.width - 96)) * NP));
        setPlaying(false);
        paint();
      });
      hover($('wm-attn'), (x, y, w, h) => {
        if (x < 22 || y < 16) return null;
        const q = pickRow(y, h, 16);
        const k = Math.min(NP - 1, Math.floor(((x - 22) / (w - 22)) * NP));
        const A = attnMatrix();
        return `<b>Patch ${q + 1} → patch ${k + 1}</b><span>attention ${(100 * A[q * NP + k]).toFixed(1)}% of patch ${q + 1}'s input</span>`;
      });
      S.query ??= 12;
    },
    tick() {
      if (anim > 0.45) {
        S.query = ((S.query ?? 0) + 1) % NP;
        anim = 0;
      }
    },
    paint() {
      const A = attnMatrix();
      const q = S.query ?? 12;
      const e = encoded();
      $('wm-arc-title').textContent = `Where patch ${q + 1} looks (${S.net === 'B' ? 'trained' : 'random'} encoder, layer ${S.layer + 1}, ${S.head < 0 ? 'average of 4 heads' : `head ${S.head + 1}`})`;
      {
        const { ctx, w, h } = fit($('wm-arcs'));
        const base = h - 96;
        const L = 96;
        const pw = (w - L) / NP;
        ctx.drawImage(frameImage(S.weld), 0, 0, M.n_frames, M.n_ch, L, base, w - L, 82);
        text(ctx, 'the weld', L - 6, base + 41, { size: 11, align: 'right', base: 'middle', color: INK, weight: 600 });
        let max = 0;
        for (let k = 0; k < NP; k++) max = Math.max(max, A[q * NP + k]);
        const xq = L + (q + 0.5) * pw;
        for (let k = 0; k < NP; k++) {
          const a = A[q * NP + k] / max;
          if (a < 0.04) continue;
          const xk = L + (k + 0.5) * pw;
          ctx.strokeStyle = `rgba(44, 98, 220, ${0.15 + 0.85 * a})`;
          ctx.lineWidth = 0.6 + 5 * a;
          ctx.beginPath();
          ctx.moveTo(xq, base - 2);
          ctx.quadraticCurveTo((xq + xk) / 2, base - 8 - Math.min(base - 14, Math.abs(xk - xq) * 0.45), xk, base - 2);
          ctx.stroke();
        }
        ctx.fillStyle = '#d99a00';
        ctx.beginPath();
        ctx.arc(xq, base - 2, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#ffcf5c';
        ctx.lineWidth = 2;
        ctx.strokeRect(L + q * pw + 1, base + 1, pw - 2, 80);
        for (const p of [1, 10, 20, 30, 40]) text(ctx, String(p), L + (p - 0.5) * pw, h - 2, { size: 10, align: 'center' });
      }
      {
        const { ctx, w, h } = fit($('wm-attn'));
        const top = 16;
        const L = 22;
        let max = 0;
        for (const v of A) max = Math.max(max, v);
        heat(ctx, L, top, w - L, h - top, NP, NP, (r, c) => A[r * NP + c], 'p', max * 0.9);
        rowsAxis(ctx, L - 4, top, h - top);
        text(ctx, 'looks at →', w, 11, { size: 10, align: 'right' });
        const rh = (h - top) / NP;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(L + 0.5, top + q * rh, w - L - 1, rh);
      }
      {
        const { ctx, w, h } = fit($('wm-stack'));
        const top = 16;
        const gap = 10;
        const bw = (w - 4 * gap) / 5;
        const scale = 3;
        e.layers.forEach((X, l) => {
          const x0 = l * (bw + gap);
          const sc = scaleOf(X, 0.98) || scale;
          heat(ctx, x0, top, bw, h - top, NP, M.dim, (r, c) => X[r * M.dim + c], 'd', sc);
          text(ctx, l === 0 ? 'input tokens' : `after layer ${l}`, x0 + bw / 2, 11, { size: 10.5, align: 'center', color: l === S.layer + 1 ? INK : MUTED, weight: l === S.layer + 1 ? 650 : 400 });
          const rh = (h - top) / NP;
          ctx.strokeStyle = '#ffcf5c';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x0 + 0.5, top + q * rh, bw - 1, rh);
        });
      }
    },
  };
  function encoded() {
    const net = S.net === 'B' ? model.snapshot(LAST).target : model.snapshot(0).encoder;
    return model.encode(net, S.weld);
  }
  function attnMatrix() {
    const a = encoded().attn[S.layer];
    if (S.head >= 0) return a.subarray(S.head * NP * NP, (S.head + 1) * NP * NP);
    const out = new Float32Array(NP * NP);
    for (let h = 0; h < M.heads; h++) for (let i = 0; i < NP * NP; i++) out[i] += a[h * NP * NP + i] / M.heads;
    return out;
  }

  // 4. the weights
  const GROUPS = [
    ['Patch embedding', (t) => [t['embed.weight']]],
    ...[0, 1, 2, 3].flatMap((l) => [
      [`Layer ${l + 1} attention`, (t) => [t[`blocks.layers.${l}.self_attn.in_proj_weight`], t[`blocks.layers.${l}.self_attn.out_proj.weight`]]],
      [`Layer ${l + 1} feed-forward`, (t) => [t[`blocks.layers.${l}.linear1.weight`], t[`blocks.layers.${l}.linear2.weight`]]],
    ]),
  ];
  function moved(k) {
    const now = model.snapshot(k).encoder.t;
    const start = model.snapshot(0).encoder.t;
    return GROUPS.map(([name, get]) => {
      let d = 0;
      let n0 = 0;
      get(now).forEach((W, i) => {
        const W0 = get(start)[i];
        for (let j = 0; j < W.length; j++) {
          d += (W[j] - W0[j]) ** 2;
          n0 += W0[j] ** 2;
        }
      });
      return [name, Math.sqrt(d / n0)];
    });
  }
  function fieldsDraw(ctx, x0, y0, w, h, k, units, perRow) {
    const W = model.snapshot(k).encoder.t['embed.weight'];
    const sc = scaleOf(model.snapshot(LAST).encoder.t['embed.weight']);
    const gap = 5;
    const rows = Math.ceil(units / perRow);
    const tw = (w - (perRow - 1) * gap) / perRow;
    const th = (h - (rows - 1) * gap) / rows;
    for (let u = 0; u < units; u++) {
      const x = x0 + (u % perRow) * (tw + gap);
      const y = y0 + Math.floor(u / perRow) * (th + gap);
      heat(ctx, x, y, tw, th, M.n_ch, M.patch, (r, c) => W[u * M.n_ch * M.patch + r * M.patch + c], 'w', sc);
    }
    return { tw, th, gap };
  }
  BOARDS.weights = {
    build() {
      head('The weights, as training moves them', 'Drag the training step or press play: the same 172,480 numbers at 9 points of the 4,000 steps.',
        `${snapSlider()}${seg('layer', [0, 1, 2, 3].map((l) => [l, `Layer ${l + 1}`]), S.layer)}${playButton('Play')}`);
      boardEl.innerHTML = `
        <div class="wm-row">
          <div class="wm-mod" style="flex: 0 0 470px"><div class="wm-mod-head"><b>Patch embedding: what each of the 64 numbers looks for</b><span>one tile per number: its 600 weights laid out like a patch, 24 rows × 25 frames. Blue adds, orange subtracts</span></div>
            <canvas id="wm-fields" class="wm-cv" style="width:470px;height:470px"></canvas></div>
          <div class="wm-mod wm-grow"><div class="wm-mod-head"><b>How far each part has moved</b><span>size of the change since step 0, relative to the random start</span></div>
            <canvas id="wm-moved" class="wm-cv" style="height:470px"></canvas></div>
        </div>
        <div class="wm-mod"><div class="wm-mod-head"><b id="wm-mats-title"></b><span>rows: outputs, columns: inputs. Same colour scale at every step (the trained weights' 99th percentile)</span></div>
          <canvas id="wm-mats" class="wm-cv" style="height:300px"></canvas></div>`;
      hover($('wm-fields'), (x, y, w, h) => {
        const per = 8;
        const gap = 5;
        const tw = (w - 7 * gap) / per;
        const th = (h - 7 * gap) / per;
        const u = Math.floor(y / (th + gap)) * per + Math.floor(x / (tw + gap));
        return u < 64 ? `<b>Number ${u + 1} of the 64</b><span>its weights on the 600 numbers of a patch, ${stepName(S.snap)}</span>` : null;
      });
    },
    tick() {
      if (anim > 1.1) {
        S.snap = (S.snap + 1) % meta.steps.length;
        anim = 0;
        const r = headEl.querySelector('[data-ctl="snap"]');
        r.value = S.snap;
        r.parentElement.querySelector('output').textContent = stepName(S.snap);
      }
    },
    paint() {
      {
        const { ctx, w, h } = fit($('wm-fields'));
        fieldsDraw(ctx, 0, 0, w, h, S.snap, 64, 8);
      }
      {
        const { ctx, w, h } = fit($('wm-moved'));
        const list = moved(S.snap);
        const fin = moved(LAST);
        const max = Math.max(...fin.map((d) => d[1]), 0.05) * 1.1;
        const L = 150;
        const rh = h / list.length;
        list.forEach(([name, v], i) => {
          const y = i * rh;
          text(ctx, name, L - 8, y + rh / 2, { size: 11.5, align: 'right', base: 'middle', color: INK });
          ctx.fillStyle = '#eef0f5';
          ctx.fillRect(L, y + rh * 0.22, w - L - 50, rh * 0.56);
          ctx.fillStyle = '#86c1ff';
          ctx.fillRect(L, y + rh * 0.22, ((w - L - 50) * fin[i][1]) / max, rh * 0.56);
          ctx.fillStyle = ACCENT;
          ctx.fillRect(L, y + rh * 0.22, ((w - L - 50) * v) / max, rh * 0.56);
          text(ctx, `${(100 * v).toFixed(0)}%`, L + ((w - L - 50) * v) / max + 5, y + rh / 2, { size: 11, base: 'middle', color: INK, weight: 600 });
        });
      }
      const t = model.snapshot(S.snap).encoder.t;
      const tf = model.snapshot(LAST).encoder.t;
      const l = S.layer;
      $('wm-mats-title').textContent = `Layer ${l + 1}'s matrices, ${stepName(S.snap)}`;
      {
        const { ctx, w, h } = fit($('wm-mats'));
        const inW = t[`blocks.layers.${l}.self_attn.in_proj_weight`];
        const inF = tf[`blocks.layers.${l}.self_attn.in_proj_weight`];
        const d = M.dim;
        const mats = [
          ['Query', inW.subarray(0, d * d), inF.subarray(0, d * d), d, d],
          ['Key', inW.subarray(d * d, 2 * d * d), inF.subarray(d * d, 2 * d * d), d, d],
          ['Value', inW.subarray(2 * d * d), inF.subarray(2 * d * d), d, d],
          ['Attn out', t[`blocks.layers.${l}.self_attn.out_proj.weight`], tf[`blocks.layers.${l}.self_attn.out_proj.weight`], d, d],
          ['FF in, 128 × 64', t[`blocks.layers.${l}.linear1.weight`], tf[`blocks.layers.${l}.linear1.weight`], 2 * d, d],
          ['FF out, 64 × 128', t[`blocks.layers.${l}.linear2.weight`], tf[`blocks.layers.${l}.linear2.weight`], d, 2 * d],
        ];
        const top = 16;
        const gap = 14;
        const unit = Math.min((h - top) / (2 * d), (w - 5 * gap) / (7 * d));
        let x = 0;
        for (const [name, W, Wf, rows, cols] of mats) {
          const mw = cols * unit;
          const mh = rows * unit;
          heat(ctx, x, top, mw, mh, rows, cols, (r, c) => W[r * cols + c], 'w', scaleOf(Wf));
          text(ctx, name, x, 11, { size: 10.5, color: INK, weight: 600 });
          x += mw + gap;
        }
      }
    },
  };

  // 5. training replayed
  BOARDS.train = {
    build() {
      head('Training, replayed', 'The same weld and the same hidden patches, at 9 points of the 4,000 steps. Everything below is recomputed at each step.',
        `${weldSelect()}${snapSlider()}${seg('kind', [['future', 'Future mask'], ['blocks', 'Block mask']], S.kind ?? 'blocks')}${playButton('Play')}`);
      boardEl.innerHTML = `
        <div class="wm-mod"><div class="wm-mod-head"><b>The error during training</b><span>line: split 0's training error, averaged over each 100 steps of batches; dots: this weld's hidden patches, computed here</span></div>
          <canvas id="wm-curve" class="wm-cv" style="height:220px"></canvas></div>
        <div class="wm-row">
          <div class="wm-mod wm-grow"><div class="wm-mod-head"><b id="wm-tr-title"></b><span>each pair of rows: the predictor's guess (top) and the answer (below) for one hidden patch</span></div>
            <canvas id="wm-pairs" class="wm-cv" style="height:300px"></canvas></div>
          <div class="wm-mod"><div class="wm-mod-head"><b>16 of the 64 patch filters</b><span>as in the weights step</span></div>
            <canvas id="wm-mini" class="wm-cv" style="width:250px;height:300px"></canvas></div>
        </div>`;
      if (S.kind == null) S.kind = 'blocks';
    },
    tick: BOARDS.weights.tick,
    paint() {
      const hidden = newMask(S.kind ?? 'blocks', 0);
      const errs = meta.steps.map((_, k) => model.game(k, S.weld, hidden).loss);
      {
        const { ctx, w, h } = fit($('wm-curve'));
        const L = 44;
        const B = 26;
        const x = (s) => L + (s / 4000) * (w - L - 12);
        const lo = 0.2;
        const hi = Math.max(1, ...errs);
        const y = (v) => 8 + (1 - (v - lo) / (hi - lo)) * (h - B - 8);
        for (const t of niceTicks(lo, hi, 4)) {
          ctx.strokeStyle = GRID;
          ctx.beginPath();
          ctx.moveTo(L, y(t));
          ctx.lineTo(w - 12, y(t));
          ctx.stroke();
          text(ctx, t.toFixed(1), L - 6, y(t), { size: 10, align: 'right', base: 'middle' });
        }
        for (const s of [0, 1000, 2000, 3000, 4000]) text(ctx, s.toLocaleString(), x(s), h - 8, { size: 10, align: s === 4000 ? 'right' : 'center' });
        const lg = meta.loss;
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 2;
        ctx.beginPath();
        lg.step.forEach((s, i) => (i ? ctx.lineTo(x(s), y(lg.all[i])) : ctx.moveTo(x(s), y(lg.all[i]))));
        ctx.stroke();
        ctx.strokeStyle = '#d99a00';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        meta.steps.forEach((s, k) => (k ? ctx.lineTo(x(s), y(errs[k])) : ctx.moveTo(x(s), y(errs[k]))));
        ctx.stroke();
        ctx.setLineDash([]);
        meta.steps.forEach((s, k) => {
          ctx.fillStyle = k === S.snap ? '#d99a00' : '#f6c75a';
          ctx.beginPath();
          ctx.arc(x(s), y(errs[k]), k === S.snap ? 6 : 4, 0, 2 * Math.PI);
          ctx.fill();
        });
        ctx.strokeStyle = 'rgba(27,32,48,0.35)';
        ctx.beginPath();
        ctx.moveTo(x(meta.steps[S.snap]), 8);
        ctx.lineTo(x(meta.steps[S.snap]), h - B);
        ctx.stroke();
        const right = x(meta.steps[S.snap]) > w - 160;
        text(ctx, `this weld: ${errs[S.snap].toFixed(2)}`, x(meta.steps[S.snap]) + (right ? -10 : 10), y(errs[S.snap]) - 12, { size: 11.5, color: INK, weight: 650, align: right ? 'right' : 'left' });
        text(ctx, 'split 0, all its welds', x(2000), y(lg.all[lg.step.indexOf(2000)]) - 10, { size: 11, align: 'center', color: ACCENT, weight: 600 });
      }
      const g = model.game(S.snap, S.weld, hidden);
      $('wm-tr-title').textContent = `Guesses and answers, ${stepName(S.snap)}: average error ${g.loss.toFixed(2)}`;
      {
        const { ctx, w, h } = fit($('wm-pairs'));
        const L = 70;
        const n = g.tgt.length;
        const rh = Math.min(16, (h - 4) / (2 * n + n * 0.6));
        let y = 0;
        g.tgt.forEach((p, j) => {
          text(ctx, `patch ${p + 1}`, L - 6, y + rh, { size: 10.5, align: 'right', base: 'middle', color: INK });
          heat(ctx, L, y, w - L - 50, rh, 1, M.dim, (r, c) => g.guess[j * M.dim + c], 'd', 2.5);
          heat(ctx, L, y + rh, w - L - 50, rh, 1, M.dim, (r, c) => g.answer[j * M.dim + c], 'd', 2.5);
          text(ctx, g.err[j].toFixed(2), w - 4, y + rh, { size: 11, align: 'right', base: 'middle', color: '#c4321f', weight: 650 });
          y += 2 * rh + rh * 0.6;
        });
      }
      {
        const { ctx, w, h } = fit($('wm-mini'));
        fieldsDraw(ctx, 0, 0, w, h, S.snap, 16, 4);
      }
    },
  };

  // 6. phase 2: summarize
  BOARDS.pool = {
    build() {
      head('Phase 2: freeze the encoder, summarize each weld', 'The trained encoder against the same encoder at its random start. Press play to go through the 18 welds.',
        `${weldSelect()}${playButton('Play')}`);
      boardEl.innerHTML = `<div class="wm-row wm-two">${['B', 'C'].map((a) => `
        <div class="wm-mod"><div class="wm-mod-head"><b>${a === 'B' ? 'Trained encoder (step 4,000)' : 'Random start (step 0)'}</b><span>${a === 'B' ? 'the world model' : 'the control'}</span></div>
          <canvas id="wm-pt-${a}" class="wm-cv" style="height:250px"></canvas>
          <div class="wm-down">↓ mean over the 40 patches, then their spread</div>
          <canvas id="wm-pb-${a}" class="wm-cv" style="height:130px"></canvas></div>`).join('')}</div>
        <div class="wm-readout" id="wm-pool-read"></div>`;
    },
    tick() {
      if (anim > 2.6) {
        S.weld = (S.weld + 1) % meta.welds.length;
        anim = 0;
        const sel = headEl.querySelector('[data-ctl="weld"]');
        if (sel) sel.value = S.weld;
        head('Phase 2: freeze the encoder, summarize each weld', `${esc(weldName(S.weld))}.`, `${weldSelect()}${playButton('Play')}`);
      }
    },
    paint() {
      const rev = S.playing ? clamp01(anim / 0.9) : 1;
      const grow = S.playing ? clamp01((anim - 0.9) / 0.6) : 1;
      const res = { B: model.pooled(model.snapshot(LAST).target, S.weld), C: model.pooled(model.snapshot(0).encoder, S.weld) };
      const sc = Math.max(...Object.values(res).map((r) => scaleOf(r.pooled, 1)));
      for (const a of ['B', 'C']) {
        const r = res[a];
        {
          const { ctx, w, h } = fit($(`wm-pt-${a}`));
          heat(ctx, 26, 0, w - 26, h, NP, M.dim, (row, c) => r.enc.out[row * M.dim + c], 'd', 2.5, (row) => clamp01(rev * NP - row));
          rowsAxis(ctx, 20, 0, h);
        }
        {
          const { ctx, w, h } = fit($(`wm-pb-${a}`));
          bars(ctx, 26, 0, w - 26, h - 16, r.pooled, 'd', sc, { split: M.dim, reveal: grow });
          text(ctx, '64 means', 26 + (w - 26) / 4, h - 3, { size: 10.5, align: 'center' });
          text(ctx, '64 spreads', 26 + (3 * (w - 26)) / 4, h - 3, { size: 10.5, align: 'center' });
        }
      }
      $('wm-pool-read').innerHTML = `<div><b>${esc(meta.classes[meta.welds[S.weld].label])}</b><span>this weld's true condition (the encoder never sees it)</span></div>
        <div><b>128</b><span>numbers per weld go on to the head; nothing above changes in phase 2</span></div>`;
    },
  };

  // 7. the head
  BOARDS.head = {
    build() {
      S.test ??= 0;
      head('Phase 2: the classification head', `Logistic regression on the 128 numbers, trained on k labeled welds per condition (split 0, draw 0). Play runs the ${model.nTest} test welds.`,
        `${seg('net', [['B', 'Trained encoder'], ['C', 'Random start']], S.net)}${seg('k', meta.ks.map((k) => [k, k === 18 ? 'All 18' : `k = ${k}`]), S.k)}
        <label class="lr-select wm-range"><span>Test weld</span><input type="range" data-ctl="test" min="0" max="${model.nTest - 1}" value="${S.test}"><output>${S.test + 1} of ${model.nTest}</output></label>
        ${playButton('Run the test welds')}`);
      boardEl.innerHTML = `
        <div class="wm-mod"><div class="wm-mod-head"><b id="wm-h-title"></b><span>the weld's 128 numbers (standardized), times each class's 128 weights, plus its bias = that class's score</span></div>
          <canvas id="wm-head" class="wm-cv" style="height:300px"></canvas></div>
        <div class="wm-row">
          <div class="wm-mod"><div class="wm-mod-head"><b>Test welds so far</b><span>rows: true condition; columns: the head's answer</span></div>
            <canvas id="wm-conf" class="wm-cv" style="width:510px;height:380px"></canvas></div>
          <div class="wm-readout wm-col" id="wm-h-read"></div>
        </div>`;
    },
    tick() {
      if (anim > 0.55) {
        if (S.test >= model.nTest - 1) {
          setPlaying(false);
          return;
        }
        S.test += 1;
        anim = 0;
        const r = headEl.querySelector('[data-ctl="test"]');
        r.value = S.test;
        r.parentElement.querySelector('output').textContent = `${S.test + 1} of ${model.nTest}`;
      }
    },
    paint() {
      const x = model.summary(S.net, S.test);
      const H = meta.heads[S.net][String(S.k)];
      const r = model.head(S.net, S.k, x);
      const truth = meta.test_labels[S.test];
      const guess = r.probs.indexOf(Math.max(...r.probs));
      $('wm-h-title').textContent = `Test weld ${S.test + 1}: ${meta.classes[truth]}`;
      {
        const { ctx, w, h } = fit($('wm-head'));
        const L = 150;
        const R = 230;
        const top = 34;
        const mw = w - L - R;
        heat(ctx, L, 0, mw, 22, 1, 2 * M.dim, (_, c) => r.z[c], 'd', 2.5);
        text(ctx, '128 numbers', L - 8, 11, { size: 11, align: 'right', base: 'middle', color: INK, weight: 600 });
        const coef = H.coef;
        const sc = scaleOf(coef.flat());
        heat(ctx, L, top, mw, h - top, 9, 2 * M.dim, (row, c) => coef[row][c], 'w', sc);
        const rh = (h - top) / 9;
        for (let c = 0; c < 9; c++) {
          const y = top + c * rh;
          text(ctx, (c === truth ? '✓ ' : '') + short(c), L - 8, y + rh / 2, { size: 11.5, align: 'right', base: 'middle', color: c === truth ? INK : MUTED, weight: c === truth ? 650 : 400 });
          const sx = L + mw + 12;
          text(ctx, r.logits[c].toFixed(1), sx + 34, y + rh / 2, { size: 10.5, align: 'right', base: 'middle', color: MUTED });
          ctx.fillStyle = '#eef0f5';
          ctx.fillRect(sx + 42, y + rh * 0.2, R - 96, rh * 0.6);
          ctx.fillStyle = c === guess ? (guess === truth ? '#1d7a43' : '#c4321f') : '#9aa1b2';
          ctx.fillRect(sx + 42, y + rh * 0.2, (R - 96) * r.probs[c], rh * 0.6);
          if (r.probs[c] > 0.02) text(ctx, pct(r.probs[c], 0), sx + 42 + (R - 96) * r.probs[c] + 4, y + rh / 2, { size: 10.5, base: 'middle', color: INK, weight: 600 });
        }
        text(ctx, 'score', L + mw + 46, 26, { size: 10, align: 'right' });
        text(ctx, 'probability', L + mw + 54, 26, { size: 10 });
      }
      const conf = Array.from({ length: 9 }, () => new Array(9).fill(0));
      let ok = 0;
      for (let i = 0; i <= S.test; i++) {
        const p = model.head(S.net, S.k, model.summary(S.net, i)).probs;
        const g = p.indexOf(Math.max(...p));
        conf[meta.test_labels[i]][g] += 1;
        if (g === meta.test_labels[i]) ok++;
      }
      {
        const { ctx, w, h } = fit($('wm-conf'));
        const L = 150;
        const top = 18;
        const cell = Math.min((w - L) / 9, (h - top) / 9);
        const maxRow = 6;
        for (let i = 0; i < 9; i++) {
          text(ctx, short(i), L - 6, top + (i + 0.5) * cell, { size: 10.5, align: 'right', base: 'middle', color: INK });
          for (let j = 0; j < 9; j++) {
            const v = conf[i][j];
            const t = v / maxRow;
            ctx.fillStyle = v === 0 ? '#f3f4f7' : i === j ? `rgba(29, 122, 67, ${0.25 + 0.75 * t})` : `rgba(196, 50, 31, ${0.3 + 0.7 * t})`;
            ctx.fillRect(L + j * cell + 1, top + i * cell + 1, cell - 2, cell - 2);
            if (v) text(ctx, String(v), L + (j + 0.5) * cell, top + (i + 0.5) * cell, { size: 11.5, align: 'center', base: 'middle', color: t > 0.5 ? '#fff' : INK, weight: 650 });
          }
          text(ctx, String(i + 1), L + (i + 0.5) * cell, 10, { size: 10, align: 'center' });
        }
        ctx.strokeStyle = '#ffcf5c';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(L + guess * cell + 1, top + truth * cell + 1, cell - 2, cell - 2);
        for (let b = 1; b < 3; b++) {
          ctx.strokeStyle = MUTED;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(L + 3 * b * cell, top);
          ctx.lineTo(L + 3 * b * cell, top + 9 * cell);
          ctx.moveTo(L, top + 3 * b * cell);
          ctx.lineTo(L + 9 * cell, top + 3 * b * cell);
          ctx.stroke();
        }
      }
      const other = S.net === 'B' ? 'C' : 'B';
      $('wm-h-read').innerHTML = `
        <div><b>${ok} of ${S.test + 1}</b><span>test welds right so far (${pct(ok / (S.test + 1))})</span></div>
        <div><b>${pct(H.test_acc)}</b><span>all ${model.nTest} test welds, ${S.net === 'B' ? 'trained encoder' : 'random start'}, k = ${S.k}</span></div>
        <div><b>${pct(meta.heads[other][String(S.k)].test_acc)}</b><span>the same with the ${other === 'B' ? 'trained encoder' : 'random start'}</span></div>
        <div><span>Columns are numbered like the rows (1 = ${esc(short(0))}). The yellow box is this weld. Each condition has 6 test welds.</span></div>`;
    },
  };

  // 8. verdict
  BOARDS.verdict = {
    build() {
      head('What it adds up to', 'All 5 splits × 10 draws of labeled welds, from the pilot (RESULTS.md); the agreement is computed here for split 0.', '');
      boardEl.innerHTML = `
        <div class="wm-mod"><div class="wm-mod-head"><b>9-condition accuracy against labeled welds per condition</b><span>mean and 95% interval over 5 splits × 10 draws; guessing scores 11.1%</span></div>
          <div id="wm-acc"></div></div>
        <div class="wm-readout" id="wm-v-read"></div>`;
    },
    paint() {
      const ks = meta.ks;
      const W = Math.min(960, boardEl.clientWidth - 4);
      const H = 320;
      const L = 56;
      const Rr = 150;
      const lx = (k) => L + (Math.log(k) / Math.log(18)) * (W - L - Rr);
      const y = (v) => 10 + (1 - (v - 0.6) / 0.4) * (H - 50);
      let s = `<svg class="lr-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
      for (const t of [0.6, 0.7, 0.8, 0.9, 1]) s += `<line x1="${L}" x2="${W - Rr}" y1="${y(t)}" y2="${y(t)}" stroke="${GRID}"/><text class="tick" x="${L - 8}" y="${y(t) + 4}" text-anchor="end">${pct(t, 0)}</text>`;
      for (const k of ks) s += `<text class="tick" x="${lx(k)}" y="${H - 22}" text-anchor="middle">${k === 18 ? '18 (all)' : k}</text>`;
      s += `<text class="axis" x="${(L + W - Rr) / 2}" y="${H - 4}" text-anchor="middle">labeled welds per condition (log scale)</text>`;
      const ends = [];
      for (const a of ['A', 'D', 'C', 'B']) {
        const r = meta.acc[a];
        if (a !== 'A') s += `<path d="${ks.map((k, i) => `${i ? 'L' : 'M'}${lx(k)},${y(r[i].hi)}`).join('')}${ks.map((k, i) => `L${lx(k)},${y(r[i].lo)}`).reverse().join('')}Z" fill="${ARM[a].color}" opacity="0.1"/>`;
        s += `<path d="${ks.map((k, i) => `${i ? 'L' : 'M'}${lx(k)},${y(r[i].mean)}`).join('')}" fill="none" stroke="${ARM[a].color}" stroke-width="${a === 'B' ? 2.6 : 2}" ${ARM[a].dash ? `stroke-dasharray="${ARM[a].dash}"` : ''}/>`;
        ks.forEach((k, i) => {
          s += `<circle cx="${lx(k)}" cy="${y(r[i].mean)}" r="4" fill="${ARM[a].color}" stroke="#fff" stroke-width="1.5"/>`;
        });
        ends.push({ a, y0: y(r.at(-1).mean), y: y(r.at(-1).mean) });
      }
      ends.sort((p, q) => p.y - q.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
      for (const e of ends) {
        s += `<path d="M${lx(18) + 6},${e.y0} L${lx(18) + 20},${e.y} L${lx(18) + 26},${e.y}" fill="none" stroke="${ARM[e.a].color}" stroke-width="1.2"/>`;
        s += `<text class="lbl strong" x="${lx(18) + 30}" y="${e.y + 4}">${ARM[e.a].short}</text>`;
      }
      s += '</svg>';
      $('wm-acc').innerHTML = s;
      const agree = ks.map((k) => {
        let same = 0;
        for (let i = 0; i < model.nTest; i++) {
          const pb = model.head('B', k, model.summary('B', i)).probs;
          const pc = model.head('C', k, model.summary('C', i)).probs;
          if (pb.indexOf(Math.max(...pb)) === pc.indexOf(Math.max(...pc))) same++;
        }
        return same;
      });
      const k5 = ks.indexOf(5);
      $('wm-v-read').innerHTML = `
        <div><b>${pct(meta.acc.B[k5].mean)} vs ${pct(meta.acc.C[k5].mean)}</b><span>trained vs random-start encoder at k = 5: phase 1 added nothing measurable</span></div>
        <div><b>${pct(meta.acc.D[k5].mean)}</b><span>48 plain frame statistics (mean and spread of each row of the frame picture), no network</span></div>
        <div><b>${agree[k5]} of ${model.nTest}</b><span>split 0 test welds on which the trained and random heads give the same answer (k = 5)</span></div>`;
    },
  };

  // ---------------------------------------------------------------- the walkthrough

  const STEPS = [
    {
      title: 'One weld, as the encoder sees it',
      board: 'signal',
      body: () => `
        <p>This tab runs the world-model pilot's network live, in your browser, with its real weights. It is a JEPA:
        a transformer encoder that learned from welds without labels.</p>
        <p>A weld is 1 second of 4 sensors at 200 kHz. The network never reads the raw signal. Each sensor is cut into
        1,000 frames of 1 ms, and each frame becomes 6 numbers: its mean, its log RMS, and its log energy in 4
        frequency bands. That is the 24 × 1,000 picture.</p>
        <p>The encoder reads the picture as <b>40 patches of 25 ms</b>. A learned matrix turns each patch's 600 numbers
        into 64, its token.</p>
        <p class="next">Press play to sweep through the weld: the patch under the playhead and its token are on the
        right. Pick another weld at the top.</p>`,
    },
    {
      title: 'Phase 1: the game of hidden patches',
      board: 'game',
      body: () => `
        <p>Phase 1 uses ${meta.n_pretrain} welds and no labels. Each training step hides some patches and asks:</p>
        <ol>
          <li>The <b>encoder</b> reads only the visible patches.</li>
          <li>The <b>predictor</b> puts a placeholder at each hidden position and guesses the 64 numbers there.</li>
          <li>The <b>answer</b> comes from the target encoder, a slowly updated copy of the encoder that reads the whole weld.</li>
          <li>The <b>error</b> is the average gap between guess and answer. Its gradient updates the encoder and the predictor.</li>
        </ol>
        <p>Half the steps hide the next 250 ms (future mask). The other half hide 4 random blocks.</p>
        <p class="next">Click patches to hide them, or press play. On the 18 welds here the trained network's error is
        0.24 to 0.60, against about 0.8 for knowing nothing and 0.82 to 0.90 before training: phase 1 really did learn
        its game.</p>`,
    },
    {
      title: 'Inside the encoder: attention',
      board: 'attention',
      body: () => `
        <p>In each of its 4 layers, every patch builds a new version of itself by mixing in the other patches. The
        attention weights say how much of each patch it takes; each row adds up to 1. There are 4 heads per layer, each
        with its own pattern.</p>
        <p>The trained encoder concentrates. In layers 1 to 3, the 3 patches a patch looks at most get about 60% of its
        attention (an even spread would give them 7.5%). Within a head, every patch looks at much the same few moments
        of the weld: one set of 3 patches takes about half of each patch's attention. It does not favour its
        neighbours: the patches within 2 of it get 11%, what an even spread gives. The random encoder spreads its
        attention almost evenly (its top 3 get 12%). Measured on the 18 welds here.</p>
        <p class="next">Click a row of the matrix, or a patch, to see where it looks. Press play to sweep through the
        patches.</p>`,
    },
    {
      title: 'The weights, as training moves them',
      board: 'weights',
      body: () => `
        <p>The encoder has ${M.encoder_params.toLocaleString()} numbers. The first matrix (600 → 64) decides what each
        of the 64 token numbers looks for in a patch. Each tile on the left is one of them, laid out like a patch:
        24 rows (sensor values) × 25 frames.</p>
        <p>At step 0 the tiles are noise. Training gives them structure along the rows: by step 4,000, which sensor
        value a weight reads (its row) explains 26% of a tile's variation, against 4% at the start. Where it sits in the
        25 ms (its column) still explains only 4%. So the 64 numbers mostly weigh sensor values, not a time pattern
        inside the patch.</p>
        <p class="next">Drag the training step, or press play. The bars on the right show how far each part of the
        encoder moved from its random start.</p>`,
    },
    {
      title: 'Training, replayed',
      board: 'train',
      body: () => `
        <p>The same weld and the same hidden patches, at 9 points of training, with every number recomputed.</p>
        <p>The error falls fast, and most of the learning is done by step 500. After that it drifts slightly up. The
        answers come from the target encoder, which keeps changing and spreading the welds apart, so the game gets a
        little harder while the encoder keeps getting better at it.</p>
        <p class="next">Press play, or drag the step. Switch between the future and the block mask.</p>`,
    },
    {
      title: 'Phase 2: freeze, then summarize',
      board: 'pool',
      body: () => `
        <p>After phase 1, the predictor is thrown away and the encoder is frozen. Each weld, including the test welds it
        never saw, goes through it once.</p>
        <p>The 40 output tokens are reduced to <b>128 numbers</b>: their mean (64) and their spread over the weld (64).
        Only these 128 numbers go on to the classification head.</p>
        <p>The right column is the control: the same encoder at its random start, never trained. Its summaries are
        not noise either: random layers still mix the patches in fixed ways, and the mean and spread keep the weld's
        overall levels.</p>
        <p class="next">Press play to go through the 18 welds.</p>`,
    },
    {
      title: 'Phase 2: the classification head',
      board: 'head',
      body: () => `
        <p>The head is a logistic regression: 9 × 128 weights plus 9 biases, trained with cross-entropy on k labeled
        welds per condition (45 when k = 5). The 128 numbers are first standardized with the ${meta.n_pretrain}
        unlabeled welds' mean and spread. The encoder does not change.</p>
        <p>Each class's score is its row of weights times the weld's 128 numbers, plus its bias; softmax turns the 9
        scores into probabilities.</p>
        <p class="next">Press play to run all ${model ? model.nTest : 54} test welds of split 0. Then switch to the
        random start: the result barely moves.</p>`,
    },
    {
      title: 'What it adds up to',
      board: 'verdict',
      body: () => `
        <p><b>Phase 1 worked as a game, but it did not help phase 2.</b> The encoder learned to predict hidden patches,
        and its weights moved far from their start. Yet at every k, the head did no better on the trained encoder than
        on the same encoder with random weights. 48 frame statistics, with no network, did as well.</p>
        <p>Our reading (a hypothesis): a weld is predictable from its neighbours for reasons all conditions share, so the
        game rewards features that do not separate them. What separates the conditions, above all clean from polished,
        is small shifts in level and spread over the weld, and averaging any reasonable features keeps those.</p>
        <p>Our data also has no actions (we never change the weld settings), so this is a self-supervised encoder, not
        a world model in the planning sense.</p>`,
    },
  ];

  // ---------------------------------------------------------------- render

  function paint() {
    if (loadState !== 'ready' || !built) return;
    BOARDS[built].paint();
  }

  function render() {
    if (loadState === 'idle') {
      load();
      return {};
    }
    if (loadState === 'loading') {
      headEl.innerHTML = '';
      boardEl.innerHTML = '<p class="wm-loading">Loading the world model: 9 snapshots of 389,280 weights (7 MB) and 18 welds…</p>';
      return {};
    }
    if (loadState === 'failed') return {};
    const want = STEPS[S.step].board;
    if (want !== built) {
      setPlaying(false);
      built = want;
      boardEl.innerHTML = '';
      BOARDS[want].build();
    } else {
      BOARDS[want].build();
    }
    paint();
    return {};
  }

  function glance() {
    const k5 = meta.ks.indexOf(5);
    return `
      <h2>The world model at a glance</h2>
      <dl class="facts">
        <div><dt>Encoder</dt><dd>${M.encoder_params.toLocaleString()} weights<small>4 transformer layers over 40 patch tokens of 64 numbers</small></dd></div>
        <div><dt>Phase 1</dt><dd>${meta.n_pretrain} welds, no labels<small>${meta.settings.STEPS.toLocaleString()} steps of the hidden-patch game</small></dd></div>
        <div><dt>9 conditions, k = 5</dt><dd>${pct(meta.acc.B[k5].mean)} vs ${pct(meta.acc.C[k5].mean)} <small>trained encoder vs its random start</small></dd></div>
      </dl>
      <p class="lr-glance"><small>Exploratory, ${esc(meta.source)}. Every heatmap is computed live in the page from the
      exported weights, which match PyTorch's outputs to within 0.002.</small></p>`;
  }

  new ResizeObserver(() => {
    if (!root.hidden) paint();
  }).observe(root);

  return { STEPS, render, glance, stop: () => setPlaying(false) };
}
