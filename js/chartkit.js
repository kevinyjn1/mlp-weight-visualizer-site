// What the chart tabs share ("Which inputs matter" and "Course methods"): colours, number formats, the small pieces
// of a chart card, and the panel that holds the cards (tooltips, clicks on data-act, changes on data-ctl, redraw on
// resize). The charts themselves are SVG strings built by each tab.

import { escapeHtml } from './md.js';

// Okabe-Ito hues: colour-blind safe. Sensors and surfaces never share a chart.
export const SENSOR_COLOR = { AE: '#0072B2', LVDT: '#E69F00', Power: '#009E73', Mic: '#CC79A7' };
export const SURFACE_COLOR = ['#D55E00', '#56B4E9', '#1b2030']; // clean, contaminated, polished
export const BLUES = ['#f3f7fd', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
export const ORANGES = ['#fdf6ee', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704'];
export const STAT = {
  entropy: ['ent', 'entropy'], mean: ['mean', 'mean'], median: ['med', 'median'], p05: ['p5', '5th percentile'],
  p25: ['p25', '25th percentile'], p75: ['p75', '75th percentile'], p95: ['p95', '95th percentile'], rms: ['rms', 'RMS'],
  var: ['var', 'variance'], std: ['std', 'standard deviation'], zero_crossing_rate: ['zcr', 'zero-crossing rate'],
  mean_crossing_rate: ['mcr', 'mean-crossing rate'],
};
export const INK = '#1b2030';
export const MUTED = '#5a6173';
export const GRID = '#e3e6ec';
export const ACCENT = '#2c62dc';
export const ASTRA = '#b84e14';

export const pct = (v, d = 1) => (v == null ? '-' : `${(100 * v).toFixed(d)}%`);
export const fx = (v, d = 2) => `${v >= 0 ? '+' : '-'}${Math.abs(v).toFixed(d)}`;
export const esc = escapeHtml;

export function ramp(stops, v) {
  const t = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const k = Math.min(Math.floor(t), stops.length - 2);
  const a = stops[k].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = stops[k + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((x, n) => Math.round(x + (b[n] - x) * (t - k))).join(',')})`;
}

export function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((v) => v >= raw) ?? 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function header(title, sub, controls = '') {
  return `<div class="lr-head"><div><h2>${title}</h2>${sub ? `<p>${sub}</p>` : ''}</div>
    ${controls ? `<div class="lr-controls">${controls}</div>` : ''}</div>`;
}

export function segmented(name, items, value) {
  return `<div class="lr-seg" role="group">${items.map(([v, label]) =>
    `<button type="button" data-act="${name}:${v}" aria-pressed="${String(v) === String(value)}">${label}</button>`).join('')}</div>`;
}

export function legendRamp(stops, left, right, label) {
  const id = `g${stops === BLUES ? 'b' : 'o'}`;
  return `<div class="lr-ramp"><span>${label}</span><span class="end">${left}</span>
    <svg width="120" height="10" aria-hidden="true"><defs><linearGradient id="${id}">${stops.map((c, n) =>
      `<stop offset="${n / (stops.length - 1)}" stop-color="${c}"/>`).join('')}</linearGradient></defs>
    <rect width="120" height="10" rx="2" fill="url(#${id})"/></svg><span class="end">${right}</span></div>`;
}

/**
 * The card that holds a chart tab's charts. onAct(name, value) gets clicks on [data-act="name:value"], onChange(el)
 * gets changes of [data-ctl] controls, onResize() is called when the panel changes size while shown.
 */
export function createChartPanel({ root, onAct, onChange, onResize }) {
  const tips = [];
  const tipEl = document.createElement('div');
  tipEl.className = 'tip';
  tipEl.hidden = true;
  const content = document.createElement('div');
  content.className = 'lr-card';
  root.append(content, tipEl);

  root.addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el) return;
    const [name, ...rest] = el.dataset.act.split(':');
    onAct(name, rest.join(':'));
  });
  root.addEventListener('change', (e) => {
    const ctl = e.target.closest('[data-ctl]');
    if (ctl) onChange(ctl);
  });
  root.addEventListener('pointermove', (e) => {
    const el = e.target.closest('[data-t]');
    if (!el) {
      tipEl.hidden = true;
      return;
    }
    tipEl.innerHTML = tips[+el.dataset.t];
    tipEl.hidden = false;
    const box = root.getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top + root.scrollTop;
    const w = tipEl.offsetWidth;
    const h = tipEl.offsetHeight;
    const left = Math.min(Math.max(x + 14, 8), box.width - w - 8);
    const top = y + 18 + h > root.scrollTop + box.height - 8 ? Math.max(y - h - 12, root.scrollTop + 8) : y + 18;
    tipEl.style.transform = `translate(${left}px, ${top}px)`;
  });
  root.addEventListener('pointerleave', () => {
    tipEl.hidden = true;
  });
  new ResizeObserver(() => {
    if (!root.hidden) onResize();
  }).observe(root);

  return {
    /** Tooltip HTML is kept in a list; the element carries its index. */
    tip(html) {
      tips.push(html);
      return `data-t="${tips.length - 1}"`;
    },
    /** The width a chart can use inside the card. */
    width: () => Math.max(320, Math.min(root.clientWidth - 2 * 22 - 2 * 20, 980)),
    /** Replace the card's content; build() may call tip(). */
    draw(build) {
      tips.length = 0;
      content.innerHTML = build();
      tipEl.hidden = true;
    },
  };
}
