// The fourth tab: which of the 528 inputs matter. Not a network view: charts of the research repo's exploratory
// logistic regression on Prof. Shao's features (weld-monitor/experiments/usmw_feature_usefulness/analyze.py, exported
// by tools/usefulness.py). Every number on it comes from data/usefulness.json, except where a comment says otherwise.

import { escapeHtml } from './md.js';

const TASKS = ['nine', 'tool_DN', 'tool_DW', 'tool_NW', 'surf_CX', 'surf_CP', 'surf_XP'];
const SHORT = {
  nine: 'All 9 conditions', tool_DN: 'Damaged vs New', tool_DW: 'Damaged vs Worn', tool_NW: 'New vs Worn',
  surf_CX: 'Clean vs Contaminated', surf_CP: 'Clean vs Polished', surf_XP: 'Contaminated vs Polished',
};
const KIND = { nine: '', tool_DN: 'Tool', tool_DW: 'Tool', tool_NW: 'Tool', surf_CX: 'Surface', surf_CP: 'Surface', surf_XP: 'Surface' };
// the pair each question compares: [attribute, first class, second class]
const PAIR = { tool_DN: ['tool', 0, 1], tool_DW: ['tool', 0, 2], tool_NW: ['tool', 1, 2], surf_CX: ['surface', 0, 1], surf_CP: ['surface', 0, 2], surf_XP: ['surface', 1, 2] };
const M_LABELS = ['1', '2', '3', '5', '10', '20', '50', 'all 528'];
// Okabe-Ito hues: colour-blind safe. Sensors and surfaces never share a chart.
const SENSOR_COLOR = { AE: '#0072B2', LVDT: '#E69F00', Power: '#009E73', Mic: '#CC79A7' };
const SURFACE_COLOR = ['#D55E00', '#56B4E9', '#1b2030']; // clean, contaminated, polished
const BLUES = ['#f3f7fd', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
const ORANGES = ['#fdf6ee', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704'];
const STAT = {
  entropy: ['ent', 'entropy'], mean: ['mean', 'mean'], median: ['med', 'median'], p05: ['p5', '5th percentile'],
  p25: ['p25', '25th percentile'], p75: ['p75', '75th percentile'], p95: ['p95', '95th percentile'], rms: ['rms', 'RMS'],
  var: ['var', 'variance'], std: ['std', 'standard deviation'], zero_crossing_rate: ['zcr', 'zero-crossing rate'],
  mean_crossing_rate: ['mcr', 'mean-crossing rate'],
};
const INK = '#1b2030';
const MUTED = '#5a6173';
const GRID = '#e3e6ec';
const ACCENT = '#2c62dc';
const ASTRA = '#b84e14';
const ASTRA_EXP19 = { acc: 0.878, lo: 15, hi: 53 }; // EXP-19 RESULTS.md: arm N, 87.8%, replies use 15 to 53 inputs
// EXP-19's logistic regression, clean<->polished mistakes per tool (its confusion matrix; RESULTS.md of this analysis)
const MISTAKES = { Damaged: 3, New: 13, Worn: 9 };

const pct = (v, d = 1) => (v == null ? '-' : `${(100 * v).toFixed(d)}%`);
const fx = (v, d = 2) => `${v >= 0 ? '+' : '-'}${Math.abs(v).toFixed(d)}`;
const esc = escapeHtml;

function ramp(stops, v) {
  const t = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const k = Math.min(Math.floor(t), stops.length - 2);
  const a = stops[k].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = stops[k + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((x, n) => Math.round(x + (b[n] - x) * (t - k))).join(',')})`;
}

export function createLr({ root, data, state, actions }) {
  const U = data.usefulness;
  const N = U.names;
  const sel = state.lr;
  const W = U.welds;
  const tips = [];

  // ---------------------------------------------------------------- names and small helpers

  const sensorOf = (i) => N.sensors[Math.floor(i / 132)];
  const bandOf = (i) => Math.floor((i % 132) / 12);
  const statOf = (i) => N.stats[i % 12];
  const bandName = (b) => N.bands[b]; // "cD3 (12.5 kHz to 25 kHz)"
  const bandShort = (b) => bandName(b).split(' ')[0];
  const bandHz = (b) => bandName(b).replace(/^\S+ \((.*)\)$/, '$1');
  const shortName = (i) => `${sensorOf(i)} ${bandShort(bandOf(i))} ${STAT[statOf(i)]?.[0] ?? statOf(i)}`;
  const longName = (i) => `${sensorOf(i)}, band ${bandShort(bandOf(i))} (${bandHz(bandOf(i))}), ${STAT[statOf(i)]?.[1] ?? statOf(i)}`;
  const task = (k = sel.task) => U.tasks[k];
  const classesOf = (k) => task(k).classes;
  const plotted = Object.keys(W.values).map(Number);

  /** Tooltip text is kept in a list; the element carries its index. */
  const tip = (html) => {
    tips.push(html);
    return `data-t="${tips.length - 1}"`;
  };

  function header(title, sub, controls = '') {
    return `<div class="lr-head"><div><h2>${title}</h2>${sub ? `<p>${sub}</p>` : ''}</div>
      ${controls ? `<div class="lr-controls">${controls}</div>` : ''}</div>`;
  }

  function taskSelect() {
    const opts = TASKS.map((k) => `<option value="${k}"${k === sel.task ? ' selected' : ''}>${KIND[k] ? `${KIND[k]}: ` : ''}${SHORT[k]}</option>`).join('');
    return `<label class="lr-select"><span>Question</span><select data-ctl="task">${opts}</select></label>`;
  }

  function segmented(name, items, value) {
    return `<div class="lr-seg" role="group">${items.map(([v, label]) =>
      `<button type="button" data-act="${name}:${v}" aria-pressed="${String(v) === String(value)}">${label}</button>`).join('')}</div>`;
  }

  function legendRamp(stops, left, right, label) {
    const id = `g${stops === BLUES ? 'b' : 'o'}`;
    return `<div class="lr-ramp"><span>${label}</span><span class="end">${left}</span>
      <svg width="120" height="10" aria-hidden="true"><defs><linearGradient id="${id}">${stops.map((c, n) =>
        `<stop offset="${n / (stops.length - 1)}" stop-color="${c}"/>`).join('')}</linearGradient></defs>
      <rect width="120" height="10" rx="2" fill="url(#${id})"/></svg><span class="end">${right}</span></div>`;
  }

  // ---------------------------------------------------------------- charts

  /** Step 1: accuracy of plain logistic regression per question, all 528 inputs. */
  function accuracyChart(width) {
    const rowH = 34;
    const left = Math.min(244, width * 0.4);
    const right = 58;
    const top = 26;
    const h = top + rowH * TASKS.length + 26;
    const x = (v) => left + v * (width - left - right);
    let s = '';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${top - 6}" y2="${h - 22}" stroke="${GRID}"/>`;
      s += `<text x="${x(v)}" y="${h - 6}" class="tick" text-anchor="middle">${v * 100}%</text>`;
    }
    TASKS.forEach((k, n) => {
      const t = task(k);
      const y = top + n * rowH;
      const on = k === sel.task;
      const chance = k === 'nine' ? 1 / 9 : 0.5;
      const acc = t.plain.mean;
      const t2 = tip(`<b>${esc(SHORT[k])}</b><span>${pct(acc)} of the test welds right (mean of 5 splits)</span>
        <span>${t.plain.per_split.map((v) => pct(v, 0)).join(' · ')}</span><span>trained on ${t.n_train}, tested on ${t.n_test} per split</span>
        <span>guessing: ${pct(chance)}</span>`);
      s += `<g class="row${on ? ' on' : ''}" data-act="task:${k}" ${t2}>
        <rect x="0" y="${y}" width="${width}" height="${rowH}" fill="transparent"/>
        <text x="${left - 12}" y="${y + rowH / 2 + 4}" text-anchor="end" class="lbl${on ? ' strong' : ''}">${KIND[k] && width >= 640 ? `<tspan class="kind">${KIND[k]} · </tspan>` : ''}${esc(SHORT[k])}</text>
        <rect x="${x(0)}" y="${y + 8}" width="${Math.max(x(acc) - x(0) - 2, 1)}" height="${rowH - 16}" rx="4" fill="${on ? ACCENT : '#9fb0d6'}"/>
        <line x1="${x(chance)}" x2="${x(chance)}" y1="${y + 4}" y2="${y + rowH - 4}" stroke="${INK}" stroke-width="2" stroke-dasharray="3 2"/>
        <text x="${x(acc) + 6}" y="${y + rowH / 2 + 4}" class="value${on ? ' strong' : ''}">${pct(acc)}</text></g>`;
    });
    s += `<text x="${x(0)}" y="${top - 10}" class="axis">Test accuracy, all 528 inputs</text>`;
    return `${header('How well logistic regression answers each question', 'Prof. Shao\'s 528 inputs · the group\'s 5 splits · dashed tick: guessing')}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>`;
  }

  /** Step 2: the 528 inputs as a map, coloured by how often the lasso keeps each one. */
  function featureMap(width) {
    const t = task();
    const left = 44;
    const gap = 12;
    const cw = Math.max(9, Math.floor((width - left - 3 * gap - 4) / 44));
    const ch = Math.max(16, Math.min(24, cw));
    const top = 40;
    const h = top + 12 * ch + 44;
    const gx = (s, b) => left + s * (11 * cw + gap) + b * cw;
    let s = '';
    N.sensors.forEach((name, si) => {
      s += `<text x="${gx(si, 0)}" y="${top - 24}" class="lbl strong">${name}</text>`;
      s += `<rect x="${gx(si, 0)}" y="${top - 18}" width="${11 * cw - 1}" height="3" rx="1.5" fill="${SENSOR_COLOR[name]}"/>`;
      s += `<text x="${gx(si, 0)}" y="${top + 12 * ch + 14}" class="tick">${bandHz(0).split(' to ')[0]}</text>`;
      s += `<text x="${gx(si, 10) + cw}" y="${top + 12 * ch + 14}" class="tick" text-anchor="end">${bandHz(10).split(' to ')[1]}</text>`;
    });
    N.stats.forEach((st, k) => {
      s += `<text x="${left - 6}" y="${top + k * ch + ch / 2 + 4}" text-anchor="end" class="tick">${STAT[st]?.[0] ?? st}</text>`;
    });
    for (let i = 0; i < 528; i++) {
      const si = Math.floor(i / 132);
      const b = bandOf(i);
      const k = i % 12;
      const v = t.half_freq[i];
      const all5 = t.split_freq[i] >= 0.999;
      const x0 = gx(si, b);
      const y0 = top + k * ch;
      const dir = t.half_toward_second?.[i];
      const toward = dir == null || v === 0 ? '' : `<span>its weight pushes toward ${esc(t.classes[dir >= 0.5 ? 1 : 0])} in ${pct(Math.max(dir, 1 - dir), 0)} of those</span>`;
      const auroc = t.single_auroc ? `<span>on its own: test AUROC ${t.single_auroc[i].toFixed(3)} (0.5 = no better than a coin)</span>` : '';
      const t2 = tip(`<b>Input ${i} · ${esc(longName(i))}</b><span>kept in ${pct(v, 0)} of 100 random halves; in ${Math.round(t.split_freq[i] * 5)} of the 5 splits</span>${toward}${auroc}
        <span>click to select it</span>`);
      const picked = sel.feature === i;
      s += `<rect x="${x0}" y="${y0}" width="${cw - 1}" height="${ch - 1}" fill="${v === 0 ? '#f7f8fa' : ramp(BLUES, 0.12 + 0.88 * v)}" class="cell${picked ? ' picked' : ''}" data-act="feature:${i}" ${t2}/>`;
      if (all5) s += `<circle cx="${x0 + (cw - 1) / 2}" cy="${y0 + (ch - 1) / 2}" r="${Math.min(cw, ch) * 0.16}" fill="${v > 0.55 ? '#fff' : INK}" pointer-events="none"/>`;
    }
    s += `<text x="${left}" y="${h - 8}" class="axis">Each block: one sensor's 11 frequency bands, low to high (cD11 to cD1). Each row: one of the 12 statistics.</text>`;
    const legend = `<div class="lr-legend">${legendRamp(BLUES, '0%', '100%', 'Kept by the lasso, of 100 random halves')}
      <span class="lr-key"><svg width="10" height="10"><circle cx="5" cy="5" r="3" fill="${INK}"/></svg>kept in all 5 of the group's splits</span></div>`;
    return `${header(`Which inputs the lasso keeps: ${esc(SHORT[sel.task])}`, `The lasso's 10 picks, ${t.n_train} training welds per split`, taskSelect())}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>${legend}${featureCard()}`;
  }

  /** The selected input, under the map. */
  function featureCard() {
    const i = sel.feature;
    if (i == null) return '<p class="lr-note">Click a cell to select an input.</p>';
    const t = task();
    const can = plotted.includes(i);
    return `<div class="lr-feature"><div><b>Input ${i}</b> · ${esc(longName(i))}<br>
      <span>${esc(SHORT[sel.task])}: kept in ${pct(t.half_freq[i], 0)} of the halves${t.single_auroc ? `, AUROC on its own ${t.single_auroc[i].toFixed(3)}` : ''}</span></div>
      <div class="lr-buttons">${can ? `<button type="button" class="secondary" data-act="plot:${i}">Plot its welds</button>` : ''}
      <button type="button" class="secondary" data-act="show:${i}">See it in the trained network</button></div></div>`;
  }

  /** Step 3: accuracy with only the lasso's first m inputs. */
  function curveChart(width) {
    const left = 48;
    const right = Math.min(170, width * 0.27);
    const top = 18;
    const h = 330;
    const bottom = 46;
    const xs = M_LABELS.map((_, n) => left + (n * (width - left - right)) / (M_LABELS.length - 1));
    const y = (v) => top + (1 - v) * (h - top - bottom);
    const series = (k) => [...['1', '2', '3', '5', '10', '20', '50'].map((m) => task(k).lasso_curve[m]), task(k).plain.mean];
    let s = '';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      s += `<line x1="${left}" x2="${xs.at(-1)}" y1="${y(v)}" y2="${y(v)}" stroke="${GRID}"/>`;
      s += `<text x="${left - 8}" y="${y(v) + 4}" text-anchor="end" class="tick">${v * 100}%</text>`;
    }
    M_LABELS.forEach((m, n) => {
      s += `<text x="${xs[n]}" y="${h - bottom + 18}" text-anchor="middle" class="tick${n === 7 ? ' strong' : ''}">${m}</text>`;
    });
    s += `<line x1="${(xs[6] + xs[7]) / 2}" x2="${(xs[6] + xs[7]) / 2}" y1="${top}" y2="${h - bottom}" stroke="${GRID}" stroke-dasharray="2 3"/>`;
    s += `<text x="${(xs[0] + xs[6]) / 2}" y="${h - 8}" text-anchor="middle" class="axis">Inputs the lasso keeps</text>`;
    if (sel.task === 'nine') {
      // Astra's EXP-19 networks: 15 to 53 inputs; placed between the m = 10, 20 and 50 ticks
      const at = (m) => (m <= 20 ? xs[4] + ((m - 10) / 10) * (xs[5] - xs[4]) : xs[5] + ((m - 20) / 30) * (xs[6] - xs[5]));
      const x0 = at(ASTRA_EXP19.lo);
      const x1 = Math.min(at(ASTRA_EXP19.hi), (xs[6] + xs[7]) / 2);
      const t2 = tip(`<b>Astra's networks in EXP-19</b><span>87.8% on 9 conditions, written from the training welds' table</span><span>they used 15 to 53 of the 528 inputs</span><span>this comparison was made after EXP-19 was scored</span>`);
      s += `<g ${t2}><line x1="${x0}" x2="${x1}" y1="${y(ASTRA_EXP19.acc)}" y2="${y(ASTRA_EXP19.acc)}" stroke="${ASTRA}" stroke-width="3" stroke-dasharray="6 3" stroke-linecap="round"/>
        <text x="${x0}" y="${y(ASTRA_EXP19.acc) + 18}" class="value halo" fill="${ASTRA}">Astra, EXP-19: 87.8%</text></g>`;
    }
    const chance = sel.task === 'nine' ? 1 / 9 : 0.5;
    s += `<line x1="${left}" x2="${xs.at(-1)}" y1="${y(chance)}" y2="${y(chance)}" stroke="${MUTED}" stroke-dasharray="4 3"/>
      <text x="${left + 4}" y="${y(chance) - 5}" class="tick">guessing</text>`;
    const order = [...TASKS.filter((k) => k !== sel.task), sel.task];
    for (const k of order) {
      const on = k === sel.task;
      const v = series(k);
      const d = v.map((p, n) => `${n ? 'L' : 'M'}${xs[n]},${y(p)}`).join(' ');
      s += `<path d="${d}" fill="none" stroke="${on ? ACCENT : '#c3c9d6'}" stroke-width="${on ? 2.5 : 1.5}" stroke-linejoin="round" data-act="task:${k}"/>`;
      if (on) {
        s += v.map((p, n) => `<circle cx="${xs[n]}" cy="${y(p)}" r="4.5" fill="${ACCENT}" stroke="#fff" stroke-width="2"/>`).join('');
        s += `<text x="${xs.at(-1) + 10}" y="${y(v.at(-1)) + 4}" class="lbl strong halo" fill="${ACCENT}">${esc(SHORT[k])}</text>`;
      }
    }
    // hover columns: every question's value at that number of inputs
    M_LABELS.forEach((m, n) => {
      const w = n === 0 ? xs[1] - xs[0] : xs[n] - xs[n - 1];
      const rows = TASKS.map((k) => `<span${k === sel.task ? ' class="on"' : ''}>${esc(SHORT[k])}: ${pct(series(k)[n])}</span>`).join('');
      s += `<rect x="${xs[n] - w / 2}" y="${top}" width="${w}" height="${h - top - bottom}" fill="transparent" class="col" ${tip(`<b>${m === 'all 528' ? 'All 528 inputs' : `The lasso's first ${m} input${m === '1' ? '' : 's'}`}</b>${rows}`)}/>`;
    });
    return `${header('How few inputs are enough', 'Test accuracy of logistic regression on only the inputs the lasso keeps first (chosen on the training welds)', taskSelect())}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>`;
  }

  /** Step 4: the share of the lasso's picks that falls on each sensor. */
  function sensorChart(width) {
    const rowH = 34;
    const left = Math.min(244, width * 0.4);
    const right = 16;
    const top = 8;
    const h = top + rowH * TASKS.length + 8;
    const x = (v) => left + v * (width - left - right);
    let s = '';
    TASKS.forEach((k, n) => {
      const t = task(k);
      const y = top + n * rowH;
      const on = k === sel.task;
      let acc = 0;
      s += `<g class="row${on ? ' on' : ''}" data-act="task:${k}"><rect x="0" y="${y}" width="${width}" height="${rowH}" fill="transparent"/>
        <text x="${left - 12}" y="${y + rowH / 2 + 4}" text-anchor="end" class="lbl${on ? ' strong' : ''}">${KIND[k] && width >= 640 ? `<tspan class="kind">${KIND[k]} · </tspan>` : ''}${esc(SHORT[k])}</text>`;
      N.sensors.forEach((name, si) => {
        const v = t.sensor_share[si];
        const x0 = x(acc);
        const w = x(acc + v) - x0;
        acc += v;
        if (w <= 0) return;
        const t2 = tip(`<b>${esc(SHORT[k])} · ${name}</b><span>${pct(v, 0)} of the lasso's picks</span>`);
        s += `<rect x="${x0}" y="${y + 6}" width="${Math.max(w - 2, 0.5)}" height="${rowH - 12}" rx="3" fill="${SENSOR_COLOR[name]}" ${t2}/>`;
        if (w > 44) s += `<text x="${x0 + 7}" y="${y + rowH / 2 + 4}" class="inbar">${name} ${pct(v, 0)}</text>`;
      });
      s += '</g>';
    });
    const legend = `<div class="lr-legend">${N.sensors.map((name) => `<span class="lr-key"><i style="background:${SENSOR_COLOR[name]}"></i>${name}</span>`).join('')}
      <span class="lr-note">AE: acoustic emission · LVDT: horn height · Power · Mic: microphone</span></div>`;
    return `${header('Which sensor answers which question', 'Share of the lasso\'s 10 picks on each sensor, over 100 random halves')}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>${legend}`;
  }

  /** Steps 5 and 7: every weld as a dot, placed by one input. */
  function dotChart(width) {
    const i = sel.plot;
    const isWindow = typeof i === 'string';
    const values = isWindow ? U.other.window_values[i] : W.values[String(i)];
    const name = isWindow ? i : shortName(i);
    const pair = PAIR[sel.task];
    const narrow = width < 560; // on a phone the tool names go above their rows
    const left = narrow ? 104 : 182;
    const right = pair?.[0] === 'surface' ? (narrow ? 50 : 74) : 20;
    const rowH = 26;
    const groupGap = narrow ? 26 : 14;
    const top = narrow ? 38 : 26;
    const h = top + 9 * rowH + 2 * groupGap + 40;
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo) * 0.04;
    const x = (v) => left + ((v - lo + pad) / (hi - lo + 2 * pad)) * (width - left - right);
    const rowY = (tool, surface) => top + tool * (3 * rowH + groupGap) + surface * rowH + rowH / 2;
    const inPair = (tool, surface) => !pair || (pair[0] === 'tool' ? [pair[1], pair[2]].includes(tool) : [pair[1], pair[2]].includes(surface));
    let s = '';
    const ticks = niceTicks(lo - pad, hi + pad, 6);
    for (const v of ticks) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${top - 6}" y2="${h - 34}" stroke="${v === 0 ? '#c9ced9' : GRID}"/>`;
      s += `<text x="${x(v)}" y="${h - 20}" text-anchor="middle" class="tick">${v}</text>`;
    }
    s += `<text x="${x(0)}" y="${h - 4}" text-anchor="middle" class="axis">${esc(name)}, standardized (0 = the average weld; 1 = one standard deviation)</text>`;
    N.tools.forEach((tool, ti) => {
      s += narrow
        ? `<text x="4" y="${rowY(ti, 0) - rowH / 2 - 4}" class="lbl strong">${tool} tool</text>`
        : `<text x="8" y="${rowY(ti, 1) + 4}" class="lbl strong">${tool}</text>`;
      N.surfaces.forEach((surface, si) => {
        const y = rowY(ti, si);
        const dim = !inPair(ti, si);
        s += `<text x="${left - 10}" y="${y + 4}" text-anchor="end" class="lbl${dim ? ' dim' : ''}">${surface}</text>`;
        s += `<line x1="${left}" x2="${width - right}" y1="${y}" y2="${y}" stroke="${GRID}" stroke-width="0.5"/>`;
      });
      if (pair?.[0] === 'surface') {
        const d = cohen(values, ti, pair[1], pair[2]);
        const t2 = tip(`<b>${tool} tool: ${esc(N.surfaces[pair[2]])} minus ${esc(N.surfaces[pair[1]])}</b><span>d = ${fx(d)} pooled standard deviations</span><span>|d| above about 2: the two groups hardly overlap</span>`);
        s += `<text x="${width - right + 10}" y="${rowY(ti, 1) + 4}" class="value" ${t2}>d ${fx(d, 1)}</text>`;
      }
    });
    W.run.forEach((run, n) => {
      const ti = W.tool[n];
      const si = W.surface[n];
      const jitter = ((((run * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * (rowH - 10);
      const cx = x(values[n]);
      const cy = rowY(ti, si) + jitter;
      const dim = !inPair(ti, si);
      const after = W.new_after_step[n];
      const color = SURFACE_COLOR[si];
      const t2 = tip(`<b>Weld ${run}</b><span>${N.tools[ti]} tool, ${N.surfaces[si]} surface</span><span>${esc(name)}: ${values[n].toFixed(2)}</span>${ti === 1 ? `<span>${after ? 'recorded after' : 'recorded before'} the horn-height step</span>` : ''}`);
      s += after
        ? `<circle cx="${cx}" cy="${cy}" r="3.6" fill="#fff" stroke="${color}" stroke-width="1.6" opacity="${dim ? 0.2 : 0.95}" ${t2}/>`
        : `<circle cx="${cx}" cy="${cy}" r="3.8" fill="${color}" opacity="${dim ? 0.18 : 0.85}" ${t2}/>`;
    });
    const options = [];
    for (const k of TASKS) {
      const t = task(k);
      const top5 = [...t.half_freq.keys()].sort((a, b) => t.half_freq[b] - t.half_freq[a]).slice(0, 5).filter((j) => plotted.includes(j));
      options.push(`<optgroup label="${esc(SHORT[k])}: the lasso's top picks">${top5.map((j) => `<option value="${j}"${j === i ? ' selected' : ''}>${esc(shortName(j))}</option>`).join('')}</optgroup>`);
    }
    options.push(`<optgroup label="Time windows (not Prof. Shao's inputs)">${Object.keys(U.other.window_values).map((w) => `<option value="${esc(w)}"${w === i ? ' selected' : ''}>${esc(w)}</option>`).join('')}</optgroup>`);
    const controls = `<label class="lr-select"><span>Input</span><select data-ctl="plot">${options.join('')}</select></label>${taskSelect()}`;
    const legend = `<div class="lr-legend">${N.surfaces.map((sf, si) => `<span class="lr-key"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${SURFACE_COLOR[si]}"/></svg>${sf}</span>`).join('')}
      <span class="lr-key"><svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="#fff" stroke="${INK}" stroke-width="1.5"/></svg>New tool, recorded after the horn-height step</span>
      ${pair ? `<span class="lr-note">Faded: welds outside the question</span>` : ''}</div>`;
    return `${header(`Every weld, placed by one input`, isWindow ? 'A time-window statistic: not one of Prof. Shao\'s 528 inputs' : esc(longName(i)), controls)}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>${legend}`;
  }

  function cohen(values, tool, a, b) {
    const pick = (sf) => values.filter((_, n) => W.tool[n] === tool && W.surface[n] === sf);
    const A = pick(a);
    const B = pick(b);
    const mean = (v) => v.reduce((p, q) => p + q, 0) / v.length;
    const ss = (v, m) => v.reduce((p, q) => p + (q - m) ** 2, 0);
    const ma = mean(A);
    const mb = mean(B);
    const pooled = Math.sqrt((ss(A, ma) + ss(B, mb)) / (A.length + B.length - 2));
    return (mb - ma) / pooled;
  }

  /** Step 6: accuracy lost when a group of inputs is shuffled across the test welds. */
  function leanChart(width) {
    const P = U.permutation;
    const metric = sel.metric;
    const metricName = ['9 conditions', 'tool', 'surface'][metric];
    const models = [['mlp', 'Trained networks', ACCENT], ['lr', 'Logistic regression', '#8f9ab3']];
    // part 1: one bar pair per sensor
    const rowH = 40;
    const left = 70;
    const right = 60;
    const h1 = 14 + rowH * 4 + 26;
    const max1 = Math.max(...models.flatMap(([m]) => P[m].sensor.map((v) => v[metric])), 0.05);
    const x1 = (v) => left + (v / max1) * (width - left - right);
    let s = '';
    for (const v of niceTicks(0, max1 * 100, 5)) {
      s += `<line x1="${x1(v / 100)}" x2="${x1(v / 100)}" y1="8" y2="${h1 - 22}" stroke="${GRID}"/><text x="${x1(v / 100)}" y="${h1 - 8}" text-anchor="middle" class="tick">${v}</text>`;
    }
    N.sensors.forEach((name, si) => {
      const y = 14 + si * rowH;
      s += `<text x="${left - 10}" y="${y + rowH / 2 + 2}" text-anchor="end" class="lbl strong">${name}</text>`;
      models.forEach(([m, label, color], mi) => {
        const v = P[m].sensor[si][metric];
        const base = P.base[m].reduce((p, q) => p + q[metric], 0) / P.base[m].length;
        const t2 = tip(`<b>${label}: all 132 ${name} inputs shuffled</b><span>${metricName} accuracy falls by ${(100 * v).toFixed(1)} points (from ${pct(base)})</span>`);
        s += `<rect x="${left}" y="${y + 3 + mi * 16}" width="${Math.max(x1(v) - left, 1)}" height="13" rx="3" fill="${color}" ${t2}/>`;
        s += `<text x="${x1(v) + 5}" y="${y + 13 + mi * 16}" class="value">${(100 * v).toFixed(1)}</text>`;
      });
    });
    const part1 = `<svg viewBox="0 0 ${width} ${h1}" width="${width}" height="${h1}" class="lr-svg">${s}</svg>`;
    // part 2: a 4 x 11 grid of sensor bands for the chosen model
    const m = sel.model;
    const vals = P[m].block.map((v) => v[metric]);
    const vmax = Math.max(...vals, 0.01);
    const cw = Math.max(26, Math.floor((width - 70) / 11));
    const ch = 30;
    const h2 = 22 + 4 * ch + 30;
    let g = '';
    for (let b = 0; b < 11; b++) {
      g += `<text x="${70 + b * cw + cw / 2}" y="14" text-anchor="middle" class="tick">${bandShort(b)}</text>`;
    }
    N.sensors.forEach((name, si) => {
      g += `<text x="60" y="${22 + si * ch + ch / 2 + 4}" text-anchor="end" class="lbl strong">${name}</text>`;
      for (let b = 0; b < 11; b++) {
        const v = vals[si * 11 + b];
        const t2 = tip(`<b>${name}, band ${esc(bandName(b))}</b><span>its 12 statistics shuffled together: ${metricName} accuracy falls by ${(100 * v).toFixed(1)} points</span>`);
        g += `<rect x="${70 + b * cw}" y="${22 + si * ch}" width="${cw - 2}" height="${ch - 2}" rx="3" fill="${v <= 0 ? '#f7f8fa' : ramp(ORANGES, 0.1 + 0.9 * (v / vmax))}" ${t2}/>`;
        if (v >= 0.005) g += `<text x="${70 + b * cw + (cw - 2) / 2}" y="${22 + si * ch + ch / 2 + 3}" text-anchor="middle" class="incell" fill="${v / vmax > 0.6 ? '#fff' : INK}">${(100 * v).toFixed(1)}</text>`;
      }
    });
    g += `<text x="70" y="${h2 - 6}" class="axis">Frequency bands, low (cD11, ${bandHz(0).split(' to ')[0]}) to high (cD1, ${bandHz(10).split(' to ')[1]})</text>`;
    const part2 = `<svg viewBox="0 0 ${width} ${h2}" width="${width}" height="${h2}" class="lr-svg">${g}</svg>`;
    const controls = `${segmented('metric', [[0, '9 conditions'], [1, 'Tool'], [2, 'Surface']], metric)}`;
    return `${header('What the trained networks lean on', `Points of ${metricName} accuracy lost when a group of inputs is shuffled across the test welds (30 shuffles × 5 splits)`, controls)}
      <div class="lr-legend">${models.map(([, label, color]) => `<span class="lr-key"><i style="background:${color}"></i>${label}</span>`).join('')}</div>
      ${part1}
      <div class="lr-subhead"><h3>One sensor band at a time</h3>${segmented('model', [['mlp', 'Trained networks'], ['lr', 'Logistic regression']], m)}</div>
      ${part2}${legendRamp(ORANGES, '0', `${(100 * vmax).toFixed(1)} points`, 'Accuracy lost')}`;
  }

  /** Step 8: inputs that keep time, and the average time course of each condition. */
  function timeChart(width) {
    const C = U.other.compare;
    const sets = Object.keys(C);
    const left = Math.min(190, width * 0.34);
    const right = 56;
    const rowH = 44;
    const h1 = 10 + rowH * sets.length + 24;
    const x = (v) => left + ((v - 0.5) / 0.5) * (width - left - right);
    const series = [['9 conditions', (c) => c.nine_tool_surface[0], '#182149'], ['Clean vs polished', (c) => c.clean_vs_polished, '#6f8fd8']];
    let s = '';
    for (const v of [0.5, 0.6, 0.7, 0.8, 0.9, 1]) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="6" y2="${h1 - 20}" stroke="${GRID}"/><text x="${x(v)}" y="${h1 - 6}" text-anchor="middle" class="tick">${Math.round(v * 100)}%</text>`;
    }
    sets.forEach((name, n) => {
      const y = 10 + n * rowH;
      const prof = n === 0;
      s += `<text x="${left - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" class="lbl${prof ? ' strong' : ''}">${esc(name === '528 wavelet features' ? "528 wavelet (Prof. Shao's)" : name)}</text>`;
      series.forEach(([label, get, color], k) => {
        const v = get(C[name]);
        const t2 = tip(`<b>${esc(name)}</b><span>${label}: ${pct(v)} (plain logistic regression, 5 splits)</span>`);
        s += `<rect x="${x(0.5)}" y="${y + 5 + k * 17}" width="${Math.max(x(v) - x(0.5), 1)}" height="14" rx="3" fill="${color}" ${t2}/>`;
        s += `<text x="${x(v) + 5}" y="${y + 16 + k * 17}" class="value">${pct(v)}</text>`;
      });
    });
    const part1 = `<svg viewBox="0 0 ${width} ${h1}" width="${width}" height="${h1}" class="lr-svg">${s}</svg>`;
    // part 2: the average signal of each condition over the weld, one panel per tool
    const T = U.other.time_course;
    const si = T.sensors.indexOf(sel.sensor);
    const bins = Math.round(sel.range / T.bin_ms);
    const panelW = Math.floor((width - 20) / 3);
    const ph = 190;
    const top = 26;
    const bottom = 30;
    const pl = 44;
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < 9; k++) {
      for (let b = 0; b < bins; b++) {
        const v = T.mean[k][si * 100 + b];
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    const span = hi - lo || 1;
    lo -= span * 0.05;
    hi += span * 0.05;
    let g = '';
    N.tools.forEach((tool, ti) => {
      const ox = ti * (panelW + 10);
      const px = (b) => ox + pl + ((b + 0.5) / bins) * (panelW - pl - 8);
      const py = (v) => top + (1 - (v - lo) / (hi - lo)) * (ph - top - bottom);
      g += `<text x="${ox + pl}" y="14" class="lbl strong">${tool} tool</text>`;
      g += `<line x1="${ox + pl}" x2="${ox + panelW - 8}" y1="${ph - bottom}" y2="${ph - bottom}" stroke="#c9ced9"/>`;
      for (const v of niceTicks(lo, hi, 4)) {
        if (ti === 0) g += `<text x="${ox + pl - 6}" y="${py(v) + 4}" text-anchor="end" class="tick">${v}</text>`;
        g += `<line x1="${ox + pl}" x2="${ox + panelW - 8}" y1="${py(v)}" y2="${py(v)}" stroke="${GRID}"/>`;
      }
      for (const ms of niceTicks(0, sel.range, 3)) {
        g += `<text x="${ox + pl + (ms / sel.range) * (panelW - pl - 8)}" y="${ph - bottom + 16}" text-anchor="middle" class="tick">${ms}</text>`;
      }
      N.surfaces.forEach((surface, sf) => {
        const k = 3 * ti + sf;
        const pts = Array.from({ length: bins }, (_, b) => `${b ? 'L' : 'M'}${px(b).toFixed(1)},${py(T.mean[k][si * 100 + b]).toFixed(1)}`).join(' ');
        g += `<path d="${pts}" fill="none" stroke="${SURFACE_COLOR[sf]}" stroke-width="2" stroke-linejoin="round" opacity="0.9" ${tip(`<b>${tool} tool, ${surface} surface</b><span>average ${esc(sel.sensor)} (${T.stat[sel.sensor]} of each 10 ms) over its 30 welds</span>`)}/>`;
      });
    });
    g += `<text x="${pl}" y="${ph - 2}" class="axis">Time into the weld (ms)</text>`;
    const part2 = `<svg viewBox="0 0 ${width} ${ph}" width="${width}" height="${ph}" class="lr-svg">${g}</svg>`;
    const sensorCtl = segmented('sensor', T.sensors.map((n) => [n, n]), sel.sensor);
    const rangeCtl = segmented('range', [[300, 'First 300 ms'], [1000, 'Whole weld']], sel.range);
    return `${header('Inputs that keep time', 'Plain logistic regression on the same splits · the 72 windows were chosen after looking at the labels, so they are optimistic')}
      <div class="lr-legend">${series.map(([label, , color]) => `<span class="lr-key"><i style="background:${color}"></i>${label}</span>`).join('')}</div>
      ${part1}
      <div class="lr-subhead"><h3>The average signal over time</h3><div class="lr-controls">${sensorCtl}${rangeCtl}</div></div>
      <div class="lr-legend">${N.surfaces.map((sf, k) => `<span class="lr-key"><i style="background:${SURFACE_COLOR[k]}"></i>${sf}</span>`).join('')}<span class="lr-note">each line: the average of 30 welds, in 10 ms steps</span></div>
      ${part2}`;
  }

  function niceTicks(lo, hi, count) {
    const raw = (hi - lo) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((v) => v >= raw) ?? 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // ---------------------------------------------------------------- the walkthrough

  const topPick = (k) => {
    const t = task(k);
    return [...t.half_freq.keys()].sort((a, b) => t.half_freq[b] - t.half_freq[a])[0];
  };
  const bestSingle = (k) => {
    const a = task(k).single_auroc;
    return a ? [...a.keys()].sort((p, q) => a[q] - a[p])[0] : null;
  };
  const sensorsLine = (k) => N.sensors.map((n, si) => [n, task(k).sensor_share[si]]).sort((a, b) => b[1] - a[1])
    .filter(([, v]) => v >= 0.1).map(([n, v]) => `${n} ${pct(v, 0)}`).join(', ');
  const curveLine = (k) => {
    const t = task(k);
    const c = t.lasso_curve;
    const firstFull = ['1', '2', '3', '5', '10', '20', '50'].find((m) => c[m] >= t.plain.mean - 0.005);
    if (firstFull) {
      return `With only <b>${firstFull} input${firstFull === '1' ? '' : 's'}</b>, ${esc(SHORT[k].toLowerCase())} reaches ${pct(c[firstFull])}, as good as all 528 (${pct(t.plain.mean)}).`;
    }
    return `${esc(SHORT[k])} never gets close: ${pct(c['50'])} with 50 inputs, against ${pct(t.plain.mean)} with all 528. What separates the two classes is spread thinly over many inputs.`;
  };

  const STEPS = [
    {
      title: 'Six small questions instead of one',
      chart: accuracyChart,
      body: () => `
        <p>Prof. Shao's network answers one big question: which of the 9 conditions made this weld? Here it is split
        into pairs: three pairs of tools and three pairs of surfaces. Each pair gets its own <b>logistic
        regression</b> on the same 528 inputs, trained on ${task('surf_CP').n_train} welds and tested on
        ${task('surf_CP').n_test}, on each of the group's 5 splits.</p>
        <p>The tool pairs are perfect, and contamination is nearly perfect. <b>Clean vs polished is the hard one:
        ${pct(task('surf_CP').plain.mean)}.</b> On all 9 conditions at once, logistic regression scores
        ${pct(task('nine').plain.mean)}, the same as in EXP-19.</p>
        <p class="next">Click a row to choose the question the next charts show.</p>`,
    },
    {
      title: 'The 528 inputs, as a map',
      chart: featureMap,
      body: () => `
        <p>Each input is one <b>statistic</b> of one <b>frequency band</b> of one <b>sensor</b>: 4 sensors × 11
        bands × 12 statistics = 528. In the map, each column is a band (low frequencies on the left) and each row a
        statistic.</p>
        <p>The <b>lasso</b> is logistic regression that pays a price for every input it uses, so it keeps only the
        few that earn it. Here it keeps 10. A dark cell was kept in almost all of 100 random halves of the welds, so
        it is not luck; a dot marks inputs kept in all 5 of the group's splits.</p>
        <p>For <b>${esc(SHORT[sel.task])}</b>, the picks fall on ${sensorsLine(sel.task)}. The surest pick is
        <b>${esc(shortName(topPick(sel.task)))}</b>, kept in ${pct(task().half_freq[topPick(sel.task)], 0)} of the
        halves.</p>
        <p class="next">Hover a cell to read it. Click one to select it, then plot its welds or find it in the
        trained network.</p>`,
    },
    {
      title: 'How few inputs are enough',
      chart: curveChart,
      body: () => `
        <p>Now the lasso keeps 1, 2, 3, 5, 10, 20 or 50 inputs, and logistic regression is trained on only those.
        The choice is made on the training welds alone.</p>
        <p>${curveLine(sel.task)}</p>
        ${sel.task === 'nine' ? `<p>On all 9 conditions, <b>20 inputs (${pct(task('nine').lasso_curve['20'])}) beat all
        528 (${pct(task('nine').plain.mean)})</b>. The dashed orange line is Astra's EXP-19 networks: 15 to 53 inputs,
        87.8%. Inputs picked by the lasso from the same welds do better, so Astra's choice of inputs is not special.
        This comparison was made after EXP-19 was scored.</p>` : `<p>Tools need one input; contamination needs two.
        Choose <b>All 9 conditions</b> above to compare with Astra's EXP-19 networks.</p>`}`,
    },
    {
      title: 'Which sensor answers which question',
      chart: sensorChart,
      body: () => `
        <p>Each bar splits the lasso's picks by sensor.</p>
        <ul>
          <li><b>Tools:</b> the microphone, mostly its lowest bands (49 to 391 Hz); for damaged vs worn, the power
          signal's highest bands.</li>
          <li><b>Contamination:</b> the horn height (LVDT), its high-frequency bands.</li>
          <li><b>Clean vs polished:</b> spread over all four, acoustic emission most (${pct(task('surf_CP').sensor_share[0], 0)}),
          above all its 12.5 to 25 kHz band, which holds the 20 kHz weld vibration.</li>
        </ul>`,
    },
    {
      title: 'Look at the welds',
      chart: dotChart,
      enter: () => {
        if (sel.plot == null || (typeof sel.plot === 'number' && !plotted.includes(sel.plot))) sel.plot = topPick('surf_CP');
      },
      body: () => `
        <p>Each dot is one weld, placed by its value of one input. Rows are the 9 conditions. Hover a dot to read
        the weld.</p>
        <p>The top clean vs polished pick, <b>${esc(shortName(topPick('surf_CP')))}</b>, tells the two surfaces
        apart well with the <b>damaged</b> tool, and hardly at all with the new and worn tools (the <i>d</i> on the
        right: how far apart the two groups sit, in standard deviations).</p>
        <p>That is where logistic regression's clean vs polished mistakes are: ${MISTAKES.New} with the new tool,
        ${MISTAKES.Worn} with the worn tool, ${MISTAKES.Damaged} with the damaged tool (of 60 each, EXP-19).</p>
        <p class="next">Pick another input above the chart. Try a microphone input from a tool question.</p>`,
    },
    {
      title: 'What the trained networks lean on',
      chart: leanChart,
      body: () => {
        const P = U.permutation;
        const power = P.mlp.sensor[2];
        const blocks = P.mlp.block.map((v, b) => [b, v[0]]).sort((a, b) => b[1] - a[1]);
        const [b0, v0] = blocks[0];
        return `
        <p>Another way to ask which inputs matter: take Prof. Shao's trained networks and <b>shuffle</b> one group
        of inputs across the test welds, so those inputs no longer belong to their weld. The accuracy lost is how
        much the network relied on them.</p>
        <p>Shuffling all 132 <b>power</b> inputs costs the most: ${(100 * power[0]).toFixed(1)} points on 9
        conditions, ${(100 * power[1]).toFixed(1)} on tool. The single most important band is
        <b>${esc(N.blocks[b0])}</b>: ${(100 * v0).toFixed(1)} points, all of it on surface.</p>
        <p>No single band costs more than about 4 points: the same information sits in many inputs. Logistic
        regression leans on the same sensors in the same order.</p>`;
      },
    },
    {
      title: 'Two cautions about recording sessions',
      chart: dotChart,
      enter: () => {
        sel.plot = bestSingle('tool_NW') ?? sel.plot;
        sel.task = 'tool_NW';
      },
      body: () => {
        const h = U.horn.nine;
        return `
        <p><b>The microphone may be hearing the room.</b> The best tool inputs are microphone bands at 49 to
        391 Hz, and they separate the new tool from the others perfectly (on screen). But each tool was recorded
        as its own session, and mains hum (60 and 120 Hz) sits in exactly these bands. A perfect split on a
        background-sound band may be telling the <i>sessions</i> apart, not the tools. The data cannot say which.</p>
        <p><b>The horn-height step does not explain the mistakes.</b> Hollow rings mark new-tool welds recorded
        after the step. Of the ${h.called_polished} new-tool clean welds called polished, ${h.called_polished_late}
        come after it (${pct(h.called_polished_late / h.called_polished, 0)}), about the same share as among all
        ${h.new_clean} new-tool clean test cases (${pct(h.new_clean_late / h.new_clean, 0)}).</p>`;
      },
    },
    {
      title: 'Inputs that keep time',
      chart: timeChart,
      body: () => {
        const C = U.other.compare;
        const w = C['72 window statistics'];
        const p = C['528 wavelet features'];
        return `
        <p>Prof. Shao's statistics are taken over the whole second, so they lose <i>when</i> things happen. The
        <b>72 window statistics</b> keep it: the average and spread of each sensor in 9 time windows, from 0-5 ms to
        700-1000 ms.</p>
        <p>With the same logistic regression they reach <b>${pct(w.nine_tool_surface[0])}</b> on 9 conditions
        (against ${pct(p.nine_tool_surface[0])}) and <b>${pct(w.clean_vs_polished)}</b> on clean vs polished
        (against ${pct(p.clean_vs_polished)}). The lasso's surest picks: the power signal's spread at 100-200 ms and
        the acoustic emission's spread in the first 5 ms.</p>
        <p><b>Caution:</b> the 9 windows were chosen on Nov 9 after looking at all 270 labeled welds, so these
        numbers are optimistic. A fair test needs welds that played no part in that choice.</p>
        <p class="next">Below the bars: each condition's average signal over the weld. Pick a sensor.</p>`;
      },
    },
  ];

  // ---------------------------------------------------------------- drawing and events

  const tipEl = document.createElement('div');
  tipEl.className = 'tip';
  tipEl.hidden = true;
  const content = document.createElement('div');
  content.className = 'lr-card';
  root.append(content, tipEl);

  function render() {
    tips.length = 0;
    const step = STEPS[sel.step];
    const width = Math.max(320, Math.min(root.clientWidth - 2 * 22 - 2 * 20, 980));
    content.innerHTML = step.chart(width);
    tipEl.hidden = true;
    return context();
  }

  function context() {
    const t = task();
    return { task: sel.task, title: SHORT[sel.task], t };
  }

  function glance() {
    const t = task();
    const best = ['1', '2', '3', '5', '10', '20', '50'].reduce((b, m) => (t.lasso_curve[m] > t.lasso_curve[b] ? m : b), '1');
    const single = bestSingle(sel.task);
    return `
      <h2>${esc(KIND[sel.task] ? `${KIND[sel.task]}: ` : '')}${esc(SHORT[sel.task])}</h2>
      <dl class="facts">
        <div><dt>All 528 inputs</dt><dd>${pct(t.plain.mean)}<small>logistic regression, test</small></dd></div>
        <div><dt>Best few</dt><dd>${pct(t.lasso_curve[best])}<small>the lasso's first ${best}</small></dd></div>
        <div><dt>Sensors picked</dt><dd>${sensorsLine(sel.task) || '-'}</dd></div>
      </dl>
      <h3>Surest single pick</h3>
      <p class="lr-glance">${esc(longName(topPick(sel.task)))}${single != null ? `<br><small>Best on its own: ${esc(shortName(single))}, AUROC ${t.single_auroc[single].toFixed(3)}</small>` : ''}</p>
      <p class="lr-glance"><small>Exploratory analysis, ${esc(U.source)} (commit ${esc(U.commit)}). Prof. Shao's
      own 528 inputs; the group's 5 splits.</small></p>`;
  }

  function act(name, value) {
    if (name === 'task') sel.task = value;
    else if (name === 'feature') sel.feature = +value;
    else if (name === 'plot') {
      sel.plot = +value;
      actions.goTo(STEPS.findIndex((s) => s.chart === dotChart));
      return;
    } else if (name === 'show') {
      actions.showInput(+value);
      return;
    } else if (name === 'metric') sel.metric = +value;
    else if (name === 'model') sel.model = value;
    else if (name === 'sensor') sel.sensor = value;
    else if (name === 'range') sel.range = +value;
    actions.changed();
  }

  root.addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el) return;
    const [name, ...rest] = el.dataset.act.split(':');
    act(name, rest.join(':'));
  });
  root.addEventListener('change', (e) => {
    const ctl = e.target.closest('[data-ctl]');
    if (!ctl) return;
    if (ctl.dataset.ctl === 'task') sel.task = ctl.value;
    if (ctl.dataset.ctl === 'plot') sel.plot = /^\d+$/.test(ctl.value) ? +ctl.value : ctl.value;
    actions.changed();
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
    if (!root.hidden) render();
  }).observe(root);

  return { STEPS, render, glance };
}
