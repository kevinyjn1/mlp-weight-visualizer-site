// The fifth tab: what the course's methods (ME 499/599 Topics 9 to 12, Lab 4) show on the welds. Charts, no 3D, from
// weld-monitor/experiments/usmw_course_methods (course_methods.py: E1 to E3; plot.py: the figure data), exported by
// tools/course_methods.py. Every number on it comes from data/course.json, except where a comment says otherwise.

import {
  SENSOR_COLOR, SURFACE_COLOR, BLUES, STAT, INK, MUTED, GRID, ACCENT, ASTRA, pct, esc, ramp, niceTicks, header,
  segmented, createChartPanel,
} from './chartkit.js';

const GROUPS = ['pooled', 'New', 'Worn', 'Damaged']; // the tools in the order they were welded
const GROUP_LABEL = { pooled: 'All tools pooled', New: 'New tool', Worn: 'Worn tool', Damaged: 'Damaged tool' };
const MODEL_COLOR = { LDA: '#0072B2', QDA: '#E69F00' };
const CLASSIFIERS = [
  'LR, all 528', 'Shrinkage LDA, all 528', 'Lasso 20 + LR', 'LDA, Fisher top k', 'QDA, Fisher top k', 'RBF SVM, all 528',
  'RBF SVM, Fisher top k', 'Tool then surface',
];
const CLASSIFIER_LABEL = {
  'LR, all 528': "Logistic regression, all 528 (EXP-19's)",
  'Shrinkage LDA, all 528': 'Shrinkage LDA, all 528',
  'Lasso 20 + LR': "The lasso's 20, then logistic regression",
  'LDA, Fisher top k': 'LDA on the Fisher top k',
  'QDA, Fisher top k': 'QDA on the Fisher top k',
  'RBF SVM, all 528': 'RBF SVM, all 528',
  'RBF SVM, Fisher top k': 'RBF SVM on the Fisher top k',
  'Tool then surface': 'Tool, then a surface rule per tool',
};
const TASK_LABEL = { nine: '9 conditions', surface: 'Surface (3)', CP: 'Clean vs polished', CX: 'Clean vs contaminated', tool: 'Tool (3)' };
const VARIANT_LABEL = { raw: 'Raw', centered: 'Centered', detrended: 'Detrended', 'detrended, all welds': 'Detrended, all welds' };
const ASTRA_EXP19 = 0.878; // EXP-19 RESULTS.md: arm N, 9 conditions
const FLOOR = 1e-3; // Fisher ratios below this are drawn at it (log axes)
// Lab 4's combinations of the top 4, in the order course_methods.py fits them (m = 2, 3, 4; LDA then QDA for each)
const COMBOS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]];
const SL = 'Shrinkage LDA, all 528';
const TS = 'Tool then surface';

export function createCourse({ root, data, state, actions }) {
  const C = data.course;
  const F = C.figures;
  const N = F.names;
  const W = F.welds;
  const sel = state.course;
  const HI = F.highlight; // the top pooled clean vs contaminated and clean vs polished features
  const valued = Object.keys(F.values).map(Number).sort((a, b) => a - b); // features with per-weld values
  if (sel.feature == null) sel.feature = HI.contamination;
  sel.picked ??= null;
  let keepFeature = false; // set when a click elsewhere chose the feature, so a step does not reset it

  const panel = createChartPanel({
    root,
    onAct: (name, value) => act(name, value),
    onChange: (ctl) => change(ctl),
    onResize: () => render(),
  });
  const tip = (html) => panel.tip(html);

  // ---------------------------------------------------------------- names and small helpers

  const parts = (i) => N.features[i].split(' '); // sensor, band, statistic
  const bandOf = (i) => Math.floor((i % 132) / 12);
  const shortName = (i) => {
    const [s, b, st] = parts(i);
    return `${s} ${b} ${STAT[st]?.[0] ?? st}`;
  };
  const longName = (i) => {
    const [s, , st] = parts(i);
    return `${s}, band ${N.bands[bandOf(i)]}, ${STAT[st]?.[1] ?? st}`;
  };
  const toolIndex = (name) => N.tools.indexOf(name);
  const f2Of = (g, i) => F.fisher[g].f2[i];
  const f1Of = (g, i) => F.fisher[g].f1[i];
  const argmax = (a) => a.reduce((b, v, k) => (v > a[b] ? k : b), 0);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const fmtJ = (v) => (v >= 10 ? v.toFixed(1) : v >= 0.1 ? v.toFixed(2) : v.toFixed(3));
  const jitter = (key, spread) => ((((key * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * spread;
  const e3 = (v, t, name) => C.e3[v][t][name];
  const better = (a, b) => a.acc.per_split.filter((x, k) => x > b.acc.per_split[k] + 1e-9).length;

  function featureSelect() {
    const opt = (i, label) => `<option value="${i}"${i === sel.feature ? ' selected' : ''}>${esc(label)}</option>`;
    const rest = valued.filter((i) => i !== HI.contamination && i !== HI.polishing);
    return `<label class="lr-select"><span>Feature</span><select data-ctl="feature">
      <optgroup label="Named in the text">${opt(HI.contamination, `${shortName(HI.contamination)} (contamination)`)}${opt(HI.polishing, `${shortName(HI.polishing)} (polishing)`)}</optgroup>
      <optgroup label="Near the top of a Fisher ranking">${rest.map((i) => opt(i, shortName(i))).join('')}</optgroup></select></label>`;
  }

  const groupButtons = () => segmented('group', GROUPS.map((g) => [g, g === 'pooled' ? 'Pooled' : g]), sel.group);
  const surfaceLegend = () => `<div class="lr-legend">${N.surfaces.map((name, k) =>
    `<span class="lr-key"><i style="background:${SURFACE_COLOR[k]}"></i>${name}</span>`).join('')}</div>`;

  function weldTip(n, value, i) {
    return tip(`<b>Weld ${W.run[n]}</b><span>${N.tools[W.tool[n]]} tool, ${N.surfaces[W.surface[n]]} surface</span>
      <span>${esc(shortName(i))}: ${value.toFixed(2)} (standardized)</span>`);
  }

  // ---------------------------------------------------------------- charts

  /** Step 1: one feature against run number, with the three tool sessions. */
  function sessionsChart(width) {
    const i = sel.feature;
    const v = F.values[String(i)];
    const left = 44;
    const right = 10;
    const top = 30;
    const bottom = 42;
    const h = 330;
    const r0 = Math.min(...W.run);
    const r1 = Math.max(...W.run);
    const x = (r) => left + ((r - r0 + 0.5) / (r1 - r0 + 1)) * (width - left - right);
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    const pad = (hi - lo) * 0.06;
    const y = (z) => top + (1 - (z - lo + pad) / (hi - lo + 2 * pad)) * (h - top - bottom);
    let s = '';
    Object.entries(F.sessions).forEach(([tool, [a, b]], k) => {
      s += `<rect x="${x(a - 0.5)}" y="${top}" width="${x(b + 0.5) - x(a - 0.5)}" height="${h - top - bottom}" fill="${k % 2 ? '#fff' : '#f3f4f7'}"/>`;
      s += `<text x="${(x(a) + x(b)) / 2}" y="${top - 10}" text-anchor="middle" class="lbl strong">${tool} tool${width >= 640 ? `<tspan class="kind"> · runs ${a}-${b}</tspan>` : ''}</text>`;
    });
    for (const t of niceTicks(lo - pad, hi + pad, 5)) {
      s += `<line x1="${left}" x2="${width - right}" y1="${y(t)}" y2="${y(t)}" stroke="${GRID}"/>`;
      s += `<text x="${left - 7}" y="${y(t) + 4}" text-anchor="end" class="tick">${t}</text>`;
    }
    for (const t of niceTicks(r0, r1, width >= 640 ? 6 : 3)) {
      s += `<text x="${x(t)}" y="${h - bottom + 17}" text-anchor="middle" class="tick">${t}</text>`;
    }
    s += `<text x="${left}" y="${h - 6}" class="axis">${width >= 600 ? `Run number (the welder's counter) · up: ${esc(shortName(i))}, standardized over the 270 welds` : `Run number · up: ${esc(shortName(i))}, standardized`}</text>`;
    W.run.forEach((run, n) => {
      s += `<circle cx="${x(run)}" cy="${y(v[n])}" r="3.6" fill="${SURFACE_COLOR[W.surface[n]]}" stroke="#fff" stroke-width="0.7" ${weldTip(n, v[n], i)}/>`;
    });
    return `${header('When each weld was made', 'All 270 welds in welding order · one dot per weld', featureSelect())}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>${surfaceLegend()}`;
  }

  /** Step 2: one feature per tool and surface, with the clean vs polished Fisher ratio. */
  function perToolChart(width) {
    const i = sel.feature;
    const v = F.values[String(i)];
    const narrow = width < 600;
    const tools = ['New', 'Worn', 'Damaged'];
    const gap = 18;
    const left = 40;
    const cols = narrow ? 1 : 3;
    const pw = (width - left - gap * (cols - 1)) / cols;
    const ph = narrow ? 180 : 250;
    const titleH = 40;
    const labelH = 22;
    const blockH = titleH + ph + labelH;
    const h = (narrow ? 3 : 1) * blockH + 6;
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    const pad = (hi - lo) * 0.06;
    let s = '';
    tools.forEach((tool, k) => {
      const ti = toolIndex(tool);
      const ox = left + (narrow ? 0 : k * (pw + gap));
      const oy = (narrow ? k : 0) * blockH + titleH;
      const y = (z) => oy + (1 - (z - lo + pad) / (hi - lo + 2 * pad)) * ph;
      const j2 = f2Of(tool, i);
      const j1 = f1Of(tool, i);
      s += `<text x="${ox}" y="${oy - 22}" class="lbl strong">${tool} tool</text>`;
      s += `<text x="${ox}" y="${oy - 7}" class="tick" ${tip(`<b>${tool} tool · ${esc(shortName(i))}</b><span>Fisher ratio, clean vs polished: ${fmtJ(j2)}</span><span>Fisher ratio, clean vs contaminated: ${fmtJ(j1)}</span><span>(gap between the two averages)² / (sum of their variances), all 90 welds of this tool</span>`)}>clean vs polished: <tspan font-weight="700" fill="${INK}">${fmtJ(j2)}</tspan></text>`;
      for (const t of niceTicks(lo - pad, hi + pad, 5)) {
        s += `<line x1="${ox}" x2="${ox + pw}" y1="${y(t)}" y2="${y(t)}" stroke="${GRID}"/>`;
        if (k === 0 || narrow) s += `<text x="${left - 7}" y="${y(t) + 4}" text-anchor="end" class="tick">${t}</text>`;
      }
      const slot = pw / 3;
      N.surfaces.forEach((surface, si) => {
        const cx = ox + (si + 0.5) * slot;
        const rows = W.run.map((_, n) => n).filter((n) => W.tool[n] === ti && W.surface[n] === si);
        const vals = rows.map((n) => v[n]).sort((a, b) => a - b);
        const med = vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2;
        for (const n of rows) {
          s += `<circle cx="${cx + jitter(W.run[n], slot * 0.45)}" cy="${y(v[n])}" r="3.3" fill="${SURFACE_COLOR[si]}" fill-opacity="0.85" ${weldTip(n, v[n], i)}/>`;
        }
        s += `<line x1="${cx - slot * 0.3}" x2="${cx + slot * 0.3}" y1="${y(med)}" y2="${y(med)}" stroke="${SURFACE_COLOR[si]}" stroke-width="3" stroke-linecap="round" pointer-events="none"/>`;
        const label = slot < 84 && surface === 'Contaminated' ? 'Contam.' : surface;
        s += `<text x="${cx}" y="${oy + ph + 16}" text-anchor="middle" class="tick">${label}</text>`;
      });
    });
    return `${header('One feature, three tools', `${esc(longName(i))} · bar = median · all 270 welds`, featureSelect())}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>`;
  }

  /** Step 3: every feature's two Fisher ratios, on log axes, pooled or inside one tool. */
  function fisherChart(width) {
    const g = sel.group;
    const f1 = F.fisher[g].f1;
    const f2 = F.fisher[g].f2;
    const left = 52;
    const right = 14;
    const top = 14;
    const bottom = 46;
    const pw = Math.min(width - left - right, 640);
    const h = top + Math.min(pw * 0.62, 400) + bottom;
    const allF1 = GROUPS.flatMap((k) => F.fisher[k].f1);
    const allF2 = GROUPS.flatMap((k) => F.fisher[k].f2);
    const xHi = Math.max(2, Math.ceil(Math.log10(Math.max(...allF1)))); // the same axes for every group
    const yHi = Math.max(1, Math.ceil(Math.log10(Math.max(...allF2))));
    const lg = (v) => Math.log10(Math.max(v, FLOOR));
    const x = (v) => left + ((lg(v) + 3.15) / (xHi + 3.15)) * pw;
    const y = (v) => top + (1 - (lg(v) + 3.15) / (yHi + 3.15)) * (h - top - bottom);
    const label = (p) => (p < 0 ? `${10 ** p}`.replace(/^0\./, '0.') : `${10 ** p}`);
    let s = '';
    for (let p = -3; p <= xHi; p++) {
      s += `<line x1="${x(10 ** p)}" x2="${x(10 ** p)}" y1="${top}" y2="${h - bottom}" stroke="${GRID}"/>`;
      s += `<text x="${x(10 ** p)}" y="${h - bottom + 16}" text-anchor="middle" class="tick">${label(p)}</text>`;
    }
    for (let p = -3; p <= yHi; p++) {
      s += `<line x1="${left}" x2="${left + pw}" y1="${y(10 ** p)}" y2="${y(10 ** p)}" stroke="${GRID}"/>`;
      s += `<text x="${left - 7}" y="${y(10 ** p) + 4}" text-anchor="end" class="tick">${label(p)}</text>`;
    }
    s += `<line x1="${x(1)}" x2="${x(1)}" y1="${top}" y2="${h - bottom}" stroke="${MUTED}" stroke-dasharray="4 3"/>`;
    s += `<line x1="${left}" x2="${left + pw}" y1="${y(1)}" y2="${y(1)}" stroke="${MUTED}" stroke-dasharray="4 3"/>`;
    s += `<text x="${left}" y="${h - 6}" class="axis">${width >= 600 ? 'Across: Fisher 1, clean vs contaminated · up: Fisher 2, clean vs polished · dashed: ratio 1 (gap = spread)' : 'Across: vs contaminated · up: vs polished · dashed: 1'}</text>`;
    const top4 = [...f1.keys()].sort((a, b) => f1[b] + f2[b] - (f1[a] + f2[a])).slice(0, 4);
    N.sensors.forEach((sensor, si) => {
      for (let i = 132 * si; i < 132 * (si + 1); i++) {
        const can = valued.includes(i);
        const t = tip(`<b>${esc(shortName(i))}</b><span>${esc(longName(i))}</span><span>Fisher 1, clean vs contaminated: ${fmtJ(f1[i])}</span>
          <span>Fisher 2, clean vs polished: ${fmtJ(f2[i])}</span><span>${GROUP_LABEL[g]} · all its welds</span>
          <span>click to select${can ? ' (its welds can be plotted)' : ''}</span>`);
        s += `<circle cx="${x(f1[i])}" cy="${y(f2[i])}" r="2.8" fill="${SENSOR_COLOR[sensor]}" fill-opacity="0.75" data-act="pick:${i}" class="pt" ${t}/>`;
      }
    });
    for (const i of top4) s += `<circle cx="${x(f1[i])}" cy="${y(f2[i])}" r="6.5" fill="none" stroke="${INK}" stroke-width="1.3" pointer-events="none"/>`;
    if (sel.picked != null) {
      const i = sel.picked;
      s += `<circle cx="${x(f1[i])}" cy="${y(f2[i])}" r="8" fill="none" stroke="#d94801" stroke-width="2.4" pointer-events="none"/>`;
    }
    const best = argmax(f2);
    s += `<text x="${left + 8}" y="${top + 16}" class="value halo">best clean vs polished: ${fmtJ(f2[best])} (${esc(shortName(best))})</text>`;
    const legend = `<div class="lr-legend">${N.sensors.map((name) => `<span class="lr-key"><i style="background:${SENSOR_COLOR[name]}"></i>${name}</span>`).join('')}
      <span class="lr-key"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="none" stroke="${INK}" stroke-width="1.3"/></svg>Lab 4's top 4 (largest sum)</span>
      <span class="lr-note">Ratios below 0.001 are drawn at 0.001.</span></div>`;
    return `${header('Every feature on both surface pairs', `528 features · ${GROUP_LABEL[g]} · Fisher ratio = (gap between the averages)² / (sum of the variances)`, groupButtons())}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>${legend}${pickedCard()}`;
  }

  function pickedCard() {
    const i = sel.picked;
    if (i == null) return '<p class="lr-note">Click a dot to select a feature.</p>';
    const can = valued.includes(i);
    const ratios = GROUPS.map((g) => `${g === 'pooled' ? 'pooled' : g} ${fmtJ(f2Of(g, i))}`).join(' · ');
    return `<div class="lr-feature"><div><b>${esc(shortName(i))}</b> · ${esc(longName(i))}<br>
      <span>Clean vs polished: ${ratios}</span></div>
      <div class="lr-buttons">${can ? `<button type="button" class="secondary" data-act="plot:${i}">Plot its welds per tool</button>`
        : '<span class="lr-note">Per-weld values are exported only for the 62 features near the top of a ranking.</span>'}</div></div>`;
  }

  /** Step 4: median |r| between the 12 statistics over the 44 sensor bands. */
  function nearChart(width) {
    const st = F.near.stats;
    const r = F.near.median_abs_r;
    const name = (k) => STAT[st[k]]?.[0] ?? st[k];
    const left = 54;
    const cell = Math.max(20, Math.min(40, Math.floor((width - left - 10) / 12)));
    const top = 8;
    const h = top + 12 * cell + 56;
    let s = '';
    for (let a = 0; a < 12; a++) {
      s += `<text x="${left - 7}" y="${top + a * cell + cell / 2 + 4}" text-anchor="end" class="tick">${name(a)}</text>`;
      const lx = left + a * cell + cell / 2;
      s += `<text x="${lx}" y="${top + 12 * cell + 12}" text-anchor="end" class="tick" transform="rotate(-45 ${lx} ${top + 12 * cell + 12})">${name(a)}</text>`;
      for (let b = 0; b < 12; b++) {
        const x0 = left + b * cell;
        const y0 = top + a * cell;
        if (a === b) {
          s += `<rect x="${x0}" y="${y0}" width="${cell - 1}" height="${cell - 1}" fill="#eef0f4"/>`;
          continue;
        }
        const v = r[a][b];
        const t = tip(`<b>${STAT[st[a]]?.[1] ?? st[a]} and ${STAT[st[b]]?.[1] ?? st[b]}</b><span>median |correlation| over the 44 sensor bands: ${v.toFixed(2)}</span>`);
        s += `<rect x="${x0}" y="${y0}" width="${cell - 1}" height="${cell - 1}" fill="${ramp(BLUES, v)}" class="cell" ${t}/>`;
        if (cell >= 26) s += `<text x="${x0 + cell / 2}" y="${y0 + cell / 2 + 3.5}" text-anchor="middle" class="incell" fill="${v > 0.6 ? '#fff' : INK}">${v.toFixed(2)}</text>`;
      }
    }
    const L = F.near.left;
    const tiles = `<div class="lr-tiles">${[['0.99', 'at |r| above 0.99'], ['0.95', 'at |r| above 0.95'], ['0.9', 'at |r| above 0.9']].map(([k, label]) =>
      `<div><b>${L[k]}</b><span>of 528 left ${label}</span></div>`).join('')}</div>`;
    return `${header('The 528 inputs repeat each other', "Prof. Shao's 12 statistics, compared within each of the 44 sensor bands · all 270 welds")}
      <div class="lr-split"><svg viewBox="0 0 ${12 * cell + left + 10} ${h}" width="${12 * cell + left + 10}" height="${h}" class="lr-svg">${s}</svg>
      <div><p class="lr-note">Features left after dropping every one that nearly copies an earlier one:</p>${tiles}</div></div>`;
  }

  /** Step 5: Lab 4's plot, number of features against test error. */
  function lab4Chart(width) {
    const d = C.e1[sel.screen][sel.group];
    const fits = d.fits.map(([model, m, split, test, train], k) => ({ model, m, split, test, train, combo: COMBOS[Math.floor((k % 22) / 2)] }));
    const left = 46;
    const right = 10;
    const top = 16;
    const bottom = 62;
    const h = 330;
    const maxErr = Math.max(0.6, ...fits.filter((z) => z.test != null).map((z) => z.test));
    const slot = (width - left - right) / 3;
    const x = (m, model) => left + (m - 2 + 0.5) * slot + (model === 'LDA' ? -0.2 : 0.2) * slot;
    const y = (e) => top + (1 - e / maxErr) * (h - top - bottom); // error up, as in Lab 4
    let s = '';
    for (const t of niceTicks(0, maxErr, 6)) {
      s += `<line x1="${left}" x2="${width - right}" y1="${y(t)}" y2="${y(t)}" stroke="${GRID}"/>`;
      s += `<text x="${left - 7}" y="${y(t) + 4}" text-anchor="end" class="tick">${Math.round(t * 100)}%</text>`;
    }
    for (const m of [2, 3, 4]) {
      s += `<text x="${left + (m - 2 + 0.5) * slot}" y="${h - bottom + 34}" text-anchor="middle" class="lbl strong">${m} features</text>`;
      for (const model of ['LDA', 'QDA']) {
        const cell = fits.filter((z) => z.m === m && z.model === model);
        const ok = cell.filter((z) => z.test != null);
        const cx = x(m, model);
        s += `<text x="${cx}" y="${h - bottom + 16}" text-anchor="middle" class="tick">${model}</text>`;
        ok.forEach((z, n) => {
          const t = tip(`<b>${model}, split ${z.split}</b><span>${z.combo.map((c) => esc(d.top4[z.split][c][0])).join(' + ')}</span>
            <span>test error ${pct(z.test)} · training error ${pct(z.train)}</span>`);
          s += `<circle cx="${cx + jitter(n * 7 + z.split * 31 + m, slot * 0.22)}" cy="${y(z.test)}" r="3.4" fill="${MODEL_COLOR[model]}" fill-opacity="0.55" ${t}/>`;
        });
        if (ok.length) {
          const mu = mean(ok.map((z) => z.test));
          s += `<line x1="${cx - slot * 0.12}" x2="${cx + slot * 0.12}" y1="${y(mu)}" y2="${y(mu)}" stroke="${MODEL_COLOR[model]}" stroke-width="3" stroke-linecap="round" pointer-events="none"/>`;
          s += `<text x="${cx + slot * 0.14}" y="${y(mu) + 4}" class="value halo" fill="${MODEL_COLOR[model]}">${pct(mu, 0)}</text>`;
        }
        const refused = cell.length - ok.length;
        if (refused) s += `<text x="${cx}" y="${h - bottom - 6}" text-anchor="middle" class="tick" style="fill:#b42318">${refused} of ${cell.length} refused</text>`;
      }
    }
    s += `<text x="${left}" y="${h - 6}" class="axis">${width >= 600 ? 'Test error, 3 surfaces · dot = one split and one combination of the top 4 · bar = mean' : 'Test error, 3 surfaces · bar = mean'}</text>`;
    const tops = d.top4.map((row, k) => `<li><b>Split ${k}:</b> ${row.map(([n, a, b]) => `${esc(n)} <span>(${a.toFixed(2)} + ${b.toFixed(2)})</span>`).join(', ')}</li>`).join('');
    const controls = `${groupButtons()}${segmented('screen', [['deduplicated', 'Near-copies skipped'], ['plain', 'Plain screen']], sel.screen)}`;
    return `${header("Lab 4's procedure on the welds", `${GROUP_LABEL[sel.group]} · top 4 by Fisher 1 + Fisher 2 on each split's training welds · every combination of 2, 3 and 4`, controls)}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>
      <div class="lr-legend">${['LDA', 'QDA'].map((m) => `<span class="lr-key"><i style="background:${MODEL_COLOR[m]}"></i>${m}</span>`).join('')}</div>
      <div class="lr-subhead"><h3>The top 4 on each split (Fisher 1 + Fisher 2)</h3></div><ul class="lr-list">${tops}</ul>`;
  }

  /** Step 6: clean vs polished per split, from one pooled model to a rule per tool, with the residuals in between. */
  function residualChart(width) {
    const e2 = C.e2;
    const steps = [
      ['One pooled model', 'raw', e2.raw.CP[SL].per_split],
      ['Pooled', 'session centered', e2.centered.CP[SL].per_split],
      ['Pooled', 'session detrended', e2.detrended.CP[SL].per_split],
      ['A rule per tool', 'raw', e3('raw', 'CP', TS).acc.per_split],
      ['A rule per tool', 'detrended', e3('detrended', 'CP', TS).acc.per_split],
    ];
    const left = 46;
    const right = 50;
    const top = 14;
    const bottom = 50;
    const h = 320;
    const xs = steps.map((_, k) => left + 20 + (k * (width - left - right - 40)) / (steps.length - 1));
    const y = (v) => top + ((1 - v) / 0.2) * (h - top - bottom); // 80% to 100%
    let s = '';
    for (const t of [0.8, 0.85, 0.9, 0.95, 1]) {
      s += `<line x1="${left}" x2="${width - right}" y1="${y(t)}" y2="${y(t)}" stroke="${GRID}"/>`;
      s += `<text x="${left - 7}" y="${y(t) + 4}" text-anchor="end" class="tick">${t * 100}%</text>`;
    }
    const narrow = width < 600; // on a phone: "Pooled / raw", "Per tool / detrended"
    steps.forEach(([a, b], k) => {
      const top1 = narrow ? (a === 'A rule per tool' ? 'Per tool' : 'Pooled') : a;
      const top2 = narrow ? b.replace('session ', '') : b;
      s += `<text x="${xs[k]}" y="${h - bottom + 18}" text-anchor="middle" class="tick strong">${top1}</text>`;
      s += `<text x="${xs[k]}" y="${h - bottom + 32}" text-anchor="middle" class="tick">${top2}</text>`;
    });
    for (let split = 0; split < 5; split++) {
      const pts = steps.map(([, , v], k) => [xs[k], y(v[split])]);
      s += `<path d="${pts.map(([px, py], k) => `${k ? 'L' : 'M'}${px},${py}`).join(' ')}" fill="none" stroke="#a7adba" stroke-width="1.2"/>`;
      steps.forEach(([a, b, v], k) => {
        s += `<circle cx="${xs[k]}" cy="${y(v[split])}" r="3.4" fill="#a7adba" ${tip(`<b>Split ${split} · ${a}, ${b}</b><span>${pct(v[split])} of 36 test welds right (${Math.round(v[split] * 36)})</span>`)}/>`;
      });
    }
    const means = steps.map(([, , v]) => mean(v));
    s += `<path d="${means.map((m, k) => `${k ? 'L' : 'M'}${xs[k]},${y(m)}`).join(' ')}" fill="none" stroke="${ACCENT}" stroke-width="3"/>`;
    means.forEach((m, k) => {
      s += `<circle cx="${xs[k]}" cy="${y(m)}" r="5.5" fill="${ACCENT}" stroke="#fff" stroke-width="2" ${tip(`<b>${steps[k][0]}, ${steps[k][1]}</b><span>mean of the 5 splits: ${pct(m)}</span>`)}/>`;
      s += `<text x="${xs[k] + 9}" y="${y(m) + 16}" class="value halo" fill="${ACCENT}">${pct(m)}</text>`;
    });
    s += `<text x="${left}" y="${h - 4}" class="axis">${width >= 600 ? 'Test accuracy, clean vs polished · shrinkage LDA on all 528 throughout · grey: one split; blue: the mean' : 'Shrinkage LDA · grey: one split; blue: mean'}</text>`;
    const variants = Object.keys(VARIANT_LABEL);
    const rows = ['surface', 'CP', 'CX', 'tool'].flatMap((t) => ['LR, all 528', SL].map((model) =>
      `<tr${t === 'tool' ? ' class="dim"' : ''}><th>${TASK_LABEL[t]}${t === 'tool' ? ' <small>(diagnostic only)</small>' : ''}</th><td>${model === SL ? 'Shrinkage LDA' : 'Logistic regression'}</td>${variants.map((v) =>
        `<td>${pct(e2[v][t][model].mean)}</td>`).join('')}</tr>`)).join('');
    const table = `<div class="lr-subhead"><h3>Every question, with each way of removing the session's level</h3></div>
      <div class="lr-scroll"><table class="lr-table"><thead><tr><th>Question</th><th>Model</th>${variants.map((v) => `<th>${VARIANT_LABEL[v]}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="lr-note">Test accuracy, mean of the 5 splits. The tool row only shows how much tool information is left after removing each session's level: the residuals use the session, which is the tool.</p>`;
    return `${header("Removing each session's level", 'Clean vs polished, on the same test welds of each split')}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>${table}`;
  }

  /** Step 7: eight classifiers on the same inputs and splits. */
  function classifierChart(width) {
    const v = sel.features;
    const t = sel.task;
    const cells = C.e3[v][t];
    const rowH = 32;
    const narrow = width < 600;
    const left = narrow ? 150 : Math.min(290, width * 0.42);
    const right = 58;
    const top = 26;
    const h = top + rowH * CLASSIFIERS.length + 30;
    const mins = CLASSIFIERS.map((n) => cells[n].acc.min);
    const lo = Math.max(0, Math.floor((Math.min(...mins) - 0.02) * 20) / 20);
    const x = (a) => left + ((a - lo) / (1 - lo)) * (width - left - right);
    const bestName = CLASSIFIERS.reduce((b, n) => (cells[n].acc.mean > cells[b].acc.mean ? n : b), CLASSIFIERS[0]);
    let s = '';
    for (const tk of niceTicks(lo, 1, narrow ? 4 : 8)) {
      s += `<line x1="${x(tk)}" x2="${x(tk)}" y1="${top - 6}" y2="${h - 24}" stroke="${GRID}"/>`;
      s += `<text x="${x(tk)}" y="${h - 8}" text-anchor="middle" class="tick">${Math.round(tk * 100)}%</text>`;
    }
    if (t === 'nine' && v === 'raw') {
      s += `<line x1="${x(ASTRA_EXP19)}" x2="${x(ASTRA_EXP19)}" y1="${top - 14}" y2="${h - 24}" stroke="${ASTRA}" stroke-width="1.5" stroke-dasharray="5 3"/>`;
      s += `<text x="${x(ASTRA_EXP19) - 5}" y="${top - 12}" text-anchor="end" class="value halo" fill="${ASTRA}">Astra, EXP-19: 87.8%</text>`;
    }
    CLASSIFIERS.forEach((name, k) => {
      const c = cells[name];
      const a = c.acc;
      const yy = top + k * rowH + rowH / 2;
      const chosen = [...new Set(c.params.map((p) => Object.entries(p).map(([kk, vv]) => `${kk.replace(/^[a-z]__/, '')} = ${typeof vv === 'number' && vv < 0.01 ? vv.toExponential(1) : vv}`).join(', ')).filter(Boolean))];
      const three = c.three ? `<span>tool ${pct(c.three[1])} · surface ${pct(c.three[2])}</span>` : '';
      const t2 = tip(`<b>${esc(CLASSIFIER_LABEL[name])}</b><span>mean ${pct(a.mean)} · splits ${a.per_split.map((p) => pct(p, 0)).join(' · ')}</span>${three}
        ${chosen.length ? `<span>chosen inside the training welds: ${esc(chosen.join('; '))}</span>` : ''}`);
      s += `<g class="row" ${t2}><rect x="0" y="${yy - rowH / 2}" width="${width}" height="${rowH}" fill="transparent"/>
        <text x="${left - 12}" y="${yy + 4}" text-anchor="end" class="lbl${name === bestName ? ' strong' : ''}">${esc(narrow ? name : CLASSIFIER_LABEL[name])}</text>
        <line x1="${x(a.min)}" x2="${x(a.max)}" y1="${yy}" y2="${yy}" stroke="${ACCENT}" stroke-opacity="0.35" stroke-width="5" stroke-linecap="round"/>
        ${a.per_split.map((p) => `<circle cx="${x(p)}" cy="${yy}" r="2.4" fill="${MUTED}"/>`).join('')}
        <circle cx="${x(a.mean)}" cy="${yy}" r="5.5" fill="${ACCENT}" stroke="#fff" stroke-width="2"/>
        <text x="${x(a.max) + 9}" y="${yy + 4}" class="value halo${name === bestName ? ' strong' : ''}">${pct(a.mean)}</text></g>`;
    });
    const tasks = Object.keys(C.e3[v]);
    const controls = `${segmented('task', tasks.map((k) => [k, TASK_LABEL[k]]), t)}${segmented('features', [['raw', 'Raw'], ['detrended', 'Detrended']], v)}`;
    return `${header('Which classifier', `${TASK_LABEL[t]} · ${v === 'raw' ? 'raw features' : 'features detrended inside each session'} · dot = mean of 5 splits, small dots = each split`, controls)}
      <svg viewBox="0 0 ${width} ${h}" width="${width}" height="${h}" class="lr-svg">${s}</svg>`;
  }

  // ---------------------------------------------------------------- the walkthrough

  const STEPS = [
    {
      title: 'When each weld was made',
      chart: sessionsChart,
      enter: () => {
        if (!keepFeature || !valued.includes(sel.feature)) sel.feature = HI.contamination;
        keepFeature = false;
      },
      body: () => {
        const s = F.sessions;
        return `
        <p>Each dot is one weld, placed by its run number (the welder's counter). Each tool was welded as one block
        of 90 runs in a row: New (${s.New[0]}-${s.New[1]}), then Worn, then Damaged. Inside each block the three
        surfaces were shuffled.</p>
        <p>So anything that drifts with time (room noise, temperature, a sensor warming up) also separates the tools.
        The data cannot tell a tool from its session. The surfaces are compared inside one session, in shuffled order,
        so a drift cannot fake a surface difference.</p>
        <p><b>For the paper:</b> tool accuracy cannot count as evidence. The surface questions can.</p>
        <p class="next">Pick the polishing feature above the chart: polishing stands out clearly only in the
        Damaged block.</p>`;
      },
    },
    {
      title: 'One feature, three tools',
      chart: perToolChart,
      enter: () => {
        if (!keepFeature || !valued.includes(sel.feature)) sel.feature = HI.polishing;
        keepFeature = false;
      },
      body: () => {
        const p = HI.polishing;
        return `
        <p>The same feature for each tool, split by surface. Its <b>Fisher ratio</b> for clean vs polished is the gap
        between the two averages, squared, over the sum of their variances: large when the groups barely overlap.</p>
        <p>For <b>${esc(shortName(p))}</b> it is <b>${fmtJ(f2Of('Damaged', p))}</b> with the Damaged tool, and only
        ${fmtJ(f2Of('New', p))} and ${fmtJ(f2Of('Worn', p))} with New and Worn.</p>
        <p>So the surfaces are not uniformly hard. Contamination is easy with every tool (98 to 99% with all 528
        inputs), and polishing is easy with the Damaged tool. The hard case is <b>clean vs polished with the New and
        Worn tools</b>.</p>
        <p><b>For the paper:</b> that hard case is where process knowledge would have to save labels.</p>`;
      },
    },
    {
      title: 'Every feature on both surface pairs',
      chart: fisherChart,
      body: () => {
        const pooled = F.fisher.pooled;
        const best = (g) => fmtJ(Math.max(...F.fisher[g].f2));
        return `
        <p>Each dot is one of the 528 features, placed by its Fisher ratio for clean vs contaminated (across) and for
        clean vs polished (up). Both axes are logarithmic.</p>
        <p>Lab 4 keeps the 4 features with the <b>largest sum</b> of the two ratios (ringed). Pooled over the tools,
        contamination ratios reach ${fmtJ(Math.max(...pooled.f1))} while polishing ratios stay at or below
        ${best('pooled')}. The sum is decided by contamination, and no feature it picks serves polishing.</p>
        <p>Inside one tool, polishing separates far better: up to ${best('Damaged')} for Damaged, ${best('New')} for
        New and ${best('Worn')} for Worn.</p>
        <p class="next">Switch tools above the chart, and click a dot to select a feature.</p>`;
      },
    },
    {
      title: 'The 528 inputs repeat each other',
      chart: nearChart,
      body: () => {
        const st = F.near.stats;
        const r = F.near.median_abs_r;
        const v = (a, b) => r[st.indexOf(a)][st.indexOf(b)].toFixed(2);
        const L = F.near.left;
        return `
        <p>Prof. Shao's 528 inputs are 12 statistics of each of 44 sensor bands. Each cell shows how strongly two of
        the statistics move together across the welds (the median over the 44 bands).</p>
        <ul>
          <li><b>std and rms: ${v('std', 'rms')}.</b> The wavelet coefficients average about zero, so the two are the
          same number.</li>
          <li><b>var</b> is std squared (${v('std', 'var')}); the two <b>crossing rates</b> agree
          (${v('zero_crossing_rate', 'mean_crossing_rate')}); <b>p5 and p95</b> mirror each other (${v('p05', 'p95')}).</li>
        </ul>
        <p>Drop every feature that nearly copies an earlier one and <b>${L['0.95']} of the 528</b> are left.</p>
        <p><b>For the paper:</b> EXP-19 gave Astra all 528 inputs, and about half of them repeat others. A prompt can
        give the distinct ones: less to read, the same information.</p>`;
      },
    },
    {
      title: "Lab 4's procedure on the welds",
      chart: lab4Chart,
      body: () => {
        const errOf = (screen, g, model, m) => {
          const ok = C.e1[screen][g].fits.filter(([md, mm, , te]) => md === model && mm === m && te != null);
          return mean(ok.map((z) => z[3]));
        };
        const plain = GROUPS.flatMap((g) => C.e1.plain[g].fits.filter(([md]) => md === 'QDA'));
        const refused = plain.filter((z) => z[3] == null).length;
        const dd = C.e1.deduplicated;
        const pooledFit = dd.pooled.fits.filter(([md, m, , te]) => md === 'LDA' && m === 4 && te != null);
        return `
        <p>Lab 4's procedure on the 3 surfaces: score every feature by Fisher 1 (clean vs contaminated) plus
        Fisher 2 (clean vs polished) on the training welds, keep the top 4, then fit LDA and QDA on every combination
        of 2, 3 or 4 of them.</p>
        <p>With the <b>plain screen</b>, the top 4 are often near-copies (the std and rms of one band), and QDA
        cannot be fit: <b>${refused} of ${plain.length}</b> QDA fits refused. Skipping near-copies fixes that.</p>
        <p>With near-copies skipped and 4 features, LDA's test error is <b>${pct(errOf('deduplicated', 'pooled', 'LDA', 4), 0)}</b>
        pooled, but <b>${pct(errOf('deduplicated', 'Damaged', 'LDA', 4), 0)}</b> inside the Damaged tool
        (${pct(errOf('deduplicated', 'New', 'LDA', 4), 0)} New, ${pct(errOf('deduplicated', 'Worn', 'LDA', 4), 0)} Worn).</p>
        <p><b>Lab 4's questions, for these welds:</b> more features mostly help; pooled training and test errors are
        about equal (${pct(mean(pooledFit.map((z) => z[4])), 0)} and ${pct(mean(pooledFit.map((z) => z[3])), 0)}), so
        4 features lack information rather than overfit; QDA beats LDA by at most a few points.</p>`;
      },
    },
    {
      title: "Removing each session's level",
      chart: residualChart,
      body: () => {
        const e2 = C.e2;
        const raw = e2.raw.CP[SL].mean;
        const pt = e3('raw', 'CP', TS).acc.mean;
        const ptd = e3('detrended', 'CP', TS).acc.mean;
        return `
        <p>Prof. Shao's 2013 monitor charted each feature after removing its trend over the weld count. Here, inside
        each tool session, a feature loses its session's median (<b>centered</b>) or a moving median of the 15
        nearest training welds (<b>detrended</b>). No surface label is used, and because the surfaces were shuffled,
        the trend cannot eat the surface gap.</p>
        <p>Clean vs polished with one pooled model: ${pct(raw)} raw, ${pct(e2.centered.CP[SL].mean)} centered,
        ${pct(e2.detrended.CP[SL].mean)} detrended. A <b>surface rule per tool</b> reaches ${pct(pt)} (better than the
        pooled model in ${better(e3('raw', 'CP', TS), { acc: e2.raw.CP[SL] })} of 5 splits), and detrending on top
        adds ${((ptd - pt) * 100).toFixed(1)} points.</p>
        <p>After centering, the tool is still told apart ${pct(e2.centered.tool['LR, all 528'].mean)} of the time
        (table below): the sessions differ in more than a constant offset.</p>
        <p><b>For the paper:</b> a rule per tool is the structure a knowledge method should be compared against.</p>`;
      },
    },
    {
      title: 'Which classifier',
      chart: classifierChart,
      body: () => {
        const n = (name) => e3('raw', 'nine', name);
        const lr = n('LR, all 528');
        return `
        <p>Eight classifiers on the same 528 inputs and splits. Every setting (k, C, gamma, reg_param) is chosen by
        cross-validation inside the training welds.</p>
        <p>On 9 conditions, <b>shrinkage LDA reaches ${pct(n(SL).acc.mean)}</b> and the RBF SVM on the Fisher top k
        ${pct(n('RBF SVM, Fisher top k').acc.mean)}, against ${pct(lr.acc.mean)} for EXP-19's logistic regression.
        Each beats it in ${better(n(SL), lr)} and ${better(n('RBF SVM, Fisher top k'), lr)} of 5 splits. Astra's
        EXP-19 network scored 87.8%.</p>
        <p>QDA is the weakest (${pct(n('QDA, Fisher top k').acc.mean)}): 18 training welds per condition cannot pin
        down a covariance per class.</p>
        <p><b>For the paper:</b> these are the baselines reviewers will expect. The comparison that counts, with 1 to
        5 labels per condition, has not been run yet.</p>`;
      },
    },
  ];

  // ---------------------------------------------------------------- drawing and events

  function render() {
    panel.draw(() => STEPS[sel.step].chart(panel.width()));
    return {};
  }

  function glance() {
    const nine = C.e3.raw.nine;
    const best = CLASSIFIERS.reduce((b, n) => (nine[n].acc.mean > nine[b].acc.mean ? n : b), CLASSIFIERS[0]);
    return `
      <h2>Course methods on the welds</h2>
      <dl class="facts">
        <div><dt>Distinct inputs</dt><dd>${F.near.left['0.95']} of 528<small>after dropping near-copies (|r| above 0.95)</small></dd></div>
        <div><dt>9 conditions</dt><dd>${pct(nine[best].acc.mean)}<small>${esc(CLASSIFIER_LABEL[best])}</small></dd></div>
        <div><dt>Clean vs polished</dt><dd>${pct(C.e2.raw.CP[SL].mean)} → ${pct(e3('raw', 'CP', TS).acc.mean)}<small>one pooled model → a rule per tool</small></dd></div>
      </dl>
      <p class="lr-glance"><small>Exploratory, ${esc(C.source)} (commit ${esc(C.commit)}). ME 499/599 Topics 9 to 12
      and Lab 4, on Prof. Shao's 528 inputs and the group's 5 splits; every model trained on all 162 training welds.</small></p>`;
  }

  function act(name, value) {
    if (name === 'group') sel.group = value;
    else if (name === 'screen') sel.screen = value;
    else if (name === 'task') {
      sel.task = value;
      if (!C.e3[sel.features][value]) sel.features = 'raw';
    } else if (name === 'features') {
      sel.features = value;
      if (!C.e3[value][sel.task]) sel.task = Object.keys(C.e3[value])[0];
    } else if (name === 'pick') sel.picked = +value;
    else if (name === 'plot') {
      sel.feature = +value;
      keepFeature = true;
      actions.goTo(STEPS.findIndex((s) => s.chart === perToolChart));
      return;
    }
    actions.changed();
  }

  function change(ctl) {
    if (ctl.dataset.ctl === 'feature') sel.feature = +ctl.value;
    actions.changed();
  }

  return { STEPS, render, glance };
}
