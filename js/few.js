// The third tab: Dennis's few-weld experiment. Not an MLP: a centroid per condition and a spread per feature, written
// by a language model from 18 labelled welds. A weld goes to the condition with the smallest sum, over the 8 features,
// of ((value - centroid) / spread)^2. Every number comes from his branch (see tools/few_welds.py).

import { className } from './net.js';
import { escapeHtml, markdown } from './md.js';

const SET_NAMES = ['Set 1: welds + full context', 'Set 2: + operator statements', 'Set 3: the bare table of welds'];
const DECIMALS = { V: 3, ms: 1, ratio: 3, mV: 2 };

export function createFew({ view, data, state, actions }) {
  const F = data.few;
  const M = data.manifest;
  const sel = state.few;
  const nf = F.features.length;
  const B = view.blocks;

  const show = (f, v) => {
    const { unit } = F.features[f];
    return `${+v.toFixed(DECIMALS[unit] ?? 3)}${unit === 'ratio' ? '' : ` ${unit}`}`;
  };
  const draw = () => F.draws[sel.draw];
  const weld = () => draw().welds[sel.weld];
  const setId = () => F.sets[sel.set].id;
  const writerName = () => (sel.writer === 'avg' ? 'No AI: plain averages' : F.models[sel.writer]);

  /** The plain rule with no model: each class's average of its 2 welds, and the pooled spread inside the classes. */
  function averages(d = sel.draw) {
    const welds = F.draws[d].welds;
    const centroids = Array.from({ length: 9 }, () => new Array(nf).fill(0));
    const spread = new Array(nf).fill(0);
    for (let k = 0; k < 9; k++) {
      const pair = welds.filter((w) => w.label === k);
      for (let f = 0; f < nf; f++) {
        centroids[k][f] = pair.reduce((s, w) => s + w.x[f], 0) / pair.length;
        for (const w of pair) spread[f] += (w.x[f] - centroids[k][f]) ** 2;
      }
    }
    return { centroids, spread: spread.map((s) => Math.sqrt(s / (welds.length - 9))) };
  }

  const model = () => (sel.writer === 'avg' ? averages() : F.replies[setId()][sel.writer][sel.draw]);

  function forward(m, x) {
    const terms = new Float32Array(nf * 9); // feature f, class k -> f * 9 + k
    const totals = new Array(9).fill(0);
    for (let f = 0; f < nf; f++) {
      for (let k = 0; k < 9; k++) {
        const t = ((x[f] - m.centroids[k][f]) / m.spread[f]) ** 2;
        terms[f * 9 + k] = t;
        totals[k] += t;
      }
    }
    let pred = 0;
    for (let k = 1; k < 9; k++) if (totals[k] < totals[pred]) pred = k;
    return { terms, totals, pred };
  }

  function accuracy() {
    const a = sel.writer === 'avg' ? F.accuracy.no_ai : F.accuracy[setId()][sel.writer];
    return { mean: a.mean, draw: a.draws[sel.draw] };
  }

  /** Everything the panel text quotes for the selection on screen. */
  function context() {
    const m = model();
    const w = weld();
    const out = forward(m, w.x);
    const avg = averages();
    const own = draw().welds.filter((v) => forward(m, v.x).pred === v.label).length;
    let near = 0;
    let worst = { d: 0, f: 0, k: 0 };
    for (let k = 0; k < 9; k++) {
      for (let f = 0; f < nf; f++) {
        const d = Math.abs(m.centroids[k][f] - avg.centroids[k][f]) / m.spread[f];
        if (d <= 0.1) near++;
        if (d > worst.d) worst = { d, f, k };
      }
    }
    return { m, w, out, avg, own, near, worst, acc: accuracy() };
  }

  // ------------------------------------------------------------------ 3D

  function render() {
    const c = context();
    const { m, w, out } = c;
    const step = STEPS[sel.step];
    const shown = state.mode.few === 'doc' ? 2 : step.shown;
    const mid = Array.from({ length: nf }, (_, f) => m.centroids.reduce((s, row) => s + row[f], 0) / 9);
    const z = new Float32Array(nf * 9);
    let top = 1e-9;
    for (let f = 0; f < nf; f++) {
      for (let k = 0; k < 9; k++) {
        z[f * 9 + k] = (m.centroids[k][f] - mid[f]) / m.spread[f];
        top = Math.max(top, Math.abs(z[f * 9 + k]));
      }
    }
    const clip = (v) => Math.max(-1, Math.min(1, v));
    const relative = m.spread.map((s, f) => {
      const column = m.centroids.map((row) => row[f]);
      return s / (Math.max(...column) - Math.min(...column) || 1);
    });
    const topSpread = Math.max(...relative);
    const topTerm = Math.max(...out.terms, 1e-9);
    const topTotal = Math.max(...out.totals, 1e-9);
    const blank = (n) => new Float32Array(n);
    view.setBlock('fx', Float32Array.from(w.x, (v, f) => clip((v - mid[f]) / m.spread[f] / top)), 'data', { duration: 700 });
    view.setBlock('fc', z.map((v) => v / top), 'weights', { duration: 900, stagger: 0.6 });
    view.setBlock('fs', Float32Array.from(relative, (v) => v / topSpread), 'weights', { duration: 700 });
    view.setBlock('fd', shown >= 1 ? out.terms.map((v) => v / topTerm) : blank(nf * 9), 'prob', { duration: 900, delay: 300 });
    view.setBlock('ft', shown >= 2 ? Float32Array.from(out.totals, (v) => v / topTotal) : blank(9), 'prob', { duration: 800, delay: 700 });
    view.setFocus(state.mode.few === 'doc' ? null : step.focus);
    state.fewShown = shown;
    const right = out.pred === w.label;
    view.updateLabel(predLabel, {
      html: `<b>${className(out.pred, M, true)}</b> is closest ${right ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>'}`,
      pos: [B.ft.rect().x1, 0, B.ft.center(out.pred)[1]],
    });
    return c;
  }

  let predLabel = null;
  function buildLabels() {
    const fly = (frame) => () => actions.fly(frame);
    const L = (def) => view.addLabel({ layout: 'few', ...def });
    const fx = B.fx.rect();
    const fc = B.fc.rect();
    const fs = B.fs.rect();
    const fd = B.fd.rect();
    const ft = B.ft.rect();
    F.features.forEach((f, k) => L({ html: f.id, cls: 'plain', pos: [B.fx.center(k)[0], 0, fx.z0], when: (c) => c.ppu >= 2.3 }));
    const far = (c) => c.ppu < 0.9; // the whole layout is small on screen: short names only
    const tag = (long, short, place) => {
      L({ html: long, ...place, when: (c) => !far(c) });
      L({ html: short, ...place, when: far });
    };
    tag('One weld · 8 features', 'Weld', { pos: [fx.x0, 0, fx.z0], align: 'l', row: (c) => (c.ppu >= 2.3 ? 1 : 0), onClick: fly('fewInput') });
    for (let k = 0; k < 9; k++) {
      L({ html: className(k, M, true), cls: 'tiny', pos: [fc.x0, 0, B.fc.center(0, k)[1]], align: 'r', v: 'middle', when: (c) => c.ppu >= 1.15 });
    }
    // the spread strip lies right under the centroid slab, so one tag below both names them
    tag('Centroids 9 × 8 · spread 8', 'Centroids', {
      pos: [fc.x0, 0, fs.z1], align: 'l', v: 'below', onClick: fly('fewCentroids'),
      alt: [{ pos: [fc.x0, 0, fc.z1], v: 'above' }], // on its own slab's corner if the ground title is too close
    });
    tag('Distance terms · 9 × 8', 'Distances', { pos: [fd.x0, 0, fd.z1], align: 'l', v: 'below', onClick: fly('fewDistance') });
    L({
      html: 'Sum', cls: 'tag small', pos: [(ft.x0 + ft.x1) / 2, 0, ft.z0], when: (c) => !far(c), onClick: fly('fewAnswer'),
      alt: [{ pos: [(ft.x0 + ft.x1) / 2, 0, ft.z1], v: 'below' }], // when the answer's tag is in the way
    });
    predLabel = L({ html: '', cls: 'tag pred', pos: [ft.x1, 0, B.ft.center(0)[1]], align: 'l', v: 'middle', clamp: true, when: () => state.fewShown >= 2 });
  }

  // ------------------------------------------------------------------ hover

  function linked({ id, i, j }) {
    if (id === 'fx' || id === 'fs') return { fx: [i, -1], fs: [i, -1], fc: [i, -1], fd: [i, -1] };
    if (id === 'ft') return { ft: [i, -1], fc: [-1, i], fd: [-1, i] };
    return { fx: [i, -1], fs: [i, -1], fc: [i, j], fd: [i, j], ft: [j, -1] };
  }

  function describe({ id, i, j }) {
    const { m, w, out, avg } = context();
    const feature = F.features[i];
    const about = feature && `${feature.id}: ${feature.meaning}`;
    if (id === 'fx') return [`${feature.id} = ${show(i, w.x[i])}`, feature.meaning, `This weld: ${className(w.label, M)}`];
    if (id === 'fs') {
      const column = m.centroids.map((row) => row[i]);
      const span = Math.max(...column) - Math.min(...column);
      return [`spread[${feature.id}] = ${show(i, m.spread[i])}`, about,
        `The nine centroids cover ${show(i, Math.min(...column))} to ${show(i, Math.max(...column))}: ${(span / m.spread[i]).toFixed(1)} spreads`];
    }
    if (id === 'fc') {
      const lines = [`centroid = ${show(i, m.centroids[j][i])}`, `${className(j, M)} · ${about}`];
      if (sel.writer !== 'avg') lines.push(`Plain average of its 2 welds: ${show(i, avg.centroids[j][i])}`);
      return lines;
    }
    if (id === 'fd') {
      if (state.fewShown < 1) return [`term[${className(j, M, true)}, ${feature.id}]`, 'Not computed yet at this step.'];
      return [`((${+w.x[i].toFixed(3)} - ${+m.centroids[j][i].toFixed(3)}) / ${m.spread[i]})² = ${out.terms[i * 9 + j].toFixed(2)}`,
        `${className(j, M)} · ${feature.id}`, `${Math.sqrt(out.terms[i * 9 + j]).toFixed(1)} spreads away on this feature`];
    }
    if (state.fewShown < 2) return [`sum[${className(i, M, true)}]`, 'Not computed yet at this step.'];
    return [`distance to ${className(i, M, true)} = ${out.totals[i].toFixed(2)}`, i === out.pred ? 'The smallest of the nine: this is the answer.' : `The smallest is ${out.totals[out.pred].toFixed(2)} (${className(out.pred, M, true)}).`];
  }

  // ------------------------------------------------------------------ text

  const accuracyTable = () => {
    const cell = (a, on, d) => `<td class="${on ? 'on' : ''}">${a.mean}%<small>${a.draws.map((v, k) => (on && k === d ? `<b>${v}</b>` : v)).join(', ')}</small></td>`;
    return `<div class="table"><table class="compare few"><thead><tr><th></th><th>Fable 5.2</th><th>Sonnet 5.5</th></tr></thead><tbody>
      ${F.sets.map((s, k) => `<tr><th>${SET_NAMES[k]}</th>${['fable', 'sonnet'].map((w) => cell(F.accuracy[s.id][w], sel.set === k && sel.writer === w, sel.draw)).join('')}</tr>`).join('')}
      <tr><th>No AI: class averages and within-class spread</th><td colspan="2" class="${sel.writer === 'avg' ? 'on' : ''}">${F.accuracy.no_ai.mean}%<small>${F.accuracy.no_ai.draws.join(', ')}</small></td></tr>
      </tbody></table></div>
      <p class="meta">Share of the 252 held-out welds put in the right one of nine conditions: mean, then draws 1, 2,
      3. Guessing gives 11%. ${escapeHtml(F.tool_surface)}</p>`;
  };

  const only = (text) => `<div class="aside few">${text}</div>`;
  const STEPS = [
    {
      title: 'A third way: write the classifier from 18 welds',
      frame: 'fewAll',
      focus: null,
      shown: 2,
      body: (c) => `
        <p>This tab is Dennis's experiment (branch <code>${F.branch}</code>, commit <code>${F.commit}</code>). It is
        <b>not</b> Prof. Shao's MLP. The classifier is far smaller: 8 hand-picked features of the power and
        horn-height signals, one <b>centroid</b> per condition (9 x 8 numbers) and one <b>spread</b> per feature
        (8 numbers). That is 80 numbers instead of 99,959.</p>
        <p>A language model was shown <b>18 labelled welds</b>, 2 per condition, and wrote those 80 numbers directly:
        no training, no tools. Claude Fable 5.2 and Claude Sonnet 5.5 each did it for three versions of the prompt and
        three draws of welds.</p>
        ${only(`<h3>On screen: ${escapeHtml(writerName())}</h3>
        <p>${SET_NAMES[sel.set]}, draw ${sel.draw + 1}. On the 252 welds of this draw that it never saw, it is right
        <b>${c.acc.draw}%</b> of the time (${c.acc.mean}% on average over the three draws). Guessing gives 11%; the
        plain formula with no AI, from the same 18 welds, gets ${F.accuracy.no_ai.mean}%.</p>`)}
        <p class="next">Change the set, the draw and who wrote the numbers with the menus above the 3D view.</p>`,
    },
    {
      title: 'One weld as 8 numbers',
      frame: 'fewInput',
      focus: ['fx'],
      shown: 0,
      body: (c) => `
        <p>Each weld is reduced to eight numbers, measured over the 1.0 s weld window with the power smoothed over
        2 ms:</p>
        <table class="compare"><tbody>${F.features.map((f, k) => `<tr><th><code>${f.id}</code></th><td>${escapeHtml(f.meaning)}<small>this weld: ${show(k, c.w.x[k])}</small></td></tr>`).join('')}</tbody></table>
        <p><b>On screen:</b> one of the 18 welds the model was shown, made with a
        ${className(c.w.label, M).toLowerCase()}. The 252 held-out welds are not in Dennis's folder, so only these 18
        can be played here.</p>`,
    },
    {
      title: 'The centroids: a typical weld of each condition',
      frame: 'fewCentroids',
      focus: ['fx', 'fc'],
      shown: 0,
      body: (c) => `
        <p>Each row is a condition and each column a feature. A cell holds the value the model wrote for a typical
        weld of that condition. The bar shows how far that is from the middle of the nine conditions, counted in
        spreads: <span class="sw w">blue</span> above, <span class="sw wn">orange</span> below.</p>
        ${sel.writer === 'avg'
          ? only('<p>These are the plain averages of each condition\'s two welds, computed in this page with no model.</p>')
          : only(`<h3>How far from plain averaging?</h3>
            <p>In <b>${c.near} of 72</b> cells, the centroid ${escapeHtml(writerName())} wrote is within a tenth of
            a spread of the plain average of that condition's two welds. The largest difference is
            ${c.worst.d.toFixed(2)} spreads (<code>${F.features[c.worst.f].id}</code>,
            ${className(c.worst.k, M, true)}). Hover a cell to compare.</p>`)}`,
    },
    {
      title: 'The spread: how much welds of one condition differ',
      frame: 'fewSpread',
      focus: ['fc', 'fs'],
      shown: 0,
      body: () => `
        <p>One number per feature, in the feature's own unit: the typical difference between two welds of the same
        condition. It decides how much a difference in that feature counts. With a small spread the feature is
        strict; with a large one it is forgiving.</p>
        <p>The bar shows the spread as a share of the range the nine centroids cover. A <b>short</b> bar is a feature
        that separates the conditions well.</p>`,
    },
    {
      title: 'Distance to every centroid',
      frame: 'fewDistance',
      focus: ['fx', 'fc', 'fs', 'fd'],
      shown: 1,
      body: () => `
        <p>For every condition and every feature, the weld's value is compared with the centroid, in spreads, and
        squared:</p>
        <p class="formula">((value - centroid) / spread)²</p>
        <p>The second slab holds these 72 terms. A tall bar is a feature on which this weld is far from that
        condition's typical weld.</p>`,
    },
    {
      title: 'The closest centroid wins',
      frame: 'fewAnswer',
      focus: ['fd', 'ft'],
      shown: 2,
      body: (c) => `
        <p>Each row of terms is added up. The condition with the <b>smallest</b> sum is the answer:
        <b>${className(c.out.pred, M)}</b> (${c.out.totals[c.out.pred].toFixed(2)}).
        ${c.out.pred === c.w.label ? 'That is what this weld is: <b class="ok">correct</b>.' : `The weld was made with a <b>${className(c.w.label, M).toLowerCase()}</b>: <b class="bad">wrong</b>.`}</p>
        <p>These numbers put <b>${c.own} of the 18</b> welds shown to the model in the right condition. That is not a
        test, since the model saw them. The test is on the 252 welds it did not see:</p>
        ${accuracyTable()}
        <p>Both models land where the formula with no AI lands, so the 18 welds carry the result. The extra context
        and the operator statements change little.</p>`,
    },
  ];

  function glance(c) {
    const top = Math.max(...c.out.totals);
    const rows = c.out.totals.map((t, k) => {
      const marks = `${k === c.out.pred ? 'pred' : ''}${k === c.w.label ? ' truth' : ''}`;
      return `<li class="${marks.trim()}"><span class="name">${className(k, M, true)}</span>
        <span class="bar"><i style="width:${(100 * t / top).toFixed(1)}%"></i></span><span class="p">${t.toFixed(1)}</span></li>`;
    }).join('');
    return `
      <h2>${escapeHtml(writerName())}</h2>
      <dl class="facts">
        <div><dt>Numbers written</dt><dd>80<small>72 centroids, 8 spreads</small></dd></div>
        <div><dt>Held-out accuracy</dt><dd>${c.acc.draw}%<small>this draw; mean ${c.acc.mean}%</small></dd></div>
        <div><dt>Its own 18 welds</dt><dd>${c.own}<small>/18 right (not a test)</small></dd></div>
      </dl>
      <h3>Distance to each condition, this weld</h3>
      <ol class="bars">${rows}</ol>
      <p class="legend"><span class="dot"></span>true condition of this weld. The smallest distance wins.</p>`;
  }

  function doc(root) {
    const reply = sel.writer === 'avg' ? null : model();
    root.innerHTML = `
      <div class="notice"><b>Exactly what went in and what came out.</b> ${SET_NAMES[sel.set]}, draw ${sel.draw + 1},
        ${escapeHtml(writerName())}. Taken from <code>${F.folder}/</code> on branch <code>${F.branch}</code>
        (${F.commit}, ${F.date}). The 18 welds in the prompt are real welds: private data.</div>
      <details class="part" open>
        <summary><i>1</i>The prompt<small>${escapeHtml(F.sets[sel.set].what)}</small></summary>
        <div class="md prompt">${markdown(F.prompts[setId()][sel.draw])}</div>
      </details>
      <details class="part" open>
        <summary><i>2</i>The reply<small>${reply ? 'verbatim' : 'no model'}</small></summary>
        ${reply
          ? `<pre class="code wide">${escapeHtml(reply.raw.trim())}</pre>`
          : `<p class="meta">Nothing was asked of a model here. The page computes each condition's centroid as the average of its two welds, and each feature's spread as the pooled standard deviation inside the conditions. Dennis's README reports ${F.accuracy.no_ai.mean}% for this kind of rule; his exact spread formula is not in the folder, so the spreads shown may differ from his.</p>`}
      </details>
      <details class="part">
        <summary><i>3</i>Accuracy on the welds it did not see<small>3 sets, 2 models, 3 draws</small></summary>
        ${accuracyTable()}
      </details>
      <details class="part">
        <summary><i>4</i>How the replies were made<small>from his README</small></summary>
        <ul class="md">
          <li>Claude Fable 5.2 (run on 2026-11-12) and Claude Sonnet 5.5 (run on 2026-11-13), as Claude Code subagents, one per prompt.</li>
          <li>Each was told to read two files and nothing else, a standard rules file for subagents and its prompt, and to reply with the JSON block only. The transcripts were checked for exactly those two reads.</li>
          <li>No training, no gradients, and no access to the held-out welds.</li>
          <li>A draw is one random choice of 2 welds per condition. The three draws are the same in all three sets, so the sets differ only in the text around the welds.</li>
          <li>The operator statements of set 2 were written from the labelled data set, as directions only. No real operator wrote them.</li>
        </ul>
      </details>`;
  }

  function options() {
    const first = {};
    return draw().welds.map((w, k) => {
      const nth = first[w.label] === undefined ? ((first[w.label] = k), 'first') : 'second';
      return [k, `${className(w.label, M, true)}: ${nth} of its 2 welds`];
    });
  }

  return { STEPS, SET_NAMES, render, context, buildLabels, linked, describe, glance, doc, options, writerName, accuracy };
}
