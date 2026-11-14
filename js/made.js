// "How it was written": the backtrace for the LLM-written tab.
//
// For one reply it shows, in the order things happened: the prompt, Astra's plan, the edits it wrote (group by group,
// split at its own comments), the explanation it gave when asked two days later, and the rule the research repo
// decoded from the finished weights. Astra's reasoning itself was not returned by the API; OpenAI's summaries of it
// were saved, and are shown as what they are: a summarizer's paraphrase.

import { LAYERS, OFFSETS, describeFeature, pct } from './net.js';
import { escapeHtml, inline, markdown, plainText } from './md.js';

const layerOf = (tensor) => +tensor[1];
const n = (v) => v.toLocaleString('en-US');

/** Passages of the prompt that the later explanation points back to: `test` looks for the reference. */
const PASSAGES = [
  { id: 'friction', label: 'contamination reduces friction', test: /reduc\w* friction|cutting fluid/i, find: 'Surface contamination reduces the friction' },
  { id: 'both', label: 'power and LVDT are nearly enough', test: /Power(?: and |\/)LVDT/i, find: 'The power and LVDT signals together come close to all four sensors' },
  { id: 'worn', label: 'what the authors found', test: /worn[- ]tool (?:behavior|uncertainty|inconsisten)|poorer surface|inconsistent worn|published fin/i, find: 'most of its errors came from confusing surface conditions' },
  { id: 'slow', label: 'power varies below 100 Hz', test: /below 100 Hz/i, find: 'almost all of its variation below 100 Hz' },
  { id: 'ae', label: 'AE peaks at 40 to 80 kHz', test: /40\s*(?:–|-|to|and)\s*80 kHz|AE (?:peak|spectrum)|dominant AE/i, find: 'its spectrum peaks between about 40 and 80 kHz' },
  { id: 'tone', label: 'the weld tone is in band 8', test: /weld[- ]tone|20[- ]?kHz|tone[- ]band|tone location/i, find: 'The 20 kHz weld vibration falls in band 8' },
  { id: 'index', label: 'the feature index formula', test: /132\s*[(×x*]|index formula|feature index/i, find: 'Feature index = 132 x sensor + 12 x band + statistic' },
  { id: 'rms', label: 'typical band RMS', test: /RMS medians|typical band RMS|reported RMS/i, find: 'Typical band RMS across all 270 welds' },
  { id: 'standard', label: 'inputs are standardized', test: /standardi[sz]ed (?:inputs|features|RMS)|centered inputs/i, find: 'Before the network, every feature is standardized' },
  { id: 'zeros', label: 'every weight starts at 0', test: /zero initiali[sz]ation/i, find: 'EVERY WEIGHT STARTS AT 0' },
  { id: 'entropy', label: 'the entropy statistic', test: /entropy/i, find: '0 entropy: Shannon entropy' },
];

const ranges = (values) => {
  const v = [...new Set(values)].sort((a, b) => a - b);
  const out = [];
  for (let k = 0; k < v.length; k++) {
    let end = k;
    while (end + 1 < v.length && v[end + 1] === v[end] + 1) end++;
    out.push(end > k + 1 ? `${v[k]} to ${v[end]}` : end > k ? `${v[k]}, ${v[end]}` : `${v[k]}`);
    k = end;
  }
  return out.join(', ');
};

export const groupTitle = (g) => g.section || `${[...new Set(g.statements.map((s) => s.tensor))].join(', ')} values`;

/** {tensor: [cell indices]} that a group's statements were the last to write. */
export function groupCells(group) {
  const out = {};
  for (const s of group.statements) if (s.read) (out[s.tensor] ??= []).push(...s.cells);
  return out;
}

/** In plain words: what a group's cells connect. */
function targets(group, manifest) {
  const lines = [];
  for (const [tensor, cells] of Object.entries(groupCells(group))) {
    if (!cells.length) continue;
    const k = layerOf(tensor);
    const out = k < 4 ? `layer ${k} unit${cells.length > 1 ? 's' : ''}` : 'class';
    if (tensor[0] === 'b') {
      lines.push(`<b>${tensor}</b>: the bias of ${out} ${ranges(cells)}.`);
      continue;
    }
    const no = LAYERS[k];
    const from = [...new Set(cells.map((c) => Math.floor(c / no)))].sort((a, b) => a - b);
    const to = cells.map((c) => c % no);
    let source;
    if (k === 1) {
      const names = from.map((i) => {
        const f = describeFeature(i, manifest);
        return `${f.sensor} ${f.band} ${f.stat}`;
      });
      source = `${from.length} input feature${from.length > 1 ? 's' : ''} (${names.slice(0, 6).join('; ')}${names.length > 6 ? `; and ${names.length - 6} more` : ''})`;
    } else {
      source = `layer ${k - 1} unit${from.length > 1 ? 's' : ''} ${ranges(from)}`;
    }
    lines.push(`<b>${tensor}</b>: ${n(cells.length)} weight${cells.length > 1 ? 's' : ''}, from ${source} to ${out} ${ranges(to)}.`);
  }
  return lines;
}

/** The later explanation, cut at its headings; each piece knows which layers its heading names. */
function chunks(text) {
  const out = [{ title: '', level: 0, layers: [], lines: [] }];
  let parent = [];
  for (const line of text.split('\n')) {
    const m = line.match(/^(#{2,4})\s+(.*)$/);
    if (!m) {
      out.at(-1).lines.push(line);
      continue;
    }
    const title = m[2];
    let layers = [
      /first[- ]layer|input branch/i.test(title) && 1,
      /second[- ]layer/i.test(title) && 2,
      /third[- ]layer/i.test(title) && 3,
      /output/i.test(title) && 4,
    ].filter(Boolean);
    if (m[1].length === 2) parent = layers;
    else if (!layers.length) layers = parent;
    out.push({ title, level: m[1].length, layers, lines: [] });
  }
  return out.filter((c) => c.title || c.lines.join('').trim());
}

/** Does a table row of the explanation talk about these units? (Only used to draw the eye; nothing is hidden.) */
function rowMentions(text, layer, units, inputs) {
  const hit = (a, b) => units.some((u) => u >= a && u <= b);
  for (const m of text.matchAll(/h([123])\[(\d+)(?::(\d+))?\]/g)) {
    if (+m[1] === layer && hit(+m[2], m[3] ? +m[3] - 1 : +m[2])) return true;
  }
  for (const m of text.matchAll(/units?\s+(\d+)\s*(?:[–-]|to)\s*(\d+)/gi)) if (hit(+m[1], +m[2])) return true;
  for (const m of text.matchAll(/\bunit\s+(\d+)\b/gi)) if (hit(+m[1], +m[1])) return true;
  if (layer >= 2) {
    for (const m of text.matchAll(/(?:Prototype|Class|Output)\s+(\d)\b/g)) if (layer === 4 && hit(+m[1], +m[1])) return true;
  }
  if (inputs.length) {
    const quoted = [...text.matchAll(/(?<![\d.])(\d{2,3})(?![\d.])/g)].map((m) => +m[1]);
    if (quoted.filter((q) => inputs.includes(q)).length >= Math.min(2, inputs.length)) return true;
  }
  return false;
}

export function createMade({ root, view, data, state, actions }) {
  const P = data.provenance;
  const cache = new Map(); // reply -> static html pieces
  const reply = () => P.replies[state.reply];
  const group = () => reply().groups[state.made.group];

  /** cell -> [group, statement] for the reply on screen. */
  function owners(k = state.reply) {
    const c = (cache.get(k) ?? cache.set(k, {}).get(k));
    if (!c.owners) {
      c.owners = new Map();
      P.replies[k].groups.forEach((g, gi) => g.statements.forEach((s, si) => {
        for (const cell of s.cells) c.owners.set(OFFSETS[s.tensor] + cell, [gi, si]);
      }));
    }
    return c.owners;
  }

  /** Flat indices (into the 99,959 numbers) written by groups 0..g. */
  function cellsUpTo(g) {
    const out = [];
    for (const [flat, [gi]] of owners()) if (gi <= g) out.push(flat);
    return out;
  }

  function staticHtml() {
    const c = (cache.get(state.reply) ?? cache.set(state.reply, {}).get(state.reply));
    if (c.html) return c.html;
    const r = reply();
    const m = data.manifest;
    const meta = m.written.replies[state.reply];
    const promptHtml = (cache.prompt ??= plainText(P.prompt.text, PASSAGES));
    const e = r.explanation;
    const d = r.decoded;
    const why = r.reasoning ?? { parts: [], items: 0, summarized: 0 };
    // one part of the summary: a bold heading the summarizer gave it, then a paragraph
    const thought = (text) => {
      const m = text.match(/^\*\*(.+?)\*\*\s*([\s\S]*)$/);
      return `<li>${m ? `<b>${inline(m[1])}</b>${markdown(m[2])}` : markdown(text)}</li>`;
    };
    const tools = ['damaged', 'new', 'worn'];
    const cell = (v) => (v == null ? '' : `${v > 0 ? '+' : ''}${v.toFixed(2)}`);
    const rule = d
      ? `<p>The research repository decoded each written network mechanically, from the weights alone, back into the
          short rule it computes (<code>decode.py</code>). That rule gives the same answer as the network on
          <b>${n(d.agreement.same_condition)} of ${n(d.agreement.welds_x_splits)}</b> weld and split combinations, so
          this is what the ${n(data.written[state.reply].nonzero)} numbers do:</p>
        <pre class="code wide">${escapeHtml(d.program)}</pre>
        ${d.tool_means_actual ? `<p>Where Astra put each tool on its own axes, against where the real welds fall on
          the same axes (averages; this comparison uses the labels, which Astra never saw):</p>
        <div class="table"><table class="num"><thead><tr><th>Axis</th>${d.tool_means_astra ? tools.map((t) => `<th>Astra: ${t}</th>`).join('') : ''}${tools.map((t) => `<th>Welds: ${t}</th>`).join('')}</tr></thead>
          <tbody>${d.axes.map((a, k) => `<tr><th>${escapeHtml(a)}</th>${d.tool_means_astra ? d.tool_means_astra.map((row) => `<td>${cell(row[k])}</td>`).join('') : ''}${d.tool_means_actual.map((row) => `<td>${cell(row[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : ''}
        <p><b>Accuracy as written: ${pct(d.accuracy_as_written)}.</b> Keeping Astra's axes and weights but moving each
          condition's centre to where the training welds are: ${pct(d.accuracy_with_label_centres)}. So the features it
          picked carry signal; the positions it guessed for the nine conditions are what fail.</p>`
      : '<p>No decoded rule is stored for this reply.</p>';

    c.html = `
      <div class="notice">
        <b>What can and cannot be traced.</b> Astra spent ${n(r.reasoning_tokens)} reasoning tokens on this reply
        (${n(r.output_tokens)} output tokens, ${r.seconds} s). The reasoning itself was not returned by the API.
        ${why.parts.length
          ? `What was saved is <b>OpenAI's summary of it</b>, ${why.parts.length} parts written by a separate summarizer: a paraphrase, shown in part 2.`
          : 'No summary of it was saved for this reply.'}
        Parts 1, 3 and 4 are the prompt and Astra's visible reply. Part 5 is Astra explaining its own numbers two days
        later: <b>an after-the-fact account, not a trace</b>.
      </div>
      <details class="part" data-part="prompt">
        <summary><i>1</i>The prompt<small>${n(P.prompt.text.length)} characters, the same for all five replies</small></summary>
        <p class="meta">System message: <code>${escapeHtml(P.prompt.system)}</code> No tools, no data, one reply.
          The typical values in it were computed over all 270 welds with the labels ignored.</p>
        <div class="prompt">${promptHtml}</div>
      </details>
      <details class="part" data-part="reasoning">
        <summary><i>2</i>Astra's reasoning, as summarized by OpenAI<small>${why.parts.length} parts, in their original order</small></summary>
        <p class="warn">A summary written by OpenAI's summarizer from the hidden reasoning; a paraphrase, not the
          reasoning itself. The full reasoning (${n(r.reasoning_tokens)} tokens) was not returned.</p>
        ${why.parts.length
          ? `<p class="meta">The saved response holds ${why.items} reasoning steps; ${why.summarized} of them carry a
              summary, so there are gaps between the parts. They come before the reply was written: this is where
              Astra weighed what to assume.</p>
            <ol class="thoughts">${why.parts.map(thought).join('')}</ol>`
          : '<p class="meta">The saved response carries no summary for this reply.</p>'}
      </details>
      <details class="part" data-part="plan">
        <summary><i>3</i>Astra's plan<small>the text of its reply, before the numbers</small></summary>
        <div class="md">${markdown(meta.design)}</div>
      </details>
      <details class="part" data-part="edits" open>
        <summary><i>4</i>The edits, group by group<small>${r.groups.length} groups, ${meta.statements} statements, ${n(data.written[state.reply].nonzero)} non-zero numbers</small></summary>
        <div id="made-group"></div>
      </details>
      <details class="part" data-part="asked">
        <summary><i>5</i>Asked afterwards: where did each number come from?<small>${e ? `${P.asked.date}, same conversation${e.complete ? '' : ', answer cut off'}` : 'not asked'}</small></summary>
        ${e ? `<p class="meta">The question, sent as a follow-up in the same conversation, with no weld, score or label
          shown:</p><blockquote>${escapeHtml(P.asked.question).replace(/\n\n/g, '<br><br>')}</blockquote>
          ${e.complete ? '' : '<p class="warn">This answer hit the 18,000-token output limit and stops in the middle of a table. The later layers are not covered.</p>'}
          <div class="md">${markdown(e.text)}</div>` : ''}
      </details>
      <details class="part" data-part="rule">
        <summary><i>6</i>The rule these weights compute<small>decoded from the weights, and compared with the welds</small></summary>
        ${rule}
      </details>`;
    return c.html;
  }

  function renderGroup() {
    const host = root.querySelector('#made-group');
    if (!host) return;
    const r = reply();
    const g = group();
    const cells = groupCells(g);
    const total = Object.values(cells).reduce((s, c) => s + c.length, 0);
    const layers = [...new Set(Object.keys(cells).map(layerOf))];
    const later = r.explanation ? chunks(r.explanation.text).filter((c) => c.layers.some((l) => layers.includes(l))) : [];
    const laterText = later.map((c) => c.lines.join('\n')).join('\n');
    const passages = PASSAGES.filter((p) => p.test.test(laterText));
    host.innerHTML = `
      <div class="stepper">
        <button class="nav" data-go="-1" ${state.made.group === 0 ? 'disabled' : ''} aria-label="Previous group">&lsaquo;</button>
        <span>Group ${state.made.group + 1} of ${r.groups.length}</span>
        <button class="nav" data-go="1" ${state.made.group === r.groups.length - 1 ? 'disabled' : ''} aria-label="Next group">&rsaquo;</button>
        <label class="check"><input type="checkbox" id="build-up" ${state.made.buildUp ? 'checked' : ''}>Build up</label>
      </div>
      ${g.notes.length ? `<p class="notes">${g.notes.map(escapeHtml).join('<br>')}</p>` : ''}
      <h4>${escapeHtml(groupTitle(g))}</h4>
      ${g.sub ? `<p class="sub">${escapeHtml(g.sub)}</p>` : ''}
      <p class="meta">Astra's own comment${g.notes.length || g.sub ? 's' : ''}, written in the edits block. The
        statements under it:</p>
      <pre class="code wide">${g.statements.map((s) => `${escapeHtml(s.text)}${s.comment ? `   <span class="c"># ${escapeHtml(s.comment)}</span>` : ''}`).join('\n')}</pre>
      <ul class="targets">${targets(g, data.manifest).map((t) => `<li>${t}</li>`).join('')}</ul>
      <p class="check-line"><span class="ok">&#10003;</span> Replayed on an all-zero network with the research
        repository's reader, these statements give exactly the ${n(total)} cell${total === 1 ? '' : 's'} lit in the 3D
        view.</p>
      <div class="later">
        <h5>What Astra said about ${layers.length > 1 ? 'these layers' : 'this layer'} when asked afterwards</h5>
        ${later.length
          ? `<p class="meta">After the fact, not a trace. Rows that name these units are marked.</p>
             ${passages.length ? `<p class="passages">It points back to the prompt: ${passages.map((p) => `<button class="chip-link" data-passage="${p.id}">${p.label}</button>`).join(' ')}</p>` : ''}
             <div class="md">${later.map((c) => `${c.title ? `<h6>${escapeHtml(c.title)}</h6>` : ''}${markdown(c.lines.join('\n'))}`).join('')}</div>`
          : `<p class="meta">${r.explanation ? 'Its answer does not reach this layer' + (r.explanation.complete ? '.' : ': it was cut off by the output limit.') : 'No follow-up was asked for this reply.'}</p>`}
      </div>`;

    const units = Object.entries(cells).flatMap(([t, c]) => (t[0] === 'b' ? c : c.map((x) => x % LAYERS[layerOf(t)])));
    const inputs = cells.W1 ? [...new Set(cells.W1.map((x) => Math.floor(x / 150)))] : [];
    for (const tr of host.querySelectorAll('.later tbody tr')) {
      if (layers.some((l) => rowMentions(tr.cells[0]?.textContent ?? '', l, units, inputs))) tr.classList.add('hit');
    }
    actions.showCells(cells, state.made.fly !== false);
    state.made.fly = true;
  }

  function render() {
    state.made.group = Math.min(state.made.group, reply().groups.length - 1);
    if (root.dataset.reply !== String(state.reply)) {
      const open = new Set([...root.querySelectorAll('details.part[open]')].map((d) => d.dataset.part));
      root.innerHTML = staticHtml();
      if (root.dataset.reply !== undefined) for (const d of root.querySelectorAll('details.part')) d.open = open.has(d.dataset.part);
      root.dataset.reply = String(state.reply);
    }
    renderGroup();
  }

  root.addEventListener('click', (e) => {
    const go = e.target.closest('[data-go]');
    if (go) {
      state.made.group += +go.dataset.go;
      actions.changed();
      return;
    }
    const passage = e.target.closest('[data-passage]');
    if (passage) {
      const part = root.querySelector('[data-part="prompt"]');
      part.open = true;
      const mark = root.querySelector(`#${passage.dataset.passage}`);
      if (mark) {
        mark.scrollIntoView({ block: 'center', behavior: 'smooth' });
        mark.classList.remove('flash');
        requestAnimationFrame(() => mark.classList.add('flash'));
      }
    }
  });
  root.addEventListener('change', (e) => {
    if (e.target.id === 'build-up') {
      state.made.buildUp = e.target.checked;
      state.made.fly = false;
      actions.changed();
    }
  });

  return {
    render,
    cellsUpTo,
    /** The group and statement that wrote one cell, or null for a cell that was never written. */
    owner: (tensor, cell) => owners().get(OFFSETS[tensor] + cell) ?? null,
    statement: (gi, si) => reply().groups[gi].statements[si],
    groupOf: (gi) => reply().groups[gi],
  };
}
