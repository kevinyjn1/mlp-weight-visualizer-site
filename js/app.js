// Glue: tabs, the panel (walkthrough or documents), the controls over the 3D view, tooltips.

import {
  LAYERS, loadData, forward, contributions, layerInput, scaleOf, normalized, describeFeature, className, fmt, pct,
  noiseInput, subsetNet, LockedError, keyIn,
} from './net.js';
import { NetworkView } from './scene.js';
import { STEPS } from './guide.js';
import { escapeHtml, markdown } from './md.js';
import { createMade, groupTitle } from './made.js';
import { createFew } from './few.js';

const $ = (sel) => document.querySelector(sel);
const TABS = ['trained', 'written', 'few'];
// the link is "#<tab>", and in the locked public copy "#<tab>&k=<key>"
const fromHash = () => {
  const tab = location.hash.slice(1).split('&')[0];
  return TABS.includes(tab) ? tab : 'trained';
};
const linkKey = () => {
  const key = keyIn(location.hash);
  return key ? `&k=${key}` : '';
};
const view = new NetworkView($('#stage'));
const state = {
  tab: fromHash(),
  reply: 0,
  input: 0,
  step: 0,
  products: false,
  noiseSeed: 11,
  shown: 8,
  fewShown: 2,
  act: null,
  mode: { trained: 'walk', written: 'walk', few: 'walk' },
  made: { group: 0, buildUp: false, fly: true },
  few: { set: 0, draw: 0, writer: 'fable', weld: 0, step: 0 },
};
const DOC_STEP = { focus: null, contrib: [], shown: 8 }; // what the 3D view shows while a document is open
let data = null;
let inputs = [];
let predLabel = null;
let made = null;
let few = null;

// room the framed blocks must leave free: the controls over the view, and a row of name tags above the blocks
view.getInsets = () => {
  const roomy = $('#stage').clientHeight > 420;
  return {
    top: $('.hud-tl').offsetHeight + (roomy ? 46 : 32),
    bottom: $('.hud-bl').offsetHeight + (roomy ? 44 : 26),
    left: 20,
    right: 20,
  };
};

// ------------------------------------------------------------------ what is on screen

const isFew = () => state.tab === 'few';
const mode = () => state.mode[state.tab];
const steps = () => (isFew() ? few.STEPS : STEPS);
const stepIndex = () => (isFew() ? state.few.step : state.step);
const currentInput = () => inputs[state.input];
const lower = (c) => className(c, data.manifest).toLowerCase();
const building = () => state.tab === 'written' && mode() === 'doc' && state.made.buildUp;

const partial = new Map();
function currentNet() {
  if (state.tab === 'trained') return data.trained;
  const net = data.written[state.reply];
  if (!building()) return net;
  const key = `${state.reply}:${state.made.group}`;
  if (!partial.has(key)) partial.set(key, subsetNet(net, made.cellsUpTo(state.made.group)));
  return partial.get(key);
}

const normCache = new WeakMap();
function normWeights(net) {
  let c = normCache.get(net);
  if (!c) {
    c = { W: [], b: [], scaleW: [], scaleB: [] };
    for (let k = 0; k < 4; k++) {
      c.scaleW.push(scaleOf(net.W[k]));
      c.W.push(normalized(net.W[k], c.scaleW[k]));
      c.scaleB.push(scaleOf(net.b[k], 1));
      c.b.push(normalized(net.b[k], c.scaleB[k]));
    }
    normCache.set(net, c);
  }
  return c;
}

function render() {
  if (isFew()) {
    const c = few.render();
    renderPanel(c);
    renderDecal(c);
    return;
  }
  const net = currentNet();
  const x = currentInput().x;
  const act = forward(net, x);
  const step = mode() === 'doc' ? DOC_STEP : STEPS[state.step];
  const shown = state.products ? 8 : step.shown;
  // while a reply is being replayed, bars keep the scale of the finished network, so they do not jump at each step
  const nw = normWeights(net);
  const full = building() ? normWeights(data.written[state.reply]) : nw;
  state.act = act;
  state.shown = shown;

  view.setBlock('x', normalized(x, scaleOf(x, 0.98)), 'data', { duration: 700 });
  for (let k = 1; k <= 4; k++) {
    const sweep = { duration: 1200, stagger: 0.8 };
    if (state.products || step.contrib.includes(k)) {
      const c = contributions(net, act, k);
      view.setBlock(`W${k}`, normalized(c, scaleOf(c, 0.999)), 'data', sweep);
    } else {
      view.setBlock(`W${k}`, building() ? normalized(net.W[k - 1], full.scaleW[k - 1]) : nw.W[k - 1], 'weights', sweep);
    }
    view.setBlock(`b${k}`, building() ? normalized(net.b[k - 1], full.scaleB[k - 1]) : nw.b[k - 1], 'weights', { duration: 700 });
    const z = act.z[k - 1];
    const scale = scaleOf(z, 1);
    const blank = new Float32Array(z.length);
    view.setBlock(`z${k}`, shown >= 2 * k - 1 ? normalized(z, scale) : blank, 'data', { duration: 800, delay: 450 });
    if (k < 4) {
      view.setBlock(`h${k}`, shown >= 2 * k ? normalized(act.h[k - 1], scale) : blank, 'data', { duration: 800, delay: 1000 });
    }
  }
  view.setBlock('p', shown >= 8 ? act.probs : new Float32Array(9), 'prob', { duration: 800, delay: 700 });
  view.setFocus(state.products ? null : step.focus);

  const c = context();
  renderPanel(c);
  renderDecal(c);
  const p = view.blocks.p;
  view.updateLabel(predLabel, {
    html: `<b>${c.predName}</b> · ${c.predP}${c.verdict}`,
    pos: [p.center(act.pred)[0], 0, p.rect().z1],
  });
}

/** The live numbers the walkthrough text quotes. */
function context() {
  const m = data.manifest;
  const net = currentNet();
  const act = state.act;
  const input = currentInput();
  const reply = m.written.replies[state.reply];
  const count = (a, test) => a.reduce((s, v) => s + (test(v) ? 1 : 0), 0);
  let wired = 0;
  for (let j = 0; j < 150; j++) {
    for (let i = 0; i < 528; i++) {
      if (net.W[0][i * 150 + j] !== 0) {
        wired++;
        break;
      }
    }
  }
  const tie = act.logits.every((v) => v === act.logits[0]);
  const right = input.kind === 'real' ? act.pred === input.label : null;
  let inputText;
  let truthText;
  if (input.kind === 'real') {
    inputText = `held-out test weld ${input.run}, made with a ${lower(input.label)}.`;
    truthText = right
      ? `The weld really was made with a ${lower(input.label)}: <b class="ok">correct</b>.`
      : `The weld was actually made with a <b>${lower(input.label)}</b>: <b class="bad">wrong</b>.`;
  } else if (input.kind === 'synthetic') {
    inputText = `a synthetic input, built so that the trained network reads it as "${lower(input.label)}". It is not
      a real weld.`;
    truthText = `This input is synthetic: it was built so that the trained network reads it as
      "${lower(input.label)}", so there is no true answer to compare with.`;
  } else {
    inputText = 'random noise, not a weld.';
    truthText = 'This input is random noise, so there is no right answer.';
  }
  return {
    tab: state.tab,
    nonzero: net.nonzero,
    nonzeroW: net.nonzeroW,
    nonzeroB: net.nonzeroB,
    wired,
    maxW1: `±${net.W[0].reduce((s, v) => Math.max(s, Math.abs(v)), 0).toFixed(2)}`,
    active: act.h.map((h) => count(h, (v) => v > 0)),
    reached: act.z.map((z) => count(z, (v) => v !== 0)),
    trainedNonzero: data.trained.nonzero,
    writtenNonzero: data.written[state.reply].nonzero,
    trainedAcc: pct(m.trained.accuracy.mean),
    replyAcc: pct(reply.accuracy.mean),
    writtenMeanAcc: pct(m.written.mean_accuracy),
    predName: className(act.pred, m),
    predP: pct(act.probs[act.pred]),
    verdict: right === null ? '' : right ? ' <span class="ok">✓</span>' : ' <span class="bad">✗</span>',
    tie,
    inputText,
    truthText,
  };
}

// ------------------------------------------------------------------ panel

function mlpGlance(c) {
  const m = data.manifest;
  const act = state.act;
  const input = currentInput();
  const trained = state.tab === 'trained';
  const reply = m.written.replies[state.reply];
  const rows = Array.from(act.probs, (p, cls) => {
    const marks = `${cls === act.pred ? ' pred' : ''}${input.kind === 'real' && cls === input.label ? ' truth' : ''}`;
    return `<li class="${marks.trim()}"><span class="name">${className(cls, m, true)}</span>
      <span class="bar"><i style="width:${(100 * p).toFixed(1)}%"></i></span><span class="p">${pct(p, 0)}</span></li>`;
  }).join('');
  return `
    <h2>${trained ? 'Trained network' : `Astra's reply ${state.reply + 1} of ${m.written.replies.length}`}</h2>
    <dl class="facts">
      <div><dt>Non-zero numbers</dt><dd>${c.nonzero.toLocaleString('en-US')}<small>${pct(c.nonzero / 99959, c.nonzero < 99959 ? 1 : 0)} of 99,959</small></dd></div>
      <div><dt>Test accuracy</dt><dd>${trained ? c.trainedAcc : c.replyAcc}<small>guessing: 11.1%</small></dd></div>
      <div><dt>Units on, this input</dt><dd>${c.active[0]}<small>/150</small> ${c.active[1]}<small>/100</small> ${c.active[2]}<small>/50</small></dd></div>
    </dl>
    <h3>Its answer for this input</h3>
    <ol class="bars">${rows}</ol>
    ${input.kind === 'real' ? '<p class="legend"><span class="dot"></span>true condition of this weld</p>' : ''}
    ${trained ? '' : `${made ? `<p class="more"><button class="link" data-action="made">How did Astra arrive at these numbers?</button>
      The prompt, its edits and its own explanation. Or click any bright cell.</p>` : ''}
      <details class="design"><summary>Astra's own description of this network</summary>${markdown(reply.design)}</details>`}`;
}

function renderPanel(c) {
  const doc = mode() === 'doc';
  const modes = { trained: null, written: made && 'How it was written', few: 'Prompt and reply' }[state.tab];
  $('#panel').dataset.mode = mode();
  $('#modes').hidden = !modes;
  if (modes) {
    $('#mode-doc').textContent = modes;
    for (const el of $('#modes').children) el.setAttribute('aria-selected', String(el.dataset.mode === mode()));
  }
  $('#made-root').hidden = !(doc && state.tab === 'written');
  $('#few-root').hidden = !(doc && isFew());
  if (doc) {
    if (isFew()) few.doc($('#few-root'));
    else made.render();
    return;
  }
  const list = steps();
  const k = stepIndex();
  $('#step-count').textContent = `Step ${k + 1} of ${list.length}`;
  $('#step-title').textContent = list[k].title;
  $('#step-body').innerHTML = list[k].body(c);
  const dots = $('#dots');
  if (dots.children.length !== list.length) {
    dots.replaceChildren(...list.map((step, n) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.addEventListener('click', () => goTo(n));
      li.append(b);
      return li;
    }));
  }
  for (const [n, dot] of [...dots.children].entries()) {
    dot.firstChild.title = list[n].title;
    dot.firstChild.setAttribute('aria-label', `Step ${n + 1}: ${list[n].title}`);
    dot.firstChild.setAttribute('aria-current', n === k ? 'step' : 'false');
    dot.firstChild.classList.toggle('done', n < k);
  }
  $('#back').disabled = $('#prev').disabled = k === 0;
  $('#next').disabled = k === list.length - 1;
  $('#continue').textContent = k === list.length - 1 ? 'Start over' : 'Continue';
  $('#glance').innerHTML = isFew() ? few.glance(c) : mlpGlance(c);
}

function renderDecal(c) {
  const m = data.manifest;
  if (isFew()) {
    view.setDecal({
      title: '18 welds → 80 numbers',
      lines: [
        { text: "Dennis's few-weld classifier · 9 centroids x 8 features, 8 spreads" },
        { text: `${few.writerName()} · set ${state.few.set + 1} · draw ${state.few.draw + 1}`, bold: true, color: '#7a5a00' },
        { text: `Right on ${c.acc.draw}% of the 252 held-out welds · guessing 11%` },
      ],
    });
    return;
  }
  const trained = state.tab === 'trained';
  view.setDecal({
    title: '528 → 150 → 100 → 50 → 9',
    lines: [
      { text: "Prof. Shao's group's weld classifier · 99,959 weights" },
      trained
        ? { text: 'Trained on 162 labelled welds', bold: true, color: '#2553c4' }
        : { text: 'Weights written by GPT-6 Astra from a text prompt', bold: true, color: '#b84e14' },
      {
        text: trained
          ? `Test accuracy ${c.trainedAcc} · guessing 11.1%`
          : `Reply ${state.reply + 1} of ${m.written.replies.length} · test accuracy ${c.replyAcc} · guessing 11.1%`,
      },
    ],
  });
}

function fly(frame, duration) {
  setTopView(false, false);
  view.flyTo(frame, duration);
}

function goTo(k) {
  const list = steps();
  const n = Math.min(Math.max(k, 0), list.length - 1);
  if (isFew()) state.few.step = n;
  else state.step = n;
  showTip(null);
  render();
  fly(list[n].frame);
  $('#scroll').scrollTop = 0;
}

function setMode(next) {
  if (next === mode()) return;
  state.mode[state.tab] = next;
  showTip(null);
  if (next === 'walk') {
    view.setMask(null);
    render();
    fly(steps()[stepIndex()].frame, 900);
  } else {
    state.made.fly = true;
    render();
    if (isFew()) fly('fewAll', 900);
  }
}

function setTab(tab) {
  if (tab === state.tab) return;
  const family = isFew();
  state.tab = tab;
  showTip(null);
  view.setMask(null);
  history.replaceState(null, '', `#${tab}${linkKey()}`);
  syncTabs();
  if (family !== isFew()) {
    view.setLayout(isFew() ? 'few' : 'mlp');
    fillInputs();
    fly(mode() === 'doc' ? (isFew() ? 'fewAll' : 'all') : steps()[stepIndex()].frame, 900);
  }
  state.made.fly = true;
  render();
}

function syncTabs() {
  for (const el of document.querySelectorAll('.tab')) el.setAttribute('aria-selected', String(el.dataset.tab === state.tab));
  $('#reply-chip').hidden = state.tab !== 'written';
  for (const el of document.querySelectorAll('.few-only')) el.hidden = !isFew();
  $('#products').hidden = isFew();
  document.body.dataset.tab = state.tab;
}

function setTopView(on, animate = true) {
  $('#top-view').setAttribute('aria-pressed', String(on));
  if (animate) view.setTopView(on);
}

// ------------------------------------------------------------------ labels in the 3D view

function buildLabels() {
  const m = data.manifest;
  const B = view.blocks;
  const go = (frame) => () => fly(frame);
  const mid = (block, a, b) => (block.center(a)[block.swap ? 1 : 0] + block.center(b)[block.swap ? 1 : 0]) / 2;

  const rx = B.x.rect();
  const inputRow = (c) => (c.bands ? 2 : c.sensors ? 1 : 0);
  view.addLabel({ html: 'Input x · 528 features', pos: [rx.x0, 0, rx.z0], align: 'l', row: inputRow, when: (c) => !c.far, onClick: go('input') });
  view.addLabel({ html: 'Input', pos: [rx.x0, 0, rx.z0], align: 'l', row: inputRow, when: (c) => c.far, onClick: go('input') });
  m.sensors.forEach((name, s) => {
    view.addLabel({ html: name, cls: 'plain', pos: [mid(B.x, 132 * s, 132 * s + 131), 0, rx.z0], row: (c) => (c.bands ? 1 : 0), when: (c) => c.sensors });
    m.bands.forEach((band, b) => {
      const i = 132 * s + 12 * b;
      view.addLabel({ html: band.name, cls: 'tiny', pos: [mid(B.x, i, i + 11), 0, rx.z0], when: (c) => c.bands });
    });
  });

  const corner = { W1: ['l', 'below', 'W1'], W2: ['r', 'above', 'L2'], W3: ['r', 'middle', 'L3'], W4: ['r', 'above', 'L4'] };
  for (const [id, [align, v, frame]] of Object.entries(corner)) {
    const r = B[id].rect();
    const k = +id[1];
    const pos = v === 'middle' ? [r.x0, 0, (r.z0 + r.z1) / 2] : [align === 'l' ? r.x0 : r.x1, 0, v === 'below' ? r.z1 : r.z0];
    // if its usual place is taken (by the ground title, say), a name tag sits on a corner of its own slab
    const alt = [{ pos: [r.x0, 0, r.z1], align: 'l', v: 'above' }, { pos: [r.x1, 0, r.z1], align: 'r', v: 'above' }];
    const place = { pos, align, v, alt, onClick: go(frame) };
    view.addLabel({ html: `${id} · ${LAYERS[k - 1]} × ${LAYERS[k]}`, ...place, when: (c) => !c.far });
    view.addLabel({ html: id, ...place, when: (c) => c.far });
  }

  const names = { 1: ['b1', 'z1', 'h1'], 2: ['b2', 'z2', 'h2'], 3: ['b3', 'z3', 'h3'], 4: ['b4', 'logits', 'softmax'] };
  const groups = { 1: 'h1 · 150', 2: 'h2 · 100', 3: 'h3 · 50', 4: 'Output · 9 classes' };
  for (let k = 1; k <= 4; k++) {
    const ids = [`b${k}`, `z${k}`, k < 4 ? `h${k}` : 'p'];
    const vertical = B[ids[0]].swap; // strips that run along z get their names at the far end
    ids.forEach((id, n) => {
      const r = B[id].rect();
      const pos = vertical ? [(r.x0 + r.x1) / 2, 0, r.z0] : [r.x0, 0, (r.z0 + r.z1) / 2];
      const shape = vertical ? { align: 'c', v: 'above' } : { align: 'r', v: 'middle' };
      view.addLabel({ html: names[k][n], cls: 'tag small', pos, ...shape, when: (c) => c.strips });
      if (n === 1) view.addLabel({ html: groups[k], cls: 'plain', pos, ...shape, when: (c) => !c.strips && !c.far });
    });
  }

  const rp = B.p.rect();
  for (let cls = 0; cls < 9; cls++) {
    view.addLabel({
      html: `${m.tools[Math.floor(cls / 3)]}<br>${m.surfaces[cls % 3]}`,
      cls: 'tiny center',
      pos: [B.p.center(cls)[0], 0, rp.z1],
      v: 'below',
      when: (c) => c.classes,
    });
  }
  predLabel = view.addLabel({
    html: '',
    cls: 'tag pred',
    pos: [B.p.center(0)[0], 0, rp.z1],
    v: 'below',
    clamp: true,
    row: (c) => (c.classes ? 2 : 0),
    when: () => state.shown >= 8,
  });
}

// ------------------------------------------------------------------ hover and click

/** Which other cells belong with the hovered one: its input, its unit, and the weights between them. */
function linked({ id, i, j }) {
  const k = id === 'x' ? 0 : id === 'p' ? 4 : +id[1];
  const out = (n) => (n < 4 ? `h${n}` : 'p');
  if (id === 'x') return { x: [i, -1], W1: [i, -1] };
  if (id[0] === 'W') {
    return { [id]: [i, j], [k === 1 ? 'x' : `h${k - 1}`]: [i, -1], [`b${k}`]: [j, -1], [`z${k}`]: [j, -1], [out(k)]: [j, -1] };
  }
  const map = { [`W${k}`]: [-1, i], [`b${k}`]: [i, -1], [`z${k}`]: [i, -1], [out(k)]: [i, -1] };
  if (id[0] === 'h') map[`W${k + 1}`] = [i, -1];
  return map;
}

/** For the written tab: which of Astra's statements set this weight or bias. */
function origin({ id, i, j }) {
  if (state.tab !== 'written' || !made || !/^[Wb]\d$/.test(id)) return null;
  const cell = id[0] === 'W' ? i * LAYERS[+id[1]] + j : i;
  const own = made.owner(id, cell);
  return { own, text: own ? `Written by: ${groupTitle(made.groupOf(own[0]))} (click for the backtrace)` : 'Never written: it keeps the starting value 0.' };
}

function describe(hit) {
  const { id, i, j } = hit;
  const m = data.manifest;
  const net = currentNet();
  const act = state.act;
  const unit = (k, n) => (k < 4 ? `unit ${n} of layer ${k}` : `class ${n}: ${className(n, m)}`);
  const feature = (n) => {
    const f = describeFeature(n, m);
    return `${f.sensor} · ${f.band} (${f.range}) · ${f.stat}`;
  };
  const from = origin(hit)?.text ?? '';
  if (id === 'x') {
    return [`x[${i}] = ${fmt(act.x[i])}`, feature(i), 'Standardized: 0 is the average training weld.'];
  }
  const k = id === 'p' ? 4 : +id[1];
  if (id[0] === 'W') {
    const w = net.W[k - 1][i * LAYERS[k] + j];
    const a = layerInput(act, k)[i];
    const known = k === 1 || state.shown >= 2 * (k - 1);
    return [
      `W${k}[${i}, ${j}] = ${fmt(w)}`,
      k === 1 ? `from input ${i}: ${feature(i)}` : `from h${k - 1}[${i}]`,
      `to ${unit(k, j)}`,
      known ? `× its input ${fmt(a)} = ${fmt(a * w)}` : '',
      from,
    ];
  }
  if (id[0] === 'b') return [`b${k}[${i}] = ${fmt(net.b[k - 1][i])}`, `bias of ${unit(k, i)}`, from];
  const hidden = 'Not computed yet at this step.';
  if (id[0] === 'z') {
    if (state.shown < 2 * k - 1) return [`${k < 4 ? `z${k}` : 'logit'}[${i}]`, hidden];
    const z = act.z[k - 1][i];
    return k < 4
      ? [`z${k}[${i}] = ${fmt(z)}`, `sum of ${LAYERS[k - 1]} products + bias ${fmt(net.b[k - 1][i])}`, `after ReLU: ${fmt(Math.max(z, 0))}`]
      : [`logit[${i}] = ${fmt(z)}`, className(i, m), `sum of 50 products + bias ${fmt(net.b[3][i])}`];
  }
  if (id[0] === 'h') {
    if (state.shown < 2 * k) return [`h${k}[${i}]`, hidden];
    return [`h${k}[${i}] = ${fmt(act.h[k - 1][i])}`, `ReLU of z${k}[${i}] = ${fmt(act.z[k - 1][i])}`, `feeds every unit of layer ${k + 1}`];
  }
  if (state.shown < 8) return [`p[${i}]`, hidden];
  return [`P(${className(i, m)}) = ${pct(act.probs[i])}`, `logit ${fmt(act.logits[i])}`];
}

const tip = $('#tip');
function showTip(hit, event) {
  if (!data || !hit) {
    tip.hidden = true;
    view.setHighlights(null);
    $('#view').classList.remove('over');
    return;
  }
  const [title, ...lines] = isFew() ? few.describe(hit) : describe(hit);
  tip.innerHTML = `<b>${escapeHtml(title)}</b>${lines.filter(Boolean).map((l) => `<span>${escapeHtml(l)}</span>`).join('')}`;
  tip.hidden = false;
  const stage = $('#stage').getBoundingClientRect();
  const x = event.clientX - stage.left;
  const y = event.clientY - stage.top;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  const left = Math.min(Math.max(x + 16, 8), stage.width - w - 8);
  const top = y + 20 + h > stage.height - 8 ? Math.max(y - h - 14, 8) : y + 20;
  tip.style.transform = `translate(${left}px, ${top}px)`;
  view.setHighlights(isFew() ? few.linked(hit) : linked(hit));
  $('#view').classList.add('over');
}

/** A click on a written weight opens its backtrace: the group of statements that set it. */
function select(hit) {
  const own = hit && origin(hit)?.own;
  if (!own) return;
  state.made.group = own[0];
  state.made.fly = false; // the reader is already looking at the cell
  if (mode() !== 'doc') state.mode.written = 'doc';
  render();
  const part = $('#made-root [data-part="edits"]');
  if (part) {
    part.open = true;
    part.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }
}

// ------------------------------------------------------------------ start

function fillInputs() {
  const m = data.manifest;
  const select = $('#input-select');
  select.replaceChildren();
  const group = (label, items) => {
    const g = document.createElement('optgroup');
    g.label = label;
    for (const [k, text] of items) g.append(new Option(text, k));
    select.append(g);
  };
  if (isFew()) {
    group('The 18 welds the model was shown', few.options());
    select.value = String(state.few.weld);
    return;
  }
  const trainedReads = (item) => forward(data.trained, item.x).pred;
  const real = inputs.map((it, k) => [k, it]).filter(([, it]) => it.kind === 'real');
  const synthetic = inputs.map((it, k) => [k, it]).filter(([, it]) => it.kind === 'synthetic');
  if (real.length) {
    group('Real test welds', real.map(([k, it]) => [
      k,
      `Weld ${it.run}: ${className(it.label, m, true)}${trainedReads(it) === it.label ? '' : ' (misread by the trained net)'}`,
    ]));
  }
  group('Synthetic, made from the trained net', synthetic.map(([k, it]) => [k, `Synthetic: ${className(it.label, m, true)}`]));
  group('Other', [[inputs.length - 1, 'Random noise']]);
  select.value = String(state.input);
}

function buildControls() {
  const m = data.manifest;
  fillInputs();
  $('#input-select').addEventListener('change', (e) => {
    if (isFew()) state.few.weld = +e.target.value;
    else state.input = +e.target.value;
    render();
  });

  const replies = $('#reply-select');
  m.written.replies.forEach((r, k) => replies.append(new Option(`${k + 1} of ${m.written.replies.length} · ${pct(r.accuracy.mean)} accuracy`, k)));
  replies.value = String(state.reply);
  replies.addEventListener('change', () => {
    state.reply = +replies.value;
    state.made.group = 0;
    render();
  });

  if (few) {
    const fill = (id, items, key) => {
      const el = $(id);
      for (const [value, text] of items) el.append(new Option(text, value));
      el.value = String(state.few[key]);
      el.addEventListener('change', () => {
        state.few[key] = key === 'writer' ? el.value : +el.value;
        if (key === 'draw') {
          state.few.weld = 0;
          fillInputs();
        }
        render();
      });
    };
    fill('#set-select', few.SET_NAMES.map((name, k) => [k, name]), 'set');
    fill('#draw-select', [0, 1, 2].map((k) => [k, `Draw ${k + 1}`]), 'draw');
    fill('#writer-select', [...Object.entries(data.few.models), ['avg', 'No AI: plain averages']], 'writer');
  }

  for (const el of document.querySelectorAll('.tab')) el.addEventListener('click', () => setTab(el.dataset.tab));
  for (const el of $('#modes').children) el.addEventListener('click', () => setMode(el.dataset.mode));
  $('#prev').addEventListener('click', () => goTo(stepIndex() - 1));
  $('#back').addEventListener('click', () => goTo(stepIndex() - 1));
  $('#next').addEventListener('click', () => goTo(stepIndex() + 1));
  $('#continue').addEventListener('click', () => goTo(stepIndex() === steps().length - 1 ? 0 : stepIndex() + 1));
  $('#scroll').addEventListener('click', (e) => {
    if (e.target.closest('[data-action="switch"]')) setTab(state.tab === 'trained' ? 'written' : 'trained');
    if (e.target.closest('[data-action="made"]')) setMode('doc');
  });
  $('#products').addEventListener('click', (e) => {
    state.products = !state.products;
    e.currentTarget.setAttribute('aria-pressed', String(state.products));
    render();
  });
  $('#height').addEventListener('input', (e) => view.setHeightFactor(+e.target.value));
  $('#top-view').addEventListener('click', (e) => setTopView(e.currentTarget.getAttribute('aria-pressed') !== 'true'));
  $('#reset-view').addEventListener('click', () => fly(mode() === 'doc' ? (isFew() ? 'fewAll' : 'all') : steps()[stepIndex()].frame, 900));
  $('#about-open').addEventListener('click', () => $('#about').showModal());
  window.addEventListener('hashchange', () => setTab(fromHash()));
  window.addEventListener('keydown', (e) => {
    if (e.target.closest('select, input, dialog, button, summary') && e.key === ' ') return;
    if (e.target.closest('select, input, dialog') || e.metaKey || e.ctrlKey || e.altKey || mode() === 'doc') return;
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      goTo(stepIndex() + 1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      goTo(stepIndex() - 1);
    }
  });
}

/** The page's forward pass must agree with numpy's, or nothing on screen can be trusted. */
function selfCheck() {
  const nets = [[data.manifest.trained, data.trained], ...data.manifest.written.replies.map((r, k) => [r, data.written[k]])];
  for (const [meta, net] of nets) {
    const got = forward(net, data.inputs[meta.check.input].x).logits;
    const worst = meta.check.logits.reduce((s, v, c) => Math.max(s, Math.abs(v - got[c])), 0);
    if (worst > 1e-3) throw new Error(`forward pass of ${meta.file} is off by ${worst}`);
  }
}

/** The public copy is locked: it opens from the full link, or from that link (or its key) pasted here. */
function askForKey(problem) {
  const box = $('#loading');
  box.classList.add('error', 'locked');
  document.body.dataset.locked = 'true'; // nothing of the page is shown behind the notice
  box.innerHTML = `<b>This page is locked.</b>
    <span>It holds data from a course that is not public. Open it with the full link you were given, or paste that
    link here.</span>
    <form><input type="text" autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="Link or key" placeholder="Paste the link">
    <button class="primary" type="submit">Open</button></form>
    <span class="bad">${escapeHtml(problem)}</span>`;
  box.querySelector('form').addEventListener('submit', (e) => {
    e.preventDefault();
    const key = keyIn(box.querySelector('input').value);
    if (!key) {
      box.querySelector('.bad').textContent = 'That does not look like the link or its key.';
      return;
    }
    location.hash = `#${fromHash()}&k=${key}`;
    location.reload();
  });
}

async function start() {
  try {
    data = await loadData();
    selfCheck();
  } catch (error) {
    if (error instanceof LockedError) {
      askForKey(error.message);
      return;
    }
    console.error(error);
    $('#loading').innerHTML = `<b>The weights could not be loaded.</b><span>${escapeHtml(String(error.message ?? error))}</span>
      <span>Run <code>tools/export_weights.py</code> to create the <code>data/</code> folder, then reload.</span>`;
    $('#loading').classList.add('error');
    return;
  }
  inputs = [...data.inputs, { kind: 'noise', x: noiseInput(state.noiseSeed) }];
  const best = data.manifest.written.replies.reduce((b, r, k, all) => (r.accuracy.mean > all[b].accuracy.mean ? k : b), 0);
  state.reply = best;
  buildLabels();
  if (data.provenance) {
    made = createMade({
      root: $('#made-root'),
      view,
      data,
      state,
      actions: {
        changed: render,
        showCells(cells, move) {
          const map = Object.fromEntries(Object.keys(view.blocks).filter((id) => /^[Wb]\d$/.test(id)).map((id) => [id, cells[id] ?? []]));
          view.setMask(map);
          const [id, list] = Object.entries(cells).sort((a, b) => b[1].length - a[1].length)[0] ?? [];
          if (move && id) {
            setTopView(false, false);
            view.flyToCells(id, list);
          }
        },
      },
    });
  }
  if (data.few) {
    few = createFew({ view, data, state, actions: { fly } });
    few.buildLabels();
  }
  for (const el of document.querySelectorAll('.tab')) if (el.dataset.tab === 'few' && !few) el.hidden = true;
  if (state.tab === 'few' && !few) state.tab = 'trained';
  buildControls();
  syncTabs();
  if (isFew()) {
    view.setLayout('few');
    fillInputs();
    view.snapTo('fewAll');
  }
  view.onPick = showTip;
  view.onSelect = select;
  view.controls.addEventListener('start', () => {
    showTip(null);
    $('#hint').classList.add('seen');
  });
  render();
  $('#loading').hidden = true;
  window.app = { state, view, data, goTo, setTab, setMode }; // handy in the console
}

start().catch((error) => {
  // anything that goes wrong while the page is put together is shown, not left behind a "Loading" notice
  console.error(error);
  $('#loading').hidden = false;
  $('#loading').classList.add('error');
  $('#loading').innerHTML = `<b>The page could not start.</b><span>${escapeHtml(String(error.message ?? error))}</span>`;
});
