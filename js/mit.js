// The sixth tab: the networks GPT-6 Astra wrote in MIT CSAIL's own experiments ("Sparks of Weight-Space
// Programming", November 2026), exactly as MIT logged them (github.com/RikiyaT/lm_understand-logs, CC BY 4.0):
//   Experiment 5  three MinAtar policies, 600 -> 32 -> 32 -> 6 with tanh, written from each game's rules and source
//   Experiment 4  a PPO-trained Kinetix car policy that never succeeds, and the same policy after Astra's one-reply
//                 repair, 679 -> 128 (ReLU) -> 5 x 128 (tanh) -> 16
// Every number comes from tools/mit_networks.py and tools/car_frames.py.

import { scaleOf, normalized, fmt, pct } from './net.js';
import { escapeHtml, markdown } from './md.js';

const GAME_SIZES = [600, 32, 32, 6];
const CAR_SIZES = [679, 128, 128, 128, 128, 128, 128, 16];
const CAR_LAYERS = ['embed', 'actor0', 'actor1', 'actor2', 'actor3', 'actor4', 'actor_head'];
const ENTRIES = [3, 3, 3, 3, 2, 2]; // the car's 16 logits: 6 action entries, each with its own softmax
const ENTRY0 = ['motor off', 'drive +1', 'drive −1'];
const CAR_VIEWS = { 'car-trained': 'trained', 'car-edited': 'edited', 'car-changes': 'changes' };
export const NETWORKS = [
  ['Experiment 5: written from the rules', [['breakout', 'Breakout'], ['seaquest', 'Seaquest'], ['asterix', 'Asterix']]],
  ['Experiment 4: a trained car policy, repaired', [
    ['car-trained', 'Car: trained by PPO'], ['car-edited', "Car: after Astra's repair"], ['car-changes', 'Car: only the changes'],
  ]],
];
// one colour per game channel: player, danger, reward, projectile, trail, gauge
const CHANNEL_COLORS = ['#2c62dc', '#d23b2a', '#d99a00', '#1b2030', '#a3abbd', '#7b4fc9'];
const POLY_SLOTS = [0, 4, 5, 6, 7, 8, 9, 10, 11];
const CIRCLE_SLOTS = [12, 13, 14, 15];
const SHAPES = { 0: 'the floor', 4: 'the blue box', 5: 'the car body', 6: 'the ramp', 7: 'the red bar', 12: 'the green wheel', 13: 'the other wheel' };
const POLY_FIELDS = [
  'position x', 'position y', 'velocity x', 'velocity y', 'inverse mass', 'inverse inertia', 'density',
  'tanh(angular velocity / 10)', 'role: none', 'role: green', 'role: blue', 'role: red', 'sin(angle)', 'cos(angle)',
  'friction', 'restitution', '0 = polygon', 'vertex 1 x', 'vertex 1 y', 'vertex 2 x', 'vertex 2 y', 'vertex 3 x',
  'vertex 3 y', 'vertex 4 x', 'vertex 4 y', '1 if at most 3 vertices',
];
const CIRCLE_FIELDS = [...POLY_FIELDS.slice(0, 16), 'radius', '1 = circle'];
const JOINT_FIELDS = [
  'active', 'fixed', 'position on shape b, x', 'position on shape b, y', 'position on shape a, x',
  'position on shape a, y', 'sin(reference angle)', 'cos(reference angle)', 'motor speed', 'motor power', 'motor on',
  'has angle limits', ...[0, 1, 2, 3].map((k) => `motor binding ${k}`), 'sin(min angle)', 'cos(min angle)',
  'sin(max angle)', 'cos(max angle)', 'min angle − reference', 'max angle − reference',
  ...Array.from({ length: 16 }, (_, k) => `shape b is ${k}`), ...Array.from({ length: 16 }, (_, k) => `shape a is ${k}`),
];
const THRUSTER_FIELDS = [
  'active', 'position x', 'position y', 'binding 0', 'binding 1', 'sin(angle)', 'cos(angle)', 'power',
  ...Array.from({ length: 16 }, (_, k) => `on shape ${k}`),
];

const n = (v) => v.toLocaleString('en-US');
const count = (a) => a.reduce((s, v) => s + (v !== 0 ? 1 : 0), 0);

function parseNet(buffer, sizes) {
  const flat = new Float32Array(buffer);
  const W = [];
  const b = [];
  let o = 0;
  for (let k = 1; k < sizes.length; k++) {
    W.push(flat.subarray(o, (o += sizes[k - 1] * sizes[k])));
    b.push(flat.subarray(o, (o += sizes[k])));
  }
  if (o !== flat.length) throw new Error(`expected ${o} numbers, got ${flat.length}`);
  return { flat, W, b, sizes, nonzero: count(flat) };
}

/** Game: tanh, tanh, logits. Car: inputs clipped to [-10, 10], ReLU, five tanh layers, logits. */
function forward(net, input, car) {
  const L = net.sizes.length - 1;
  const x = car ? Float32Array.from(input, (v) => Math.min(Math.max(v, -10), 10)) : input;
  const z = [];
  const h = [];
  let a = x;
  for (let k = 0; k < L; k++) {
    const no = net.sizes[k + 1];
    const W = net.W[k];
    const acc = Float64Array.from(net.b[k]);
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      if (ai === 0) continue;
      const row = i * no;
      for (let j = 0; j < no; j++) acc[j] += ai * W[row + j];
    }
    z.push(Float32Array.from(acc));
    if (k < L - 1) {
      a = Float32Array.from(acc, (v) => (car && k === 0 ? Math.max(v, 0) : Math.tanh(v)));
      h.push(a);
    }
  }
  const logits = z[L - 1];
  const probs = new Float32Array(logits.length);
  let o = 0;
  for (const size of car ? ENTRIES : [logits.length]) {
    const part = logits.subarray(o, o + size);
    const top = Math.max(...part);
    const e = Array.from(part, (v) => Math.exp(v - top));
    const sum = e.reduce((s, v) => s + v, 0);
    e.forEach((v, m) => (probs[o + m] = v / sum));
    o += size;
  }
  const best = (lo, hi) => {
    let m = lo;
    for (let c = lo + 1; c < hi; c++) if (probs[c] > probs[m]) m = c;
    return m;
  };
  return { x, z, h, logits, probs, pred: car ? best(0, 3) : best(0, logits.length) };
}

function contributions(net, act, k) {
  const W = net.W[k - 1];
  const a = k === 1 ? act.x : act.h[k - 2];
  const no = net.sizes[k];
  const out = new Float32Array(W.length);
  for (let i = 0; i < a.length; i++) {
    if (a[i] === 0) continue;
    for (let j = 0; j < no; j++) out[i * no + j] = a[i] * W[i * no + j];
  }
  return out;
}

/** Which shape and which number an observation entry is. */
export function obsName(i) {
  if (i < 234) {
    const slot = POLY_SLOTS[Math.floor(i / 26)];
    return `polygon slot ${slot}${SHAPES[slot] ? ` (${SHAPES[slot]})` : ', empty'} · ${POLY_FIELDS[i % 26]}`;
  }
  if (i < 306) {
    const slot = CIRCLE_SLOTS[Math.floor((i - 234) / 18)];
    return `circle slot ${slot}${SHAPES[slot] ? ` (${SHAPES[slot]})` : ', empty'} · ${CIRCLE_FIELDS[(i - 234) % 18]}`;
  }
  if (i < 630) {
    const slot = Math.floor((i - 306) / 54);
    const what = ['joint 0 (car body to the other wheel)', 'joint 1 (car body to the green wheel)'][slot] ?? `joint ${slot}, empty`;
    return `${what} · ${JOINT_FIELDS[(i - 306) % 54]}`;
  }
  if (i < 678) return `thruster ${Math.floor((i - 630) / 24)}, empty · ${THRUSTER_FIELDS[(i - 630) % 24]}`;
  return 'gravity y / 10';
}

export function createMit({ view, data, state, actions }) {
  const D = data.mit;
  const M = D.meta;
  const sel = state.mit;
  const games = Object.fromEntries(M.games.map((g) => [g.id, { meta: g, net: parseNet(D.buffers[g.file], GAME_SIZES) }]));
  const car = {
    trained: parseNet(D.buffers[M.car.files.trained], CAR_SIZES),
    edited: parseNet(D.buffers[M.car.files.edited], CAR_SIZES),
  };
  const isCar = () => sel.net.startsWith('car');
  const carView = () => CAR_VIEWS[sel.net];
  const game = () => games[sel.net];
  const prefix = () => (isCar() ? 'c' : 'g');
  const layout = () => (isCar() ? 'car' : 'game');
  const depth = () => (isCar() ? 7 : 3);
  const net = () => (isCar() ? car[carView() === 'trained' ? 'trained' : 'edited'] : game().net);
  const reply = () => (isCar() ? M.car.reply : game().meta.reply);
  const tensorOf = (k, part) => (isCar() ? `${CAR_LAYERS[k - 1]}.${part}` : `${part}${k}`);
  const layerName = (k) => (isCar() ? CAR_LAYERS[k - 1] : `layer ${k}`);

  // ------------------------------------------------------------------ the episode on screen

  const episode = () => {
    if (!isCar()) return { frames: game().meta.episode.frames, offset: 0 };
    const E = D.frames.episodes;
    return carView() === 'trained' ? { ...E.trained, offset: 0 } : { ...E.edited, offset: E.trained.steps };
  };
  const frameCount = () => episode().frames.length;
  function frameX(k = sel.frame) {
    if (isCar()) {
      const o = (episode().offset + k) * 679;
      return D.frames.x.subarray(o, o + 679);
    }
    const x = new Float32Array(600);
    for (const i of episode().frames[k].on) x[i] = 1;
    return x;
  }

  // ------------------------------------------------------------------ who wrote each weight

  const ownerCache = new Map();
  /** "W1:cell" -> [group, statement], for the reply on screen. */
  function owners() {
    const key = isCar() ? 'car' : sel.net;
    if (!ownerCache.has(key)) {
      const map = new Map();
      reply().groups.forEach((g, gi) => g.statements.forEach((s, si) => {
        for (const cell of s.cells) map.set(`${s.tensor}:${cell}`, [gi, si]);
      }));
      ownerCache.set(key, map);
    }
    return ownerCache.get(key);
  }
  const groupTitle = (g) => (isCar() ? `${g.section} (${g.statements.length} statements)` : g.section || 'Untitled group');

  // ------------------------------------------------------------------ what the bars show

  const shown = new Map();
  /** The numbers a weight or bias block shows, normalized to [-1, 1] (cached: the car's blocks are large). */
  function weightBars(k, part) {
    const key = `${sel.net}:${part}${k}`;
    if (shown.has(key)) return shown.get(key);
    const pick = (nn) => (part === 'W' ? nn.W[k - 1] : nn.b[k - 1]);
    let out;
    if (!isCar()) out = normalized(pick(game().net), scaleOf(pick(game().net)));
    else if (carView() !== 'changes') {
      // the repaired network keeps the trained network's scale: what Astra kept looks the same, what it set stands out
      out = normalized(pick(car[carView()]), scaleOf(pick(car.trained)));
    } else {
      const before = pick(car.trained);
      const after = pick(car.edited);
      const diff = new Float32Array(before.length);
      for (const cell of M.car.changed[tensorOf(k, part)]) diff[cell] = after[cell] - before[cell];
      out = normalized(diff, scaleOf(diff));
    }
    shown.set(key, out);
    return out;
  }

  function context() {
    const nn = net();
    const x = frameX();
    const act = forward(nn, x, isCar());
    const frame = episode().frames[sel.frame];
    return { nn, act, frame, steps: frameCount() };
  }

  let decalKey = '';
  function setDecal() {
    const key = sel.net;
    if (key === decalKey) return;
    decalKey = key;
    if (!isCar()) {
      const g = game().meta;
      view.setDecal({
        title: '600 → 32 → 32 → 6',
        lines: [
          { text: `MIT's ${g.name} policy · ${n(20486)} weights · Experiment 5` },
          { text: "Every weight written by GPT-6 Astra from the game's rules", bold: true, color: '#b84e14' },
          { text: `MIT: ${g.scores.mit[0]} points per episode · PPO after 500M steps: ${g.scores.ppo[1]}` },
        ],
      });
      return;
    }
    const v = carView();
    const changed = Object.values(M.car.changed).reduce((s, c) => s + c.length, 0);
    view.setDecal({
      title: '679 → 128 → 5 × 128 → 16',
      lines: [
        { text: `MIT's Kinetix car policy · ${n(171664)} weights · Experiment 4` },
        v === 'trained'
          ? { text: 'Trained by PPO for 1 billion steps', bold: true, color: '#2553c4' }
          : { text: v === 'edited' ? "After GPT-6 Astra's one-reply repair" : `What Astra changed: ${n(changed)} weights`, bold: true, color: '#b84e14' },
        { text: `Reaches the goal: ${pct(M.car.result.before.success, 0)} before, ${pct(M.car.result.after.success, 0)} of ${n(M.car.result.after.episodes)} episodes after` },
      ],
    });
  }

  // ------------------------------------------------------------------ 3D

  let predLabels = {};
  function render() {
    const c = context();
    const { nn, act } = c;
    const P = prefix();
    const L = depth();
    const quick = sel.playing;
    const ms = (d) => (quick ? Math.min(d, 280) : d);
    view.setBlock(`${P}x`, normalized(act.x, scaleOf(act.x, 0.98)), 'data', { duration: ms(700) });
    for (let k = 1; k <= L; k++) {
      if (state.products) {
        const prod = contributions(nn, act, k);
        view.setBlock(`${P}W${k}`, normalized(prod, scaleOf(prod, 0.999)), 'data', { duration: ms(1200), stagger: 0.8 });
      } else {
        view.setBlock(`${P}W${k}`, weightBars(k, 'W'), 'weights', { duration: ms(1200), stagger: 0.8 });
      }
      view.setBlock(`${P}b${k}`, weightBars(k, 'b'), 'weights', { duration: ms(700) });
      const z = act.z[k - 1];
      view.setBlock(`${P}z${k}`, normalized(z, scaleOf(z, 1)), 'data', { duration: ms(800), delay: quick ? 0 : 450 });
      if (k < L) {
        const h = act.h[k - 1];
        const relu = isCar() && k === 1;
        view.setBlock(`${P}h${k}`, relu ? normalized(h, scaleOf(h, 1)) : normalized(h, 1), 'data', { duration: ms(800), delay: quick ? 0 : 900 });
      }
    }
    view.setBlock(`${P}p`, act.probs, 'prob', { duration: ms(800), delay: quick ? 0 : 700 });
    const step = state.mode.mit === 'doc' ? null : STEPS[sel.step];
    view.setFocus(step?.focus ? step.focus.map((id) => P + id) : null);
    setDecal();
    const pb = view.blocks[`${P}p`];
    const label = predLabels[layout()];
    if (label) {
      const what = isCar() ? `entry 0: <b>${ENTRY0[act.pred]}</b>` : `<b>${M.actions[act.pred]}</b>`;
      view.updateLabel(label, { html: `${what} · ${pct(act.probs[act.pred], 0)}`, pos: [pb.rect().x1, 0, pb.center(act.pred)[1]] });
    }
    return c;
  }

  function buildLabels() {
    for (const [lay, P, sizes] of [['game', 'g', GAME_SIZES], ['car', 'c', CAR_SIZES]]) {
      const L = sizes.length - 1;
      const B = (id) => view.blocks[P + id];
      const add = (def) => view.addLabel({ layout: lay, ...def });
      const fly = (frame) => () => actions.fly(P + frame);
      const rx = B('x').rect();
      const far = (c) => c.ppu < 0.62;
      const inputName = lay === 'game' ? 'Input x · one screen, 10 × 10 × 6' : 'Input x · the scene, 679 numbers';
      add({ html: inputName, pos: [rx.x0, 0, rx.z0], align: 'l', row: (c) => (c.ppu >= 2.4 ? 1 : 0), when: (c) => !far(c), onClick: fly('Input') });
      add({ html: 'Input', pos: [rx.x0, 0, rx.z0], align: 'l', when: far, onClick: fly('Input') });
      if (lay === 'game') {
        for (let r = 0; r < 10; r++) {
          const mid = (B('x').center(60 * r)[0] + B('x').center(60 * r + 59)[0]) / 2;
          add({ html: `row ${r}`, cls: 'tiny', pos: [mid, 0, rx.z0], when: (c) => c.ppu >= 2.4 });
        }
      } else {
        for (const [a, b, name] of [[0, 233, 'polygons'], [234, 305, 'circles'], [306, 629, 'joints'], [630, 677, 'thrusters']]) {
          const mid = (B('x').center(a)[0] + B('x').center(b)[0]) / 2;
          add({ html: name, cls: 'tiny', pos: [mid, 0, rx.z0], when: (c) => c.ppu >= 1.6 });
        }
      }
      for (let k = 1; k <= L; k++) {
        const r = B(`W${k}`).rect();
        const across = k % 2 === 1;
        const name = lay === 'game' ? `W${k}` : `${CAR_LAYERS[k - 1]}.W`;
        const place = k === 1
          ? { pos: [r.x0, 0, r.z1], align: 'l', v: 'below' }
          : across ? { pos: [r.x0, 0, (r.z0 + r.z1) / 2], align: 'r', v: 'middle' } : { pos: [r.x1, 0, r.z0], align: 'r', v: 'above' };
        const alt = [{ pos: [r.x0, 0, r.z1], align: 'l', v: 'above' }, { pos: [r.x1, 0, r.z1], align: 'r', v: 'above' }];
        add({ html: `${name} · ${sizes[k - 1]} × ${sizes[k]}`, ...place, alt, when: (c) => !far(c), onClick: fly(`L${k}`) });
        add({ html: name, ...place, alt, when: far, onClick: fly(`L${k}`) });
        const ids = [`b${k}`, `z${k}`, k < L ? `h${k}` : 'p'];
        const fn = lay === 'car' && k === 1 ? 'ReLU' : 'tanh';
        const names = [`b${k}`, `z${k}`, k < L ? `h${k}` : 'softmax'];
        ids.forEach((id, m) => {
          const s = B(id).rect();
          const pos = across ? [(s.x0 + s.x1) / 2, 0, s.z0] : [s.x0, 0, (s.z0 + s.z1) / 2];
          const shape = across ? { align: 'c', v: 'above' } : { align: 'r', v: 'middle' };
          add({ html: names[m], cls: 'tag small', pos, ...shape, when: (c) => c.strips });
          if (m === 1) {
            const group = k < L ? `h${k} · ${sizes[k]} ${fn}` : lay === 'game' ? 'Output · 6 actions' : 'Output · 16 logits';
            add({ html: group, cls: 'plain', pos, ...shape, when: (c) => !c.strips && !far(c) });
          }
        });
      }
      const pb = B('p');
      const pr = pb.rect();
      const outs = lay === 'game' ? M.actions : ENTRY0;
      outs.forEach((name, k) => add({ html: name, cls: 'tiny', pos: [pr.x1, 0, pb.center(k)[1]], align: 'l', v: 'middle', when: (c) => c.ppu >= (lay === 'game' ? 1.3 : 2.4) }));
      predLabels[lay] = add({ html: '', cls: 'tag pred', pos: [pr.x1, 0, pb.center(0)[1]], align: 'l', v: 'middle', clamp: true, row: () => 0 });
    }
  }

  // ------------------------------------------------------------------ hover

  function linked({ id, i, j }) {
    const P = prefix();
    const name = id.slice(1);
    const L = depth();
    const out = (k) => (k < L ? `h${k}` : 'p');
    const map = {};
    const put = (b, v) => (map[P + b] = v);
    if (name === 'x') {
      put('x', [i, -1]);
      put('W1', [i, -1]);
      return map;
    }
    if (name[0] === 'W') {
      const k = +name.slice(1);
      put(name, [i, j]);
      put(k === 1 ? 'x' : `h${k - 1}`, [i, -1]);
      put(`b${k}`, [j, -1]);
      put(`z${k}`, [j, -1]);
      put(out(k), [j, -1]);
      return map;
    }
    const k = name === 'p' ? L : +name.slice(1);
    put(`W${k}`, [-1, i]);
    put(`b${k}`, [i, -1]);
    put(`z${k}`, [i, -1]);
    put(out(k), [i, -1]);
    if (name[0] === 'h') put(`W${k + 1}`, [i, -1]);
    return map;
  }

  function inputName(i) {
    if (isCar()) return obsName(i);
    const cell = Math.floor(i / 6);
    const ch = i % 6;
    const what = game().meta.channels[ch];
    return `row ${Math.floor(cell / 10)}, column ${cell % 10} · channel ${ch} (${M.channel_names[ch]}: ${what ? what.replace(/_/g, ' ') : 'always empty in this game'})`;
  }
  const unitName = (k, u) => {
    if (k < depth()) return `unit ${u} of ${layerName(k)}`;
    if (!isCar()) return `the logit of “${M.actions[u]}”`;
    let o = 0;
    for (const [e, size] of ENTRIES.entries()) {
      if (u < o + size) return `the logit of action entry ${e}, value ${u - o}${e === 0 ? ` (${ENTRY0[u - o]})` : ' (no effect here)'}`;
      o += size;
    }
    return '';
  };

  /** Where one weight came from: the statement that wrote it, or training. */
  function origin(tensor, cell) {
    const own = owners().get(`${tensor}:${cell}`);
    const g = own && reply().groups[own[0]];
    if (!isCar()) return own ? `Written by: ${groupTitle(g)} (click for the statement)` : 'Never written: it stays 0.';
    if (carView() === 'trained') return 'Trained by PPO.';
    return own ? `Set by Astra: ${g.statements[own[1]].text} (click to see it)` : 'Kept from the trained network.';
  }

  function describe({ id, i, j }) {
    const { nn, act } = context();
    const name = id.slice(1);
    const L = depth();
    if (name === 'x') {
      const shownX = isCar() ? `x[${i}] = ${fmt(act.x[i])}` : `x[${i}] = ${act.x[i]}`;
      return [shownX, inputName(i)];
    }
    const k = name === 'p' ? L : +name.slice(1);
    if (name[0] === 'W') {
      const tensor = tensorOf(k, 'W');
      const w = nn.W[k - 1][i * nn.sizes[k] + j];
      const a = k === 1 ? act.x[i] : act.h[k - 2][i];
      const lines = [`${tensor}[${i}, ${j}] = ${fmt(w)}`, k === 1 ? `from input ${i}: ${inputName(i)}` : `from unit ${i} of ${layerName(k - 1)}`, `to ${unitName(k, j)}`, `× its input ${fmt(a)} = ${fmt(a * w)}`];
      if (isCar() && carView() !== 'trained') {
        const before = car.trained.W[k - 1][i * nn.sizes[k] + j];
        if (before !== w) lines.push(`trained value: ${fmt(before)}`);
      }
      return [...lines, origin(tensor, i * nn.sizes[k] + j)];
    }
    if (name[0] === 'b') return [`${tensorOf(k, 'b')}[${i}] = ${fmt(nn.b[k - 1][i])}`, `bias of ${unitName(k, i)}`, origin(tensorOf(k, 'b'), i)];
    if (name[0] === 'z') {
      const z = act.z[k - 1][i];
      const fn = k === L ? '' : isCar() && k === 1 ? `after ReLU: ${fmt(Math.max(z, 0))}` : `after tanh: ${fmt(Math.tanh(z))}`;
      return [`z${k}[${i}] = ${fmt(z)}`, `${unitName(k, i)}: sum of ${nn.sizes[k - 1]} products + bias`, fn];
    }
    if (name[0] === 'h') return [`h${k}[${i}] = ${fmt(act.h[k - 1][i])}`, `output of ${unitName(k, i)}`, `feeds every unit of ${layerName(k + 1)}`];
    return [`P = ${pct(act.probs[i])}`, unitName(L, i), `logit ${fmt(act.logits[i])}`];
  }

  /** Hovering the first layer of a game shows that unit's weights as the game grid, in the panel. */
  function hover(hit) {
    if (isCar() || !hit) return;
    const name = hit.id.slice(1);
    let unit = null;
    if (name === 'W1') unit = hit.j;
    else if (['b1', 'z1', 'h1'].includes(name)) unit = hit.i;
    else if (name === 'W2') unit = hit.i;
    if (unit === null || unit === sel.unit) return;
    sel.unit = unit;
    const host = document.querySelector('#mit-rf');
    if (host) host.innerHTML = fieldHtml();
  }

  /** A click on a written or changed weight opens Astra's reply at the statement that set it. */
  function select(hit) {
    if (!hit || !/^[Wb]\d$/.test(hit.id.slice(1))) return false;
    const name = hit.id.slice(1);
    const k = +name.slice(1);
    const tensor = tensorOf(k, name[0]);
    const cell = name[0] === 'W' ? hit.i * net().sizes[k] + hit.j : hit.i;
    const own = owners().get(`${tensor}:${cell}`);
    if (!own || (isCar() && carView() === 'trained')) return false;
    sel.group = own[0];
    sel.fly = false;
    return true;
  }

  // ------------------------------------------------------------------ pictures in the panel

  function screenSvg(x) {
    const g = game().meta;
    const cells = [];
    for (let r = 0; r < 10; r++) {
      for (let col = 0; col < 10; col++) {
        const on = [0, 1, 2, 3, 4, 5].filter((ch) => x[(r * 10 + col) * 6 + ch]);
        const [first, ...rest] = on;
        const X = col * 10;
        const Y = r * 10;
        cells.push(`<rect x="${X + 0.5}" y="${Y + 0.5}" width="9" height="9" rx="1.2" fill="${first === undefined ? '#eceef3' : CHANNEL_COLORS[first]}"/>`);
        rest.forEach((ch, m) => cells.push(`<circle cx="${X + 3 + 4 * m}" cy="${Y + 7}" r="1.7" fill="${CHANNEL_COLORS[ch]}" stroke="#fff" stroke-width="0.6"/>`));
      }
    }
    const legend = g.channels.map((what, ch) => (what ? `<span><i style="background:${CHANNEL_COLORS[ch]}"></i>${escapeHtml(M.channel_names[ch])}: ${escapeHtml(what.replace(/_/g, ' '))}</span>` : '')).join('');
    return `<div class="mit-screen"><svg viewBox="0 0 100 100" role="img" aria-label="The game screen the network sees">${cells.join('')}</svg>
      <p class="mit-legend">${legend}<span class="meta">A cell in two channels shows the second as a dot.</span></p></div>`;
  }

  /** What one first-layer unit looks at: its 600 weights as the 10 × 10 grid, one small grid per channel. */
  function fieldHtml() {
    const g = game();
    const W = g.net.W[0];
    const u = sel.unit ?? 0;
    let top = 0;
    for (let i = 0; i < 600; i++) top = Math.max(top, Math.abs(W[i * 32 + u]));
    const used = g.meta.channels.map((what, ch) => [what, ch]).filter(([what]) => what);
    const grids = used.map(([what, ch]) => {
      const rects = [];
      for (let cell = 0; cell < 100; cell++) {
        const w = W[(cell * 6 + ch) * 32 + u];
        const t = top ? Math.min(Math.abs(w) / top, 1) : 0;
        const fill = w === 0 ? '#eceef3' : w > 0 ? `rgba(44, 98, 220, ${0.18 + 0.82 * t})` : `rgba(217, 96, 31, ${0.18 + 0.82 * t})`;
        rects.push(`<rect x="${(cell % 10) * 10 + 0.5}" y="${Math.floor(cell / 10) * 10 + 0.5}" width="9" height="9" rx="1" fill="${fill}"><title>${fmt(w)}</title></rect>`);
      }
      return `<figure><svg viewBox="0 0 100 100" role="img" aria-label="Weights from channel ${ch}">${rects.join('')}</svg><figcaption>${escapeHtml(M.channel_names[ch])}<small>${escapeHtml(what.replace(/_/g, ' '))}</small></figcaption></figure>`;
    }).join('');
    // the groups of statements that wrote this column, in the order of the reply: the first one usually names the unit
    const written = new Set();
    for (let i = 0; i < 600; i++) {
      const own = owners().get(`W1:${i * 32 + u}`);
      if (own) written.add(own[0]);
    }
    const order = [...written].sort((a, b) => a - b);
    const more = order.length > 1 ? `, then ${order.length - 1} more group${order.length > 2 ? 's' : ''}` : '';
    const comment = order.length ? `<br>Written by: “${escapeHtml(groupTitle(reply().groups[order[0]]))}”${more}` : '<br>Astra left this unit at 0.';
    const nonzero = [...Array(600).keys()].filter((i) => W[i * 32 + u] !== 0).length;
    return `<p class="mit-unit"><b>Unit ${u}</b> · ${nonzero} non-zero weight${nonzero === 1 ? '' : 's'}, bias ${fmt(g.net.b[0][u])}${comment}</p>
      <div class="mit-field${used.length <= 4 ? ' two' : ''}">${grids}</div>
      <p class="mit-legend"><span><i style="background:#2c62dc"></i>pushes the unit up</span><span><i style="background:#d9601f"></i>pushes it down</span><span class="meta">Hover W1 in the 3D view to pick a unit.</span></p>`;
  }

  /** The car scene, drawn from the 679 numbers the policy reads. */
  let sceneBox = null;
  function sceneSvg(x) {
    const shapes = [];
    const roleColor = (o, dynamic) => (x[o + 9] ? '#2fa84f' : x[o + 10] ? '#2c62dc' : x[o + 11] ? '#d23b2a' : dynamic ? '#5a6173' : '#b3b9c8');
    POLY_SLOTS.forEach((slot, s) => {
      const o = 26 * s;
      if (!x[o + 6]) return; // an empty slot has density 0
      const [sin, cos] = [x[o + 12], x[o + 13]];
      const nv = x[o + 25] ? 3 : 4;
      const pts = [];
      for (let v = 0; v < nv; v++) {
        const [vx, vy] = [x[o + 17 + 2 * v], x[o + 18 + 2 * v]];
        pts.push([x[o] + cos * vx - sin * vy, x[o + 1] + sin * vx + cos * vy]);
      }
      shapes.push({ kind: 'poly', pts, fill: roleColor(o, x[o + 4] > 0), slot });
    });
    CIRCLE_SLOTS.forEach((slot, s) => {
      const o = 234 + 18 * s;
      if (!x[o + 6]) return;
      shapes.push({ kind: 'circle', cx: x[o], cy: x[o + 1], r: x[o + 16], sin: x[o + 12], cos: x[o + 13], fill: roleColor(o, x[o + 4] > 0), slot });
    });
    if (!sceneBox) {
      // one frame for the whole episode: x from 0 to 5, y from the floor to the highest point the car reaches
      let top = 1.4;
      const { frames } = episode();
      for (let k = 0; k < frames.length; k++) {
        const f = frameX(k);
        for (const s of [1, 2]) top = Math.max(top, f[26 * s + 1] + 0.45);
        for (const s of [0, 1]) top = Math.max(top, f[234 + 18 * s + 1] + 0.3);
      }
      sceneBox = { x0: -0.05, x1: 5.05, y0: 0.2, y1: Math.min(top, 4) };
    }
    const { x0, x1, y0, y1 } = sceneBox;
    const P = (px, py) => `${(px - x0).toFixed(3)},${(y1 - py).toFixed(3)}`;
    const body = shapes.map((s) => (s.kind === 'poly'
      ? `<polygon points="${s.pts.map(([a, b]) => P(a, b)).join(' ')}" fill="${s.fill}"><title>${SHAPES[s.slot] ?? `slot ${s.slot}`}</title></polygon>`
      : `<g><circle cx="${(s.cx - x0).toFixed(3)}" cy="${(y1 - s.cy).toFixed(3)}" r="${s.r.toFixed(3)}" fill="${s.fill}"><title>${SHAPES[s.slot] ?? `slot ${s.slot}`}</title></circle>
         <line x1="${(s.cx - x0).toFixed(3)}" y1="${(y1 - s.cy).toFixed(3)}" x2="${(s.cx - x0 + s.r * s.cos).toFixed(3)}" y2="${(y1 - s.cy - s.r * s.sin).toFixed(3)}" stroke="#fff" stroke-width="0.03"/></g>`)).join('');
    return `<div class="mit-scene"><svg viewBox="0 0 ${(x1 - x0).toFixed(2)} ${(y1 - y0).toFixed(2)}" role="img" aria-label="The car scene at this step">${body}</svg>
      <p class="mit-legend"><span><i style="background:#2fa84f"></i>green wheel: must touch blue</span><span><i style="background:#2c62dc"></i>blue box (goal)</span><span><i style="background:#d23b2a"></i>red: touching it fails</span></p></div>`;
  }

  // ------------------------------------------------------------------ panel

  const scores = (g) => `${g.scores.mit[0]} <small>± ${g.scores.mit[1]}</small>`;

  function glance(c) {
    const { act, frame } = c;
    if (!isCar()) {
      const g = game().meta;
      const rows = M.actions.map((name, k) => {
        const marks = `${k === act.pred ? 'pred' : ''}${k === frame.chosen ? ' truth' : ''}`;
        return `<li class="${marks.trim()}"><span class="name">${name}</span><span class="bar"><i style="width:${(100 * act.probs[k]).toFixed(1)}%"></i></span><span class="p">${pct(act.probs[k], 0)}</span></li>`;
      }).join('');
      return `
        <h2>MIT's ${g.name} network</h2>
        <dl class="facts">
          <div><dt>Non-zero numbers</dt><dd>${n(g.nonzero)}<small>${pct(g.nonzero / 20486)} of 20,486</small></dd></div>
          <div><dt>Points per episode</dt><dd>${g.scores.mit[0]}<small>MIT, 1,000 episodes · PPO 500M: ${g.scores.ppo[1]}</small></dd></div>
          <div><dt>This screen</dt><dd>step ${sel.frame + 1}<small>of ${c.steps} shown · score so far ${frame.score}</small></dd></div>
        </dl>
        <h3>The screen it sees</h3>
        ${screenSvg(act.x)}
        <h3>Its answer for this screen</h3>
        <ol class="bars">${rows}</ol>
        <p class="legend"><span class="dot"></span>the action it sampled at this step${frame.taken !== frame.chosen ? ` (the game repeated “${M.actions[frame.taken]}” instead)` : ''}</p>
        <h3>What a first-layer unit looks at</h3>
        <div id="mit-rf">${fieldHtml()}</div>`;
    }
    const v = carView();
    const R = M.car.result;
    const changed = Object.values(M.car.changed).reduce((s, cells) => s + cells.length, 0);
    const ep = episode();
    const rows = ENTRY0.map((name, k) => {
      const marks = `${k === act.pred ? 'pred' : ''}${k === frame.action[0] ? ' truth' : ''}`;
      return `<li class="${marks.trim()}"><span class="name">${name}</span><span class="bar"><i style="width:${(100 * act.probs[k]).toFixed(1)}%"></i></span><span class="p">${pct(act.probs[k], 0)}</span></li>`;
    }).join('');
    const last = sel.frame === c.steps - 1;
    return `
      <h2>MIT's car policy · ${v === 'trained' ? 'trained by PPO' : v === 'edited' ? "after Astra's repair" : 'what Astra changed'}</h2>
      <dl class="facts">
        <div><dt>Changed by Astra</dt><dd>${n(changed)}<small>${pct(changed / 171664)} of 171,664 weights</small></dd></div>
        <div><dt>Reaches the goal</dt><dd>${pct(R.before.success, 0)} → ${pct(R.after.success, 0)}<small>before → after, ${n(R.after.episodes)} test episodes</small></dd></div>
        <div><dt>This episode</dt><dd>step ${sel.frame + 1}<small>of ${c.steps} · ${last ? `ends: ${ep.outcome === 'success' ? '<b class="ok">success</b>' : ep.outcome}` : `ends in ${ep.outcome}`}</small></dd></div>
      </dl>
      <h3>The scene it reads</h3>
      ${sceneSvg(act.x)}
      <h3>Action entry 0, the wheels' motors</h3>
      <ol class="bars">${rows}</ol>
      <p class="legend"><span class="dot"></span>the value it sampled at this step. Entries 1 to 5 drive nothing in this scene.</p>`;
  }

  const only = (text) => `<div class="aside">${text}</div>`;
  const weldsAside = () => only(`<h3>Compare: the same method on the welds</h3>
    <p>Astra's five weld networks have 308 to 473 non-zero numbers out of 99,959 (under 0.5%) and score 2.2% to
    14.4% on 9 conditions, where guessing scores 11.1%. The game prompt contains the game's exact source code, so the meaning of
    every input and every rule is given. The weld prompt describes the process but says nothing about how the nine
    conditions differ.</p>`);
  const layerOneComments = () => {
    const seen = new Set();
    for (const g of reply().groups) if (g.statements.some((s) => s.tensor === 'W1' || s.tensor === 'b1')) seen.add(g.section);
    return [...seen].filter(Boolean);
  };
  const embedUnits = () => {
    const W = car.edited.W[0];
    return [6, 7, 8].map((u) => {
      const rows = [];
      for (let i = 0; i < 679; i++) if (W[i * 128 + u] !== 0) rows.push(i);
      return `<li>unit ${u} ← ${rows.map((i) => `x[${i}], ${escapeHtml(obsName(i))}`).join('; ')}${car.edited.b[0][u] ? `, bias ${fmt(car.edited.b[0][u])}` : ''}</li>`;
    }).join('');
  };

  const STEPS = [
    {
      title: "MIT's own networks",
      get frame() { return `${prefix()}All`; },
      focus: null,
      body: () => (isCar()
        ? `<p>These are the networks from MIT CSAIL's post itself, exactly as MIT logged them. In their
            <b>Experiment 4</b>, a car policy trained by PPO for 1 billion steps never reaches its goal. GPT-6 Astra
            read all 171,664 of its weights as text, plus one recorded episode, explained why it fails, and repaired
            it with ${reply().statements} statements in one reply: no tools, no simulator, no feedback.</p>
          <p>Before the repair it reaches the goal in ${pct(M.car.result.before.success, 0)} of 256 episodes; after it,
            in <b>${pct(M.car.result.after.success, 0)} of ${n(M.car.result.after.episodes)}</b>.</p>
          <p class="next">The Network menu above the 3D view switches between the trained policy, the repaired one,
            and only what Astra changed. ▶ plays the episode.</p>`
        : `<p>These are the networks from MIT CSAIL's post itself, exactly as MIT logged them. In their
            <b>Experiment 5</b>, GPT-6 Astra was given a game's rules, its full source code and the network's shape,
            and wrote every weight of a 600 → 32 → 32 → 6 policy in one reply: no data, no gradients, no game play, no
            tools.</p>
          ${only(`<h3>On screen: ${game().meta.name}</h3>
            <p>It sets <b>${n(game().meta.nonzero)} of 20,486</b> numbers (${pct(game().meta.nonzero / 20486)}); all
            the others stay 0. MIT scored it <b>${scores(game().meta)}</b> points per episode over 1,000 episodes.
            Networks of the same shape trained by PPO score ${game().meta.scores.ppo[0]} after 100 million steps and
            ${game().meta.scores.ppo[1]} after 500 million.</p>`)}
          ${weldsAside()}
          <p class="next">Switch games with the Network menu above the 3D view. ▶ plays the episode.</p>`),
    },
    {
      title: 'The input',
      get frame() { return `${prefix()}Input`; },
      focus: ['x', 'W1'],
      body: () => (isCar()
        ? `<p>At every step the policy reads <b>679 numbers</b> describing the scene. For every shape: its position,
            velocity, angle, role (green, blue or red), size and outline. For every joint: its motor settings.
            Polygons fill inputs 0 to 233, circles 234 to 305, joints 306 to 629, thrusters 630 to 677, and the last
            one is gravity. Most slots are empty in this scene.</p>
          <p>The strip above <code>embed.W</code> is this step's input. The panel draws the scene from the same
            numbers. Hover a cell to see which shape and which number it is.</p>`
        : `<p>Each screen is <b>600 numbers</b>, each 0 or 1: a 10 × 10 grid with 6 channels. Input
            i = (row × 10 + column) × 6 + channel. In ${game().meta.name}:</p>
          <ul>${game().meta.channels.map((what, ch) => `<li>channel ${ch} (${M.channel_names[ch]}): ${what ? escapeHtml(what.replace(/_/g, ' ')) : 'always empty'}</li>`).join('')}</ul>
          <p>The strip above W1 is this screen, one block of 60 inputs per grid row. The panel shows it as a
            picture.</p>`),
    },
    {
      title: 'Layer 1: what each unit looks at',
      get frame() { return `${prefix()}L1`; },
      focus: ['x', 'W1', 'b1', 'z1', 'h1'],
      body: () => (isCar()
        ? `<p><code>embed.W</code> is where the trained policy reads the scene. Astra kept almost all of it and
            rewired three of its 128 units. It zeroed their weights, then gave each one input:</p>
          <ul>${embedUnits()}</ul>
          <p>So three units now carry the car's position and speed directly. Choose “only what Astra changed” to see
            nothing but these edits.</p>`
        : `<p>A column of W1 is one first-layer unit. Its 600 weights say which cells of which channel push it up
            (blue) or down (orange). The panel turns the column back into the 10 × 10 grid, one grid per channel.
            Hover a cell of W1 to pick its unit.</p>
          <p>Astra's own comments for this layer, written in its reply:</p>
          <ul>${layerOneComments().slice(0, 5).map((t) => `<li>${escapeHtml(t)}</li>`).join('')}</ul>
          ${layerOneComments().length > 5 ? `<p class="meta">…and ${layerOneComments().length - 5} more, in “Astra's reply” above.</p>` : ''}`),
    },
    {
      title: 'The middle layers',
      get frame() { return `${prefix()}Mid`; },
      get focus() { return isCar() ? ['h1', 'W2', 'b2', 'z2', 'h2', 'W3', 'b3', 'z3', 'h3', 'W4', 'b4', 'z4', 'h4', 'W5', 'b5', 'z5', 'h5', 'W6', 'b6', 'z6', 'h6'] : ['h1', 'W2', 'b2', 'z2', 'h2']; },
      body: () => (isCar()
        ? `<p>In <code>actor0</code> to <code>actor4</code>, Astra took over a few units at the end of each layer:
            122 to 127 in actor0, 124 to 127 in actor1, then 126 and 127. It set their links to the other units to 0
            and wired them into a small controller of its own, with large weights such as 80 and biases such as
            −876. The rest keeps its trained weights, minus its links to these units.</p>
          <p>In Astra's words, the override backs right from rest, switches to leftward drive around body x = 3.8,
            and switches off once the green wheel's centre passes x = 0.95.</p>
          <p>${{
            trained: "This is the network before the repair. Choose “after Astra's repair” to see the controller.",
            edited: "The bars keep the trained network's scale, so the new weights stand at full height. Hover a cell to see the statement that set it, and its trained value.",
            changes: 'Only the weights Astra changed are drawn, as new value minus old; everything else lies flat. They sit along the edges of each block, in the last rows and columns.',
          }[carView()]}</p>`
        : `<p>W2 combines the first-layer units into decisions. Every weight Astra did not set stays 0: here
            ${n(game().meta.nonzero_by_tensor.W2)} of 1,024 are set.</p>
          <p>Hover a cell to see which of Astra's statements wrote it, or click it to open the statement in
            “Astra's reply”.</p>`),
    },
    {
      title: 'The output',
      get frame() { return `${prefix()}L${depth()}`; },
      get focus() { const L = depth(); return [`h${L - 1}`, `W${L}`, `b${L}`, `z${L}`, 'p']; },
      body: () => {
        if (isCar()) {
          return `<p><code>actor_head</code> gives 16 logits for 6 action entries, each with its own softmax. Only
            entry 0 drives anything here: both wheel motors, as off, +1 or −1.</p>
          <p>Astra's two override units add −100 or +200 to entry 0's three logits, and its new bias offsets cancel
            them while the override is off. The yellow strip is the probability of each value at this step.</p>`;
        }
        const b = game().net.b[2];
        return `<p>W3 turns the second layer into 6 logits, one per action. The action is sampled from their
            softmax, and 10% of the time the game repeats the previous action instead.</p>
          <p>Astra's output biases in ${game().meta.name}: ${M.actions.map((a, k) => `${a} ${fmt(b[k])}`).join(', ')}.
            A logit 100 below another is a probability of about e<sup>−100</sup>: never chosen.</p>`;
      },
    },
    {
      title: 'How they did',
      get frame() { return `${prefix()}All`; },
      focus: null,
      body: () => (isCar()
        ? `<div class="table"><table class="compare"><thead><tr><th></th><th>Trained by PPO</th><th>After Astra's repair</th></tr></thead><tbody>
            <tr><th>Reaches the goal</th><td>${pct(M.car.result.before.success, 0)}<small>256 episodes</small></td><td>${pct(M.car.result.after.success, 0)}<small>${n(M.car.result.after.episodes)} test episodes</small></td></tr>
            <tr><th>The episode shown here</th><td>${D.frames.episodes.trained.outcome}<small>after ${D.frames.episodes.trained.steps} steps</small></td><td>${D.frames.episodes.edited.outcome}<small>after ${D.frames.episodes.edited.steps} steps</small></td></tr>
            <tr><th>Weights</th><td>171,664</td><td>${n(Object.values(M.car.changed).reduce((s, c) => s + c.length, 0))} changed</td></tr>
          </tbody></table></div>
          <p class="meta">Success rates from MIT's logged evaluation. The episodes shown were replayed here with the
            replication's Kinetix environment (seed ${D.frames.seed}).</p>
          ${only(`<h3>What it had to work with</h3><p>The prompt held every weight, an episode showing the failure,
            and the exact scene. The repair is a short controller written into weights, in a few units taken over from
            the trained network.</p>`)}`
        : `<div class="table"><table class="compare"><thead><tr><th></th>${M.games.map((g) => `<th>${g.name}</th>`).join('')}</tr></thead><tbody>
            <tr><th>MIT's network, MIT's score</th>${M.games.map((g) => `<td>${scores(g)}</td>`).join('')}</tr>
            <tr><th>The same network, our scoring</th>${M.games.map((g) => `<td>${g.scores.ours[0]} <small>± ${g.scores.ours[1]}</small></td>`).join('')}</tr>
            <tr><th>Our 5 new replies</th>${M.games.map((g) => `<td>${(g.scores.replies.reduce((s, v) => s + v, 0) / 5).toFixed(1)}<small>${g.scores.replies.join(', ')}</small></td>`).join('')}</tr>
            <tr><th>PPO, 100M / 500M steps</th>${M.games.map((g) => `<td>${g.scores.ppo[0]} / ${g.scores.ppo[1]}</td>`).join('')}</tr>
            <tr><th>Non-zero weights</th>${M.games.map((g) => `<td>${n(g.nonzero)}<small>of 20,486</small></td>`).join('')}</tr>
          </tbody></table></div>
          <p class="meta">Points per episode, mean over 1,000 episodes (± standard error). MIT's and PPO's numbers are
            from MIT's post; the other rows are the replication in the research repository
            (<code>experiments/exp5_minatar</code>).</p>
          ${weldsAside()}`),
    },
  ];

  // ------------------------------------------------------------------ Astra's reply

  function doc(root) {
    const r = reply();
    const key = sel.net === 'car-trained' || sel.net === 'car-edited' || sel.net === 'car-changes' ? 'car' : sel.net;
    if (root.dataset.net !== key) {
      const src = M.source;
      const promptPath = isCar() ? 'exp4_fix_car/runs/oneshot_repair/reply/prompt.txt' : `exp5_write_from_scratch/runs/${sel.net}/message.txt`;
      const gave = isCar()
        ? `<ul class="md"><li>The task: change the weights by hand so the car succeeds; first explain why the trained
            network fails, then fix only what causes it.</li><li>The scene: every shape, joint and motor, the action
            format and the observation layout.</li><li>One recorded episode, every 16 steps, with the policy's
            probabilities, and the result over 256 episodes: 0 successes.</li><li>All 171,664 actor weights as
            integers times a scale per tensor (${n(r.input_tokens)} input tokens in all).</li><li>No tools, no
            simulator, nothing to run, one reply, no feedback.</li></ul>`
        : `<ul class="md"><li>The game's rules in a paragraph, and the game's full source code (MinAtar 1.0.15).</li>
            <li>The input layout (10 × 10 × 6), the action meanings and the sticky actions.</li><li>The network:
            600 → 32 → 32 → 6 with tanh, every weight starting at 0.</li><li>How it is scored, and the edits format.</li>
            <li>No data, no gradients, no game play, no tools, one reply.</li></ul>`;
      root.innerHTML = `
        <div class="notice"><b>Exactly what MIT logged.</b> From MIT CSAIL's
          <a href="${src.post}" target="_blank" rel="noopener">Sparks of Weight-Space Programming</a>, logs at
          <a href="${src.url}" target="_blank" rel="noopener">${escapeHtml(src.repo)}</a> (commit
          <code>${src.commit.slice(0, 7)}</code>, ${escapeHtml(src.license)}). Replayed with the research
          repository's reader, the reply's statements rebuild MIT's network exactly.</div>
        <details class="part" data-part="gave">
          <summary><i>1</i>What MIT gave Astra<small>the prompt, in short</small></summary>
          ${gave}
          <p class="meta">The full prompt is in MIT's repository:
            <a href="https://github.com/${src.repo}/blob/${src.commit}/${promptPath}" target="_blank" rel="noopener">${escapeHtml(promptPath)}</a>.
            ${isCar() ? '' : 'It quotes MinAtar\'s source (GPL-3.0), so it is linked, not copied.'}</p>
        </details>
        <details class="part" data-part="plan" open>
          <summary><i>2</i>Astra's ${isCar() ? 'diagnosis and plan' : 'plan'}<small>its reply, before the numbers</small></summary>
          <div class="md">${markdown(r.design)}</div>
        </details>
        <details class="part" data-part="edits" open>
          <summary><i>3</i>The edits, group by group<small>${r.groups.length} groups, ${r.statements} statements</small></summary>
          <div id="mit-group"></div>
        </details>
        ${r.after ? `<details class="part" data-part="after" open><summary><i>4</i>Astra's closing note<small>after the edits</small></summary><div class="md">${markdown(r.after)}</div></details>` : ''}
        <details class="part" data-part="run">
          <summary><i>${r.after ? 5 : 4}</i>How the reply was made<small>from MIT's log</small></summary>
          <ul class="md"><li>GPT-6 Astra, reasoning effort xhigh, through <code>codex exec</code>.</li>
            <li>${n(r.output_tokens)} output tokens, ${n(r.reasoning_tokens)} of them reasoning; ${r.seconds} s.</li>
            <li>Tools used: ${r.tool_free ? 'none' : 'some'}.</li></ul>
        </details>`;
      root.dataset.net = key;
    }
    renderGroup(root);
  }

  function renderGroup(root) {
    const host = root.querySelector('#mit-group');
    const r = reply();
    sel.group = Math.min(sel.group, r.groups.length - 1);
    const g = r.groups[sel.group];
    const cells = {};
    for (const s of g.statements) {
      if (!s.read) continue;
      const [layer, part] = isCar() ? s.tensor.split('.') : [s.tensor.slice(1), s.tensor[0]];
      const k = isCar() ? CAR_LAYERS.indexOf(layer) + 1 : +layer;
      (cells[`${prefix()}${part}${k}`] ??= []).push(...s.cells);
    }
    const total = Object.values(cells).reduce((s, c) => s + c.length, 0);
    host.innerHTML = `
      <div class="stepper">
        <button class="nav" data-go="-1" ${sel.group === 0 ? 'disabled' : ''} aria-label="Previous group">&lsaquo;</button>
        <span>Group ${sel.group + 1} of ${r.groups.length}</span>
        <button class="nav" data-go="1" ${sel.group === r.groups.length - 1 ? 'disabled' : ''} aria-label="Next group">&rsaquo;</button>
      </div>
      ${g.notes?.length ? `<p class="notes">${g.notes.map(escapeHtml).join('<br>')}</p>` : ''}
      <h4>${escapeHtml(isCar() ? `${g.section}: ${g.statements.length} statement${g.statements.length === 1 ? '' : 's'}` : g.section || 'Untitled group')}</h4>
      ${g.sub ? `<p class="sub">${escapeHtml(g.sub)}</p>` : ''}
      <p class="meta">${isCar() ? 'The reply has no comments in its edits block; its blank lines separate the layers.' : "Astra's own comment, written in the edits block."} The statements:</p>
      <pre class="code wide">${g.statements.map((s) => `${escapeHtml(s.text)}${s.comment ? `   <span class="c"># ${escapeHtml(s.comment)}</span>` : ''}`).join('\n')}</pre>
      <p class="check-line"><span class="ok">&#10003;</span> These statements are the last to write the ${n(total)}
        cell${total === 1 ? '' : 's'} lit in the 3D view.</p>`;
    const mask = Object.fromEntries(Object.keys(view.blocks).filter((id) => id[0] === prefix() && /^[Wb]\d$/.test(id.slice(1))).map((id) => [id, cells[id] ?? []]));
    view.setMask(mask);
    const [id, list] = Object.entries(cells).sort((a, b) => b[1].length - a[1].length)[0] ?? [];
    if (sel.fly !== false && id) actions.flyToCells(id, list);
    sel.fly = true;
  }

  function bindDoc(root) {
    root.addEventListener('click', (e) => {
      const go = e.target.closest('[data-go]');
      if (!go) return;
      sel.group += +go.dataset.go;
      renderGroup(root);
    });
  }

  /** The page's forward pass must give the logits numpy gives, or nothing on screen can be trusted. */
  function selfCheck() {
    const close = (got, want, what) => {
      const worst = want.reduce((s, v, k) => Math.max(s, Math.abs(v - got[k])), 0);
      if (worst > 1e-3 + 1e-5 * Math.max(...want.map(Math.abs))) throw new Error(`${what}: forward pass off by ${worst}`);
    };
    for (const g of M.games) {
      const x = new Float32Array(600);
      for (const i of g.episode.frames[g.check.frame].on) x[i] = 1;
      close(forward(games[g.id].net, x, false).logits, g.check.logits, g.name);
    }
    if (M.car.check && D.frames) {
      for (const name of ['trained', 'edited']) {
        const k = M.car.check[name].frame;
        close(forward(car[name], D.frames.x.subarray(k * 679, (k + 1) * 679), true).logits, M.car.check[name].logits, `car ${name}`);
      }
    }
  }

  function setNet(id) {
    sel.net = id;
    sel.frame = Math.min(sel.frame, frameCount() - 1);
    sel.unit = null;
    sel.group = 0;
    sceneBox = null;
    if (sel.unit === null && !isCar()) {
      const W = game().net.W[0];
      sel.unit = [...Array(32).keys()].find((u) => [...Array(600).keys()].some((i) => W[i * 32 + u] !== 0)) ?? 0;
    }
  }
  setNet(sel.net);

  return {
    STEPS, render, glance, describe, linked, hover, select, doc, bindDoc, buildLabels, selfCheck, setNet, layout,
    frameCount, allFrame: () => `${prefix()}All`, isCar,
  };
}
