// The network itself: loading the exported weights, the forward pass, and names for inputs and classes.

export const LAYERS = [528, 150, 100, 50, 9];
export const N_PARAMS = 99959;
/** Where each tensor starts in a weight file. */
export const OFFSETS = { W1: 0, b1: 79200, W2: 79350, b2: 94350, W3: 94450, b3: 99450, W4: 99500, b4: 99950 };

const STAT_NAMES = {
  entropy: 'entropy',
  mean: 'mean',
  median: 'median',
  p05: '5th percentile',
  p25: '25th percentile',
  p75: '75th percentile',
  p95: '95th percentile',
  rms: 'RMS',
  var: 'variance',
  std: 'standard deviation',
  zero_crossing_rate: 'zero-crossing rate',
  mean_crossing_rate: 'mean-crossing rate',
};

async function getBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

// ---------------------------------------------------------------- the lock of the public copy
//
// The public copy of this site (tools/build_public.mjs) holds every data file encrypted with AES-256-GCM. The key
// is not on the server: it travels in the part of the link after '#', which browsers never send. So the page opens
// for anyone who was given the full link and for nobody else.

const KEY_STORE = 'mlp-weight-visualizer-key';

/** Thrown when the data is locked and no key, or the wrong key, is at hand. */
export class LockedError extends Error {}

/** The key in a link (…#trained&k=KEY), or in whatever was pasted. */
export const keyIn = (text) => (text.match(/(?:^|[#&?])k=([A-Za-z0-9_-]{20,})/) ?? text.trim().match(/^([A-Za-z0-9_-]{20,})$/))?.[1] ?? null;

function storedKey() {
  try {
    return localStorage.getItem(KEY_STORE);
  } catch {
    return null; // private windows may refuse storage; the link still works
  }
}

function remember(key) {
  try {
    if (key) localStorage.setItem(KEY_STORE, key);
    else localStorage.removeItem(KEY_STORE);
  } catch {
    /* nothing to do */
  }
}

/** A function that reads one data file: straight from the server, or decrypted when the copy is locked. */
async function reader(base) {
  const lock = await fetch(`${base}lock.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!lock?.locked) return getBuffer;
  const text = keyIn(location.hash) ?? storedKey();
  if (!text) throw new LockedError('');
  let key;
  try {
    const b64 = text.replace(/-/g, '+').replace(/_/g, '/');
    const raw = Uint8Array.from(atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4)), (c) => c.charCodeAt(0));
    key = await crypto.subtle.importKey('raw', raw, 'AES-GCM', false, ['decrypt']);
  } catch {
    remember(null);
    throw new LockedError('That is not a key for this page.');
  }
  return async (url) => {
    const sealed = new Uint8Array(await getBuffer(`${url}.enc`)); // 12-byte iv, then ciphertext and tag
    try {
      const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: sealed.subarray(0, 12) }, key, sealed.subarray(12));
      remember(text);
      return plain;
    } catch {
      remember(null);
      throw new LockedError('This key does not open the page.');
    }
  };
}

/** One exported weight file -> {W: [W1..W4], b: [b1..b4]} as views into one Float32Array. */
export function parseWeights(buffer) {
  const flat = new Float32Array(buffer);
  if (flat.length !== N_PARAMS) throw new Error(`expected ${N_PARAMS} numbers, got ${flat.length}`);
  const W = [];
  const b = [];
  let o = 0;
  for (let k = 0; k < 4; k++) {
    const n = LAYERS[k] * LAYERS[k + 1];
    W.push(flat.subarray(o, o + n));
    o += n;
    b.push(flat.subarray(o, o + LAYERS[k + 1]));
    o += LAYERS[k + 1];
  }
  const count = (a) => {
    let c = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== 0) c++;
    return c;
  };
  return { flat, W, b, nonzeroW: W.map(count), nonzeroB: b.map(count), nonzero: count(flat) };
}

/** The same network with only the numbers at `cells` (indices into the 99,959) kept; every other one is zero. */
export function subsetNet(net, cells) {
  const flat = new Float32Array(N_PARAMS);
  for (const k of cells) flat[k] = net.flat[k];
  return parseWeights(flat.buffer);
}

export async function loadData(base = 'data/') {
  const read = await reader(base);
  const json = async (file) => JSON.parse(new TextDecoder().decode(await read(base + file)));
  const manifest = await json('manifest.json');
  const [trained, inputs, ...written] = await Promise.all([
    read(base + manifest.trained.file),
    read(base + manifest.inputs.file),
    ...manifest.written.replies.map((r) => read(base + r.file)),
  ]);
  const x = new Float32Array(inputs);
  const optional = (file) => json(file).catch((error) => {
    if (error instanceof LockedError) throw error;
    return null; // a copy exported without this part
  });
  const [provenance, few] = await Promise.all([optional('provenance.json'), optional('few_welds.json')]);
  return {
    manifest,
    provenance,
    few,
    trained: parseWeights(trained),
    written: written.map(parseWeights),
    inputs: manifest.inputs.items.map((meta, k) => ({ ...meta, x: x.subarray(k * 528, (k + 1) * 528) })),
  };
}

/** h1 = relu(x @ W1 + b1), ..., logits = h3 @ W4 + b4, then softmax. Ties go to the lowest class. */
export function forward(net, x) {
  const z = [];
  const h = [];
  let a = x;
  for (let k = 0; k < 4; k++) {
    const no = LAYERS[k + 1];
    const W = net.W[k];
    const acc = new Float64Array(no);
    for (let j = 0; j < no; j++) acc[j] = net.b[k][j];
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      if (ai === 0) continue;
      const row = i * no;
      for (let j = 0; j < no; j++) acc[j] += ai * W[row + j];
    }
    const zk = Float32Array.from(acc);
    z.push(zk);
    if (k < 3) {
      a = zk.map((v) => (v > 0 ? v : 0));
      h.push(a);
    }
  }
  const logits = z[3];
  let top = -Infinity;
  for (const v of logits) top = Math.max(top, v);
  const e = Array.from(logits, (v) => Math.exp(v - top));
  const sum = e.reduce((s, v) => s + v, 0);
  const probs = Float32Array.from(e, (v) => v / sum);
  let pred = 0;
  for (let c = 1; c < 9; c++) if (logits[c] > logits[pred]) pred = c;
  return { x, z, h, logits, probs, pred };
}

/** The input of layer k (1-based): x for the first layer, the previous ReLU output otherwise. */
export const layerInput = (act, k) => (k === 1 ? act.x : act.h[k - 2]);

/** Weight times input, cell by cell: what each cell of W_k adds to its unit for this input. */
export function contributions(net, act, k) {
  const W = net.W[k - 1];
  const a = layerInput(act, k);
  const no = LAYERS[k];
  const out = new Float32Array(W.length);
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    if (ai === 0) continue;
    const row = i * no;
    for (let j = 0; j < no; j++) out[row + j] = ai * W[row + j];
  }
  return out;
}

/** A robust "largest value": the q-quantile of the non-zero magnitudes (1 if everything is zero). */
export function scaleOf(values, q = 0.995) {
  const mags = [];
  for (let i = 0; i < values.length; i++) if (values[i] !== 0) mags.push(Math.abs(values[i]));
  if (!mags.length) return 1;
  const sorted = Float32Array.from(mags).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] || 1;
}

export function normalized(values, scale) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] / scale;
    out[i] = v > 1 ? 1 : v < -1 ? -1 : v;
  }
  return out;
}

const hz = (v) => (v >= 1000 ? `${+(v / 1000).toPrecision(2)} kHz` : `${Math.round(v)} Hz`);

/** Input index = 132 x sensor + 12 x band + statistic. */
export function describeFeature(i, manifest) {
  const sensor = Math.floor(i / 132);
  const band = Math.floor((i % 132) / 12);
  const stat = i % 12;
  const b = manifest.bands[band];
  return {
    sensor: manifest.sensors[sensor],
    band: b.name,
    range: `${hz(b.hz[0])} to ${hz(b.hz[1])}`,
    stat: STAT_NAMES[manifest.stats[stat]] ?? manifest.stats[stat],
  };
}

export function className(c, manifest, short = false) {
  const tool = manifest.tools[Math.floor(c / 3)];
  const surface = manifest.surfaces[c % 3];
  return short ? `${tool} / ${surface}` : `${tool} tool, ${surface} surface`;
}

export function fmt(v, digits = 3) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  const s = a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 0.001 ? v.toFixed(digits) : v.toExponential(1);
  return v > 0 ? `+${s}` : s;
}

export const pct = (p, digits = 1) => `${(100 * p).toFixed(digits)}%`;

/** A repeatable standard-normal vector (for the "random noise" input). */
export function noiseInput(seed, n = 528) {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  return out;
}
