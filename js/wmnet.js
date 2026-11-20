// The world-model pilot's networks, run in the page: the JEPA encoder (a 4-layer pre-norm transformer over 40 patch
// tokens), its predictor, and the logistic-regression head of phase 2. Plain loops over Float32Arrays, written to match
// PyTorch's nn.TransformerEncoderLayer(norm_first=True, activation="gelu", dropout=0) and nn.MultiheadAttention. The
// page checks every number against outputs computed in Python before it shows anything (tools/world_model.py).

const EPS = 1e-5; // nn.LayerNorm and F.layer_norm

// float16 -> float32, by table
const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 31;
    const f = h & 1023;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function decodeHalf(buffer, byteOffset, count) {
  const u = new Uint16Array(buffer, byteOffset, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = HALF[u[i]];
  return out;
}

/** y (n x outD) = x (n x inD) W^T + b, with W stored out x in (PyTorch's layout). */
export function linear(x, n, inD, W, b, outD) {
  const y = new Float32Array(n * outD);
  for (let i = 0; i < n; i++) {
    const xi = i * inD;
    for (let o = 0; o < outD; o++) {
      let s = b ? b[o] : 0;
      const wo = o * inD;
      for (let j = 0; j < inD; j++) s += x[xi + j] * W[wo + j];
      y[i * outD + o] = s;
    }
  }
  return y;
}

export function layerNorm(x, n, d, g = null, b = null) {
  const y = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = 0; j < d; j++) m += x[i * d + j];
    m /= d;
    let v = 0;
    for (let j = 0; j < d; j++) v += (x[i * d + j] - m) ** 2;
    const r = 1 / Math.sqrt(v / d + EPS);
    for (let j = 0; j < d; j++) y[i * d + j] = (x[i * d + j] - m) * r * (g ? g[j] : 1) + (b ? b[j] : 0);
  }
  return y;
}

// erf by Abramowitz and Stegun 7.1.26 (error below 1.5e-7), for the exact GELU PyTorch uses
function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1 - p * Math.exp(-a * a));
}
const gelu = (v) => 0.5 * v * (1 + erf(v / Math.SQRT2));

const add = (a, b) => {
  const y = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
  return y;
};

/** nn.MultiheadAttention without masks: also returns the attention weights, heads x n x n (rows: who looks). */
function attention(x, n, d, H, P) {
  const qkv = linear(x, n, d, P.inW, P.inB, 3 * d);
  const hd = d / H;
  const scale = 1 / Math.sqrt(hd);
  const attn = new Float32Array(H * n * n);
  const ctx = new Float32Array(n * d);
  const row = new Float32Array(n);
  for (let h = 0; h < H; h++) {
    const q0 = h * hd;
    const k0 = d + h * hd;
    const v0 = 2 * d + h * hd;
    for (let i = 0; i < n; i++) {
      let max = -Infinity;
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let c = 0; c < hd; c++) s += qkv[i * 3 * d + q0 + c] * qkv[j * 3 * d + k0 + c];
        row[j] = s * scale;
        if (row[j] > max) max = row[j];
      }
      let sum = 0;
      for (let j = 0; j < n; j++) {
        row[j] = Math.exp(row[j] - max);
        sum += row[j];
      }
      const a = h * n * n + i * n;
      for (let j = 0; j < n; j++) attn[a + j] = row[j] / sum;
      for (let c = 0; c < hd; c++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += attn[a + j] * qkv[j * 3 * d + v0 + c];
        ctx[i * d + q0 + c] = s;
      }
    }
  }
  return { out: linear(ctx, n, d, P.outW, P.outB, d), attn };
}

/** One pre-norm transformer layer: x + attention(norm1(x)), then + linear2(gelu(linear1(norm2(x)))). */
function layer(x, n, d, H, P) {
  const { out, attn } = attention(layerNorm(x, n, d, P.n1g, P.n1b), n, d, H, P);
  const x1 = add(x, out);
  const f = linear(layerNorm(x1, n, d, P.n2g, P.n2b), n, d, P.l1W, P.l1b, 2 * d);
  for (let i = 0; i < f.length; i++) f[i] = gelu(f[i]);
  return { x: add(x1, linear(f, n, 2 * d, P.l2W, P.l2b, d)), attn };
}

const pick = (x, d, rows) => {
  const y = new Float32Array(rows.length * d);
  rows.forEach((r, i) => y.set(x.subarray(r * d, (r + 1) * d), i * d));
  return y;
};

/** The tensors of one network, by name, as views into the decoded snapshot. */
function bind(meta, flat, part, nLayers) {
  const t = {};
  for (const s of meta.tensors) {
    if (!s.name.startsWith(`${part}.`)) continue;
    const size = s.shape.reduce((a, b) => a * b, 1);
    t[s.name.slice(part.length + 1)] = flat.subarray(s.offset, s.offset + size);
  }
  const layers = [];
  for (let l = 0; l < nLayers; l++) {
    const g = (k) => t[`blocks.layers.${l}.${k}`];
    layers.push({
      n1g: g('norm1.weight'), n1b: g('norm1.bias'), inW: g('self_attn.in_proj_weight'), inB: g('self_attn.in_proj_bias'),
      outW: g('self_attn.out_proj.weight'), outB: g('self_attn.out_proj.bias'), n2g: g('norm2.weight'), n2b: g('norm2.bias'),
      l1W: g('linear1.weight'), l1b: g('linear1.bias'), l2W: g('linear2.weight'), l2b: g('linear2.bias'),
    });
  }
  return { t, layers };
}

export function createModel(meta, weightBuffer, weldBuffer) {
  const M = meta.model;
  const nP = M.n_patches;
  const patchLen = M.n_ch * M.patch;
  const posE = Float32Array.from(meta.pos.encoder.flat());
  const posP = Float32Array.from(meta.pos.predictor.flat());
  const snaps = new Map();

  function snapshot(k) {
    if (!snaps.has(k)) {
      const flat = decodeHalf(weightBuffer, k * meta.per_snapshot * 2, meta.per_snapshot);
      snaps.set(k, {
        flat,
        encoder: bind(meta, flat, 'encoder', M.depth),
        predictor: bind(meta, flat, 'predictor', M.p_depth),
        target: bind(meta, flat, 'target', M.depth),
      });
    }
    return snaps.get(k);
  }

  // the welds: frames (n x 24 x 1000), then envelopes (n x 4 x 1000 x 2), both float16; then the test welds' summaries
  const nW = meta.welds.length;
  const frameCount = nW * M.n_ch * M.n_frames;
  const envCount = nW * meta.sensors.length * M.n_frames * 2;
  const frames = decodeHalf(weldBuffer, 0, frameCount);
  const env = decodeHalf(weldBuffer, frameCount * 2, envCount);
  const nTest = meta.test_labels.length;
  const summaries = new Float32Array(weldBuffer, (frameCount + envCount) * 2, 2 * nTest * 2 * M.dim);
  const summary = (arm, i) => summaries.subarray(((arm === 'B' ? 0 : nTest) + i) * 2 * M.dim, ((arm === 'B' ? 0 : nTest) + i + 1) * 2 * M.dim);

  const weldFrames = (w) => frames.subarray(w * M.n_ch * M.n_frames, (w + 1) * M.n_ch * M.n_frames);
  const weldEnv = (w, s) => env.subarray((w * meta.sensors.length + s) * M.n_frames * 2, (w * meta.sensors.length + s + 1) * M.n_frames * 2);

  /** 40 patches x 600: patch i holds channel c's frames i*25 .. i*25+24 at c*25 + t (wm.patches). */
  const patchCache = new Map();
  function patches(w) {
    if (!patchCache.has(w)) {
      const F = weldFrames(w);
      const P = new Float32Array(nP * patchLen);
      for (let i = 0; i < nP; i++)
        for (let c = 0; c < M.n_ch; c++)
          for (let t = 0; t < M.patch; t++) P[i * patchLen + c * M.patch + t] = F[c * M.n_frames + i * M.patch + t];
      patchCache.set(w, P);
    }
    return patchCache.get(w);
  }

  /** Embed the patches, keep some (or all), run the layers: every layer's tokens and attention, and the output. */
  function encode(net, w, keep = null) {
    const d = M.dim;
    let h = linear(patches(w), nP, patchLen, net.t['embed.weight'], net.t['embed.bias'], d);
    for (let i = 0; i < h.length; i++) h[i] += posE[i];
    const embedded = h;
    if (keep) h = pick(h, d, keep);
    const n = keep ? keep.length : nP;
    const layers = [h];
    const attn = [];
    for (const P of net.layers) {
      const r = layer(h, n, d, M.heads, P);
      h = r.x;
      layers.push(h);
      attn.push(r.attn);
    }
    return { n, rows: keep ?? [...Array(nP).keys()], embedded, layers, attn, out: layerNorm(h, n, d, net.t['norm.weight'], net.t['norm.bias']) };
  }

  /** The predictor: context tokens at their places, a learned placeholder (plus position) at each hidden patch. */
  function predict(net, ctxOut, ctxIdx, tgtIdx) {
    const d = M.dim;
    const pd = M.p_dim;
    const nc = ctxIdx.length;
    const nt = tgtIdx.length;
    const z = new Float32Array((nc + nt) * pd);
    z.set(linear(ctxOut, nc, d, net.t['inp.weight'], net.t['inp.bias'], pd));
    for (let i = 0; i < nc; i++) for (let c = 0; c < pd; c++) z[i * pd + c] += posP[ctxIdx[i] * pd + c];
    for (let j = 0; j < nt; j++) for (let c = 0; c < pd; c++) z[(nc + j) * pd + c] = net.t.mask[c] + posP[tgtIdx[j] * pd + c];
    let h = z;
    for (const P of net.layers) h = layer(h, nc + nt, pd, M.heads, P).x;
    const last = layerNorm(h.subarray(nc * pd), nt, pd, net.t['norm.weight'], net.t['norm.bias']);
    return linear(last, nt, pd, net.t['out.weight'], net.t['out.bias'], d);
  }

  const gameCache = new Map();
  /** One round of the game at snapshot k: what the encoder sees, the predictor's guesses, the answers, the errors. */
  function game(k, w, hidden) {
    const key = `${k}|${w}|${hidden.join(',')}`;
    if (gameCache.has(key)) return gameCache.get(key);
    const S = snapshot(k);
    const set = new Set(hidden);
    const ctx = [...Array(nP).keys()].filter((p) => !set.has(p));
    const tgt = [...hidden].sort((a, b) => a - b);
    const seen = encode(S.encoder, w, ctx);
    const guess = tgt.length ? predict(S.predictor, seen.out, ctx, tgt) : new Float32Array(0);
    const whole = encode(S.target, w);
    const answer = layerNorm(pick(whole.out, M.dim, tgt), tgt.length, M.dim);
    const err = tgt.map((_, j) => {
      let s = 0;
      for (let c = 0; c < M.dim; c++) s += Math.abs(guess[j * M.dim + c] - answer[j * M.dim + c]);
      return s / M.dim;
    });
    let zeros = 0;
    for (let i = 0; i < answer.length; i++) zeros += Math.abs(answer[i]);
    const r = { ctx, tgt, seen, guess, answer, err, loss: err.length ? err.reduce((a, b) => a + b, 0) / err.length : null, zeros: answer.length ? zeros / answer.length : null };
    if (gameCache.size > 60) gameCache.delete(gameCache.keys().next().value);
    gameCache.set(key, r);
    return r;
  }

  /** Phase 2: the frozen encoder's 128 numbers for a weld: mean and (n - 1) spread of its 40 output tokens. */
  function pooled(net, w) {
    const e = encode(net, w);
    const d = M.dim;
    const out = new Float32Array(2 * d);
    for (let c = 0; c < d; c++) {
      let m = 0;
      for (let i = 0; i < nP; i++) m += e.out[i * d + c];
      m /= nP;
      let v = 0;
      for (let i = 0; i < nP; i++) v += (e.out[i * d + c] - m) ** 2;
      out[c] = m;
      out[d + c] = Math.sqrt(v / (nP - 1));
    }
    return { enc: e, pooled: out };
  }

  /** The head: standardize with the unlabeled welds' mean and std, then logits = coef z + bias, then softmax. */
  function head(arm, k, x) {
    const H = meta.heads[arm][String(k)];
    const S = meta.stats[arm];
    const z = Float32Array.from(x, (v, i) => (v - S.mu[i]) / S.sd[i]);
    const logits = H.coef.map((row, c) => row.reduce((s, w, i) => s + w * z[i], H.bias[c]));
    const max = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - max));
    const sum = e.reduce((a, b) => a + b, 0);
    return { z, logits, probs: e.map((v) => v / sum) };
  }

  /** The page's numbers must match PyTorch's on the same float16 weights, or nothing it draws can be trusted. */
  function selfCheck() {
    const C = meta.check;
    const last = meta.steps.length - 1;
    const hidden = C.target;
    const g = game(last, C.weld, hidden);
    const worst = (a, b) => b.flat().reduce((m, v, i) => Math.max(m, Math.abs(v - a[i])), 0);
    const gaps = {
      guess: worst(g.guess, C.predicted),
      answer: worst(g.answer, C.answers),
      trained: worst(pooled(snapshot(last).target, C.weld).pooled, C.pooled.B),
      random: worst(pooled(snapshot(0).encoder, C.weld).pooled, C.pooled.C),
    };
    const bad = Object.entries(gaps).filter(([, v]) => !(v < 2e-3));
    if (bad.length) throw new Error(`the world model's forward pass is off: ${bad.map(([k, v]) => `${k} ${v}`).join(', ')}`);
    return gaps;
  }

  return { meta, M, snapshot, patches, weldFrames, weldEnv, encode, predict, game, pooled, head, summary, selfCheck, nTest };
}
