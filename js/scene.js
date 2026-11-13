// The 3D view: every weight and every value is one instanced bar on a dark slab.
//
// Layout (ground plane, x to the right, z towards the viewer). Each matrix sits between the strip that feeds it and
// the strip it produces, so a unit's sum can be read straight along its row or column:
//
//   x  (528, along x)
//   W1 (528 across, 150 down)  b1 z1 h1 (along z)  W2 (150 down, 100 across)
//                                                  b2 z2 h2 (along x)
//                                                  W3 (100 across, 50 down)  b3 z3 h3 (along z)  W4 (50 down, 9 across)
//                                                                                                b4 logits softmax

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LAYERS } from './net.js';

export const BG = '#f1f2f4';
export const PALETTES = {
  weights: { pos: '#86c1ff', neg: '#ff8f50', zero: '#2c365c', plate: '#1b2240' },
  data: { pos: '#74e695', neg: '#ff74b3', zero: '#254231', plate: '#162a1d' },
  prob: { pos: '#ffcf5c', neg: '#ffcf5c', zero: '#453b22', plate: '#2b2413' },
};

const PITCH = [1, 1.5, 2.2, 3.6, 14]; // size of one index of each layer, in world units
const T = 10; // thickness of a strip
const TX = 14; // thickness of the input and softmax strips
const G = 4; // gap between neighbouring blocks
const IN_GROUPS = [[132, 6], [12, 0.6]]; // extra gaps between sensors and between wavelet bands
const NO_GROUPS = [[1e9, 0], [1e9, 0]];
const BASE = 0.3; // thickness of a zero cell
const H_MATRIX = 5.5;
const H_VECTOR = 12;
const HEIGHT_POWER = 0.8;
const SPARSE_BELOW = 0.03; // a matrix with fewer non-zero cells than this share gets markers
const MARKER_PX = 5;

const rgb = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
};
const cellGap = (pitch) => Math.min(0.16 * pitch, 0.5);
const smooth = (t) => t * t * (3 - 2 * t);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

const VERT = /* glsl */ `
uniform vec2 uOrigin;
uniform float uSwap;
uniform int uNj;
uniform float uNi;
uniform float uSweepJ;
uniform vec2 uPitch;
uniform vec2 uCell;
uniform vec2 uGroupA;
uniform vec2 uGroupB;
uniform float uHeight;
uniform float uBase;
uniform float uT;
uniform float uStagger;
uniform vec3 uHover;
uniform vec3 uPosA;
uniform vec3 uNegA;
uniform vec3 uZeroA;
uniform vec3 uPosB;
uniform vec3 uNegB;
uniform vec3 uZeroB;
uniform float uFade;
uniform vec3 uBg;
uniform float uMinPx;
uniform float uPxScale;
uniform float uMaskOn;
attribute float aMask;
attribute float aA;
attribute float aB;
varying vec3 vColor;

void main() {
  int id = gl_InstanceID;
  int ii = id / uNj;
  int jj = id - ii * uNj;
  float fi = float(ii);
  float fj = float(jj);

  // every change is a sweep from the old value (aA) to the new one (aB)
  float frac = uSweepJ > 0.5 ? fj / float(uNj) : fi / uNi;
  float t = clamp(uT * (1.0 + uStagger) - uStagger * frac, 0.0, 1.0);
  t = t * t * (3.0 - 2.0 * t);
  float v = mix(aA, aB, t);
  float m = min(abs(v), 1.0);
  float h = uBase + uHeight * pow(m, ${HEIGHT_POWER.toFixed(2)});

  vec3 cz = mix(uZeroA, uZeroB, t);
  vec3 cv = v >= 0.0 ? mix(uPosA, uPosB, t) : mix(uNegA, uNegB, t);
  float tint = pow(m, 0.55);
  if (uMinPx > 0.0 && m > 0.0) tint = max(tint, 0.6); // a marker must stand out from the empty slab
  vec3 col = mix(cz, cv, tint);

  if (uMaskOn > 0.5) {
    if (aMask > 0.5) col = mix(col, vec3(1.0), 0.22);
    else col = mix(col, cz, 0.72);
  }
  if (uHover.z > 0.5) {
    bool onI = uHover.x > -0.5 && abs(fi - uHover.x) < 0.5;
    bool onJ = uHover.y > -0.5 && abs(fj - uHover.y) < 0.5;
    bool both = (onI || uHover.x < -0.5) && (onJ || uHover.y < -0.5);
    if (both) col = mix(col, vec3(1.0), 0.45);
    else if (onI || onJ) col = mix(col, vec3(1.0), 0.10);
    else col = mix(col, cz, 0.55);
  }

  float nx = uSwap > 0.5 ? abs(normal.z) : abs(normal.x);
  float light = normal.y > 0.5 ? 1.0 : (nx > 0.5 ? 0.82 : 0.68);
  if (normal.y < 0.5) light *= mix(0.78, 1.0, position.y);
  vColor = mix(col * light, uBg, uFade);

  float gi = fi * uPitch.x + floor((fi + 0.5) / uGroupA.x) * uGroupA.y + floor((fi + 0.5) / uGroupB.x) * uGroupB.y;
  vec2 corner = vec2(gi, fj * uPitch.y);
  vec2 cell = uCell;
  if (uMinPx > 0.0 && m > 0.0 && (uMaskOn < 0.5 || aMask > 0.5)) {
    // a lone non-zero cell in an almost empty matrix never shrinks below uMinPx on screen
    vec2 c = corner + 0.5 * uCell;
    vec3 cw = uSwap > 0.5 ? vec3(uOrigin.x + c.y, 0.0, uOrigin.y + c.x) : vec3(uOrigin.x + c.x, 0.0, uOrigin.y + c.y);
    float depth = -(viewMatrix * vec4(cw, 1.0)).z;
    float unit = depth / uPxScale; // world units per pixel at this cell
    cell = max(uCell, vec2(uMinPx * unit));
    corner -= 0.5 * (cell - uCell);
    h = max(h, uBase + 1.6 * uMinPx * unit * step(0.001, uHeight));
  }
  vec2 loc = corner + vec2(position.x * cell.x, position.z * cell.y);
  vec3 world = uSwap > 0.5
    ? vec3(uOrigin.x + loc.y, position.y * h, uOrigin.y + loc.x)
    : vec3(uOrigin.x + loc.x, position.y * h, uOrigin.y + loc.y);
  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}
`;

const FRAG = /* glsl */ `
varying vec3 vColor;
void main() {
  gl_FragColor = vec4(vColor, 1.0);
}
`;

/** A unit cube on [0,1]^3 without its bottom face. */
function openBox() {
  const faces = [
    [[0, 1, 0], [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]]],
    [[1, 0, 0], [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]],
    [[-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]],
    [[0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]],
    [[0, 0, -1], [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]]],
  ];
  const position = [];
  const normal = [];
  const index = [];
  faces.forEach(([n, corners], f) => {
    for (const c of corners) {
      position.push(...c);
      normal.push(...n);
    }
    index.push(4 * f, 4 * f + 1, 4 * f + 2, 4 * f, 4 * f + 2, 4 * f + 3);
  });
  return { position: new Float32Array(position), normal: new Float32Array(normal), index };
}

class Block {
  constructor(def, box) {
    Object.assign(this, { groups: NO_GROUPS, swap: false, nj: 1, layout: 'mlp' }, def);
    this.isMatrix = this.nj > 1;
    this.cellI = this.pitchI - cellGap(this.pitchI);
    this.cellJ = this.isMatrix ? this.pitchJ - cellGap(this.pitchJ) : this.pitchJ;
    this.maxHeight = def.maxHeight ?? (this.isMatrix ? H_MATRIX : H_VECTOR);
    this.extI = this.posI(this.ni - 1) + this.pitchI;
    this.extJ = this.nj * this.pitchJ;
    const n = this.ni * this.nj;
    this.from = new Float32Array(n);
    this.to = new Float32Array(n);
    this.palette = PALETTES[this.family];
    this.fade = 0;
    this.fadeTarget = 0;
    this.anim = null;

    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setIndex(box.index);
    geometry.setAttribute('position', new THREE.BufferAttribute(box.position, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(box.normal, 3));
    this.aA = new THREE.InstancedBufferAttribute(this.from, 1);
    this.aB = new THREE.InstancedBufferAttribute(this.to, 1);
    geometry.setAttribute('aA', this.aA);
    geometry.setAttribute('aB', this.aB);
    this.aMask = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    geometry.setAttribute('aMask', this.aMask);
    geometry.instanceCount = n;

    const p = this.palette;
    this.uniforms = {
      uOrigin: { value: new THREE.Vector2(this.ox, this.oz) },
      uSwap: { value: this.swap ? 1 : 0 },
      uNj: { value: this.nj },
      uNi: { value: this.ni },
      uSweepJ: { value: this.isMatrix ? 1 : 0 },
      uPitch: { value: new THREE.Vector2(this.pitchI, this.pitchJ) },
      uCell: { value: new THREE.Vector2(this.cellI, this.cellJ) },
      uGroupA: { value: new THREE.Vector2(...this.groups[0]) },
      uGroupB: { value: new THREE.Vector2(...this.groups[1]) },
      uHeight: { value: this.maxHeight },
      uBase: { value: BASE },
      uT: { value: 1 },
      uStagger: { value: 0.6 },
      uHover: { value: new THREE.Vector3(-1, -1, 0) },
      uPosA: { value: rgb(p.pos) },
      uNegA: { value: rgb(p.neg) },
      uZeroA: { value: rgb(p.zero) },
      uPosB: { value: rgb(p.pos) },
      uNegB: { value: rgb(p.neg) },
      uZeroB: { value: rgb(p.zero) },
      uFade: { value: 0 },
      uBg: { value: rgb(BG) },
      uMinPx: { value: 0 },
      uMaskOn: { value: 0 },
      uPxScale: { value: 1000 },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;

    const r = this.rect();
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(r.x1 - r.x0 + 2, 0.8, r.z1 - r.z0 + 2),
      new THREE.MeshBasicMaterial({ color: p.plate }),
    );
    plate.position.set((r.x0 + r.x1) / 2, -0.42, (r.z0 + r.z1) / 2);
    this.plate = plate;
    this.plateColor = new THREE.Color(p.plate);
  }

  posI(i) {
    const [[sa, ga], [sb, gb]] = this.groups;
    return i * this.pitchI + Math.floor(i / sa) * ga + Math.floor(i / sb) * gb;
  }

  /** The last cell that starts at or before the distance u along the i axis. */
  floorI(u) {
    let lo = 0;
    let hi = this.ni - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.posI(mid) <= u) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** The cell under the distance u along the i axis, or -1 in a gap. */
  indexI(u) {
    if (u < 0 || u > this.extI) return -1;
    const i = this.floorI(u);
    return u - this.posI(i) <= this.cellI ? i : -1;
  }

  rect() {
    const [w, d] = this.swap ? [this.extJ, this.extI] : [this.extI, this.extJ];
    return { x0: this.ox, x1: this.ox + w, z0: this.oz, z1: this.oz + d };
  }

  /** World x, z of the centre of cell (i, j). */
  center(i, j = 0) {
    const u = this.posI(i) + this.cellI / 2;
    const w = j * this.pitchJ + this.cellJ / 2;
    return this.swap ? [this.ox + w, this.oz + u] : [this.ox + u, this.oz + w];
  }

  heightOf(index, factor) {
    return BASE + this.maxHeight * factor * Math.pow(Math.min(Math.abs(this.to[index]), 1), HEIGHT_POWER);
  }

  progress(now) {
    if (!this.anim) return 1;
    return Math.min(Math.max((now - this.anim.start) / this.anim.duration, 0), 1);
  }

  /** Sweep from what is on screen now to `values` (normalized to [-1, 1]). */
  set(values, palette, now, { duration = 900, delay = 0, stagger = 0.6 } = {}) {
    const u = this.uniforms;
    const next = PALETTES[palette];
    let same = next === this.palette;
    if (same) {
      for (let k = 0; k < values.length; k++) {
        if (values[k] !== this.to[k]) {
          same = false;
          break;
        }
      }
    }
    if (same) return false;

    const tGlobal = this.progress(now);
    if (tGlobal < 1) {
      const s = u.uStagger.value;
      const nAlong = this.isMatrix ? this.nj : this.ni;
      for (let k = 0; k < this.from.length; k++) {
        const along = this.isMatrix ? k % this.nj : k;
        const t = smooth(Math.min(Math.max(tGlobal * (1 + s) - (s * along) / nAlong, 0), 1));
        this.from[k] += (this.to[k] - this.from[k]) * t;
      }
      for (const key of ['Pos', 'Neg', 'Zero']) u[`u${key}A`].value.lerp(u[`u${key}B`].value, smooth(tGlobal));
    } else {
      this.from.set(this.to);
      for (const key of ['Pos', 'Neg', 'Zero']) u[`u${key}A`].value.copy(u[`u${key}B`].value);
    }
    this.to.set(values);
    let filled = 0;
    for (let k = 0; k < values.length; k++) if (values[k] !== 0) filled++;
    this.sparse = this.isMatrix && filled < SPARSE_BELOW * values.length;
    if (!this.sparse) u.uMinPx.value = 0; // markers come on only once the sweep to a sparse matrix has finished
    u.uPosB.value.copy(rgb(next.pos));
    u.uNegB.value.copy(rgb(next.neg));
    u.uZeroB.value.copy(rgb(next.zero));
    this.palette = next;
    this.aA.needsUpdate = true;
    this.aB.needsUpdate = true;
    u.uStagger.value = stagger;
    u.uT.value = 0;
    this.anim = { start: now + delay, duration };
    return true;
  }

  pick(o, d, factor) {
    const { x0, x1, z0, z1 } = this.rect();
    const top = BASE + this.maxHeight * factor;
    let t0 = 0;
    let t1 = Infinity;
    for (const [oo, dd, lo, hi] of [[o.x, d.x, x0, x1], [o.y, d.y, 0, top], [o.z, d.z, z0, z1]]) {
      if (Math.abs(dd) < 1e-9) {
        if (oo < lo || oo > hi) return null;
        continue;
      }
      let a = (lo - oo) / dd;
      let b = (hi - oo) / dd;
      if (a > b) [a, b] = [b, a];
      t0 = Math.max(t0, a);
      t1 = Math.min(t1, b);
      if (t0 > t1) return null;
    }
    const step = 0.3 * Math.min(this.cellI, this.cellJ);
    for (let t = t0; t <= t1 + step; t += step) {
      const px = o.x + d.x * t;
      const pz = o.z + d.z * t;
      const i = this.indexI(this.swap ? pz - this.oz : px - this.ox);
      if (i < 0) continue;
      const w = this.swap ? px - this.ox : pz - this.oz;
      const j = Math.floor(w / this.pitchJ);
      if (j < 0 || j >= this.nj || w - j * this.pitchJ > this.cellJ) continue;
      if (o.y + d.y * t <= this.heightOf(i * this.nj + j, factor)) return { id: this.id, i, j, t };
    }
    // the ray slipped between two bars: take the cell next to the point where it reaches the slab
    const tg = (BASE - o.y) / d.y;
    const px = o.x + d.x * tg;
    const pz = o.z + d.z * tg;
    if (!(tg > 0) || px < x0 || px > x1 || pz < z0 || pz > z1) return null;
    const w = this.swap ? px - this.ox : pz - this.oz;
    const j = Math.min(Math.max(Math.floor(w / this.pitchJ), 0), this.nj - 1);
    return { id: this.id, i: this.floorI(this.swap ? pz - this.oz : px - this.ox), j, t: tg };
  }
}

function buildBlocks(box) {
  const blocks = {};
  const add = (def) => (blocks[def.id] = new Block(def, box));
  const strip = (id, ni, ox, oz, swap, pitchI, family, thick = T) =>
    add({ id, ni, ox, oz, swap, pitchI, pitchJ: thick, family });

  add({ id: 'x', ni: 528, ox: 0, oz: -(G + TX), pitchI: 1, pitchJ: TX, groups: IN_GROUPS, family: 'data' });
  const w1 = add({ id: 'W1', ni: 528, nj: 150, ox: 0, oz: 0, pitchI: 1, pitchJ: PITCH[1], groups: IN_GROUPS, family: 'weights' });

  let x = w1.rect().x1 + G;
  strip('b1', 150, x, 0, true, PITCH[1], 'weights');
  strip('z1', 150, (x += T + G), 0, true, PITCH[1], 'data');
  strip('h1', 150, (x += T + G), 0, true, PITCH[1], 'data');
  const w2 = add({ id: 'W2', ni: 150, nj: 100, ox: (x += T + G), oz: 0, swap: true, pitchI: PITCH[1], pitchJ: PITCH[2], family: 'weights' });

  const x2 = w2.ox;
  let z = w2.rect().z1 + G;
  strip('b2', 100, x2, z, false, PITCH[2], 'weights');
  strip('z2', 100, x2, (z += T + G), false, PITCH[2], 'data');
  strip('h2', 100, x2, (z += T + G), false, PITCH[2], 'data');
  const w3 = add({ id: 'W3', ni: 100, nj: 50, ox: x2, oz: (z += T + G), pitchI: PITCH[2], pitchJ: PITCH[3], family: 'weights' });

  const z3 = w3.oz;
  x = w3.rect().x1 + G;
  strip('b3', 50, x, z3, true, PITCH[3], 'weights');
  strip('z3', 50, (x += T + G), z3, true, PITCH[3], 'data');
  strip('h3', 50, (x += T + G), z3, true, PITCH[3], 'data');
  const w4 = add({ id: 'W4', ni: 50, nj: 9, ox: (x += T + G), oz: z3, swap: true, pitchI: PITCH[3], pitchJ: PITCH[4], family: 'weights' });

  z = w4.rect().z1 + G;
  strip('b4', 9, w4.ox, z, false, PITCH[4], 'weights');
  strip('z4', 9, w4.ox, (z += T + G), false, PITCH[4], 'data');
  strip('p', 9, w4.ox, (z += T + G), false, PITCH[4], 'prob', TX);
  return blocks;
}

/**
 * The few-weld classifier (not an MLP): features across, classes down.
 *
 *   fx  one weld's 8 features
 *   fc  9 x 8 centroids          fd  9 x 8 terms ((value - centroid) / spread)^2     ft  their sum per class
 *   fs  8 spreads
 */
function buildFewBlocks(box) {
  const blocks = {};
  const PF = 30;
  const PC = 16;
  const TH = 16;
  const add = (def) => (blocks[def.id] = new Block({ layout: 'few', maxHeight: 14, ...def }, box));
  add({ id: 'fx', ni: 8, ox: 0, oz: -(G + TH), pitchI: PF, pitchJ: TH, family: 'data' });
  const fc = add({ id: 'fc', ni: 8, nj: 9, ox: 0, oz: 0, pitchI: PF, pitchJ: PC, family: 'weights' });
  add({ id: 'fs', ni: 8, ox: 0, oz: fc.rect().z1 + G, pitchI: PF, pitchJ: TH, family: 'weights' });
  const fd = add({ id: 'fd', ni: 8, nj: 9, ox: fc.rect().x1 + 26, oz: 0, pitchI: PF, pitchJ: PC, family: 'prob' });
  add({ id: 'ft', ni: 9, ox: fd.rect().x1 + G, oz: 0, swap: true, pitchI: PC, pitchJ: TH, family: 'prob' });
  return blocks;
}

export class NetworkView {
  constructor(stage) {
    this.stage = stage;
    this.canvas = stage.querySelector('canvas');
    this.labelLayer = stage.querySelector('.labels');
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(BG);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 2, 9000);
    this.heightFactor = 1;
    this.dirty = true;
    this.onPick = null;
    this.onFrame = null;

    const box = openBox();
    this.blocks = { ...buildBlocks(box), ...buildFewBlocks(box) };
    this.blockList = Object.values(this.blocks);
    this.layout = 'mlp';
    for (const b of this.blockList) {
      b.mesh.visible = b.plate.visible = b.layout === this.layout;
      this.scene.add(b.plate, b.mesh);
    }
    this.bgColor = new THREE.Color(BG);

    this.#buildDecal();
    this.#buildFrames();
    this.labels = [];
    this.#buildControls();
    this.#bindPointer();

    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
    this.snapTo('all');
    requestAnimationFrame((t) => this.#tick(t));
  }

  rect(...ids) {
    const out = { x0: Infinity, x1: -Infinity, z0: Infinity, z1: -Infinity };
    for (const id of ids) {
      const r = this.blocks[id].rect();
      out.x0 = Math.min(out.x0, r.x0);
      out.x1 = Math.max(out.x1, r.x1);
      out.z0 = Math.min(out.z0, r.z0);
      out.z1 = Math.max(out.z1, r.z1);
    }
    return out;
  }

  // ---------------------------------------------------------------- scene furniture

  #buildDecal() {
    const w1 = this.blocks.W1.rect();
    const fs = this.blocks.fs.rect();
    // where the title lies on the ground in each layout: x0, z0, width
    this.decalPlaces = {
      mlp: [w1.x0 + 14, w1.z1 + 66, w1.x1 - w1.x0 - 130], // leaves room for W1's name tag, stops short of W3's
      few: [fs.x0, fs.z1 + 84, this.blocks.ft.rect().x1 - fs.x0],
    };
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 640;
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }),
    );
    plane.rotation.x = -Math.PI / 2;
    this.scene.add(plane);
    this.decal = { canvas, texture, plane };
    this.#placeDecal();
  }

  #placeDecal() {
    const { canvas, plane } = this.decal;
    const [x0, z0, width] = this.decalPlaces[this.layout];
    const depth = (width * canvas.height) / canvas.width;
    plane.scale.set(width, depth, 1);
    plane.position.set(x0 + width / 2, -0.8, z0 + depth / 2);
  }

  /** Show one of the two sets of blocks ('mlp' or 'few'). */
  setLayout(name) {
    if (name === this.layout) return;
    this.layout = name;
    for (const b of this.blockList) b.mesh.visible = b.plate.visible = b.layout === name;
    this.#placeDecal();
    this.dirty = true;
  }

  setDecal({ title, lines }) {
    const { canvas, texture } = this.decal;
    const ctx = canvas.getContext('2d');
    const font = (weight, px) => `${weight} ${px}px system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.textBaseline = 'alphabetic';
    // every line shrinks until it fits the width of the decal
    const fit = (text, weight, px) => {
      ctx.font = font(weight, px);
      const width = ctx.measureText(text).width;
      if (width > canvas.width - 16) ctx.font = font(weight, Math.floor((px * (canvas.width - 16)) / width));
    };
    ctx.fillStyle = '#1d2337';
    fit(title, 700, 168);
    ctx.fillText(title, 0, 170);
    const inked = [[0, 34, ctx.measureText(title).width, 196]]; // x0, y0, x1, y1 of each piece of text, in canvas px
    let y = 292;
    let widest = 0;
    for (const line of lines) {
      ctx.fillStyle = line.color ?? '#4a5166';
      fit(line.text, line.bold ? 650 : 450, 64);
      ctx.fillText(line.text, 4, y);
      widest = Math.max(widest, ctx.measureText(line.text).width + 4);
      y += 98;
    }
    if (lines.length) inked.push([0, 236, widest, y - 98 + 20]);
    this.decal.inked = inked;
    texture.needsUpdate = true;
    this.dirty = true;
  }

  #buildFrames() {
    const R = (...ids) => this.rect(...ids);
    const pad = (r, p) => [r.x0 - p, r.x1 + p, r.z0 - p, r.z1 + p];
    const x = R('x');
    const w1 = R('W1');
    const h1 = R('h1');
    const b1 = R('b1');
    const all = R(...Object.keys(this.blocks));
    this.frames = {
      all: { box: pad(all, 4), theta: -0.1, phi: 0.74 },
      input: { box: [x.x0 - 3, x.x0 + 124, x.z0 - 8, x.z1 + 8], theta: 0.12, phi: 0.7 },
      W1: { box: pad(R('x', 'W1', 'h1'), 3), theta: -0.08, phi: 0.66 },
      sum1: { box: [w1.x1 - 120, h1.x1 + 3, -3, 96], theta: 0.3, phi: 0.92 },
      relu1: { box: [b1.x0 - 12, h1.x1 + 4, -3, 52], theta: 0.26, phi: 0.98 },
      L2: { box: pad(R('h1', 'W2', 'h2'), 4), theta: -0.12, phi: 0.72 },
      L3: { box: pad(R('h2', 'W3', 'h3'), 4), theta: -0.12, phi: 0.74 },
      L4: { box: pad(R('h3', 'W4', 'p'), 5), theta: -0.1, phi: 0.78 },
      out: { box: [R('p').x0 - 34, R('p').x1 + 6, R('b4').z0 - 6, R('p').z1 + 8], theta: 0.0, phi: 0.72 },
    };
    const few = R('fx', 'fc', 'fs', 'fd', 'ft');
    const fc = R('fc');
    const fd = R('fd');
    Object.assign(this.frames, {
      fewAll: { box: [few.x0 - 70, few.x1 + 30, few.z0 - 16, few.z1 + 84], theta: -0.1, phi: 0.72 },
      fewInput: { box: [fc.x0 - 86, fc.x1 + 8, R('fx').z0 - 18, fc.z0 + 40], theta: 0.1, phi: 0.82 },
      fewCentroids: { box: [fc.x0 - 86, fc.x1 + 8, R('fx').z0 - 16, R('fs').z1 + 8], theta: -0.06, phi: 0.66 },
      fewSpread: { box: [fc.x0 - 86, fc.x1 + 8, fc.z1 - 60, R('fs').z1 + 12], theta: 0.1, phi: 0.86 },
      fewDistance: { box: [fc.x1 - 60, R('ft').x1 + 12, fd.z0 - 30, fd.z1 + 12], theta: 0.1, phi: 0.74 },
      fewAnswer: { box: [fd.x0 - 8, R('ft').x1 + 90, fd.z0 - 14, fd.z1 + 10], theta: 0.14, phi: 0.76 },
    });
  }

  #buildControls() {
    const c = new OrbitControls(this.camera, this.canvas);
    c.enableDamping = true;
    c.dampingFactor = 0.14;
    c.screenSpacePanning = false;
    c.zoomToCursor = true;
    c.minDistance = 10;
    c.maxDistance = 3200;
    c.minPolarAngle = 0.02;
    c.maxPolarAngle = 1.45;
    c.zoomSpeed = 1.1;
    c.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    c.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
    c.addEventListener('start', () => {
      this.tween = null;
      this.userMoved = true;
    });
    this.controls = c;
  }

  #bindPointer() {
    const el = this.canvas;
    let down = null;
    el.addEventListener('pointerdown', (e) => {
      down = { x: e.clientX, y: e.clientY, time: performance.now() };
    });
    el.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse' || e.buttons) return;
      this.onPick?.(this.pick(e.clientX, e.clientY), e);
    });
    el.addEventListener('pointerleave', (e) => {
      if (e.pointerType === 'mouse') this.onPick?.(null, e);
    });
    el.addEventListener('pointerup', (e) => {
      if (!down) return;
      const still = Math.hypot(e.clientX - down.x, e.clientY - down.y) < 7 && performance.now() - down.time < 450;
      down = null;
      if (!still) return;
      const hit = this.pick(e.clientX, e.clientY);
      this.onPick?.(hit, e, true);
      this.onSelect?.(hit, e);
    });
    el.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  // ---------------------------------------------------------------- values

  setBlock(id, values, palette, options) {
    if (this.blocks[id].set(values, palette, performance.now(), options)) this.dirty = true;
  }

  /** Blocks outside `ids` recede into the background (null: nothing recedes). */
  setFocus(ids) {
    for (const b of this.blockList) b.fadeTarget = !ids || ids.includes(b.id) ? 0 : 0.72;
    this.dirty = true;
  }

  setHeightFactor(f) {
    this.heightFactor = f;
    for (const b of this.blockList) b.uniforms.uHeight.value = b.maxHeight * f;
    this.dirty = true;
  }

  /** {blockId: [i, j]} with -1 for "any"; blocks not named are left alone. */
  setHighlights(map) {
    for (const b of this.blockList) {
      const h = map?.[b.id];
      b.uniforms.uHover.value.set(h ? h[0] : -1, h ? h[1] : -1, h ? 1 : 0);
    }
    this.dirty = true;
  }

  /** Light up arbitrary cells: {blockId: [cell indices]}. Blocks not named go back to normal. */
  setMask(map) {
    for (const b of this.blockList) {
      const cells = map?.[b.id];
      b.uniforms.uMaskOn.value = cells ? 1 : 0;
      if (!cells) continue;
      b.aMask.array.fill(0);
      for (const k of cells) b.aMask.array[k] = 1;
      b.aMask.needsUpdate = true;
    }
    this.dirty = true;
  }

  pick(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    let best = null;
    for (const b of this.blockList) {
      if (b.layout !== this.layout) continue;
      const hit = b.pick(ray.ray.origin, ray.ray.direction, this.heightFactor);
      if (hit && (!best || hit.t < best.t)) best = hit;
    }
    if (best) this.#snapToMarker(best, ray.ray);
    return best;
  }

  /** In a sparse matrix the markers are drawn larger than their cells: a near miss still means the marker. */
  #snapToMarker(hit, ray) {
    const b = this.blocks[hit.id];
    if (!b.sparse || b.to[hit.i * b.nj + hit.j] !== 0) return;
    const depth = ray.direction.dot(this.camera.getWorldDirection(new THREE.Vector3())) * hit.t;
    const reach = (0.5 * MARKER_PX * depth) / this.focal + 0.2;
    const [ri, rj] = [Math.ceil(reach / b.pitchI), Math.ceil(reach / b.pitchJ)];
    const { i: i0, j: j0 } = hit;
    let nearest = Infinity;
    for (let i = Math.max(i0 - ri, 0); i <= Math.min(i0 + ri, b.ni - 1); i++) {
      for (let j = Math.max(j0 - rj, 0); j <= Math.min(j0 + rj, b.nj - 1); j++) {
        if (b.to[i * b.nj + j] === 0) continue;
        const d = Math.hypot((b.posI(i) - b.posI(i0)) / b.pitchI, j - j0);
        if (d < nearest) [nearest, hit.i, hit.j] = [d, i, j];
      }
    }
  }

  // ---------------------------------------------------------------- camera

  resize() {
    const w = Math.max(this.stage.clientWidth, 50);
    const h = Math.max(this.stage.clientHeight, 50);
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.focal = h / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)); // px per world unit at depth 1
    for (const b of this.blockList) b.uniforms.uPxScale.value = this.focal;
    if (this.frameName && !this.userMoved) this.snapTo(this.frameName);
    this.dirty = true;
  }

  /** Where the camera must stand to see a frame's box at its viewing angles. */
  #pose(name) {
    const f = this.frames[name];
    const [x0, x1, z0, z1] = f.box;
    const target = new THREE.Vector3((x0 + x1) / 2, 0, (z0 + z1) / 2);
    const dir = new THREE.Vector3().setFromSphericalCoords(1, f.phi, f.theta);
    const probe = this.camera.clone();
    const insets = this.getInsets?.() ?? { top: 58, right: 18, bottom: 26, left: 18 };
    const lim = {
      left: -1 + (2 * insets.left) / this.width,
      right: 1 - (2 * insets.right) / this.width,
      top: 1 - (2 * insets.top) / this.height,
      bottom: -1 + (2 * insets.bottom) / this.height,
    };
    const corners = [];
    for (const cx of [x0, x1]) for (const cz of [z0, z1]) for (const cy of [0, 8]) corners.push(new THREE.Vector3(cx, cy, cz));
    const v = new THREE.Vector3();
    const fits = (r) => {
      probe.position.copy(target).addScaledVector(dir, r);
      probe.lookAt(target);
      probe.updateMatrixWorld();
      for (const c of corners) {
        v.copy(c).project(probe);
        if (v.x < lim.left || v.x > lim.right || v.y < lim.bottom || v.y > lim.top) return false;
      }
      return true;
    };
    let lo = 10;
    let hi = 6000;
    for (let k = 0; k < 30; k++) {
      const mid = (lo + hi) / 2;
      if (fits(mid)) hi = mid;
      else lo = mid;
    }
    return { target, r: hi, theta: f.theta, phi: f.phi };
  }

  #applyPose(p) {
    this.camera.position.setFromSphericalCoords(p.r, p.phi, p.theta).add(p.target);
    this.controls.target.copy(p.target);
    this.camera.lookAt(p.target);
    this.dirty = true;
  }

  #currentPose() {
    const target = this.controls.target.clone();
    const s = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(target));
    return { target, r: s.radius, theta: s.theta, phi: s.phi };
  }

  snapTo(name) {
    this.frameName = name;
    this.userMoved = false;
    this.tween = null;
    this.#applyPose(this.#pose(name));
    this.controls.update();
  }

  flyTo(name, duration = 1400) {
    this.frameName = name;
    this.userMoved = false;
    this.tween = { from: this.#currentPose(), to: this.#pose(name), start: performance.now(), duration };
  }

  /** Fly to the cells `indices` of one block (for example the cells one group of statements wrote). */
  flyToCells(id, indices, duration = 1200) {
    const b = this.blocks[id];
    let [x0, x1, z0, z1] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const k of indices) {
      const [x, z] = b.center(Math.floor(k / b.nj), k % b.nj);
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      z0 = Math.min(z0, z);
      z1 = Math.max(z1, z);
    }
    if (x0 > x1) return;
    const grow = (lo, hi, least) => (hi - lo < least ? [(lo + hi - least) / 2, (lo + hi + least) / 2] : [lo, hi]);
    [x0, x1] = grow(x0 - 10, x1 + 10, 120);
    [z0, z1] = grow(z0 - 10, z1 + 10, 70);
    this.frames.cells = { box: [x0, x1, z0, z1], theta: 0.1, phi: 0.62 };
    this.flyTo('cells', duration);
  }

  /** Tilt between a straight-down view and the 3D view without moving the target. */
  setTopView(on) {
    const from = this.#currentPose();
    this.tween = {
      from,
      to: { ...from, phi: on ? 0.03 : 0.74, theta: on ? 0 : -0.1 },
      start: performance.now(),
      duration: 900,
    };
    this.userMoved = true;
  }

  // ---------------------------------------------------------------- labels

  /**
   * def: { html, pos: [x, y, z], align: 'l' | 'c' | 'r', v: 'above' | 'middle' | 'below', cls, when(ctx), row(ctx),
   *        clamp, onClick }
   */
  addLabel(def) {
    const el = document.createElement(def.onClick ? 'button' : 'div');
    el.className = `label ${def.cls ?? 'tag'}`;
    el.innerHTML = def.html ?? '';
    if (def.onClick) {
      el.type = 'button';
      el.addEventListener('click', def.onClick);
    }
    this.labelLayer.append(el);
    const cls = def.cls ?? 'tag';
    const priority = def.priority ?? (cls.includes('pred') ? 10 : cls.includes('small') ? 6 : cls.includes('tag') ? 8 : cls.includes('plain') ? 5 : 3);
    const label = {
      align: 'c', v: 'above', layout: 'mlp', ...def, priority, el, on: false, width: 0, height: 0,
      pos: new THREE.Vector3(...def.pos),
      alt: (def.alt ?? []).map((a) => ({ ...a, pos: a.pos && new THREE.Vector3(...a.pos) })),
    };
    this.labels.push(label);
    this.labels.sort((a, b) => b.priority - a.priority); // stable: equal priorities keep the order they were added in
    this.dirty = true;
    return label;
  }

  updateLabel(label, { html, pos }) {
    if (html !== undefined && html !== label.html) {
      label.html = html;
      label.el.innerHTML = html;
      label.width = 0;
      label.height = 0;
    }
    if (pos) label.pos.set(...pos);
    this.dirty = true;
  }

  /** Where the ground title's text is on screen right now: labels are kept off it. */
  #titleRects() {
    const { canvas, inked } = this.decal;
    if (!inked) return [];
    const [x0, z0, width] = this.decalPlaces[this.layout];
    const unit = width / canvas.width;
    const v = new THREE.Vector3();
    const out = [];
    const SLICES = 10; // the title lies slanted on screen: narrow slices hug it better than one box around it all
    for (const [a, b, c, d] of inked) {
      for (let k = 0; k < SLICES; k++) {
        const left = a + ((c - a) * k) / SLICES;
        const right = a + ((c - a) * (k + 1)) / SLICES;
        const r = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
        let visible = true;
        for (const [px, py] of [[left, b], [right, b], [left, d], [right, d]]) {
          v.set(x0 + px * unit, 0, z0 + py * unit).project(this.camera);
          if (v.z < -1 || v.z > 1) visible = false;
          const sx = (v.x * 0.5 + 0.5) * this.width;
          const sy = (-v.y * 0.5 + 0.5) * this.height;
          r.left = Math.min(r.left, sx);
          r.right = Math.max(r.right, sx);
          r.top = Math.min(r.top, sy);
          r.bottom = Math.max(r.bottom, sy);
        }
        if (visible) out.push(r);
      }
    }
    return out;
  }

  /**
   * Put every label where it belongs, most important first. A label that would cover a more important one, or the
   * ground title, tries its alternative places (`alt`) and is hidden if none is free.
   */
  #placeLabels() {
    const w = this.width;
    const h = this.height;
    const focal = h / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
    const ppu = focal / this.camera.position.distanceTo(this.controls.target);
    const ctx = {
      ppu,
      far: ppu < 0.62, // the whole network is small on screen: only the shortest names fit
      sensors: 132 * ppu >= 56,
      bands: 12.6 * ppu >= 34,
      strips: T * ppu >= 18,
      classes: PITCH[4] * ppu >= 62,
    };
    const v = new THREE.Vector3();
    const ax = { l: 0, c: -0.5, r: -1 };
    const ay = { above: -1, middle: -0.5, below: 0 };
    const taken = this.#titleRects();
    const free = (r) => !taken.some((o) => r.left < o.right + 3 && r.right > o.left - 3 && r.top < o.bottom + 1 && r.bottom > o.top - 1);

    const rectAt = (L, place) => {
      v.copy(place.pos ?? L.pos).project(this.camera);
      let x = (v.x * 0.5 + 0.5) * w;
      let y = (-v.y * 0.5 + 0.5) * h;
      if (v.z < -1 || v.z > 1 || x < 0 || x > w || y < 0 || y > h) return null;
      const align = place.align ?? L.align;
      const side = place.v ?? L.v;
      const rowOf = place.row ?? L.row;
      const row = (typeof rowOf === 'function' ? rowOf(ctx) : (rowOf ?? 0)) * 22 + 5;
      if (side === 'above') y -= row;
      else if (side === 'below') y += row;
      else x += align === 'r' ? -6 : align === 'l' ? 6 : 0;
      let left = x + ax[align] * L.width;
      const top = y + ay[side] * L.height;
      if (L.clamp) left = Math.min(Math.max(left, 8), Math.max(w - L.width - 8, 8));
      else if (left < 2 || left + L.width > w - 2) return null; // a name cut off by the edge is worse than no name
      return { left, top, right: left + L.width, bottom: top + L.height };
    };

    for (const L of this.labels) {
      let rect = null;
      if (L.layout === this.layout && (L.when ? L.when(ctx) : true)) {
        if (!L.width) {
          L.width = L.el.offsetWidth;
          L.height = L.el.offsetHeight;
        }
        for (const place of [L, ...L.alt]) {
          const r = rectAt(L, place === L ? {} : place);
          if (r && free(r)) {
            rect = r;
            break;
          }
        }
      }
      if (rect) {
        taken.push(rect);
        L.el.style.transform = `translate(${rect.left.toFixed(1)}px, ${rect.top.toFixed(1)}px)`;
      }
      if (!!rect !== L.on) {
        L.on = !!rect;
        L.el.classList.toggle('on', L.on);
      }
    }
  }

  // ---------------------------------------------------------------- loop

  #tick(now) {
    requestAnimationFrame((t) => this.#tick(t));
    let dirty = this.dirty;

    if (this.tween) {
      const { from, to, start, duration } = this.tween;
      const t = easeInOut(Math.min((now - start) / duration, 1));
      let dTheta = to.theta - from.theta;
      dTheta = ((dTheta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      this.#applyPose({
        target: from.target.clone().lerp(to.target, t),
        r: Math.exp(Math.log(from.r) + (Math.log(to.r) - Math.log(from.r)) * t),
        theta: from.theta + dTheta * t,
        phi: from.phi + (to.phi - from.phi) * t,
      });
      if (now - start >= duration) this.tween = null;
      dirty = true;
    }
    if (this.controls.update()) dirty = true;

    for (const b of this.blockList) {
      if (b.anim) {
        const t = b.progress(now);
        b.uniforms.uT.value = t;
        if (t >= 1) {
          b.anim = null;
          b.uniforms.uMinPx.value = b.sparse ? MARKER_PX : 0;
        }
        dirty = true;
      }
      if (Math.abs(b.fade - b.fadeTarget) > 0.002) {
        b.fade += (b.fadeTarget - b.fade) * 0.14;
        b.uniforms.uFade.value = b.fade;
        b.plate.material.color.copy(b.plateColor).lerp(this.bgColor, b.fade);
        dirty = true;
      }
    }

    if (dirty) {
      this.dirty = false;
      this.renderer.render(this.scene, this.camera);
      this.#placeLabels();
      this.onFrame?.();
    }
  }
}

export { T as STRIP, PITCH, LAYERS };
