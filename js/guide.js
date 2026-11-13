// The walkthrough: what each step shows and says.
//
// A step names the camera frame, the blocks in focus, the matrices that show products (weight x input) instead of
// weights, and how far the forward pass has been computed:
//   shown  0 nothing, 1 z1, 2 h1, 3 z2, 4 h2, 5 z3, 6 h3, 7 logits, 8 softmax.
// body(c) gets the live numbers of the network and input on screen (see context() in app.js).

const n = (v) => v.toLocaleString('en-US');
const only = (c, trained, written) => `<div class="aside ${c.tab}">${c.tab === 'trained' ? trained : written}</div>`;

export const STEPS = [
  {
    id: 'overview',
    title: 'A weld classifier you can walk through',
    frame: 'all',
    focus: null,
    contrib: [],
    shown: 8,
    body: (c) => `
      <p>This is the classifier Prof. Shao's group uses on their ultrasonic metal welding data. It reads
      <b>528 numbers</b> that describe one weld and returns <b>9 scores</b>, one for each combination of tool state
      (new, worn, damaged) and surface state (clean, contaminated, polished). In between sit
      <b>99,959 weights</b>.</p>
      <p>Each <span class="sw w">blue and orange</span> slab is a weight matrix. Each
      <span class="sw d">green</span> strip is a list of values for one weld on its way through. A taller, brighter
      bar is a larger number; a flat dark cell is zero.</p>
      ${only(
        c,
        `<h3>This tab: trained</h3>
        <p>Gradient descent set every one of these weights, using 162 labelled welds. On welds they have never seen,
        the group's saved networks are right <b>${c.trainedAcc}</b> of the time.</p>`,
        `<h3>This tab: written by an LLM</h3>
        <p>GPT-6 Astra was given a text description of the welding process and of this architecture, and replied with
        the weights as text: no data, no training, no gradients. This is MIT CSAIL's <i>weight-space programming</i>.</p>
        <p>It set <b>${n(c.nonzero)}</b> of the 99,959 numbers and left the rest at zero. On held-out welds this reply
        scores <b>${c.replyAcc}</b>; its five replies average ${c.writtenMeanAcc}. Guessing gives 11.1%.</p>`,
      )}
      <p class="next">Press <b>Continue</b> to follow one weld through the network.</p>`,
  },
  {
    id: 'input',
    title: 'One weld becomes 528 numbers',
    frame: 'input',
    focus: ['x'],
    contrib: [],
    shown: 0,
    body: (c) => `
      <p>Four sensors record each weld for one second: acoustic emission (AE), the horn's displacement (LVDT),
      electrical power and a microphone.</p>
      <p>A wavelet transform splits each signal into 11 frequency bands, from cD11 (49 to 98 Hz) up to cD1
      (50 to 100 kHz), and 12 statistics summarize each band: entropy, mean, percentiles, RMS, variance and so on.</p>
      <p class="formula">4 sensors x 11 bands x 12 statistics = 528 features</p>
      <p>The features are standardized, so 0 means "like the average training weld". A
      <span class="sw d">green</span> bar is above average, a <span class="sw dn">pink</span> bar below.</p>
      <p><b>On screen:</b> ${c.inputText} Change it with the <i>Input</i> menu above the 3D view, and hover any bar
      to see which feature it is.</p>`,
  },
  {
    id: 'w1',
    title: 'The first weight matrix',
    frame: 'W1',
    focus: ['x', 'W1', 'b1'],
    contrib: [],
    shown: 0,
    body: (c) => `
      <p><b>W1</b> has a column for each of the 528 inputs and a row for each of the 150 units of the first hidden
      layer: 79,200 weights. One cell says how strongly one feature pushes one unit,
      <span class="sw w">blue</span> for up and <span class="sw wn">orange</span> for down.</p>
      ${only(
        c,
        `<p>A trained matrix is <b>dense</b>. All 79,200 cells are non-zero and none stands out: the largest is about
        ${c.maxW1}. What the network knows is spread thinly over the whole slab, which is why it looks like static.
        No single cell means anything on its own.</p>`,
        `<p>Astra's matrix is almost <b>empty</b>: ${n(c.nonzeroW[0])} of 79,200 cells are set and the rest are
        exactly zero. Every short streak is a rule it wrote on purpose, and only ${c.wired} of the 150 units receive
        any input at all.</p>
        <p>Hover a bright cell to read which feature it listens to.</p>`,
      )}`,
  },
  {
    id: 'sum1',
    title: 'Multiply, add up, add the bias',
    frame: 'sum1',
    focus: ['x', 'W1', 'b1', 'z1'],
    contrib: [1],
    shown: 1,
    body: (c) => `
      <p>Each unit multiplies every input by its own weight, adds the 528 products along its row, then adds its
      bias:</p>
      <p class="formula">z1 = x · W1 + b1</p>
      <p>The slab now shows the <b>products</b> (weight x input) in <span class="sw d">green</span> and
      <span class="sw dn">pink</span> instead of the weights. The strip <b>b1</b> holds the 150 biases and
      <b>z1</b> the 150 sums.</p>
      ${only(
        c,
        `<p>Every unit gets hundreds of small pushes, up and down, that mostly cancel. What is left over is the
        signal.</p>`,
        `<p>Only the units Astra wired up receive anything: ${c.reached[0]} of the 150 sums are non-zero.</p>`,
      )}`,
  },
  {
    id: 'relu1',
    title: 'ReLU keeps the positive part',
    frame: 'relu1',
    focus: ['W1', 'b1', 'z1', 'h1'],
    contrib: [1],
    shown: 2,
    body: (c) => `
      <p class="formula">h1 = max(z1, 0)</p>
      <p>A negative sum (<span class="sw dn">pink</span>) becomes zero and the unit goes quiet. A positive sum
      (<span class="sw d">green</span>) passes through unchanged. This is the only non-linear step, and it is what
      lets stacked layers compute more than one big weighted sum.</p>
      <p>For this input, <b>${c.active[0]} of 150</b> units stay on.</p>
      ${only(
        c,
        '',
        `<p>Astra often writes units in pairs with opposite weights, so that one of the pair carries the positive
        part of a quantity and the other its negative part.</p>`,
      )}`,
  },
  {
    id: 'l2',
    title: 'Layer 2: 150 values become 100',
    frame: 'L2',
    focus: ['h1', 'W2', 'b2', 'z2', 'h2'],
    contrib: [2],
    shown: 4,
    body: (c) => `
      <p>The 150 outputs of layer 1 enter <b>W2</b> (150 x 100) from the left. Each of its 100 columns is one unit
      of layer 2: the products add up down the column, the bias <b>b2</b> is added, and ReLU gives <b>h2</b> in the
      strips below.</p>
      <p><b>${c.active[1]} of 100</b> units are on for this input.</p>
      ${only(
        c,
        `<p>Dense again: all 15,000 weights are in use.</p>`,
        `<p>${n(c.nonzeroW[1])} of 15,000 weights are set, and ${c.nonzeroB[1]} of the 100 biases. Astra's own
        account of what these units compute is in the box below.</p>`,
      )}`,
  },
  {
    id: 'l3',
    title: 'Layer 3: 100 values become 50',
    frame: 'L3',
    focus: ['h2', 'W3', 'b3', 'z3', 'h3'],
    contrib: [3],
    shown: 6,
    body: (c) => `
      <p><b>W3</b> (100 x 50) takes h2 from above. Each row is one unit of layer 3, and its sum lands in the strips
      on the right.</p>
      <p><b>${c.active[2]} of 50</b> units are on.</p>
      ${only(
        c,
        `<p>By now each value mixes all 528 inputs, twice over. Nothing here maps to a single sensor any more.</p>`,
        `<p>${n(c.nonzeroW[2])} of 5,000 weights are set.</p>`,
      )}`,
  },
  {
    id: 'l4',
    title: 'The output layer: nine scores',
    frame: 'L4',
    focus: ['h3', 'W4', 'b4', 'z4'],
    contrib: [4],
    shown: 7,
    body: (c) => `
      <p><b>W4</b> (50 x 9) turns the last 50 values into one score, or <i>logit</i>, per condition. There is no
      ReLU here, so a logit can be negative. The largest logit wins.</p>
      ${only(
        c,
        `<p>All 450 weights are in use.</p>`,
        `<p>Astra set ${n(c.nonzeroW[3])} of these 450 weights.</p>`,
      )}`,
  },
  {
    id: 'softmax',
    title: 'Softmax and the answer',
    frame: 'out',
    focus: ['b4', 'z4', 'p'],
    contrib: [],
    shown: 8,
    body: (c) => `
      <p>Softmax turns the nine logits into probabilities that add up to 1:</p>
      <p class="formula">p = exp(logit) / sum of exp(logits)</p>
      <p>${c.tie
        ? `All nine logits are equal here, so every class gets 11.1% and the tie goes to the first class,
          <b>${c.predName}</b>.`
        : `The network's answer for this input: <b>${c.predName}</b> (${c.predP}).`}</p>
      <p>${c.truthText}</p>`,
  },
  {
    id: 'compare',
    title: 'Trained against written',
    frame: 'all',
    focus: null,
    contrib: [],
    shown: 8,
    body: (c) => `
      <table class="compare">
        <thead><tr><th></th><th>Trained</th><th>Written by Astra</th></tr></thead>
        <tbody>
          <tr><th>Non-zero numbers</th><td>${n(c.trainedNonzero)}<small>of 99,959</small></td>
            <td>${n(c.writtenNonzero)}<small>of 99,959</small></td></tr>
          <tr><th>Set by</th><td>gradient descent<small>162 labelled welds</small></td>
            <td>one text reply<small>no data</small></td></tr>
          <tr><th>Test accuracy</th><td>${c.trainedAcc}</td>
            <td>${c.replyAcc}<small>5 replies: ${c.writtenMeanAcc}</small></td></tr>
        </tbody>
      </table>
      <p>Same architecture, same input, opposite pictures. The trained weights are dense and unreadable, and they
      work. The written weights are sparse and every one has a stated reason, but the network
      <b>scores near chance</b> (guessing gives 11.1%): Astra's guesses about how each condition changes the
      signals did not hold on the real welds.</p>
      <p>One guess did hold: pooled over the five replies, the written networks name a contaminated surface 72% of
      the time (chance: 33%).</p>
      <p><button class="link" data-action="switch">Show the ${c.tab === 'trained' ? 'written' : 'trained'}
      network</button> The camera stays where it is, so you can flip back and forth.</p>`,
  },
];
