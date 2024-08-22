// ----------------------------------------------------------------------------
// utils

/*
Class that mimics the random interface in Python, fully deterministic,
and in a way that we also control fully, and can also use in C, etc.
*/
class RNG {
  constructor(seed) {
      this.state = BigInt(seed);
  }

  random_u32() {
      // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
      this.state ^= (this.state >> 12n) & 0xFFFFFFFFFFFFFFFFn;
      this.state ^= (this.state << 25n) & 0xFFFFFFFFFFFFFFFFn;
      this.state ^= (this.state >> 27n) & 0xFFFFFFFFFFFFFFFFn;
      return Number((this.state * 0x2545F4914F6CDD1Dn >> 32n) & 0xFFFFFFFFn);
  }

  random() {
      // random float32 in [0, 1)
      return (this.random_u32() >>> 8) / 16777216.0;
  }

  uniform(a = 0.0, b = 1.0) {
      // random float32 in [a, b)
      return a + (b - a) * this.random();
  }
}

/*
Simple dataset generation function that generates a dataset of n points
in 2D space, with labels 0, 1, 2. The dataset is split into training,
validation, and test sets (80%, 10%, 10%).
*/
function genData(random, n = 100) {
  const pts = [];
  for (let i = 0; i < n; i++) {
      const x = random.uniform(-2.0, 2.0);
      const y = random.uniform(-2.0, 2.0);
      // Very simple dataset
      const label = x < 0 ? 0 : y < 0 ? 1 : 2;
      // Uncomment the following line and comment out the above line to use concentric circles instead
      // const label = x**2 + y**2 < 1 ? 0 : x**2 + y**2 < 2 ? 1 : 2;
      pts.push([[x, y], label]);
  }
  // Create train/val/test splits of the data (80%, 10%, 10%)
  const tr = pts.slice(0, Math.floor(0.8 * n));
  const val = pts.slice(Math.floor(0.8 * n), Math.floor(0.9 * n));
  const te = pts.slice(Math.floor(0.9 * n));
  return { train: tr, validation: val, test: te };
}

// ----------------------------------------------------------------------------
// Value

/*
The Value object stores a single scalar number and its gradient.
*/
class Value {
  constructor(data, _children = [], _op = '') {
      this.data = data;
      this.grad = 0;
      // internal variables used for autograd graph construction
      this._backward = () => {};
      this._prev = new Set(_children);
      this._op = _op; // the op that produced this node, for graphviz / debugging / etc
  }

  add(other) {
      other = other instanceof Value ? other : new Value(other);
      const out = new Value(this.data + other.data, [this, other], '+');

      out._backward = () => {
          this.grad += out.grad;
          other.grad += out.grad;
      };

      return out;
  }

  mul(other) {
      other = other instanceof Value ? other : new Value(other);
      const out = new Value(this.data * other.data, [this, other], '*');

      out._backward = () => {
          this.grad += other.data * out.grad;
          other.grad += this.data * out.grad;
      };

      return out;
  }

  pow(other) {
      if (typeof other !== 'number') {
          throw new Error("only supporting number powers for now");
      }
      const out = new Value(Math.pow(this.data, other), [this], `**${other}`);

      out._backward = () => {
          this.grad += (other * Math.pow(this.data, other - 1)) * out.grad;
      };

      return out;
  }

  relu() {
      const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');

      out._backward = () => {
          this.grad += (out.data > 0 ? 1 : 0) * out.grad;
      };

      return out;
  }

  tanh() {
      const out = new Value(Math.tanh(this.data), [this], 'tanh');

      out._backward = () => {
          this.grad += (1 - out.data ** 2) * out.grad;
      };

      return out;
  }

  exp() {
      const out = new Value(Math.exp(this.data), [this], 'exp');

      out._backward = () => {
          this.grad += Math.exp(this.data) * out.grad;
      };

      return out;
  }

  log() {
      const out = new Value(Math.log(this.data), [this], 'log');

      out._backward = () => {
          this.grad += (1 / this.data) * out.grad;
      };

      return out;
  }

  neg() {
      return this.mul(-1);
  }

  sub(other) {
      return this.add(other.neg());
  }

  div(other) {
      return this.mul(other.pow(-1));
  }

  backward() {
      // topological order all of the children in the graph
      const topo = [];
      const visited = new Set();

      const buildTopo = (v) => {
          if (!visited.has(v)) {
              visited.add(v);
              for (const child of v._prev) {
                  buildTopo(child);
              }
              topo.push(v);
          }
      };

      buildTopo(this);

      // go one variable at a time and apply the chain rule to get its gradient
      this.grad = 1;
      for (const v of topo.reverse()) {
          v._backward();
      }
  }

  toString() {
      return `Value(data=${this.data}, grad=${this.grad})`;
  }

  // Alias for toString to mimic Python's __repr__
  [Symbol.for('nodejs.util.inspect.custom')]() {
      return this.toString();
  }
}

// ----------------------------------------------------------------------------
// MLP Module

class Module {
  zeroGrad() {
      for (const p of this.parameters()) {
          p.grad = 0;
      }
  }

  parameters() {
      return [];
  }
}

class Neuron extends Module {
  constructor(nin, nonlin = true) {
      super();
      const scale = Math.pow(nin, -0.5);
      this.w = Array(nin).fill().map(() => new Value(random.uniform(-1, 1) * scale));
      this.b = new Value(0);
      this.nonlin = nonlin;
  }

  forward(x) {
      const act = x.reduce((sum, xi, i) => sum.add(this.w[i].mul(xi)), this.b);
      return this.nonlin ? act.tanh() : act;
  }

  parameters() {
      return [...this.w, this.b];
  }

  toString() {
      return `${this.nonlin ? 'TanH' : 'Linear'}Neuron(${this.w.length})`;
  }
}

class Layer extends Module {
  constructor(nin, nout, kwargs = {}) {
      super();
      this.neurons = Array(nout).fill().map(() => new Neuron(nin, kwargs.nonlin));
  }

  forward(x) {
      const out = this.neurons.map(n => n.forward(x));
      return out.length === 1 ? out[0] : out;
  }

  parameters() {
      return this.neurons.flatMap(n => n.parameters());
  }

  toString() {
      return `Layer of [${this.neurons.join(', ')}]`;
  }
}

class MLP extends Module {
  constructor(nin, nouts) {
      super();
      const sz = [nin, ...nouts];
      this.layers = sz.slice(0, -1).map((_, i) =>
          new Layer(sz[i], sz[i+1], { nonlin: i !== nouts.length - 1 })
      );
  }

  forward(x) {
      return this.layers.reduce((acc, layer) => layer.forward(acc), x);
  }

  parameters() {
      return this.layers.flatMap(layer => layer.parameters());
  }

  toString() {
      return `MLP of [${this.layers.join(', ')}]`;
  }
}

// ----------------------------------------------------------------------------
// loss function

function crossEntropy(logits, target) {
  // subtract the max for numerical stability (avoids overflow)
  const maxVal = Math.max(...logits.map(val => val.data));
  logits = logits.map(val => val.add(-maxVal));
  // 1) evaluate elementwise e^x
  const ex = logits.map(x => x.exp());
  // 2) compute the sum of the above
  const denom = ex.reduce((a, b) => a.add(b));
  // 3) normalize by the sum to get probabilities
  const probs = ex.map(x => x.div(denom));
  // 4) log the probabilities at target
  const logp = probs[target].log();
  // 5) the negative log likelihood loss (invert so we get a loss - lower is better)
  const nll = logp.mul(-1);
  return nll;
}

// ----------------------------------------------------------------------------
// evaluation utility

function evalSplit(model, split) {
  // evaluate the loss of a split
  let loss = new Value(0);
  for (const [x, y] of split) {
    const logits = model.forward([new Value(x[0]), new Value(x[1])]);
    loss = loss.add(crossEntropy(logits, y));
  }
  loss = loss.mul(1.0 / split.length); // normalize the loss
  return loss.data;
}

// ----------------------------------------------------------------------------
// Create an instance of RNG with seed 42
const random = new RNG(42);
// Generate data using the genData function
const dataSplits = genData(random, 100);
const trainSplit = dataSplits.train;
const valSplit = dataSplits.validation;
const testSplit = dataSplits.test;

// init the model: 2D inputs, 16 neurons, 3 outputs (logits)
const model = new MLP(2, [16, 3]);

// optimize using Adam
const learningRate = 1e-1;
const beta1 = 0.9;
const beta2 = 0.95;
const weightDecay = 1e-4;
for (const p of model.parameters()) {
  p.m = 0.0;
  p.v = 0.0;
}

// train
for (let step = 0; step < 100; step++) {

  // evaluate the validation split every few steps
  if (step % 10 === 0) {
      const valLoss = evalSplit(model, valSplit);
      console.log(`step ${step}, val loss ${valLoss.toFixed(6)}`);
  }

  // forward the network (get logits of all training datapoints)
  let loss = new Value(0);
  for (const [x, y] of trainSplit) {
      const logits = model.forward([new Value(x[0]), new Value(x[1])]);
      loss = loss.add(crossEntropy(logits, y));
  }
  loss = loss.mul(1.0 / trainSplit.length); // normalize the loss
  // backward pass (deposit the gradients)
  loss.backward();
  // update with AdamW
  for (const p of model.parameters()) {
      p.m = beta1 * p.m + (1 - beta1) * p.grad;
      p.v = beta2 * p.v + (1 - beta2) * p.grad ** 2;
      const mHat = p.m / (1 - beta1 ** (step + 1));  // bias correction
      const vHat = p.v / (1 - beta2 ** (step + 1));
      p.data -= learningRate * (mHat / (Math.sqrt(vHat) + 1e-8) + weightDecay * p.data);
  }
  model.zeroGrad(); // never forget to clear those gradients! happens to everyone

  console.log(`step ${step}, train loss ${loss.data}`);
}
