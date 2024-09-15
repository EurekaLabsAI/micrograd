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
      // random number in [0, 1)
      return (this.random_u32() >>> 8) / 16777216.0;
  }

  uniform(a = 0.0, b = 1.0) {
      // random number in [a, b)
      return a + (b - a) * this.random();
  }
}

/*
Generates the Yin Yang dataset.
Thank you https://github.com/lkriener/yin_yang_data_set
*/
function genDataYinYang(random, n = 1000, rSmall = 0.1, rBig = 0.5) {
  const pts = [];

  function distToRightDot(x, y) {
    return Math.sqrt((x - 1.5 * rBig)**2 + (y - rBig)**2);
  }

  function distToLeftDot(x, y) {
    return Math.sqrt((x - 0.5 * rBig)**2 + (y - rBig)**2);
  }

  function whichClass(x, y) {
    const dRight = distToRightDot(x, y);
    const dLeft = distToLeftDot(x, y);
    const criterion1 = dRight <= rSmall;
    const criterion2 = dLeft > rSmall && dLeft <= 0.5 * rBig;
    const criterion3 = y > rBig && dRight > 0.5 * rBig;
    const isYin = criterion1 || criterion2 || criterion3;
    const isCircles = dRight < rSmall || dLeft < rSmall;

    if (isCircles) return 2;
    return isYin ? 0 : 1;
  }

  function getSample(goalClass = null) {
    while (true) {
      // Generate x and y in the range [0, 2*rBig]
      const x = random.uniform(0, 2 * rBig);
      const y = random.uniform(0, 2 * rBig);

      if (Math.sqrt((x - rBig)**2 + (y - rBig)**2) > rBig) {
        continue;
      }

      const c = whichClass(x, y);
      if (goalClass === null || c === goalClass) {
        // Scale and shift x and y to span [-2, 2]
        const scaledX = (x / rBig - 1) * 2;
        const scaledY = (y / rBig - 1) * 2;
        return [scaledX, scaledY, c];
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const goalClass = i % 3;
    const [x, y, c] = getSample(goalClass);
    pts.push([[x, y], c]);
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
  constructor(data, _prev = [], _op = '') {
      this.data = data;
      this.grad = 0;
      // internal variables used for autograd graph construction
      this._backward = () => {};
      this._prev = _prev;
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
          this.grad += out.data * out.grad;
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
      return this.mul(-1.0);
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
// for visualization purposes, type could be 'param', 'input', 'loss' etc.
function setValuesType(nodes, type) {
  nodes.forEach(n => {
    n._type = type;
  });
}

class Neuron extends Module {
  constructor(nin, nonlin = true) {
      super();
      const scale = Math.pow(nin, -0.5);
      this.w = Array(nin).fill().map(() => new Value(random.uniform(-1, 1) * scale));
      this.b = new Value(0);
      this.nonlin = nonlin;
      setValuesType([this.b].concat(this.w), "param");
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
      return out;
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
  // commenting these two lines out to get a cleaner visualization
  // const maxVal = Math.max(...logits.map(val => val.data));
  // logits = logits.map(val => val.add(-maxVal));
  // 1) evaluate elementwise e^x
  const ex = logits.map(x => x.exp());
  // 2) compute the sum of the above
  const denom = ex.reduce((a, b) => a.add(b), new Value(0));
  // 3) normalize by the sum to get probabilities
  const probs = ex.map(x => x.div(denom));
  // 4) log the probabilities at target
  const logp = probs[target].log();
  // 5) the negative log likelihood loss (invert so we get a loss - lower is better)
  const nll = logp.mul(-1);
  setValuesType([nll], "loss");
  return nll;
}

// ----------------------------------------------------------------------------
// optimizer

class AdamW {
  constructor(parameters, lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.0) {
    this.parameters = parameters;
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.t = 0;
    for (const p of this.parameters) {
      p.m = 0;
      p.v = 0;
    }
  }

  step() {
    this.t += 1;
    for (const p of this.parameters) {
      if (p.grad === null) {
        continue;
      }
      p.m = this.beta1 * p.m + (1 - this.beta1) * p.grad;
      p.v = this.beta2 * p.v + (1 - this.beta2) * (p.grad ** 2);
      const mHat = p.m / (1 - this.beta1 ** this.t);
      const vHat = p.v / (1 - this.beta2 ** this.t);
      p.data -= this.lr * (mHat / (Math.sqrt(vHat) + this.eps) + this.weightDecay * p.data);
    }
  }

  zeroGrad() {
    for (const p of this.parameters) {
      p.grad = 0;
    }
  }
}
