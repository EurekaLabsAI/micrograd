// Defines a simple autograd engine and uses it to classify points in the plane
// to 3 classes (red, green, blue) using a simple multilayer perceptron (MLP).
const { RNG, gen_data } = require('./utils.js');

const random = new RNG(42);

class Value {
    // stores a single scalar value and its gradient
    constructor(data, _children = [], _op = '') {
        this.data = data;
        this.grad = 0;
        this._backward = () => {};
        this._prev = new Set(_children);
        this._op = _op;
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
        if (typeof other !== 'number') throw new Error("only supporting number powers for now");
        const out = new Value(Math.pow(this.data, other), [this], `**${other}`);

        out._backward = () => {
            this.grad += other * Math.pow(this.data, other - 1) * out.grad;
        };

        return out;
    }

    relu() {
        const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');
        out._backward = () => {
            this.grad += (out.data > 0) * out.grad;
        };

        return out;
    }

    tanh() {
        const x = Math.tanh(this.data);
        const out = new Value(x, [this], 'tanh');
        out._backward = () => {
            this.grad += (1 - x * x) * out.grad;
        };

        return out;
    }

    exp() {
        const x = Math.exp(this.data);
        const out = new Value(x, [this], 'exp');
        out._backward = () => {
            this.grad += x * out.grad;
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

    backward() {
        const topo = [];
        const visited = new Set();

        function buildTopo(v) {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._prev) {
                    buildTopo(child);
                }
                topo.push(v);
            }
        }

        buildTopo(this);

        this.grad = 1;
        for (const v of topo.reverse()) {
            v._backward();
        }
    }

    neg() { return this.mul(-1); }
    sub(other) { return this.add(other instanceof Value ? other.neg() : new Value(-other)); }
    div(other) { return this.mul(other instanceof Value ? other.pow(-1) : new Value(Math.pow(other, -1))); }

    toString() {
        return `Value(data=${this.data}, grad=${this.grad})`;
    }
}


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
    constructor(nin, kwargs) {
        super();
        this.w = Array(nin).fill().map(() => new Value(random.uniform(-1, 1) * Math.pow(nin, -0.5)));
        this.b = new Value(0);
        this.nonlin = kwargs.nonlin !== undefined ? kwargs.nonlin : true;
    }

    call(x) {
        const act = this.w.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), this.b);
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
        this.neurons = Array(nout).fill().map(() => new Neuron(nin, kwargs));
    }

    call(x) {
        const out = this.neurons.map(n => n.call(x));
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
        this.layers = sz.slice(0, -1).map((s, i) => new Layer(s, sz[i + 1], { nonlin: i !== nouts.length - 1 }));

    }

    call(x) {
        for (const layer of this.layers) {
            x = layer.call(x);
        }
        return x;
    }

    parameters() {
        return this.layers.flatMap(l => l.parameters());
    }

    toString() {
        return `MLP of [${this.layers.join(', ')}]`;
    }
}

function crossEntropy(logits, target) {
    // subtract the max for numerical stability (avoids overflow)
    const maxVal = Math.max(...logits.map(v => v.data));
    const shiftedLogits = logits.map(v => v.sub(maxVal));
    // 1) evaluate elementwise e^x
    const ex = shiftedLogits.map(x => x.exp());
    // 2) compute the sum of the above
    const denom = ex.reduce((sum, x) => sum.add(x));
    // 3) normalize by the sum to get probabilities
    const probs = ex.map(x => x.div(denom));
    // 4) log the probabilities at target
    const logp = probs[target].log();
    // 5) the negative log likelihood loss (invert so we get a loss - lower is better)
    const nll = logp.neg();
    return nll;
}

// evaluation utility to compute the loss on a given split of the dataset
function evalSplit(model, split) {
    // evaluate the loss of a split
    let loss = new Value(0);
    for (const [x, y] of split) {
        const logits = model.call([new Value(x[0]), new Value(x[1])]);
        loss = loss.add(crossEntropy(logits, y));
    }
    loss = loss.mul(1.0 / split.length); // normalize the loss
    return loss.data;
}

// let's train!
if (require.main === module) {
    // generate a random dataset with 100 2-dimensional datapoints in 3 classes
    const [train_split, val_split, test_split] = gen_data(random, 100);

    // init the model: 2D inputs, 16 neurons, 3 outputs (logits)
    const model = new MLP(2, [16, 3]);

    // optimize using Adam
    const learning_rate = 1e-1;
    const beta1 = 0.9;
    const beta2 = 0.95;
    const weight_decay = 1e-4;
    const eps = 1e-8;
    for (const p of model.parameters()) {
        p.m = 0.0;
        p.v = 0.0;
    }

    // training loop
    for (let step = 0; step < 100; step++) {
        // evaluate the validation split every few steps
        if (step % 10 === 0) {
            const val_loss = evalSplit(model, val_split);
            console.log(`step ${step}, val loss ${val_loss.toFixed(6)}`);
        }

        // forward the network (get logits of all training datapoints)
        let loss = new Value(0);
        for (const [x, y] of train_split) {
            const logits = model.call([new Value(x[0]), new Value(x[1])]);
            loss = loss.add(crossEntropy(logits, y));
        }
        loss = loss.mul(1.0 / train_split.length); // normalize the loss

        // backward pass (deposit the gradients)
        loss.backward();

        // update with AdamW
        for (const p of model.parameters()) {
            p.m = beta1 * p.m + (1 - beta1) * p.grad;
            p.v = beta2 * p.v + (1 - beta2) * Math.pow(p.grad, 2);
            const m_hat = p.m / (1 - Math.pow(beta1, step + 1));
            const v_hat = p.v / (1 - Math.pow(beta2, step + 1));
            p.data -= learning_rate * (m_hat / (Math.sqrt(v_hat) + eps) + weight_decay * p.data);
        }
        model.zeroGrad();

        console.log(`step ${step}, train loss ${loss.data.toFixed(20)}`);
    }
}