"""
Defines a simple autograd engine and uses it to classify points in the plane
to 3 classes (red, green, blue) using a simple multilayer perceptron (MLP).
"""
import math

# -----------------------------------------------------------------------------

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def uniform(self, a=0.0, b=1.0):
        # random float32 in [a, b)
        return a + (b-a) * self.random()

random = RNG(42)

# -----------------------------------------------------------------------------
# Value

class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op="", layer_name=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.layer_name = layer_name

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, layer_name="const")
        out = Value(
            self.data + other.data,
            (self, other),
            "+",
            f"{self.layer_name}+{other.layer_name}",
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, layer_name="const")
        out = Value(
            self.data * other.data,
            (self, other),
            "*",
            f"{self.layer_name}*{other.layer_name}",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(
            self.data**other, (self,), f"**{other}", f"{self.layer_name}**{other}"
        )

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(
            0 if self.data < 0 else self.data,
            (self,),
            "ReLU",
            f"ReLU({self.layer_name})",
        )

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh", f"tanh({self.layer_name})")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp", f"exp({self.layer_name})")

        def _backward():
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward

        return out

    def log(self):
        # (this is the natural log)
        out = Value(math.log(self.data), (self,), "log", f"log({self.layer_name})")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op}, layer_name={self.layer_name})"


# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) network


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True, layer_name=""):
        r = random.uniform(-1, 1) * nin**-0.5
        self.w = [Value(r, layer_name=f"{layer_name}-W{i}") for i in range(nin)]
        self.b = Value(0, layer_name=f"{layer_name}-B")
        self.nonlin = nonlin
        self.layer_name = layer_name

    def __call__(self, x):
        # Convert x to Value if it's not already to add layer_name
        if not all(isinstance(xi, Value) for xi in x):
            x = [Value(xi, layer_name=f"I{i}") for i, xi in enumerate(x)]

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, layer_name="", **kwargs):
        self.neurons = [
            Neuron(nin, layer_name=f"{layer_name}-N{i}", **kwargs) for i in range(nout)
        ]
        self.layer_name = layer_name

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], f"L{i}", nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# -----------------------------------------------------------------------------
# loss function: the negative log likelihood (NLL) loss

def nll_loss(logits, target):
    # subtract the max for numerical stability (avoids overflow)
    max_val = max(val.data for val in logits)
    logits = [val - max_val for val in logits]
    # 1) evaluate elementwise e^x
    ex = [x.exp() for x in logits]
    # 2) compute the sum of the above
    denom = sum(ex)
    # 3) normalize by the sum to get probabilities
    probs = [x / denom for x in ex]
    # 4) log the probabilities at target
    logp = (probs[target]).log()
    # 5) the negative log likelihood loss (invert so we get a loss - lower is better)
    nll = -logp
    return nll

# -----------------------------------------------------------------------------
# let's train!

# generate a random dataset with 100 2-dimensional datapoints in 3 classes
def gen_data(n=100):
    pts = []
    for _ in range(n):
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        # concentric circles
        # label = 0 if x**2 + y**2 < 1 else 1 if x**2 + y**2 < 2 else 2
        # very simple dataset
        label = 0 if x < 0 else 1 if y < 0 else 2
        pts.append(([x, y], label))
    # create train/val/test splits of the data (80%, 10%, 10%)
    tr = pts[:int(0.8*n)]
    val = pts[int(0.8*n):int(0.9*n)]
    te = pts[int(0.9*n):]
    return tr, val, te
train_split, val_split, test_split = gen_data()

def main():
    # init the model: 2D inputs, 16 neurons, 3 outputs (logits)
    model = MLP(2, [16, 3])

    def eval_split(model, split):
        # evaluate the loss of a split
        loss = Value(0)
        for x, y in split:
            logits = model([Value(x[0]), Value(x[1])])
            loss += nll_loss(logits, y)
        loss = loss * (1.0/len(split)) # normalize the loss
        return loss.data

    # optimize using Adam
    learning_rate = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-4
    for p in model.parameters():
        p.m = 0.0
        p.v = 0.0

    # train
    for step in range(100):

        # evaluate the validation split every few steps
        if step % 10 == 0:
            val_loss = eval_split(model, val_split)
            print(f"step {step}, val loss {val_loss}")

        # forward the network (get logits of all training datapoints)
        loss = Value(0)
        for x, y in train_split:
            logits = model([Value(x[0]), Value(x[1])])
            loss += nll_loss(logits, y)
        loss = loss * (1.0/len(train_split)) # normalize the loss
        # backward pass (deposit the gradients)
        loss.backward()
        # update with Adam
        for p in model.parameters():
            p.m = beta1 * p.m + (1 - beta1) * p.grad
            p.v = beta2 * p.v + (1 - beta2) * p.grad**2
            p.data -= learning_rate * p.m / (p.v**0.5 + 1e-8)
            p.data -= weight_decay * p.data # weight decay
        model.zero_grad()

        print(f"step {step}, train loss {loss.data}")
