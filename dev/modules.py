import math


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
        return a + (b - a) * self.random()


# generate a random dataset with 100 2-dimensional datapoints in 3 classes
def gen_data(random: RNG, n=100, type="simple"):
    pts = []
    for _ in range(n):
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        if type == "circle":
            # concentric circles
            label = 0 if x**2 + y**2 < 1 else 1 if x**2 + y**2 < 2 else 2
        else:
            # very simple dataset
            label = 0 if x < 0 else 1 if y < 0 else 2
        pts.append(([x, y], label))
    # create train/val/test splits of the data (80%, 10%, 10%)
    tr = pts[: int(0.8 * n)]
    val = pts[int(0.8 * n) : int(0.9 * n)]
    te = pts[int(0.9 * n) :]
    return tr, val, te

# Fixed random seed for reproducibility
random = RNG(42)

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
        self.w = [Value(r, layer_name=f"{layer_name}_[W{i}]") for i in range(nin)]
        self.b = Value(0, layer_name=f"{layer_name}_[B]")
        self.nonlin = nonlin
        self.layer_name = layer_name

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, layer_name="", **kwargs):
        self.neurons = [
            Neuron(nin, layer_name=f"{layer_name}_[N{i}]", **kwargs)
            for i in range(nout)
        ]
        self.layer_name = layer_name

    def __call__(self, x):
        out = []
        for i, n in enumerate(self.neurons):
            out.append(n(x))

        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], f"[L{i}]", nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
