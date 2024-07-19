"""
Defines a simple autograd engine and uses it to classify points in the plane
to 3 classes (red, green, blue) using a simple multilayer perceptron (MLP).
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils import RNG, gen_data

random = RNG(42)

# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) network

class Neuron(nn.Module):

    def __init__(self, nin, nonlin=True):
        super().__init__()
        self.w = Parameter(torch.tensor([random.uniform(-1, 1) * nin**-0.5 for _ in range(nin)]))
        self.b = Parameter(torch.zeros(1))
        self.nonlin = nonlin

    def forward(self, x):
        act = torch.sum(self.w * x) + self.b
        return act.tanh() if self.nonlin else act

    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(nn.Module):

    def __init__(self, nin, nout, **kwargs):
        super().__init__()
        self.neurons = nn.ModuleList([Neuron(nin, **kwargs) for _ in range(nout)])

    def forward(self, x):
        out = [n(x) for n in self.neurons]
        return torch.stack(out, dim=-1)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(nn.Module):

    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        self.layers = nn.ModuleList([Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# -----------------------------------------------------------------------------
# let's train!

train_split, val_split, test_split = gen_data(random, n=100)

# init the model: 2D inputs, 16 neurons, 3 outputs (logits)
model = MLP(2, [16, 3])

@torch.no_grad()
def eval_split(model, split):
    model.eval()
    # evaluate the loss of a split
    loss = 0.0
    for x, y in split:
        logits = model(torch.tensor(x))
        y = torch.tensor(y).view(-1)
        loss += F.cross_entropy(logits, y).item()
    loss = loss * (1.0/len(split)) # normalize the loss
    return loss

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
    losses = []
    model.train()
    for x, y in train_split:
        logits = model(torch.tensor(x))
        loss = F.cross_entropy(logits, torch.tensor(y).view(-1))
        losses.append(loss)
    loss = torch.stack(losses).mean()
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
