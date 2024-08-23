"""
Same as micrograd.py, but uses PyTorch for the autograd engine.
This is a way for us to check and verify correctness, and also
shows some of the similarities/differences in how PyTorch would
implement the same MLP. PyTorch lets you specify the forward pass,
records all the operations performed, and then calls backward()
"under the hood" inside its autograd engine.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils import RNG, gen_data_yinyang

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

# generate a dataset with 100 2-dimensional datapoints in 3 classes
train_split, val_split, test_split = gen_data_yinyang(random, n=100)

# init the model: 2D inputs, 8 neurons, 3 outputs (logits)
model = MLP(2, [8, 3])
model.to(torch.float64) # ensure we're using double precision

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
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-1,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=1e-4
)

# train
for step in range(100):

    # evaluate the validation split every few steps
    if step % 10 == 0:
        val_loss = eval_split(model, val_split)
        print(f"step {step}, val loss {val_loss}")

    # forward the network (get logits of all training datapoints)
    model.train()
    losses = []
    for x, y in train_split:
        logits = model(torch.tensor(x))
        loss = F.cross_entropy(logits, torch.tensor(y).view(-1))
        losses.append(loss)
    loss = torch.stack(losses).mean()
    # backward pass (deposit the gradients)
    loss.backward()
    # update with AdamW
    optimizer.step()
    optimizer.zero_grad()

    print(f"step {step}, train loss {loss.data}")
