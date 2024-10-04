
# micrograd

> micrograd is the only piece of code you need to train neural networks. Everything else is just efficiency.

In this module we build a tiny "autograd" engine (short for automatic gradient) that implements the backpropagation algorithm, as it was prominently popularized for training neural networks in the 1986 paper [Learning Internal Representations by Error Propagation](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf) by Rumelhart, Hinton and Williams. This repository builds on the earlier repo [karpathy/micrograd](https://github.com/karpathy/micrograd), but modifies it into an LLM101n module.

The code we build here is the heart of neural network training - it allows us to calculate how we should update the parameters of a neural network in order to make it better at some task, such as the one of next token prediction in autoregressive language models. This exact same algorithm is used in all modern deep learning libraries, such as PyTorch, TensorFlow, JAX, and others, except that those libraries are much more optimized and feature-rich.

!WIP!

Very early draft, putting up a possible first version of what I have in mind. In simple terms, 3-way classification for 2D training data. Very visual, easy to understand, good for intuition building.

Covers:
- Autograd engine (micrograd)
- Neural network (NN) with 1 hidden layer (MLP) built on top of it
- Training loop: loss function, backpropagation, parameter updates

**Interactive Demo**. There is a nice visaulization of the "action" during training, showing the exact computational graph. The full computational graph includes the entire dataset which is too much. So instead, here we pick a single datapoint to forward through the network and show that graph alone. In this case we're forwarding the origin (0,0) and a fictional label 0, running the forward pass and the loss, doing backward, and then showing the data/grads. The process to get this visualization working:

1. First run `python micrograd.py` and make sure it saves the `graph.svg`, which is the connectivity graph. You'll need to install graphviz if you don't have it. E.g. on my MacBook this is `brew install graphviz` followed by `pip install graphviz`.
2. Once we have the `graph.svg` file, we load it from the HTML. Because of cross-origin security issues when loading the svg from the file system, we can't just open up the HTML page directly and need to serve these files. The easiest way is to run a simple python webserver with `python -m http.server`. Then open up the localhost URL it gives you, and open the `micrograd.html` page. You'll see a really cool visualization of the training process yay!

TODOs:
- Parallel implementation in C that prints the same thing
- Many improvements to the interactive web demo. Should be able to step through the optimization in more detail, select the "fictional" example to forward through the graph, possibly add/delete datapoints and see how the network responds to it, etc. Probably the entire visualization can be made significantly nicer.

### License

MIT
