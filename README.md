
# micrograd

> micrograd is the only piece of code you need to train neural networks. Everything else is just efficiency.

In this module we build a tiny "autograd" engine (short for automatic gradient) that implements the backpropagation algorithm, as it was prominently popularized for training neural networks in the 1986 paper [Learning Internal Representations by Error Propagation](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf) by Rumelhart, Hinton and Williams. This repository builds on the earlier repo [karpathy/micrograd](https://github.com/karpathy/micrograd), but modifies into into an LLM101n module.

The code we build here is the heart of neural network training - it allows us to calculate how we should update the parameters of a neural network in order to make it better at some task, such as the one of next token prediction in autoregressive language models. This exact same algorithm is used in all modern deep learning libraries, such as PyTorch, TensorFlow, JAX, and others., except that those libraries are much more optimized and feature-rich.

!WIP!

Very early draft, putting up a possible first version of what I have in mind. In simple terms, 3-way classification for 2D training data. Very visual, easy to understand, good for intuition building.

Covers:
- Autograd engine (micrograd)
- Neural network (NN) with 1 hidden layer (MLP) built on top of it
- Training loop: loss function, backpropagation, parameter updates

I really want this module to incorporate this [JavaScript web demo](https://cs.stanford.edu/~karpathy/svmjs/demo/demonn.html) I built ages ago, where student can interactively add/modify datapoints and play/pause the optimization to see how the neural network responds to it. However, instead of it being built in JavaScript, today it would probably be a nice streamlit app that uses this code. Even better, the computational graph could be shown on the side, with the full details of the data/grads in all of the nodes.

TODOs:
- finalize the training loop details (probably use batches?)
- streamlit app for interactive demo, including the original micrograd repo goodies: showing the computational graph
- parallel implementation `pytorch_reference.py` in PyTorch that prints the exact same results as micrograd.py
- parallel implementation in C that prints the same thing
