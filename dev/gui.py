import ast
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from graphviz import Digraph

# add current directory to access micrograd module
sys.path.append(".")
from micrograd import MLP, RNG, Value, nll_loss  # noqa: E402


def trace(root):
    # Traverse the computation graph to extract nodes and edges
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    # Create a graphviz Digraph object to visualize the computation graph
    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)

    # Create clusters for layers and neurons
    layer_clusters = {}
    neuron_clusters = {}

    for n in nodes:
        layer_name = n.layer_name.split("-")[0]
        if layer_name not in layer_clusters:
            layer_clusters[layer_name] = Digraph(name=f"cluster_{layer_name}")
            layer_clusters[layer_name].attr(
                label=layer_name, style="filled", color="lightgrey"
            )

        neuron_name = "-".join(n.layer_name.split("-")[:2])
        if neuron_name not in neuron_clusters:
            neuron_clusters[neuron_name] = Digraph(name=f"cluster_{neuron_name}")
            neuron_clusters[neuron_name].attr(
                label=neuron_name, style="filled", color="white"
            )

        # Generate unique identifier for each node
        uid = str(id(n))

        # Truncate layer name if it is too long
        layer_name = (
            f"{n.layer_name[:5]}...{n.layer_name[-5:]}"
            if len(n.layer_name) > 10
            else n.layer_name
        )

        # Add node to appropriate neuron cluster
        neuron_clusters[neuron_name].node(
            uid,
            label=f"{{ {layer_name} | data: {n.data:.4f} | grad: {n.grad:.4f} }}",
            shape="record",
        )

        if n._op:
            # If value is the result of an operation, add a node for it and an edge from the operation to the final value
            neuron_clusters[neuron_name].node(name=uid + n._op, label=n._op)
            neuron_clusters[neuron_name].edge(uid + n._op, uid)

    # Add neuron clusters to layer clusters
    for neuron_name, neuron_cluster in neuron_clusters.items():
        layer_name = neuron_name.split("-")[0]
        layer_clusters[layer_name].subgraph(neuron_cluster)

    # Add layer clusters to main graph
    for layer_cluster in layer_clusters.values():
        dot.subgraph(layer_cluster)

    # Add edges between nodes
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


st.set_page_config(layout="wide", page_title="Micrograd - MLP", page_icon="🧠")
st.write(
    """# Micrograd

This application allows you to train a Multi-Layer Perceptron (MLP) to classify points in a 2D plane using an automatic differentiation engine to implement backpropagation.

You can adjust the hyperparameters and visualize the decision boundary of the trained model on the training, validation, and test data splits.

The computation graph of the forward and backward pass of the model on a single data point in the validation set is also visualized to help you understand the flow of data and gradients during training.
"""
)


def gen_data(n, random, type="simple"):
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


def plot_decision_boundary(model, split, h=0.1, title="Decision Boundary"):
    # Create a grid of points
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict the class for each point in the grid
    Z = []
    logits = np.array(
        [model([Value(point[0]), Value(point[1])]) for point in grid_points]
    )
    logits_data = np.array(
        [[logit.data for logit in logit_list] for logit_list in logits]
    )
    Z = np.argmax(logits_data, axis=1).reshape(xx.shape)

    # Plot the decision boundary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.6, cmap="viridis")

    ax.scatter(
        [x[0] for x, _ in split],
        [x[1] for x, _ in split],
        c=[y for _, y in split],
        edgecolors="k",
        marker="o",
        cmap="viridis",
        s=100,
        alpha=0.8,
    )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


# Function to train the model
def train_model(model, data, n_iters):
    learning_rate = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-4
    for p in model.parameters():
        p.m = 0.0
        p.v = 0.0

    progress_bar = st.progress(0, text="")
    for step in range(n_iters):
        loss = Value(0)
        for x, y in data:
            logits = model([Value(x[0]), Value(x[1])])
            loss += nll_loss(logits, y)

        loss = loss * (1.0 / len(data))
        loss.backward()
        for p in model.parameters():
            p.m = beta1 * p.m + (1 - beta1) * p.grad
            p.v = beta2 * p.v + (1 - beta2) * p.grad**2
            p.data -= learning_rate * p.m / (p.v**0.5 + 1e-8)
            p.data -= weight_decay * p.data  # weight decay
        model.zero_grad()

        progress_bar.progress(
            (step + 1) / n_iters,
            text=f"Training: Step {step + 1} | Loss {loss.data:.4f}",
        )

    return model


st.sidebar.markdown("## Training options")
model_layers_text = st.sidebar.text_input(
    "Model layers", "[16]", help="List of hidden layer sizes. Default is [16]."
)

n_iters = st.sidebar.number_input("Number of iterations", 1, 100, 50)
n_datapoints = st.sidebar.number_input("Number of data points", 1, 500, 100)
random_seed = st.sidebar.number_input(
    "Random seed", min_value=None, max_value=None, value=42
)


# Initialize random number generator
random = RNG(random_seed)

gen_data_type = st.sidebar.selectbox(
    "Data type",
    ["simple", "circle"],
)

# Generate data
train_data, val_data, test_data = gen_data(
    n=n_datapoints, type=gen_data_type, random=random
)


try:
    model_layers = ast.literal_eval(model_layers_text) + [
        3
    ]  # Add output layer of 3 neurons
except (SyntaxError, ValueError):
    st.sidebar.error("model layers must be valid python syntax.")
    st.stop()

model = MLP(2, model_layers)

if st.button("Train Model"):
    # Train the model
    model = train_model(model, train_data, n_iters=n_iters)

    # Plot decision boundaries
    with st.spinner("Plotting decision boundaries..."):
        fig_train = plot_decision_boundary(
            model, train_data, h=0.1, title="Decision Boundary [Train Data]"
        )
        fig_val = plot_decision_boundary(
            model, val_data, h=0.1, title="Decision Boundary [Validation Data]"
        )
        fig_test = plot_decision_boundary(
            model, test_data, h=0.1, title="Decision Boundary [Test Data]"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(fig_train)

    with col2:
        st.pyplot(fig_val)

    with col3:
        st.pyplot(fig_test)

    # Computation graph visualization
    st.markdown(
        """## Computation Graph
                
The computation graph below shows the forward and backward pass of the model on one random data point from the training set.

Each node represents a value in the computation graph with its data and gradient values, and the edges represent the flow of data during the forward and backward pass."""
    )
    with st.spinner("Drawing computation graph..."):
        random_train_sample = train_data[int(random.uniform(0, len(train_data)))]

        toy_prediction = model(random_train_sample[0])
        toy_loss = nll_loss(toy_prediction, random_train_sample[1])
        dot = draw_dot(toy_loss)

    col1, col2 = st.columns([0.2, 0.8])

    with col1:
        input_data = random_train_sample[0]
        predicted_logits = toy_prediction
        predicted_output = np.argmax([logit.data for logit in predicted_logits])

        st.write(f"Input Data: {[round(x, 4) for x in input_data]}")
        st.write(
            f"Predicted Logits: {[round(logit.data, 4) for logit in predicted_logits]}"
        )
        st.write(f"Predicted Output (Class): {predicted_output}")
        st.write(f"True Output (Class): {random_train_sample[1]}")

    with col2:
        st.graphviz_chart(dot, use_container_width=True)
        st.download_button(
            label="Download Computation Graph",
            data=dot.pipe(format="png"),
            file_name="computation_graph.png",
            mime="image/png",
            type="primary",
        )
