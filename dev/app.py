import ast
import re
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyvis.network import Network
from streamlit.components.v1 import html

# This is imported separately to avoid redefining classes when the script is reloaded
from modules import MLP, RNG, Value, gen_data

# Constants
LEARNING_RATE = 1e-1
BETA1 = 0.9
BETA2 = 0.95
WEIGHT_DECAY = 1e-4
COLOR_MAPPING = {0: "red", 1: "green", 2: "blue"}


def init_model(MODEL_CONFIG):
    model = MLP(2, MODEL_CONFIG)
    for p in model.parameters():
        p.m = 0.0
        p.v = 0.0

    return model


# Initialize session state
def init_session_state(MODEL_CONFIG, GEN_DATA_TYPE, SEED):
    if "data" not in st.session_state:
        train_data, val_data, test_data = gen_data(RNG(SEED), n=300, type=GEN_DATA_TYPE)
        st.session_state.data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

    if "model" not in st.session_state:
        st.session_state.model = init_model(ast.literal_eval(MODEL_CONFIG) + [3])

    if "step" not in st.session_state:
        st.session_state.step = 0

    if "train_loss" not in st.session_state:
        st.session_state.train_loss = 0

    if "train_last_loss" not in st.session_state:
        st.session_state.train_last_loss = None

    if "val_loss" not in st.session_state:
        st.session_state.val_loss = 0

    if "test_loss" not in st.session_state:
        st.session_state.test_loss = 0

    if "val_acc" not in st.session_state:
        st.session_state.val_acc = 0

    if "test_acc" not in st.session_state:
        st.session_state.test_acc = 0

    if "decision_boundary" not in st.session_state:
        st.session_state.decision_boundary = None

    if "loss_history" not in st.session_state:
        st.session_state.loss_history = {"train": [], "val": [], "test": []}

    if "accuracy_history" not in st.session_state:
        st.session_state.accuracy_history = {"val": [], "test": []}


# Model operations
def cross_entropy(logits, target):
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


def forward_pass(model, data):
    loss = Value(0)
    last_loss = loss
    predicted_outputs = []
    for x, y in data:
        logits = model([Value(x[0], layer_name="[I0]"), Value(x[1], layer_name="[I1]")])
        last_loss = cross_entropy(logits, y)
        loss += last_loss

        predicted_outputs.append(np.argmax([logit.data for logit in logits]))
    loss = loss * (1.0 / len(data))

    accuracy = sum(
        [1 for i, (_, y) in enumerate(data) if y == predicted_outputs[i]]
    ) / len(data)

    return last_loss, loss, predicted_outputs, accuracy


def backward_pass(model, loss, step):
    loss.backward()
    for p in model.parameters():
        p.m = BETA1 * p.m + (1 - BETA1) * p.grad
        p.v = BETA2 * p.v + (1 - BETA2) * p.grad**2
        m_hat = p.m / (1 - BETA1 ** (step + 1))
        v_hat = p.v / (1 - BETA2 ** (step + 1))
        p.data -= LEARNING_RATE * (m_hat / (v_hat**0.5 + 1e-8) + WEIGHT_DECAY * p.data)
    model.zero_grad()
    return model, loss


# Training functions
def train_step():
    train_last_loss, train_loss, _, _ = forward_pass(
        st.session_state.model, st.session_state.data["train"]
    )
    st.session_state.model, train_loss = backward_pass(
        st.session_state.model, train_loss, st.session_state.step
    )
    st.session_state.step += 1
    st.session_state.train_loss = train_loss.data
    st.session_state.train_last_loss = train_last_loss

    _, val_loss, _, val_acc = forward_pass(
        st.session_state.model, st.session_state.data["val"]
    )
    st.session_state.val_loss = val_loss.data
    st.session_state.val_acc = val_acc

    _, test_loss, _, test_acc = forward_pass(
        st.session_state.model, st.session_state.data["test"]
    )
    st.session_state.test_loss = test_loss.data
    st.session_state.test_acc = test_acc

    st.session_state.loss_history["train"].append(st.session_state.train_loss)
    st.session_state.loss_history["val"].append(st.session_state.val_loss)
    st.session_state.loss_history["test"].append(st.session_state.test_loss)

    st.session_state.accuracy_history["val"].append(st.session_state.val_acc)
    st.session_state.accuracy_history["test"].append(st.session_state.test_acc)


# Visualization functions
def get_decision_boundary(model):
    x1_min, x1_max = -2, 2
    x2_min, x2_max = -2, 2
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    logits = np.array([model([Value(point[0]), Value(point[1])]) for point in grid])
    logits_data = np.array(
        [[logit.data for logit in logit_list] for logit_list in logits]
    )
    Z = np.argmax(logits_data, axis=1).reshape(xx1.shape)
    return xx1, xx2, Z


def plot_decision_boundary(data, contour_data, title):
    x = np.array([x for x, _ in data])
    y = np.array([y for _, y in data])
    xx1, xx2, Z = contour_data
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx1[0],
            y=xx2[:, 0],
            z=Z,
            colorscale=[[0, "red"], [0.5, "green"], [1, "blue"]],
            opacity=0.3,
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode="markers",
            marker=dict(
                color=[COLOR_MAPPING[y] for y in y],
                size=10,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x1",
        yaxis_title="x2",
        xaxis=dict(range=[-2, 2]),
        yaxis=dict(range=[-2, 2]),
        height=400,
    )
    return fig


def get_activation(model, layer_idx, neuron_idx):
    neuron = model.layers[layer_idx].neurons[neuron_idx]
    x1_min, x1_max = -2, 2
    x2_min, x2_max = -2, 2
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = np.array(
        [neuron([Value(point[0]), Value(point[1])]).data for point in grid]
    ).reshape(xx1.shape)
    return xx1, xx2, Z


def plot_activation_layer(model, step, layer_idx):
    num_neurons = len(model.layers[layer_idx].neurons)
    fig = make_subplots(
        rows=num_neurons,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"Neuron {j}" for j in range(num_neurons)],
    )
    for j, neuron in enumerate(model.layers[layer_idx].neurons):
        xx1, xx2, Z = get_activation(model, layer_idx, j)
        fig.add_trace(
            go.Heatmap(
                z=Z,
                x=xx1[0],
                y=xx2[:, 0],
                colorscale=[[0, "red"], [0.5, "green"], [1, "blue"]],
                showscale=False,
            ),
            row=j + 1,
            col=1,
        )
    fig.update_layout(
        title=f"Neuron activations | Layer {layer_idx} | Step {step}",
        height=400 * num_neurons,
        width=400,
        showlegend=False,
    )
    return fig


def plot_loss_curves():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(0, st.session_state.step + 1)),
            y=st.session_state.loss_history["train"],
            mode="lines",
            name="Train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(0, st.session_state.step + 1)),
            y=st.session_state.loss_history["val"],
            mode="lines",
            name="Validation",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(0, st.session_state.step + 1)),
            y=st.session_state.loss_history["test"],
            mode="lines",
            name="Test",
        )
    )
    fig.update_layout(title="Loss Curves", xaxis_title="Step", yaxis_title="Loss")
    return fig


def plot_accuracy_curves():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(0, st.session_state.step + 1)),
            y=st.session_state.accuracy_history["val"],
            mode="lines",
            name="Validation",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(0, st.session_state.step + 1)),
            y=st.session_state.accuracy_history["test"],
            mode="lines",
            name="Test",
        )
    )
    fig.update_layout(
        title="Accuracy Curves", xaxis_title="Step", yaxis_title="Accuracy"
    )
    return fig


# Visualization of the computational graph
def trace(root):
    nodes, edges, levels = set(), set(), {}

    def build(v, level=0):
        # Iterate over a given node, check if it's already in the set of nodes, if not, add it
        if v not in nodes:
            nodes.add(v)
            # Use the operation as a fallback for the layer name if it's missing
            levels[v] = level
            # If node contains children, then 1. add an edge, 2. recursively look at the child nodes for their children
            for child in v._prev:
                edges.add((child, v))
                build(child, level + 1)

    build(root)
    return nodes, edges, levels


def draw_dot(root):
    G = Network(
        directed=True,
        layout=True,
        cdn_resources="remote",
    )

    nodes, edges, levels = trace(root)

    for n in nodes:
        # Generate unique identifier for each node
        uid = str(id(n))

        def get_node_color(node):
            # Get the color of the node based on the layer name
            # W is blue, B is yellow, I is black

            if re.search(r"W\d]$", node.layer_name):
                return "green"
            elif re.search(r"B]$", node.layer_name):
                return "yellow"
            elif re.search(r"^\[I\d]$", node.layer_name):
                return "red"
            elif node.layer_name == "const":
                return "gray"
            else:
                return "lightblue"

        G.add_node(
            uid,
            label=f"Level: {levels[n]}\nData: {n.data:.4f}\nGrad: {n.grad:.4f}",
            title=f"{n}",
            level=levels[n],
            shape="box",
            color=get_node_color(n),
        )

        if n._op:
            # If value is the result of an operation, add a node for it and an edge from the operation to the final value
            op_uid = uid + n._op
            G.add_node(
                op_uid, label=n._op, title=f"{n}", level=levels[n], shape="ellipse"
            )

            G.add_edge(op_uid, uid)

    for n1, n2 in edges:
        # For end node, add an edge from the initial value to the operator node, identified by the `uid` + `operator`
        G.add_edge(str(id(n1)), str(id(n2)) + n2._op)

    G.options.layout.hierarchical.direction = "RL"
    G.options.layout.hierarchical.levelSeparation = 300

    return G


# Main app entrypoint
def main():
    st.set_page_config(layout="wide", page_title="Neural Network Playground")
    st.markdown("# Neural Network Playground")
    st.markdown(
        "This app demonstrates how a neural network learns to classify data points. It is based on the [micrograd](https://github.com/EurekaLabsAI/micrograd) repository. The objective is to visualize the training process, and show how the model learns through backpropagation."
    )

    # User input section
    inp_col1, inp_col2, inp_col3 = st.columns(3)
    with inp_col1:
        GEN_DATA_TYPE = st.selectbox(
            "Data type",
            ["simple", "circle"],
            help="Select the type of data to generate.",
            on_change=lambda: [
                st.session_state.pop(k) for k in list(st.session_state.keys())
            ],
        )

    with inp_col2:
        MODEL_CONFIG = st.text_input(
            "Model layers",
            "[3]",
            help="List of hidden layer sizes. Default is [3]. The output layer of 3 neurons is added automatically to the end.",
            on_change=lambda: [
                st.session_state.pop(k) for k in list(st.session_state.keys())
            ],
        )

    with inp_col3:
        SEED = st.number_input(
            "Random seed",
            min_value=None,
            max_value=None,
            value=42,
            help="Random seed for data generation.",
            on_change=lambda: [
                st.session_state.pop(k) for k in list(st.session_state.keys())
            ],
        )

    init_session_state(MODEL_CONFIG, GEN_DATA_TYPE, SEED)

    inp_col1, inp_col2 = st.columns(2)
    with inp_col1:
        st.write("Click the buttons below to train the model for one or ten steps.")

        st.write(
            "After each step (or batch of steps), the computational graph, decision boundary, neuron activations, and loss curves will be updated."
        )

    with inp_col2:
        st.write("#### Model Configuration")
        layer_neuron_info = """

""".join(
            [
                f"Layer `{i}`: `{len(layer.neurons)} neurons`"
                for i, layer in enumerate(st.session_state.model.layers)
            ]
        )
        st.markdown(layer_neuron_info)

    col1, col2, col3, _ = st.columns([1, 1, 1, 3])

    with col1:
        if st.button(
            "Train one step", type="primary", help="Train the model for one step"
        ):
            train_step()

    with col2:
        if st.button(
            "Train ten steps", type="primary", help="Train the model for ten steps"
        ):
            for _ in range(10):
                train_step()
                time.sleep(0.5)

    with col3:
        if st.button("Reset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state(MODEL_CONFIG, GEN_DATA_TYPE, SEED)

    # Backward pass :: Computational graph view
    with st.container(border=True):
        st.markdown("## Computational Graph")
        st.markdown(
            """The computational graph below shows the model structure.
        The nodes are colored according to their type:"""
        )
        st.markdown(
            """
        - `Red`: Input nodes
        - `Green`: Weight nodes
        - `Yellow`: Bias nodes
        - `Gray`: Constant nodes
        - `Blue`: Operation nodes
            """
        )

        with st.expander("Show Graph"):
            if st.session_state.train_last_loss:
                st.subheader(f"Step: {st.session_state.step}")
                net = draw_dot(st.session_state.train_last_loss)
                net.write_html("graph.html")
                html(net.html, height=800, scrolling=True)
            else:
                st.write("No graph to display yet. Please train the model.")

    # Forward  pass :: Decision boundary, neuron activations, loss curves
    with st.container(border=True):
        st.markdown("## Model Visualization")
        decision_boundary_col, activation_col, loss_curve_col = st.columns(3)

        with decision_boundary_col:
            st.markdown("### Decision Boundary")
            with st.expander("Show Decision Boundary", expanded=True):
                st.session_state.decision_boundary = get_decision_boundary(
                    st.session_state.model
                )

                train_data, val_data, test_data = st.tabs(
                    ["Train", "Validation", "Test"]
                )

                with train_data:
                    fig = plot_decision_boundary(
                        st.session_state.data["train"],
                        st.session_state.decision_boundary,
                        title=f"Training Data | Step: {st.session_state.step}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with val_data:
                    fig = plot_decision_boundary(
                        st.session_state.data["val"],
                        st.session_state.decision_boundary,
                        title=f"Validation Data | Step: {st.session_state.step}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with test_data:
                    fig = plot_decision_boundary(
                        st.session_state.data["test"],
                        st.session_state.decision_boundary,
                        title=f"Test Data | Step: {st.session_state.step}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with activation_col:
            st.markdown("### Neuron Activations")
            with st.expander("Show Neuron Activations", expanded=True):
                layer_tabs = st.tabs(
                    [f"Layer {i}" for i in range(len(st.session_state.model.layers))]
                )
                for i, layer_tab in enumerate(layer_tabs):
                    with layer_tab:
                        if i == len(st.session_state.model.layers) - 1:
                            st.write("Output layer activations are not visualized.")
                            continue

                        # st.write(f"Layer: {i}")
                        fig = plot_activation_layer(
                            st.session_state.model, st.session_state.step, i
                        )
                        st.plotly_chart(fig, use_container_width=True)

        with loss_curve_col:
            st.markdown("### Loss / Accuracy Curves")
            with st.expander("Show Loss / Accuracy Curves", expanded=True):
                loss_c, acc_c = st.tabs(["Loss", "Accuracy"])

                with loss_c:
                    fig = plot_loss_curves()
                    st.plotly_chart(fig, use_container_width=True)

                with acc_c:
                    fig = plot_accuracy_curves()
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
