import ast
import sys
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.append(".")

from micrograd import MLP, Value, cross_entropy
from utils import RNG, gen_data

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
def init_session_state(MODEL_CONFIG, GEN_DATA_TYPE):
    if "data" not in st.session_state:
        train_data, val_data, test_data = gen_data(RNG(42), n=300, type=GEN_DATA_TYPE)
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
def forward_pass(model, data):
    loss = Value(0)
    predicted_outputs = []
    for x, y in data:
        logits = model([Value(x[0]), Value(x[1])])
        loss += cross_entropy(logits, y)
        predicted_outputs.append(np.argmax([logit.data for logit in logits]))
    loss = loss * (1.0 / len(data))

    accuracy = sum(
        [1 for i, (_, y) in enumerate(data) if y == predicted_outputs[i]]
    ) / len(data)

    return loss, predicted_outputs, accuracy


def backward_pass(model, loss, step):
    loss.backward()
    for p in model.parameters():
        p.m = BETA1 * p.m + (1 - BETA1) * p.grad
        p.v = BETA2 * p.v + (1 - BETA2) * p.grad**2
        m_hat = p.m / (1 - BETA1 ** (step + 1))
        v_hat = p.v / (1 - BETA2 ** (step + 1))
        p.data -= LEARNING_RATE * (m_hat / (v_hat**0.5 + 1e-8) + WEIGHT_DECAY * p.data)
    model.zero_grad()
    return model


def train_step():
    train_loss, _, _ = forward_pass(
        st.session_state.model, st.session_state.data["train"]
    )
    st.session_state.model = backward_pass(
        st.session_state.model, train_loss, st.session_state.step
    )
    st.session_state.step += 1
    st.session_state.train_loss = train_loss.data

    val_loss, _, val_acc = forward_pass(
        st.session_state.model, st.session_state.data["val"]
    )
    st.session_state.val_loss = val_loss.data
    st.session_state.val_acc = val_acc

    test_loss, _, test_acc = forward_pass(
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


# Main app
def main():
    st.set_page_config(layout="wide", page_title="Neural Network Playground")
    st.markdown("# Neural Network Playground")
    st.markdown(
        "This app demonstrates how a neural network learns to classify data points. It is based on the [micrograd](https://github.com/EurekaLabsAI/micrograd) repository. The objective is to visualize the training process, and show how the model learns through backpropagation."
    )

    inp_col1, inp_col2 = st.columns(2)
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
            "[16]",
            help="List of hidden layer sizes. Default is [16]. The output layer of 3 neurons is added automatically to the end.",
            on_change=lambda: [
                st.session_state.pop(k) for k in list(st.session_state.keys())
            ],
        )

    init_session_state(MODEL_CONFIG, GEN_DATA_TYPE)

    inp_col1, inp_col2 = st.columns(2)
    with inp_col1:
        st.write(
            """Click the buttons below to train the model for one or ten steps. After each step (or batch of steps), the decision boundary, neuron activations, and loss curves will be updated. 
        The model configuration and layer information are displayed on the right."""
        )

    with inp_col2:
        st.write("#### Model Configuration")
        layer_neuron_info = ", ".join(
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
            init_session_state(MODEL_CONFIG, GEN_DATA_TYPE)

    decision_boundary_col, activation_col, loss_curve_col = st.columns(3)

    with decision_boundary_col:
        st.markdown("## Decision Boundary")
        st.session_state.decision_boundary = get_decision_boundary(
            st.session_state.model
        )

        train_data, val_data, test_data = st.tabs(["Train", "Validation", "Test"])

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
        st.markdown("## Neuron Activations")
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
        st.markdown("## Loss / Accuracy Curves")
        loss_c, acc_c = st.tabs(["Loss", "Accuracy"])

        with loss_c:
            fig = plot_loss_curves()
            st.plotly_chart(fig, use_container_width=True)

        with acc_c:
            fig = plot_accuracy_curves()
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
