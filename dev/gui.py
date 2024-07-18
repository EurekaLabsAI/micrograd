import ast
import sys


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# add current directory to access micrograd module
sys.path.append(".")

from micrograd import MLP, RNG, Value, nll_loss  # noqa: E402

st.set_page_config(layout="wide", page_title="Micrograd - MLP", page_icon="🧠")
st.write(
    """# Micrograd

This application allows you to train a Multi-Layer Perceptron (MLP) to classify points in a 2D plane using an automatic differentiation engine to implement backpropagation.

You can adjust the hyperparameters and visualize the decision boundary of the trained model on the training, validation, and test data splits.
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

# Plot decision boundary
if st.button("Train Model"):
    model = train_model(model, train_data, n_iters=n_iters)

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
