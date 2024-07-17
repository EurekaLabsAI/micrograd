import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from micrograd_streamlit import MLP, Value, gen_data, nll_loss

st.set_page_config(layout="wide", page_title="MLP in your browser")


@st.cache_data
def generate_initial_data(n):
    return gen_data(n=n)


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
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.6)
    ax.scatter(
        [x[0] for x, _ in split],
        [x[1] for x, _ in split],
        c=[y for _, y in split],
        edgecolors="k",
        marker="o",
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)

    return fig


# Function to train the model
def train_model(model, data, n_iters=20):
    learning_rate = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-4
    for p in model.parameters():
        p.m = 0.0
        p.v = 0.0

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

        print(f"step {step}, train loss {loss.data}")

    return model


st.title("Micrograd - MLP")

st.sidebar.markdown("### Training options")
n_iters = st.sidebar.slider("Number of iterations", 1, 100, 10)
n_datapoints = st.sidebar.slider("Number of data points", 1, 100, 50)

# Initialize session state for data if not already done
if "data" not in st.session_state:
    st.session_state.data, _, _ = generate_initial_data(n=n_datapoints)

st.sidebar.write("### Data generation options")
new_point_min = st.sidebar.slider("Minimum value for new points", -2.0, 2.0, -2.0)
new_point_max = st.sidebar.slider("Maximum value for new points", -2.0, 2.0, 2.0)
num_new_points = st.sidebar.slider("Number of new points to add", 1, 100, 10)
new_points_button = st.sidebar.button("Add new points")

left_column, right_column = st.columns(2)

with left_column:
    # Plot decision boundary
    model = MLP(2, [16, 3])
    with st.spinner("Training model..."):
        model = train_model(model, st.session_state.data, n_iters=n_iters)
        fig = plot_decision_boundary(model, st.session_state.data, h=0.1)
        st.pyplot(fig)

with right_column:
    if new_points_button:
        with st.spinner("Adding new points..."):
            new_points = np.random.uniform(
                new_point_min, new_point_max, (num_new_points, 2)
            )

            new_labels = [
                0 if point[0] < 0 else 1 if point[1] < 0 else 2 for point in new_points
            ]
            new_data = [(point, label) for point, label in zip(new_points, new_labels)]
            new_data = st.session_state.data + new_data

        with st.spinner("Training model with new points..."):
            # Retrain the model with the new data
            model = train_model(model, new_data, n_iters=n_iters)
            fig = plot_decision_boundary(
                model,
                new_data,
                h=0.1,
                title="Updated decision boundary",
            )
            st.pyplot(fig)
    else:
        st.write("No new points added yet - use the sidebar to add new points")
