// Create an instance of RNG with seed 42
const random = new RNG(42);
// Generate data using the genData function
const dataSplits = genDataYinYang(random, 100);
const trainSplit = dataSplits.train;
const valSplit = dataSplits.validation;
const testSplit = dataSplits.test;
// init the model: 2D inputs, 8 neurons, 3 outputs (logits)
const model = new MLP(2, [8, 3]);
// optimize using AdamW
const optimizer = new AdamW(model.parameters(), 1e-1, [0.9, 0.95], 1e-8, 1e-4);

let currentPage = 1;
const rowsPerPage = 10;

function renderOptimizerParam() {
  const table = document
    .getElementById("param-table")
    .getElementsByTagName("tbody")[0];

  const stepCounter = document.getElementById("step-counter-optimizer");
  stepCounter.textContent = `Step ${step + 1}`;

  while (table.rows.length > 0) {
    table.deleteRow(0);
  }

  const start = (currentPage - 1) * rowsPerPage;
  const end = start + rowsPerPage;
  const paginatedParams = optimizer.parameters.slice(start, end);

  paginatedParams.forEach((param, index) => {
    const row = table.insertRow();
    row.insertCell().textContent = (start + index + 1).toString();
    row.insertCell().textContent = param.data.toFixed(6);
    row.insertCell().textContent = param.grad.toFixed(6);
    row.insertCell().textContent = param.m.toFixed(6);
    row.insertCell().textContent = param.v.toFixed(6);
  });

  renderPaginationControls();
}

function renderPaginationControls() {
  const paginationContainer = document.getElementById("pagination-controls");
  paginationContainer.innerHTML = "";

  const totalPages = Math.ceil(optimizer.parameters.length / rowsPerPage);

  for (let i = 1; i <= totalPages; i++) {
    const button = document.createElement("button");
    button.textContent = i;
    button.classList.add("pagination-button");
    if (i === currentPage) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => {
      currentPage = i;
      renderOptimizerParam();
    });
    paginationContainer.appendChild(button);
  }
}

function train_step() {
  // forward the network and the loss function on all training datapoints
  let loss = new Value(0);
  for (const [x, y] of trainSplit) {
    const logits = model.forward([new Value(x[0]), new Value(x[1])]);
    loss = loss.add(crossEntropy(logits, y));
  }
  loss = loss.mul(1.0 / trainSplit.length); // normalize the loss
  // backward pass (deposit the gradients)
  loss.backward();
  // update with AdamW
  optimizer.step();

  // render the optimizer parameters
  renderOptimizerParam();

  optimizer.zeroGrad();
  return loss.data;
}
function render(minX = -2, minY = -2, maxX = 2, maxY = 2) {
  // first render the datapoints
  const canvas = document.getElementById("decision-canvas");
  const ctx = canvas.getContext("2d");

  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Function to map data points to canvas coordinates
  function mapToCanvas(x, y) {
    const canvasX = (x - minX) * (canvas.width / (maxX - minX));
    const canvasY = (maxY - y) * (canvas.height / (maxY - minY));
    return [canvasX, canvasY];
  }

  // Render the current decision surface
  const stepSize = 0.1;
  const rectWidth = (stepSize * canvas.width) / (maxX - minX);
  const rectHeight = (stepSize * canvas.height) / (maxY - minY);
  for (let x = minX; x < maxX; x += stepSize) {
    for (let y = minY; y < maxY; y += stepSize) {
      const centerX = x + stepSize / 2;
      const centerY = y + stepSize / 2;
      const logits = model.forward([new Value(centerX), new Value(centerY)]);
      const exps = logits.map((logit) => Math.exp(logit.data));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((exp) => exp / sumExps);

      const r = Math.floor(probs[0] * 255);
      const g = Math.floor(probs[1] * 255);
      const b = Math.floor(probs[2] * 255);

      const [canvasX, canvasY] = mapToCanvas(x, y);
      const [canvasX2, canvasY2] = mapToCanvas(x + stepSize, y + stepSize);
      const width = canvasX2 - canvasX;
      const height = canvasY - canvasY2;
      const mutedR = Math.floor(r + (255 - r) * 0.5);
      const mutedG = Math.floor(g + (255 - g) * 0.5);
      const mutedB = Math.floor(b + (255 - b) * 0.5);
      ctx.fillStyle = `rgb(${mutedR},${mutedG},${mutedB})`;
      ctx.strokeStyle = `rgb(${mutedR},${mutedG},${mutedB})`;
      ctx.fillRect(canvasX, canvasY2, width, height);
      ctx.strokeRect(canvasX, canvasY2, width, height);
    }
  }

  // Render training data points
  for (const [x, y] of trainSplit) {
    const [canvasX, canvasY] = mapToCanvas(x[0], x[1]);
    ctx.fillStyle = y === 0 ? "red" : y === 1 ? "green" : "blue";
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "black";
    ctx.stroke();
  }

  // Render the 0-level set of all individual neurons
  for (const neuron of model.layers[0].neurons) {
    const w0 = neuron.w[0].data;
    const w1 = neuron.w[1].data;
    const b = neuron.b.data;
    const x1 = -2;
    const y1 = (-b - w0 * x1) / w1;
    const x2 = 2;
    const y2 = (-b - w0 * x2) / w1;
    const [canvasX1, canvasY1] = mapToCanvas(x1, y1);
    const [canvasX2, canvasY2] = mapToCanvas(x2, y2);
    ctx.strokeStyle = "white";
    ctx.beginPath();
    ctx.moveTo(canvasX1, canvasY1);
    ctx.lineTo(canvasX2, canvasY2);
    ctx.stroke();
  }
}

let step = 0;
function trainAndRenderStep() {
  if (step < 100) {
    // evaluate the validation split every few steps
    if (step % 10 === 0) {
      const valLoss = evalSplit(model, valSplit);
      console.log(`step ${step + 1}, val loss ${valLoss.toFixed(6)}`);
    }
    // train for one iteration
    const trainLoss = train_step();
    console.log(`step ${step + 1}, train loss ${trainLoss}`);

    // render the current state
    render();
    document.getElementById("step-counter-decision").textContent =
      `Step ${step + 1}`;

    step++;
    setTimeout(trainAndRenderStep, 100);
  }
}

trainAndRenderStep();
