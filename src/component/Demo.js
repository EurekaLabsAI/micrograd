import React, { useRef, useEffect } from 'react'
import { crossEntropy, AdamW, MLP, RNG, genDataYinYang, Value, evalSplit } from './micrograd';





const Demo = props => {

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
        optimizer.zeroGrad();
        return loss.data;
    }
    function render(ctx, minX = -2, minY = -2, maxX = 2, maxY = 2) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

        // Function to map data points to canvas coordinates
        function mapToCanvas(x, y) {
            const canvasX = (x - minX) * (ctx.canvas.width / (maxX - minX));
            const canvasY = (maxY - y) * (ctx.canvas.height / (maxY - minY));
            return [canvasX, canvasY];
        }

        // Render the current decision surface
        const stepSize = 0.1;
        const rectWidth = stepSize * ctx.canvas.width / (maxX - minX);
        const rectHeight = stepSize * ctx.canvas.height / (maxY - minY);
        for (let x = minX; x < maxX; x += stepSize) {
            for (let y = minY; y < maxY; y += stepSize) {
                const centerX = x + stepSize / 2;
                const centerY = y + stepSize / 2;
                const logits = model.forward([new Value(centerX), new Value(centerY)]);
                const exps = logits.map(logit => Math.exp(logit.data));
                const sumExps = exps.reduce((a, b) => a + b, 0);
                const probs = exps.map(exp => exp / sumExps);

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
            ctx.fillStyle = y === 0 ? 'red' : y === 1 ? 'green' : 'blue';
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = 'black';
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
            ctx.strokeStyle = 'white';
            ctx.beginPath();
            ctx.moveTo(canvasX1, canvasY1);
            ctx.lineTo(canvasX2, canvasY2);
            ctx.stroke();
        }
    }


    function trainAndRenderStep(ctx, step) {
        if (step < 100) {
            // evaluate the validation split every few steps
            if (step % 10 === 0) {
                const valLoss = evalSplit(model, valSplit);
                console.log(`step ${step}, val loss ${valLoss.toFixed(6)}`);
            }
            // train for one iteration
            const trainLoss = train_step();
            console.log(`step ${step}, train loss ${trainLoss}`);

            // render the current state
            render(ctx);

            step++;
            setTimeout(trainAndRenderStep, 100);
        }
    }



    const canvasRef = useRef(null)

    const draw = (ctx, frameCount) => {


        let step = 0;
        trainAndRenderStep(ctx, step);








    }

    useEffect(() => {

        const canvas = canvasRef.current
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        const context = canvas.getContext('2d')
        let frameCount = 0
        let animationFrameId

        //Our draw came here
        const render = () => {
            frameCount++
            draw(context, frameCount)
            animationFrameId = window.requestAnimationFrame(render)
        }
        render()

        return () => {
            window.cancelAnimationFrame(animationFrameId)
        }
    }, [draw])

    return <canvas ref={canvasRef} {...props} />
}

export default Demo