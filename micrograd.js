/*
Class that mimics the random interface in Python, fully deterministic,
and in a way that we also control fully, and can also use in C, etc.
*/
class RNG {
    constructor(seed) {
        this.state = BigInt(seed);
    }

    random_u32() {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        this.state ^= (this.state >> 12n) & 0xFFFFFFFFFFFFFFFFn;
        this.state ^= (this.state << 25n) & 0xFFFFFFFFFFFFFFFFn;
        this.state ^= (this.state >> 27n) & 0xFFFFFFFFFFFFFFFFn;
        return Number((this.state * 0x2545F4914F6CDD1Dn >> 32n) & 0xFFFFFFFFn);
    }

    random() {
        // random float32 in [0, 1)
        return (this.random_u32() >>> 8) / 16777216.0;
    }

    uniform(a = 0.0, b = 1.0) {
        // random float32 in [a, b)
        return a + (b - a) * this.random();
    }
}

/*
Simple dataset generation function that generates a dataset of n points
in 2D space, with labels 0, 1, 2. The dataset is split into training,
validation, and test sets (80%, 10%, 10%).
*/
function genData(random, n = 100) {
    const pts = [];
    for (let i = 0; i < n; i++) {
        const x = random.uniform(-2.0, 2.0);
        const y = random.uniform(-2.0, 2.0);
        // Very simple dataset
        const label = x < 0 ? 0 : y < 0 ? 1 : 2;
        // Uncomment the following line and comment out the above line to use concentric circles instead
        // const label = x**2 + y**2 < 1 ? 0 : x**2 + y**2 < 2 ? 1 : 2;
        pts.push([[x, y], label]);
    }
    // Create train/val/test splits of the data (80%, 10%, 10%)
    const tr = pts.slice(0, Math.floor(0.8 * n));
    const val = pts.slice(Math.floor(0.8 * n), Math.floor(0.9 * n));
    const te = pts.slice(Math.floor(0.9 * n));
    return { train: tr, validation: val, test: te };
}

// Create an instance of RNG with seed 42
const random = new RNG(42);
// Generate data using the genData function
const dataSplits = genData(random, 100);
const trainSplit = dataSplits.train;
const valSplit = dataSplits.validation;
const testSplit = dataSplits.test;

// Function to format a data point for console output
function formatDataPoint(dataPoint) {
    const [[x, y], label] = dataPoint;
    return `[${x.toFixed(4)}, ${y.toFixed(4)}], ${label}`;
}

// Print the first 3 elements of all splits to console
console.log("First 3 elements of training split:");
trainSplit.slice(0, 3).forEach((dataPoint, index) => {
    console.log(`  ${index + 1}: ${formatDataPoint(dataPoint)}`);
});

console.log("\nFirst 3 elements of validation split:");
valSplit.slice(0, 3).forEach((dataPoint, index) => {
    console.log(`  ${index + 1}: ${formatDataPoint(dataPoint)}`);
});

console.log("\nFirst 3 elements of test split:");
testSplit.slice(0, 3).forEach((dataPoint, index) => {
    console.log(`  ${index + 1}: ${formatDataPoint(dataPoint)}`);
});
