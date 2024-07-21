// mimic micrograd RNG
class RNG {
    constructor(seed) {
        this.state = BigInt(seed);
    }

    random_u32() {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        this.state = BigInt.asUintN(64, this.state);
        this.state ^= (this.state >> 12n) & 0xFFFFFFFFFFFFFFFFn;
        this.state ^= (this.state << 25n) & 0xFFFFFFFFFFFFFFFFn;
        this.state ^= (this.state >> 27n) & 0xFFFFFFFFFFFFFFFFn;
        
        return Number((this.state * 0x2545F4914F6CDD1Dn >> 32n) & 0xFFFFFFFFn);
    }

    random() {
        // random Number in [0, 1)
        return (this.random_u32() >>> 8) / 16777216.0;
    }

    uniform(a = 0.0, b = 1.0) {
        // random Number in [a, b)
        return a + (b - a) * this.random();
    }
}

function gen_data(random, n = 100) {
    let pts = [];
    for (let i = 0; i < n; i++) {
        let x = random.uniform(-2.0, 2.0);
        let y = random.uniform(-2.0, 2.0);
        // concentric circles
        // label = 0 if x**2 + y**2 < 1 else 1 if x**2 + y**2 < 2 else 2
        // very simple dataset
        let label = x < 0 ? 0 : y < 0 ? 1 : 2;
        pts.push([[x, y], label]);
    }
    // create train/val/test splits of the data (80%, 10%, 10%)
    let tr = pts.slice(0, Math.floor(0.8 * n));
    let val = pts.slice(Math.floor(0.8 * n), Math.floor(0.9 * n));
    let te = pts.slice(Math.floor(0.9 * n));
    return [tr, val, te];
}

// Export the RNG and gen_data function
module.exports = { RNG, gen_data };

// For equivalence testing
if (require.main === module) {
    let rng = new RNG(42);  // Use seed 42
    console.log("Testing random():");
    for (let i = 0; i < 1000; i++) {
        console.log(rng.random().toFixed(20));
    }
    
    console.log("\nTesting uniform(-10.0, 10.0):");
    for (let i = 0; i < 1000; i++) {
        console.log(rng.uniform(-10.0, 10.0).toFixed(20));
    }
    
    console.log("\nTesting random_u32():");
    for (let i = 0; i < 1000; i++) {
        console.log(rng.random_u32());
    }
}
