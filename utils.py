# -----------------------------------------------------------------------------

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def uniform(self, a=0.0, b=1.0):
        # random float32 in [a, b)
        return a + (b-a) * self.random()

# generate a random dataset with 100 2-dimensional datapoints in 3 classes
def gen_data(random: RNG, n=100):
    pts = []
    for _ in range(n):
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        # concentric circles
        # label = 0 if x**2 + y**2 < 1 else 1 if x**2 + y**2 < 2 else 2
        # very simple dataset
        label = 0 if x < 0 else 1 if y < 0 else 2
        pts.append(([x, y], label))
    # create train/val/test splits of the data (80%, 10%, 10%)
    tr = pts[:int(0.8*n)]
    val = pts[int(0.8*n):int(0.9*n)]
    te = pts[int(0.9*n):]
    return tr, val, te