# proper tests
import pytest
from collections import Counter
from utils import RNG, gen_data

# Fixtures
@pytest.fixture
def rng():
    return RNG(seed=42)

def test_rng_initialization(rng):
    assert rng.state == 42

def test_random_u32(rng):
    """Test random_u32 for determinism and range."""
    rng2 = RNG(42)
    for _ in range(1000):
        num = rng.random_u32()
        assert num == rng2.random_u32()  # Determinism
        assert isinstance(num, int)
        assert 0 <= num < 2**32  # Range

def test_random(rng):
    """Test random for determinism and range."""
    rng2 = RNG(42)
    for _ in range(1000):
        num = rng.random()
        assert num == rng2.random()  # Determinism
        assert isinstance(num, float)
        assert 0.0 <= num < 1.0  # Range

def test_uniform(rng):
    """Test uniform for determinism, range, and distribution."""
    rng2 = RNG(42)
    a, b = -3.5, 8.2
    samples = [rng.uniform(a, b) for _ in range(10000)]
    for _ in range(1000):
        assert rng.uniform(a, b) == rng2.uniform(a, b)  # Determinism
        assert a <= rng.uniform(a, b) < b  # Range
    assert abs(sum(samples) / len(samples) - (a + b) / 2) < 0.1  # Approx. mean
    assert abs(max(samples) - b) < 1e-6  # Max should be close to b
    assert abs(min(samples) - a) < 1e-6  # Min should be close to a

def test_rng_reproducibility():
    rng1 = RNG(42)
    rng2 = RNG(42)
    assert [rng1.random() for _ in range(10)] == [rng2.random() for _ in range(10)]

def test_gen_data_determinism():
    """Test gen_data for determinism."""
    rng1 = RNG(101112)
    rng2 = RNG(101112)
    data1 = gen_data(rng1)
    data2 = gen_data(rng2)
    assert data1 == data2

def test_gen_data_splits():
    rng = RNG(232425)
    n = 200
    tr, val, te = gen_data(rng, n)
    assert len(tr) == int(0.8 * n)
    assert len(val) == int(0.1 * n)
    assert len(te) == int(0.1 * n)

def test_gen_data_labels():
    rng = RNG(343536)
    _, _, te = gen_data(rng)
    for pt, label in te:
        x, y = pt
        assert label == (0 if x < 0 else 1 if y < 0 else 2)

def test_gen_data_output():
    rng = RNG(42)
    tr, val, te = gen_data(rng, n=100)
    
    assert len(tr) == 80
    assert len(val) == 10
    assert len(te) == 10

    for dataset in [tr, val, te]:
        for point, label in dataset:
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)
            assert label in [0, 1, 2]

def test_gen_data_distribution():
    rng = RNG(42)
    tr, val, te = gen_data(rng, n=1000)
    all_data = tr + val + te
    
    x_values = [point[0] for point, _ in all_data]
    y_values = [point[1] for point, _ in all_data]
    
    assert -2 <= min(x_values) < -1.9
    assert 1.9 < max(x_values) <= 2
    assert -2 <= min(y_values) < -1.9
    assert 1.9 < max(y_values) <= 2

def test_gen_data_label_assignment():
    rng = RNG(42)
    tr, val, te = gen_data(rng, n=1000)
    all_data = tr + val + te
    
    for point, label in all_data:
        x, y = point
        if x < 0:
            assert label == 0
        elif y < 0:
            assert label == 1
        else:
            assert label == 2

@pytest.mark.parametrize("n", [10, 100, 1000])
def test_gen_data_different_sizes(n):
    rng = RNG(42)
    tr, val, te = gen_data(rng, n=n)
    assert len(tr) == int(0.8 * n)
    assert len(val) == int(0.1 * n)
    assert len(te) == n - len(tr) - len(val)

def test_gen_data_labels_distribution(rng):
    tr, val, te = gen_data(rng, n=100)

    # Check if the labels are somewhat evenly distributed (this is not a strict test)
    all_labels = [label for _, label in tr + val + te]
    label_counts = Counter(all_labels)
    assert len(label_counts) == 3
    # the label is defined as `0 if x < 0 else 1 if y < 0 else 2`
    # so about 50% in label 0, 25% in label 1 and 2
    expected = [50, 25, 25]
    assert all([expected[label] - 10 <= label_counts[label] <= expected[label] + 10 for label in (0, 1, 2)])

def test_gen_data_split(rng):
    tr, val, te = gen_data(rng, n=100)

    # Ensure there is no overlap between train, val, and test sets
    tr_set = set((tuple(point), label) for point, label in tr)
    val_set = set((tuple(point), label) for point, label in val)
    te_set = set((tuple(point), label) for point, label in te)

    assert tr_set.isdisjoint(val_set)
    assert tr_set.isdisjoint(te_set)
    assert val_set.isdisjoint(te_set)

# to test the consistency across languages (Python and C), we hardcode the expected values
def test_random(rng):
    assert rng.random() == 0.3390852212905884
    assert rng.random() == 0.7822558283805847
    assert rng.random() == 0.790136992931366

def test_uniform(rng):
    assert rng.uniform(-1, 1) == -0.32182955741882324
    assert rng.uniform(-1, 1) == 0.5645116567611694
    assert rng.uniform(-1, 1) == 0.5802739858627319

def test_gen_data(rng):
    tr, val, te = gen_data(rng, n=100)
    assert len(tr) == 80
    assert len(val) == 10
    assert len(te) == 10
    assert tr[0] == ([-0.6436591148376465, 1.1290233135223389], 0)
    assert val[0] == ([-0.6666977405548096, 1.137477159500122], 0)
    assert te[0] == ([-1.871429443359375, 0.952826976776123], 0)
    assert tr[-1] == ([0.6179506778717041, -0.6032657623291016], 1)
    assert val[-1] == ([-0.5639073848724365, -0.4837164878845215], 0)
    assert te[-1] == ([-0.9766521453857422, -0.15662121772766113], 0)

if __name__ == "__main__":
    pytest.main()
