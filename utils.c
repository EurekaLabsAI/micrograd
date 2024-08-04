#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// RNG structure to mimic Python's random interface
typedef struct {
    uint64_t state;
} RNG;

// Initialize RNG with a seed
void rng_init(RNG* rng, uint64_t seed) {
    rng->state = seed;
}

// Generate a random 32-bit unsigned integer
uint32_t rng_random_u32(RNG* rng) {
    rng->state ^= (rng->state >> 12);
    rng->state ^= (rng->state << 25);
    rng->state ^= (rng->state >> 27);
    return (uint32_t)((rng->state * 0x2545F4914F6CDD1DULL) >> 32);
}

// Generate a random float in [0, 1)
float rng_random(RNG* rng) {
    return (rng_random_u32(rng) >> 8) / 16777216.0f;
}

// Generate a random float in [a, b)
float rng_uniform(RNG* rng, float a, float b) {
    return a + (b - a) * rng_random(rng);
}

// Structure to hold a data point
typedef struct {
    float x;
    float y;
    int label;
} DataPoint;

// Generate random dataset
void gen_data(RNG* random, int n, DataPoint** train, int* train_size, 
              DataPoint** val, int* val_size, DataPoint** test, int* test_size) {
    DataPoint* pts = malloc(n * sizeof(DataPoint));
    
    for (int i = 0; i < n; i++) {
        float x = rng_uniform(random, -2.0f, 2.0f);
        float y = rng_uniform(random, -2.0f, 2.0f);
        
        // Very simple dataset
        int label = (x < 0) ? 0 : (y < 0) ? 1 : 2;
        
        pts[i] = (DataPoint){x, y, label};
    }
    
    // Create train/val/test splits (80%, 10%, 10%)
    *train_size = (int)(0.8f * n);
    *val_size = (int)(0.1f * n);
    *test_size = n - *train_size - *val_size;
    
    *train = malloc(*train_size * sizeof(DataPoint));
    *val = malloc(*val_size * sizeof(DataPoint));
    *test = malloc(*test_size * sizeof(DataPoint));
    
    for (int i = 0; i < *train_size; i++) {
        (*train)[i] = pts[i];
    }
    for (int i = 0; i < *val_size; i++) {
        (*val)[i] = pts[*train_size + i];
    }
    for (int i = 0; i < *test_size; i++) {
        (*test)[i] = pts[*train_size + *val_size + i];
    }
    
    free(pts);
}
