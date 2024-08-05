#include <stdlib.h>

typedef struct RNG
{
    unsigned long long int state;
} RNG;

u_int32_t rng_random_u32(RNG* rng) {
    rng->state ^= (rng->state >> 12) & 0xFFFFFFFFFFFFFFFF;
    rng->state ^= (rng->state << 25) & 0xFFFFFFFFFFFFFFFF;
    rng->state ^= (rng->state >> 27) & 0xFFFFFFFFFFFFFFFF;
    return ((rng->state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF;
}

double rng_random(RNG *rng) {
    return (rng_random_u32(rng) >> 8) / 16777216.0;
}

double rng_uniform(RNG *rng, float beg, int end) {
    return beg + (end - beg) * rng_random(rng);
}

typedef struct DataPoint{
   double x, y;
   int label; 
} DataPoint;

typedef struct DataSet {
    int tr_size, val_size, te_size;
    DataPoint **tr, **val, **te;
} DataSet;


DataPoint* construct_datapoint(double x, double y, int label) {
    DataPoint* p = malloc(sizeof(DataPoint));
    p->x = x;
    p->y = y;
    p->label = label;
    return p;
}

DataSet* construct_dataset(DataPoint **tr, DataPoint **val, DataPoint **te,
int tr_size, int val_size, int te_size) {
    DataSet *dataset = malloc(sizeof(DataSet));
    dataset->tr_size = tr_size;
    dataset->val_size = val_size;
    dataset->te_size = te_size;
    dataset->tr = tr;
    dataset->val = val;
    dataset->te = te;
    return dataset;
}

DataPoint* deep_copy_datapoint(DataPoint *ptr) {
    DataPoint *copy = malloc(sizeof(DataPoint));
    copy->x = ptr->x;
    copy->y = ptr->y;
    copy->label = ptr->label;
    return copy;
}

DataPoint** slice(DataPoint **ptr, int beg, int limit) {
    int size = limit - beg;
    DataPoint **result = calloc(size, sizeof(DataPoint));
    for(int i = beg; i < limit; i++) {
        result[i - beg] = deep_copy_datapoint(ptr[i]);
    }
    return result;
}
DataSet* generate(int n, RNG *rng) {
    DataPoint **pts = calloc(n, sizeof(DataPoint*));
    for(int i = 0; i < n; i++) {
        double x = rng_uniform(rng, -2.0, 2.0);
        double y = rng_uniform(rng, -2.0, 2.0);
        int label = (x < 0) ? 0: (y < 0)? 1 : 2;
        pts[i] = construct_datapoint(x, y, label);
    }
    DataPoint **tr = slice(pts, 0, 0.8 * n);
    DataPoint **val = slice(pts, 0.8 * n , 0.9 * n);
    DataPoint **te = slice(pts, 0.9 * n, n);
    DataSet *dataset = construct_dataset(tr, val, te, 0.8 * n, 0.1 * n, 0.1 * n);
    return dataset;
}
