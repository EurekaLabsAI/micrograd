/*
Compile and run:
gcc -O3 -Wall -Wextra -Wpedantic -o micrograd micrograd.c && ./micrograd
*/
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------
// utils

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file,
                line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// random number generation

// RNG structure to mimic Python's random interface
typedef struct {
    uint64_t state;
} RNG;

// Initialize RNG with a seed
void rng_init(RNG *rng, uint64_t seed) { rng->state = seed; }

// Generate a random 32-bit unsigned integer
uint32_t rng_random_u32(RNG *rng) {
    rng->state ^= (rng->state >> 12);
    rng->state ^= (rng->state << 25);
    rng->state ^= (rng->state >> 27);
    return (uint32_t)((rng->state * 0x2545F4914F6CDD1DULL) >> 32);
}

// Generate a random float in [0, 1)
float rng_random(RNG *rng) { return (rng_random_u32(rng) >> 8) / 16777216.0f; }

// Generate a random float in [a, b)
float rng_uniform(RNG *rng, float a, float b) {
    return a + (b - a) * rng_random(rng);
}

// ----------------------------------------------------------------------------
// random dataset generation

// Structure to hold a data point
typedef struct {
    float x;
    float y;
    int label;
} DataPoint;

// Generate random dataset
void gen_data(RNG *random, int n, DataPoint **train, int *train_size, DataPoint **val, int *val_size, DataPoint **test, int *test_size) {
    DataPoint *pts = mallocCheck(n * sizeof(DataPoint));

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

    *train = mallocCheck(*train_size * sizeof(DataPoint));
    *val = mallocCheck(*val_size * sizeof(DataPoint));
    *test = mallocCheck(*test_size * sizeof(DataPoint));

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

// ----------------------------------------------------------------------------
// micrograd engine

// stores a single scalar value and its gradient
typedef struct Value {
    double data;
    double grad;
    struct Value **_prev;
    int _prev_count;
    void (*_backward)(struct Value *);
    char *_op;
} Value;

// a list to track all temporary Values that need to be freed using free_tracked_values()
typedef struct {
    Value **values;
    int size;
    int capacity;
} ValueTracker;

// Global temporary Value tracker
// These values are generated during each training iteration of the neural network
// (eg: activations, intermediate outputs, etc.) and are not part of the network's
// parameters. They need to be freed after each training step.
ValueTracker g_value_tracker;

// Initialize the global Value tracker
void value_tracker_init(void) {
    g_value_tracker.size = 0;
    g_value_tracker.capacity = 1024;  // Initial capacity
    g_value_tracker.values = mallocCheck(g_value_tracker.capacity * sizeof(Value *));
}

// Append a Value to the global Value tracker
// It also automatically grows the capacity of the tracker if the current capacity
// is exceeded
void value_tracker_append(Value *v) {
    if (g_value_tracker.size >= g_value_tracker.capacity) {
        g_value_tracker.capacity *= 2;
        g_value_tracker.values = realloc(g_value_tracker.values, g_value_tracker.capacity * sizeof(Value *));
        if (!g_value_tracker.values) {
            fprintf(stderr, "Error: realloc failed in value_tracker_append\n");
            exit(EXIT_FAILURE);
        }
    }
    g_value_tracker.values[g_value_tracker.size++] = v;
}

// Free all Values in the global tracker
void free_tracked_values(void) {
    for (int i = 0; i < g_value_tracker.size; i++) {
        Value *v = g_value_tracker.values[i];
        if (v) {
            free(v->_prev);
            free(v->_op);
            free(v);
        }
    }
    g_value_tracker.size = 0;  // Reset the tracker for the next step
}

// Create a new Value
Value *create_value(double data, Value **children, int n_children, const char *op) {
    Value *v = mallocCheck(sizeof(Value));
    v->data = data;
    v->grad = 0.0;

    if (n_children > 0) {
        v->_prev = mallocCheck(n_children * sizeof(Value *));
        for (int i = 0; i < n_children; i++) {
            v->_prev[i] = children[i];
        }
    } else {
        v->_prev = NULL;
    }
    v->_prev_count = n_children;
    v->_backward = NULL;
    v->_op = strdup(op);
    return v;
}

// Create a new Value and add it to the global Value tracker
Value *create_tracked_value(double data, Value **children, int n_children, const char *op) {
    Value *v = create_value(data, children, n_children, op);
    value_tracker_append(v);
    return v;
}

void value_free(Value *v) {
    if (!v) return;
    free(v->_prev);
    free(v->_op);
    free(v);
}

// Helper function to check if a Value* is in an array
int contains(Value **arr, int size, Value *v) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == v) return 1;
    }
    return 0;
}

// Helper function to grow an array
Value **grow_array(Value **array, int *capacity) {
    *capacity *= 2;
    Value **new_array = realloc(array, *capacity * sizeof(Value *));
    if (!new_array) {
        fprintf(stderr, "Error: realloc failed in grow_array\n");
        exit(EXIT_FAILURE);
    }
    return new_array;
}

void dfs_topo_sort(Value *v, Value ***topo, int *topo_size, int *topo_capacity,
                   Value ***visited, int *visited_size, int *visited_capacity) {
    if (contains(*visited, *visited_size, v)) return;

    if (*visited_size >= *visited_capacity) {
        *visited = grow_array(*visited, visited_capacity);
    }
    (*visited)[(*visited_size)++] = v;

    for (int i = 0; i < v->_prev_count; i++) {
        if (v->_prev[i]) {
            dfs_topo_sort(v->_prev[i], topo, topo_size, topo_capacity, visited,
                          visited_size, visited_capacity);
        }
    }

    if (*topo_size >= *topo_capacity) {
        *topo = grow_array(*topo, topo_capacity);
    }
    (*topo)[(*topo_size)++] = v;
}

Value **build_topo(Value *v, int *topo_size) {
    int topo_capacity = 16000;  // Initial capacity
    int visited_capacity = 16000;

    Value **topo = mallocCheck(topo_capacity * sizeof(Value *));
    *topo_size = 0;
    Value **visited = mallocCheck(visited_capacity * sizeof(Value *));
    int visited_size = 0;

    dfs_topo_sort(v, &topo, topo_size, &topo_capacity, &visited, &visited_size,
                  &visited_capacity);

    free(visited);
    return topo;
}

void value_backward(Value *v) {
    int topo_size;
    Value **topo = build_topo(v, &topo_size);

    // Initialize gradient
    v->grad = 1.0;
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->_backward) {
            topo[i]->_backward(topo[i]);
        }
    }

    free(topo);
}

// Addition
void backward_add(Value *v) {
    if (v->_prev_count != 2) return;
    v->_prev[0]->grad += 1.0 * v->grad;
    v->_prev[1]->grad += 1.0 * v->grad;
}

Value *value_add(Value *a, Value *b) {
    Value *out = create_tracked_value(a->data + b->data, (Value *[]){a, b}, 2, "+");
    out->_backward = backward_add;
    return out;
}

// Multiplication
void backward_mul(Value *v) {
    if (v->_prev_count != 2) return;
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

Value *value_mul(Value *a, Value *b) {
    Value *out = create_tracked_value(a->data * b->data, (Value *[]){a, b}, 2, "*");
    out->_backward = backward_mul;
    return out;
}

// Power
void backward_pow(Value *v) {
    double base = v->_prev[0]->data;
    double exponent = v->_prev[1]->data;
    if (base == 0.0 && exponent < 1.0) {
        // Handle undefined behavior gracefully
        return;
    }
    v->_prev[0]->grad += exponent * pow(base, exponent - 1) * v->grad;
}

Value *value_pow(Value *a, Value *b) {
    Value *out = create_tracked_value(pow(a->data, b->data), (Value *[]){a, b}, 2, "**");
    out->_backward = backward_pow;
    return out;
}

// ReLU
void backward_relu(Value *v) {
    v->_prev[0]->grad += (v->data > 0.0) ? 1.0 * v->grad : 0.0;
}

Value *value_relu(Value *a) {
    Value *out = create_tracked_value((a->data > 0.0) ? a->data : 0.0, (Value *[]){a}, 1, "ReLU");
    out->_backward = backward_relu;
    return out;
}

// Tanh
void backward_tanh(Value *v) {
    if (v->_prev_count != 1) return;
    double tanh_val = tanh(v->_prev[0]->data);
    v->_prev[0]->grad += (1.0 - tanh_val * tanh_val) * v->grad;
}

Value *value_tanh(Value *a) {
    double t = tanh(a->data);
    Value *out = create_tracked_value(t, (Value *[]){a}, 1, "tanh");
    out->_backward = backward_tanh;
    return out;
}

// Exponential
void backward_exp(Value *v) {
    if (v->_prev_count != 1) return;
    double exp_val = exp(v->_prev[0]->data);
    v->_prev[0]->grad += exp_val * v->grad;
}

Value *value_exp(Value *a) {
    double e = exp(a->data);
    Value *out = create_tracked_value(e, (Value *[]){a}, 1, "exp");
    out->_backward = backward_exp;
    return out;
}

// Logarithm
void backward_log(Value *v) {
    if (v->_prev_count != 1) return;
    if (v->_prev[0]->data == 0.0) {
        // Handle log(0) gracefully
        return;
    }
    v->_prev[0]->grad += (1.0 / v->_prev[0]->data) * v->grad;
}

Value *value_log(Value *a) {
    Value *out = create_tracked_value(log(a->data), (Value *[]){a}, 1, "log");
    out->_backward = backward_log;
    return out;
}

// ----------------------------------------------------------------------------
// neural network

// Forward declarations for neuron_call and neuron_parameters
typedef struct Neuron Neuron;  // Forward declaration

// Neuron struct implementation
struct Neuron {
    Value **w;
    Value *b;
    int nin;
    int nonlin;
};

// Function to create a new neuron
Neuron *neuron_new(RNG *random, int nin, int nonlin) {
    Neuron *n = mallocCheck(sizeof(Neuron));
    n->w = mallocCheck(nin * sizeof(Value *));
    for (int i = 0; i < nin; i++) {
        double init_weight = rng_uniform(random, -1.0, 1.0) * pow(nin, -0.5);
        n->w[i] = create_value(init_weight, NULL, 0, "weight");
    }
    n->b = create_value(0.0, NULL, 0, "bias");
    n->nin = nin;
    n->nonlin = nonlin;
    return n;
}

void neuron_free(Neuron *n) {
    for (int i = 0; i < n->nin; i++) {
        value_free(n->w[i]);
    }
    value_free(n->b);
    free(n->w);
    free(n);
}

// Function to perform a forward pass through a neuron
Value *neuron_call(Neuron *n, Value **x) {
    Value *act = n->b;
    for (int i = 0; i < n->nin; i++) {
        Value *mul_result = value_mul(n->w[i], x[i]);
        Value *new_act = value_add(act, mul_result);
        act = new_act;
    }

    if (n->nonlin) {
        Value *tanh_result = value_tanh(act);
        return tanh_result;
    } else {
        return act;
    }
}

// Function to retrieve neuron parameters
Value **neuron_parameters(Neuron *n) {
    Value **params = mallocCheck((n->nin + 1) * sizeof(Value *));
    for (int i = 0; i < n->nin; i++) {
        params[i] = n->w[i];
    }
    params[n->nin] = n->b;
    return params;
}

// Layer struct implementation
typedef struct Layer {
    Neuron **neurons;
    int nin;
    int nout;
} Layer;

// Function to create a new layer
Layer *layer_new(RNG *random, int nin, int nout, int nonlin) {
    Layer *l = mallocCheck(sizeof(Layer));
    l->neurons = mallocCheck(nout * sizeof(Neuron *));
    for (int i = 0; i < nout; i++) {
        l->neurons[i] = neuron_new(random, nin, nonlin);
    }
    l->nin = nin;
    l->nout = nout;
    return l;
}

// Function to free a layer
void layer_free(Layer *l) {
    for (int i = 0; i < l->nout; i++) {
        neuron_free(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

// Function to perform a forward pass through a layer
Value **layer_call(Layer *l, Value **x) {
    Value **out = mallocCheck(l->nout * sizeof(Value *));
    for (int i = 0; i < l->nout; i++) {
        out[i] = neuron_call(l->neurons[i], x);
    }
    return out;
}

// Function to retrieve layer parameters
Value **layer_parameters(Layer *l) {
    Value **params = mallocCheck(l->nout * (l->nin + 1) * sizeof(Value *));
    int idx = 0;
    for (int i = 0; i < l->nout; i++) {
        Value **neuron_params = neuron_parameters(l->neurons[i]);
        for (int j = 0; j < l->nin + 1; j++) {
            params[idx++] = neuron_params[j];
        }
        free(neuron_params);
    }
    return params;
}

// MLP struct implementation
typedef struct MLP {
    Layer **layers;
    int n_layers;
} MLP;

// Function to create a new MLP
MLP *mlp_new(RNG *random, int nin, int *nouts, int n_layers) {
    MLP *m = mallocCheck(sizeof(MLP));
    m->layers = mallocCheck(n_layers * sizeof(Layer *));
    m->n_layers = n_layers;
    int sizes[n_layers + 1];
    sizes[0] = nin;
    for (int i = 0; i < n_layers; i++) {
        sizes[i + 1] = nouts[i];
    }
    for (int i = 0; i < n_layers; i++) {
        // Last layer is linear (no nonlinearity)
        m->layers[i] =
            layer_new(random, sizes[i], sizes[i + 1], i != n_layers - 1);
    }
    return m;
}

// Function to free an MLP
void mlp_free(MLP *m) {
    for (int i = 0; i < m->n_layers; i++) {
        layer_free(m->layers[i]);
    }
    free(m->layers);
    free(m);
}

// Function to perform a forward pass through the MLP
Value **mlp_call(MLP *m, Value **x) {
    Value **out = x;
    for (int i = 0; i < m->n_layers; i++) {
        Value **out2 = layer_call(m->layers[i], out);
        if (i > 0)  // Free the previous layer's output but not the input
            free(out);
        out = out2;
    }
    return out;
}

// Function to retrieve all MLP parameters
Value **mlp_parameters(MLP *m) {
    int total_params = 0;
    for (int i = 0; i < m->n_layers; i++) {
        total_params += m->layers[i]->nout * (m->layers[i]->nin + 1);
    }
    Value **params = mallocCheck(total_params * sizeof(Value *));
    int idx = 0;
    for (int i = 0; i < m->n_layers; i++) {
        Value **layer_params = layer_parameters(m->layers[i]);
        for (int j = 0; j < m->layers[i]->nout * (m->layers[i]->nin + 1); j++) {
            params[idx++] = layer_params[j];
        }
        free(layer_params);
    }
    return params;
}

// Loss function
Value *cross_entropy(Value **logits, int n_logits, int target) {
    double max_val = -DBL_MAX;
    for (int i = 0; i < n_logits; i++) {
        if (logits[i]->data > max_val) max_val = logits[i]->data;
    }
    Value **ex = mallocCheck(n_logits * sizeof(Value *));
    Value *denom = create_tracked_value(0.0, NULL, 0, "denominator");
    for (int i = 0; i < n_logits; i++) {
        Value *shifted = value_add(logits[i], create_tracked_value(-max_val, NULL, 0, "shift"));
        ex[i] = value_exp(shifted);
        denom = value_add(denom, ex[i]);
    }
    Value *exponent = create_tracked_value(-1.0, NULL, 0, "exponent");
    Value *prob_target = value_mul(ex[target], value_pow(denom, exponent));
    Value *logp = value_log(prob_target);
    Value *nll = value_mul(logp, create_tracked_value(-1.0, NULL, 0, "nll"));

    free(ex);
    return nll;
}

// Evaluation function
double eval_split(MLP *model, DataPoint *split, int split_size) {
    Value *loss = create_tracked_value(0.0, NULL, 0, "eval_loss");
    for (int i = 0; i < split_size; i++) {
        Value *x[2] = {create_tracked_value(split[i].x, NULL, 0, "input_x"),
                       create_tracked_value(split[i].y, NULL, 0, "input_y")};
        Value **logits = mlp_call(model, x);
        Value *ce = cross_entropy(logits, 3, split[i].label);
        loss = value_add(loss, ce);
        free(logits);  // Free the array of logits (Values are tracked)
    }
    double result = loss->data / split_size;
    free_tracked_values();
    return result;
}

// Training function
void train(MLP *model, DataPoint *train_split, int train_size, DataPoint *val_split, int val_size) {
    double learning_rate = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.95;
    double weight_decay = 1e-4;

    Value **params = mlp_parameters(model);
    int n_params = 0;
    for (int i = 0; i < model->n_layers; i++) {
        n_params += model->layers[i]->nout * (model->layers[i]->nin + 1);
    }
    double *m = calloc(n_params, sizeof(double));
    double *v = calloc(n_params, sizeof(double));

    for (int step = 0; step < 100; step++) {
        if (step % 10 == 0) {
            double val_loss = eval_split(model, val_split, val_size);
            printf("step %d, val loss %.12f\n", step, val_loss);
        }

        Value *loss = create_tracked_value(0.0, NULL, 0, "train_loss");
        for (int i = 0; i < train_size; i++) {
            Value *x[2] = {
                create_tracked_value(train_split[i].x, NULL, 0, "input_x"),
                create_tracked_value(train_split[i].y, NULL, 0, "input_y")};
            Value **logits = mlp_call(model, x);
            Value *ce = cross_entropy(logits, 3, train_split[i].label);
            loss = value_add(loss, ce);
            free(logits);  // Free the array of logits (Values are tracked)
        }
        loss = value_mul(loss, create_tracked_value(1.0 / train_size, NULL, 0, "scale"));

        value_backward(loss);  // Compute gradients

        for (int i = 0; i < n_params; i++) {
            double grad = params[i]->grad;
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            double m_hat = m[i] / (1 - pow(beta1, step + 1));
            double v_hat = v[i] / (1 - pow(beta2, step + 1));
            params[i]->data -= learning_rate * (m_hat / (sqrt(v_hat) + 1e-8) +
                                                weight_decay * params[i]->data);
            params[i]->grad = 0;  // Zero the gradients
        }

        printf("step %d, train loss %.12f\n", step, loss->data);
        free_tracked_values();
    }

    free(m);
    free(v);
    free(params);
}

// ----------------------------------------------------------------------------

#ifndef TESTING
// if we are TESTING (see test_micrograd.c), we'll skip the int main below
int main(void) {
    value_tracker_init();

    RNG rng;
    rng_init(&rng, 42);

    DataPoint *train_split, *val_split, *test_split;
    int train_size, val_size, test_size;
    gen_data(&rng, 100, &train_split, &train_size, &val_split, &val_size, &test_split, &test_size);

    MLP *model = mlp_new(&rng, 2, (int[]){16, 3}, 2);
    train(model, train_split, train_size, val_split, val_size);

    // Clean up
    free(train_split);
    free(val_split);
    free(test_split);
    mlp_free(model);

    // Free the Value tracker itself 
    free(g_value_tracker.values);

    return 0;
}
#endif
