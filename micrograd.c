#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Value {
    double data;
    double grad;
    void (*_backward)(struct Value*);
    struct Value* _prev;
    struct Value* _op;
} Value;

typedef struct Neuron {
    Value* weights;
    Value bias;
    int n_inputs;
    bool is_linear;
} Neuron;

Value* value_add(Value* a, Value* b);
Value* value_mul(Value* a, Value* b);
Value* value_tanh(Value* v);
// ... other operations

void neuron_forward(Neuron* n, Value* inputs[], Value* output);

Value* create_value(double data);
void free_value(Value* v);

Neuron* create_neuron(int num_inputs, bool nonlin);
void free_neuron(Neuron* n);

typedef struct Layer {
    Neuron* neurons;
    int n_neurons;
} Layer;

typedef struct MLP {
    Layer* layers;
    int n_layers;
} MLP;

// -----------------------------------------------------------------------------
// ...
int main() {
    printf("Hello world\n");
    return 0;
}