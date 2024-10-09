/*
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// -----------------------------------------------------------------------------
// MICROGRAD ENGINE. ...

// enum Operation{
//     ADD,
//     MUL,
//     TANH,
//     // ...
// };

#define MAX_PREV 2

typedef struct Value {
    double data;
    double grad;
    char* label;
    struct Value* _prev[MAX_PREV];
    int n_prev;
    //enum Operation _op;
    char* _op;
} Value;

// Value* value_add(Value* a, Value* b);
// Value* value_mul(Value* a, Value* b);
// Value* value_tanh(Value* v);
// // ... other operations

int main() {
    Value v = {2.0};

    printf("Value: {data: %f}\n", v.data);
    return 0;
}

// -----------------------------------------------------------------------------
// NEURAL NETWORK.

// typedef struct Neuron {
//     Value* weights;
//     Value bias;
//     int n_inputs;
//     bool is_linear;
// } Neuron;

// void neuron_forward(Neuron* n, Value* inputs[], Value* output);

// Value* create_value(double data);
// void free_value(Value* v);

// Neuron* create_neuron(int num_inputs, bool nonlin);
// void free_neuron(Neuron* n);

// typedef struct Layer {
//     Neuron* neurons;
//     int n_neurons;
// } Layer;

// typedef struct MLP {
//     Layer* layers;
//     int n_layers;
// } MLP;

// -----------------------------------------------------------------------------
// GRAPHING EXAMPLE.

// #define START -5.0
// #define END 5.0
// #define STEP 0.25

// double f(double x) {
//     return 3 * pow(x, 2) - 4 * x + 5;
// }

// int main() {
//     int num_points = (int)((END - START) / STEP) + 1;
//     double *xs = malloc(num_points * sizeof(double));
//     double *ys = malloc(num_points * sizeof(double));

//     if (xs == NULL || ys == NULL) {
//         fprintf(stderr, "Memory allocation failed\n");
//         return 1;
//     }

//     // Generate x values and calculate corresponding y values
//     for (int i = 0; i < num_points; i++) {
//         xs[i] = START + i * STEP;
//         ys[i] = f(xs[i]);
//     }

//     // Output data to a file
//     FILE *fp = fopen("plot_data.txt", "w");
//     if (fp == NULL) {
//         fprintf(stderr, "Failed to open file\n");
//         free(xs);
//         free(ys);
//         return 1;
//     }

//     for (int i = 0; i < num_points; i++) {
//         fprintf(fp, "%f %f\n", xs[i], ys[i]);
//     }

//     fclose(fp);
//     free(xs);
//     free(ys);

//     printf("Data written to plot_data.txt\n");
//     return 0;
// }