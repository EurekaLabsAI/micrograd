#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include "utils.c"

// Value struct implementation
typedef struct Value
{
    double data;
    double grad;
    struct Value **_prev;
    int _prev_count;
    void (*_backward)(struct Value *);
    char *_op;
} Value;

Value *value_new(double data, Value **children, int n_children, const char *op)
{
    Value *v = malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->_prev = malloc(n_children * sizeof(Value *));
    for (int i = 0; i < n_children; i++)
    {
        v->_prev[i] = children[i];
    }
    v->_prev_count = n_children;
    v->_backward = NULL;
    v->_op = strdup(op);
    return v;
}

void value_free(Value *v)
{
    free(v->_prev);
    free(v->_op);
    free(v);
}

void backward_add(Value *v)
{
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
}

Value *value_add(Value *a, Value *b)
{
    Value *out = value_new(a->data + b->data, (Value *[]){a, b}, 2, "+");
    out->_backward = backward_add;
    return out;
}

void backward_mul(Value *v)
{
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

Value *value_mul(Value *a, Value *b)
{
    Value *out = value_new(a->data * b->data, (Value *[]){a, b}, 2, "*");
    out->_backward = backward_mul;
    return out;
}

void backward_pow(Value *v)
{
    double b = v->_prev[1]->data;
    v->_prev[0]->grad += (b * pow(v->_prev[0]->data, b - 1)) * v->grad;
}

Value *value_pow(Value *a, double b)
{
    Value *b_val = value_new(b, NULL, 0, "");
    Value *out = value_new(pow(a->data, b), (Value *[]){a, b_val}, 2, "**");
    out->_backward = backward_pow;
    return out;
}

void backward_relu(Value *v)
{
    v->_prev[0]->grad += (v->data > 0) * v->grad;
}

Value *value_relu(Value *a)
{
    Value *out = value_new(a->data < 0 ? 0 : a->data, (Value *[]){a}, 1, "ReLU");
    out->_backward = backward_relu;
    return out;
}

void backward_tanh(Value *v)
{
    v->_prev[0]->grad += (1 - v->data * v->data) * v->grad;
}

Value *value_tanh(Value *a)
{
    double t = tanh(a->data);
    Value *out = value_new(t, (Value *[]){a}, 1, "tanh");
    out->_backward = backward_tanh;
    return out;
}

void backward_exp(Value *v)
{
    v->_prev[0]->grad += v->data * v->grad;
}

Value *value_exp(Value *a)
{
    double e = exp(a->data);
    Value *out = value_new(e, (Value *[]){a}, 1, "exp");
    out->_backward = backward_exp;
    return out;
}

void backward_log(Value *v)
{
    v->_prev[0]->grad += (1 / v->_prev[0]->data) * v->grad;
}

Value *value_log(Value *a)
{
    Value *out = value_new(log(a->data), (Value *[]){a}, 1, "log");
    out->_backward = backward_log;
    return out;
}

// Helper function to check if a Value* is in an array
int contains(Value **arr, int size, Value *v)
{
    for (int i = 0; i < size; i++)
    {
        if (arr[i] == v)
            return 1;
    }
    return 0;
}

void build_topo(Value *v, Value **topo, int *topo_size, Value **visited, int *visited_size, int max_nodes)
{
    if (contains(visited, *visited_size, v))
        return;
    if (*visited_size >= max_nodes)
    {
        fprintf(stderr, "Error: Exceeded maximum number of nodes in topological sort."
                        " Current max_nodes limit is %d\n",
                max_nodes);
        exit(1);
    }
    visited[(*visited_size)++] = v;
    for (int i = 0; i < v->_prev_count; i++)
    {
        build_topo(v->_prev[i], topo, topo_size, visited, visited_size, max_nodes);
    }
    topo[(*topo_size)++] = v;
}

void value_backward(Value *v)
{
    // Topological sort
    // Adjust size as needed, for `mlp_new(&rng, 2, (int[]){16, 3}, 2)` it's 15,942 nodes!
    int max_nodes = 16000;
    Value **topo = malloc(max_nodes * sizeof(Value *));
    int topo_size = 0;
    Value **visited = malloc(max_nodes * sizeof(Value *));
    int visited_size = 0;

    build_topo(v, topo, &topo_size, visited, &visited_size, max_nodes);

    // Go one variable at a time and apply the chain rule to get its gradient
    v->grad = 1.0;
    for (int i = topo_size - 1; i >= 0; i--)
    {
        if (topo[i]->_backward)
        {
            topo[i]->_backward(topo[i]);
        }
    }

    free(topo);
    free(visited);
}

// Neuron struct implementation
typedef struct Neuron
{
    Value **w;
    Value *b;
    int nin;
    int nonlin;
} Neuron;

Neuron *neuron_new(RNG *random, int nin, int nonlin)
{
    Neuron *n = malloc(sizeof(Neuron));
    n->w = malloc(nin * sizeof(Value *));
    for (int i = 0; i < nin; i++)
    {
        n->w[i] = value_new(rng_uniform(random, -1, 1) * pow(nin, -0.5), NULL, 0, "");
    }
    n->b = value_new(0, NULL, 0, "");
    n->nin = nin;
    n->nonlin = nonlin;
    return n;
}

void neuron_free(Neuron *n)
{
    for (int i = 0; i < n->nin; i++)
    {
        value_free(n->w[i]);
    }
    free(n->w);
    value_free(n->b);
    free(n);
}

Value *neuron_call(Neuron *n, Value **x)
{
    Value *act = n->b;
    for (int i = 0; i < n->nin; i++)
    {
        act = value_add(act, value_mul(n->w[i], x[i]));
    }
    return n->nonlin ? value_tanh(act) : act;
}

Value **neuron_parameters(Neuron *n)
{
    Value **params = malloc((n->nin + 1) * sizeof(Value *));
    for (int i = 0; i < n->nin; i++)
    {
        params[i] = n->w[i];
    }
    params[n->nin] = n->b;
    return params;
}

// Layer struct implementation
typedef struct Layer
{
    Neuron **neurons;
    int nin;
    int nout;
} Layer;

Layer *layer_new(RNG *random, int nin, int nout, int nonlin)
{
    Layer *l = malloc(sizeof(Layer));
    l->neurons = malloc(nout * sizeof(Neuron *));
    for (int i = 0; i < nout; i++)
    {
        l->neurons[i] = neuron_new(random, nin, nonlin);
    }
    l->nin = nin;
    l->nout = nout;
    return l;
}

void layer_free(Layer *l)
{
    for (int i = 0; i < l->nout; i++)
    {
        neuron_free(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

Value **layer_call(Layer *l, Value **x)
{
    Value **out = malloc(l->nout * sizeof(Value *));
    for (int i = 0; i < l->nout; i++)
    {
        out[i] = neuron_call(l->neurons[i], x);
    }
    return out;
}

Value **layer_parameters(Layer *l)
{
    Value **params = malloc(l->nout * (l->nin + 1) * sizeof(Value *));
    int idx = 0;
    for (int i = 0; i < l->nout; i++)
    {
        Value **neuron_params = neuron_parameters(l->neurons[i]);
        for (int j = 0; j < l->nin + 1; j++)
        {
            params[idx++] = neuron_params[j];
        }
        free(neuron_params);
    }
    return params;
}

// MLP struct implementation
typedef struct MLP
{
    Layer **layers;
    int n_layers;
} MLP;

MLP *mlp_new(RNG *random, int nin, int *nouts, int n_layers)
{
    MLP *m = malloc(sizeof(MLP));
    m->layers = malloc(n_layers * sizeof(Layer *));
    m->n_layers = n_layers;
    int sizes[n_layers + 1];
    sizes[0] = nin;
    for (int i = 0; i < n_layers; i++)
    {
        sizes[i + 1] = nouts[i];
    }
    for (int i = 0; i < n_layers; i++)
    {
        m->layers[i] = layer_new(random, sizes[i], sizes[i + 1], i != n_layers - 1);
    }
    return m;
}

void mlp_free(MLP *m)
{
    for (int i = 0; i < m->n_layers; i++)
    {
        layer_free(m->layers[i]);
    }
    free(m->layers);
    free(m);
}

Value **mlp_call(MLP *m, Value **x)
{
    Value **out = x;
    for (int i = 0; i < m->n_layers; i++)
    {
        out = layer_call(m->layers[i], out);
    }
    return out;
}

Value **mlp_parameters(MLP *m)
{
    int total_params = 0;
    for (int i = 0; i < m->n_layers; i++)
    {
        total_params += m->layers[i]->nout * (m->layers[i]->nin + 1);
    }
    Value **params = malloc(total_params * sizeof(Value *));
    int idx = 0;
    for (int i = 0; i < m->n_layers; i++)
    {
        Value **layer_params = layer_parameters(m->layers[i]);
        for (int j = 0; j < m->layers[i]->nout * (m->layers[i]->nin + 1); j++)
        {
            params[idx++] = layer_params[j];
        }
        free(layer_params);
    }
    return params;
}

// Loss function
Value *cross_entropy(Value **logits, int n_logits, int target)
{
    double max_val = -DBL_MAX;
    for (int i = 0; i < n_logits; i++)
    {
        if (logits[i]->data > max_val)
            max_val = logits[i]->data;
    }
    Value **ex = malloc(n_logits * sizeof(Value *));
    Value *denom = value_new(0, NULL, 0, "");
    for (int i = 0; i < n_logits; i++)
    {
        ex[i] = value_exp(value_add(logits[i], value_new(-max_val, NULL, 0, "")));
        denom = value_add(denom, ex[i]);
    }
    Value **probs = malloc(n_logits * sizeof(Value *));
    for (int i = 0; i < n_logits; i++)
    {
        probs[i] = value_mul(ex[i], value_pow(denom, -1));
    }
    Value *logp = value_log(probs[target]);
    Value *nll = value_mul(logp, value_new(-1, NULL, 0, ""));
    free(ex);
    free(probs);
    return nll;
}

// Evaluation function
double eval_split(MLP *model, DataPoint *split, int split_size)
{
    Value *loss = value_new(0, NULL, 0, "");
    for (int i = 0; i < split_size; i++)
    {
        Value *x[2] = {value_new(split[i].x, NULL, 0, ""), value_new(split[i].y, NULL, 0, "")};
        Value **logits = mlp_call(model, x);
        loss = value_add(loss, cross_entropy(logits, 3, split[i].label));
    }
    double result = loss->data / split_size;
    value_free(loss);
    return result;
}

// Training function
void train(MLP *model, DataPoint *train_split, int train_size, DataPoint *val_split, int val_size)
{
    double learning_rate = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.95;
    double weight_decay = 1e-4;

    Value **params = mlp_parameters(model);
    int n_params = 0;
    for (int i = 0; i < model->n_layers; i++)
    {
        n_params += model->layers[i]->nout * (model->layers[i]->nin + 1);
    }
    double *m = calloc(n_params, sizeof(double));
    double *v = calloc(n_params, sizeof(double));

    for (int step = 0; step < 100; step++)
    {
        if (step % 10 == 0)
        {
            double val_loss = eval_split(model, val_split, val_size);
            printf("step %d, val loss %.12f\n", step, val_loss);
        }

        Value *loss = value_new(0, NULL, 0, "");
        for (int i = 0; i < train_size; i++)
        {
            Value *x[2] = {value_new(train_split[i].x, NULL, 0, ""), value_new(train_split[i].y, NULL, 0, "")};
            Value **logits = mlp_call(model, x);
            loss = value_add(loss, cross_entropy(logits, 3, train_split[i].label));
        }
        loss = value_mul(loss, value_new(1.0 / train_size, NULL, 0, ""));

        value_backward(loss);

        for (int i = 0; i < n_params; i++)
        {
            double grad = params[i]->grad;
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            double m_hat = m[i] / (1 - pow(beta1, step + 1));
            double v_hat = v[i] / (1 - pow(beta2, step + 1));
            params[i]->data -= learning_rate * (m_hat / (sqrt(v_hat) + 1e-8) + weight_decay * params[i]->data);
            params[i]->grad = 0;
        }

        printf("step %d, train loss %.12f\n", step, loss->data);
        value_free(loss);
    }

    free(m);
    free(v);
    free(params);
}

#ifndef TESTING
// if we are TESTING (see test_micrograd.c), we'll skip the int main below
int main(int argc, const char * argv[])
{
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

    return 0;
}
#endif
