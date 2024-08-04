#define TESTING
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "micrograd.c"

#define TOL 1e-6

void test_value_creation()
{
    Value *v = value_new(5.0, NULL, 0, "test");
    assert(fabs(v->data - 5.0) < TOL);
    assert(fabs(v->grad) < TOL);
    assert(v->_prev_count == 0);
    assert(strcmp(v->_op, "test") == 0);
    value_free(v);
    printf("test_value_creation passed\n");
}

void test_value_add()
{
    Value *a = value_new(3.0, NULL, 0, "a");
    Value *b = value_new(2.0, NULL, 0, "b");
    Value *c = value_add(a, b);
    assert(fabs(c->data - 5.0) < TOL);
    value_backward(c);
    assert(fabs(a->grad - 1.0) < TOL);
    assert(fabs(b->grad - 1.0) < TOL);
    value_free(a);
    value_free(b);
    value_free(c);
    printf("test_value_add passed\n");
}

void test_value_mul()
{
    Value *a = value_new(3.0, NULL, 0, "a");
    Value *b = value_new(2.0, NULL, 0, "b");
    Value *c = value_mul(a, b);
    assert(fabs(c->data - 6.0) < TOL);
    value_backward(c);
    assert(fabs(a->grad - 2.0) < TOL);
    assert(fabs(b->grad - 3.0) < TOL);
    value_free(a);
    value_free(b);
    value_free(c);
    printf("test_value_mul passed\n");
}

void test_value_pow()
{
    Value *a = value_new(2.0, NULL, 0, "a");
    Value *b = value_pow(a, 3.0);
    assert(fabs(b->data - 8.0) < TOL);
    value_backward(b);
    assert(fabs(a->grad - 12.0) < TOL);
    value_free(a);
    value_free(b);
    printf("test_value_pow passed\n");
}

void test_value_relu()
{
    Value *a = value_new(-2.0, NULL, 0, "a");
    Value *b = value_relu(a);
    assert(fabs(b->data) < TOL);
    value_backward(b);
    assert(fabs(a->grad) < TOL);

    Value *c = value_new(3.0, NULL, 0, "c");
    Value *d = value_relu(c);
    assert(fabs(d->data - 3.0) < TOL);
    value_backward(d);
    assert(fabs(c->grad - 1.0) < TOL);

    value_free(a);
    value_free(b);
    value_free(c);
    value_free(d);
    printf("test_value_relu passed\n");
}

void test_value_tanh()
{
    Value *a = value_new(0.0, NULL, 0, "a");
    Value *b = value_tanh(a);
    assert(fabs(b->data) < TOL);
    value_backward(b);
    assert(fabs(a->grad - 1.0) < TOL);
    value_free(a);
    value_free(b);
    printf("test_value_tanh passed\n");
}

void test_neuron()
{
    RNG rng;
    rng_init(&rng, 42);
    Neuron *n = neuron_new(&rng, 2, 1);
    Value *x1 = value_new(2.0, NULL, 0, "x1");
    Value *x2 = value_new(3.0, NULL, 0, "x2");
    Value *result = neuron_call(n, (Value *[]){x1, x2});
    assert(result != NULL);
    value_free(x1);
    value_free(x2);
    value_free(result);
    // TODO: Free neuron
    printf("test_neuron passed\n");
}

void test_layer()
{
    RNG rng;
    rng_init(&rng, 42);
    Layer *l = layer_new(&rng, 2, 3, 1);
    Value *x1 = value_new(2.0, NULL, 0, "x1");
    Value *x2 = value_new(3.0, NULL, 0, "x2");
    Value **result = layer_call(l, (Value *[]){x1, x2});
    assert(result != NULL);
    for (int i = 0; i < 3; i++)
    {
        assert(result[i] != NULL);
        value_free(result[i]);
    }
    free(result);
    value_free(x1);
    value_free(x2);
    // TODO: Free layer
    printf("test_layer passed\n");
}

void test_mlp()
{
    RNG rng;
    rng_init(&rng, 42);
    MLP *m = mlp_new(&rng, 2, (int[]){3, 1}, 2);
    Value *x1 = value_new(2.0, NULL, 0, "x1");
    Value *x2 = value_new(3.0, NULL, 0, "x2");
    Value **result = mlp_call(m, (Value *[]){x1, x2});
    assert(result != NULL);
    assert(result[0] != NULL);
    value_free(result[0]);
    free(result);
    value_free(x1);
    value_free(x2);
    // TODO: Free MLP
    printf("test_mlp passed\n");
}

void test_sanity_check()
{
    Value *x = value_new(-4.0, NULL, 0, "x");
    Value *two = value_new(2.0, NULL, 0, "2");
    Value *z = value_add(value_add(value_mul(two, x), value_new(2.0, NULL, 0, "2")), x);
    Value *q = value_add(value_relu(z), value_mul(z, x));
    Value *h = value_relu(value_mul(z, z));
    Value *y = value_add(value_add(h, q), value_mul(q, x));

    value_backward(y);

    // Forward pass check
    assert(fabs(y->data - (-20.0)) < TOL);
    // Backward pass check
    assert(fabs(x->grad - 46.0) < TOL);

    // Clean up
    value_free(x);
    value_free(two);
    value_free(z);
    value_free(q);
    value_free(h);
    value_free(y);

    printf("test_sanity_check passed\n");
}

void test_more_ops()
{
    Value *a = value_new(-4.0, NULL, 0, "a");
    Value *b = value_new(2.0, NULL, 0, "b");
    Value *c = value_add(a, b);
    Value *d = value_add(value_mul(a, b), value_pow(b, 3.0));
    c = value_add(c, value_add(c, value_new(1.0, NULL, 0, "1")));
    c = value_add(value_add(value_add(c, value_new(1.0, NULL, 0, "1")), c), value_mul(a, value_new(-1.0, NULL, 0, "-1")));
    d = value_add(d, value_add(value_mul(d, value_new(2.0, NULL, 0, "2")), value_relu(value_add(b, a))));
    d = value_add(d, value_add(value_mul(value_new(3.0, NULL, 0, "3"), d), value_relu(value_add(b, value_mul(a, value_new(-1.0, NULL, 0, "-1"))))));
    Value *e = value_add(c, value_mul(d, value_new(-1.0, NULL, 0, "-1")));
    Value *f = value_pow(e, 2.0);
    Value *g = value_mul(f, value_new(0.5, NULL, 0, "0.5"));
    g = value_add(g, value_mul(value_new(10.0, NULL, 0, "10"), value_pow(f, -1.0)));

    value_backward(g);

    // Forward pass check
    assert(fabs(g->data - 24.704082) < TOL);
    // Backward pass check
    assert(fabs(a->grad - 138.833819) < TOL);
    assert(fabs(b->grad - 645.577259) < TOL);

    // Clean up
    value_free(a);
    value_free(b);
    value_free(c);
    value_free(d);
    value_free(e);
    value_free(f);
    value_free(g);

    printf("test_more_ops passed\n");
}

int main()
{
    test_sanity_check();
    test_more_ops();
    test_value_creation();
    test_value_add();
    test_value_mul();
    test_value_pow();
    test_value_relu();
    test_value_tanh();
    test_neuron();
    test_layer();
    test_mlp();
    printf("All tests passed!\n");
    return 0;
}
