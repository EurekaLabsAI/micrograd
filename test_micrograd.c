/*
Compile and run:
gcc -O3 -Wall -Wextra -Wpedantic -o test_micrograd test_micrograd.c && ./test_micrograd
*/

#define TESTING
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "micrograd.c"

#define TOL 1e-6

// ----------------------------------------------------------------------------
// random number generation tests

void test_rng_initialization(void)
{
    printf("test_rng_initialization\n");
    RNG rng;
    rng_init(&rng, 42);
    assert(rng.state == 42);
    printf("test_rng_initialization passed\n");
}

void test_random_u32(void)
{
    RNG rng1, rng2;
    rng_init(&rng1, 42);
    rng_init(&rng2, 42);

    for (int i = 0; i < 1000; i++)
    {
        uint32_t num1 = rng_random_u32(&rng1);
        uint32_t num2 = rng_random_u32(&rng2);
        assert(num1 == num2);
    }
    printf("test_random_u32 passed\n");
}

void test_random(void)
{
    RNG rng1, rng2;
    rng_init(&rng1, 42);
    rng_init(&rng2, 42);

    for (int i = 0; i < 1000; i++)
    {
        float num1 = rng_random(&rng1);
        float num2 = rng_random(&rng2);
        assert(num1 == num2);
    }
    printf("test_random passed\n");
}

void test_uniform(void)
{
    RNG rng1, rng2;
    rng_init(&rng1, 42);
    rng_init(&rng2, 42);
    float a = -3.5f, b = 8.2f;

    for (int i = 0; i < 1000; i++)
    {
        float num1 = rng_uniform(&rng1, a, b);
        float num2 = rng_uniform(&rng2, a, b);
        assert(num1 == num2);
    }
    printf("test_uniform passed\n");
}

// ----------------------------------------------------------------------------
// data generation tests

void test_gen_data_determinism(void)
{
    RNG rng1, rng2;
    rng_init(&rng1, 101112);
    rng_init(&rng2, 101112);

    DataPoint *train1, *val1, *test1;
    DataPoint *train2, *val2, *test2;
    int train_size1, val_size1, test_size1;
    int train_size2, val_size2, test_size2;

    gen_data(&rng1, 100, &train1, &train_size1, &val1, &val_size1, &test1, &test_size1);
    gen_data(&rng2, 100, &train2, &train_size2, &val2, &val_size2, &test2, &test_size2);

    assert(train_size1 == train_size2);
    assert(val_size1 == val_size2);
    assert(test_size1 == test_size2);

    for (int i = 0; i < train_size1; i++)
    {
        assert(train1[i].x == train2[i].x);
        assert(train1[i].y == train2[i].y);
        assert(train1[i].label == train2[i].label);
    }

    free(train1);
    free(val1);
    free(test1);
    free(train2);
    free(val2);
    free(test2);

    printf("test_gen_data_determinism passed\n");
}

void test_gen_data_splits(void)
{
    RNG rng;
    rng_init(&rng, 232425);
    int n = 200;
    DataPoint *train, *val, *test;
    int train_size, val_size, test_size;

    gen_data(&rng, n, &train, &train_size, &val, &val_size, &test, &test_size);

    assert(train_size == 160); // 80% of 200
    assert(val_size == 20);    // 10% of 200
    assert(test_size == 20);   // 10% of 200

    free(train);
    free(val);
    free(test);

    printf("test_gen_data_splits passed\n");
}

void test_gen_data_labels(void)
{
    RNG rng;
    rng_init(&rng, 343536);
    DataPoint *train, *val, *test;
    int train_size, val_size, test_size;

    gen_data(&rng, 100, &train, &train_size, &val, &val_size, &test, &test_size);

    for (int i = 0; i < test_size; i++)
    {
        float x = test[i].x;
        float y = test[i].y;
        int label = test[i].label;
        assert(label == (x < 0 ? 0 : (y < 0 ? 1 : 2)));
    }

    free(train);
    free(val);
    free(test);

    printf("test_gen_data_labels passed\n");
}

void test_random_hardcoded(void)
{
    RNG rng;
    rng_init(&rng, 42);

    assert(rng_random(&rng) == 0.3390852212905884f);
    assert(rng_random(&rng) == 0.7822558283805847f);
    assert(rng_random(&rng) == 0.790136992931366f);

    printf("test_random_hardcoded passed\n");
}

void test_uniform_hardcoded(void)
{
    RNG rng;
    rng_init(&rng, 42);

    assert(rng_uniform(&rng, -1, 1) == -0.32182955741882324f);
    assert(rng_uniform(&rng, -1, 1) == 0.5645116567611694f);
    assert(rng_uniform(&rng, -1, 1) == 0.5802739858627319f);

    printf("test_uniform_hardcoded passed\n");
}

void test_gen_data_hardcoded(void)
{
    RNG rng;
    rng_init(&rng, 42);
    DataPoint *train, *val, *test;
    int train_size, val_size, test_size;

    gen_data(&rng, 100, &train, &train_size, &val, &val_size, &test, &test_size);

    assert(train_size == 80);
    assert(val_size == 10);
    assert(test_size == 10);

    assert(train[0].x == -0.6436591148376465f);
    assert(train[0].y == 1.1290233135223389f);
    assert(train[0].label == 0);

    assert(val[0].x == -0.6666977405548096f);
    assert(val[0].y == 1.137477159500122f);
    assert(val[0].label == 0);

    assert(test[0].x == -1.871429443359375f);
    assert(test[0].y == 0.952826976776123f);
    assert(test[0].label == 0);

    assert(train[train_size - 1].x == 0.6179506778717041f);
    assert(train[train_size - 1].y == -0.6032657623291016f);
    assert(train[train_size - 1].label == 1);

    assert(val[val_size - 1].x == -0.5639073848724365f);
    assert(val[val_size - 1].y == -0.4837164878845215f);
    assert(val[val_size - 1].label == 0);

    assert(test[test_size - 1].x == -0.9766521453857422f);
    assert(test[test_size - 1].y == -0.15662121772766113f);
    assert(test[test_size - 1].label == 0);

    free(train);
    free(val);
    free(test);

    printf("test_gen_data_hardcoded passed\n");
}

// ----------------------------------------------------------------------------
// micrograd tests

void test_value_creation(void) {
    Value *v = create_tracked_value(5.0, NULL, 0, "test");
    assert(fabs(v->data - 5.0) < TOL);
    assert(fabs(v->grad) < TOL);
    assert(v->_prev_count == 0);
    assert(strcmp(v->_op, "test") == 0);
    free_tracked_values();
    printf("test_value_creation passed\n");
}

void test_value_add(void) {
    Value *a = create_tracked_value(3.0, NULL, 0, "a");
    Value *b = create_tracked_value(2.0, NULL, 0, "b");
    Value *c = value_add(a, b);
    assert(fabs(c->data - 5.0) < TOL);
    value_backward(c);
    assert(fabs(a->grad - 1.0) < TOL);
    assert(fabs(b->grad - 1.0) < TOL);
    free_tracked_values();
    printf("test_value_add passed\n");
}

void test_value_mul(void) {
    Value *a = create_tracked_value(3.0, NULL, 0, "a");
    Value *b = create_tracked_value(2.0, NULL, 0, "b");
    Value *c = value_mul(a, b);
    assert(fabs(c->data - 6.0) < TOL);
    value_backward(c);
    assert(fabs(a->grad - 2.0) < TOL);
    assert(fabs(b->grad - 3.0) < TOL);
    free_tracked_values();
    printf("test_value_mul passed\n");
}

void test_value_pow(void) {
    Value *a = create_tracked_value(2.0, NULL, 0, "a");
    Value *b = value_pow(a, create_tracked_value(3.0, NULL, 0, "3"));
    assert(fabs(b->data - 8.0) < TOL);
    value_backward(b);
    assert(fabs(a->grad - 12.0) < TOL);
    free_tracked_values();
    printf("test_value_pow passed\n");
}

void test_value_relu(void)
{
    Value *a = create_tracked_value(-2.0, NULL, 0, "a");
    Value *b = value_relu(a);
    assert(fabs(b->data) < TOL);
    value_backward(b);
    assert(fabs(a->grad) < TOL);

    Value *c = create_tracked_value(3.0, NULL, 0, "c");
    Value *d = value_relu(c);
    assert(fabs(d->data - 3.0) < TOL);
    value_backward(d);
    assert(fabs(c->grad - 1.0) < TOL);

    free_tracked_values();
    printf("test_value_relu passed\n");
}

void test_value_tanh(void)
{
    Value *a = create_tracked_value(0.0, NULL, 0, "a");
    Value *b = value_tanh(a);
    assert(fabs(b->data) < TOL);
    value_backward(b);
    assert(fabs(a->grad - 1.0) < TOL);

    free_tracked_values();
    printf("test_value_tanh passed\n");
}

void test_sanity_check(void)
{
    Value *x = create_tracked_value(-4.0, NULL, 0, "x");
    Value *two = create_tracked_value(2.0, NULL, 0, "2");
    Value *z = value_add(value_add(value_mul(two, x), create_tracked_value(2.0, NULL, 0, "2")), x);
    Value *q = value_add(value_relu(z), value_mul(z, x));
    Value *h = value_relu(value_mul(z, z));
    Value *y = value_add(value_add(h, q), value_mul(q, x));

    value_backward(y);

    // Forward pass check
    assert(fabs(y->data - (-20.0)) < TOL);
    // Backward pass check
    assert(fabs(x->grad - 46.0) < TOL);

    free_tracked_values();
    printf("test_sanity_check passed\n");
}

void test_more_ops(void)
{
    Value *a = create_tracked_value(-4.0, NULL, 0, "a");
    Value *b = create_tracked_value(2.0, NULL, 0, "b");
    Value *c = value_add(a, b);
    Value *d = value_add(value_mul(a, b), value_pow(b, create_tracked_value(3.0, NULL, 0, "3")));
    c = value_add(c, value_add(c, create_tracked_value(1.0, NULL, 0, "1")));
    c = value_add(value_add(value_add(c, create_tracked_value(1.0, NULL, 0, "1")), c), value_mul(a, create_tracked_value(-1.0, NULL, 0, "-1")));
    d = value_add(d, value_add(value_mul(d, create_tracked_value(2.0, NULL, 0, "2")), value_relu(value_add(b, a))));
    d = value_add(d, value_add(value_mul(create_tracked_value(3.0, NULL, 0, "3"), d), value_relu(value_add(b, value_mul(a, create_tracked_value(-1.0, NULL, 0, "-1"))))));
    Value *e = value_add(c, value_mul(d, create_tracked_value(-1.0, NULL, 0, "-1")));
    Value *f = value_pow(e, create_tracked_value(2.0, NULL, 0, "2"));
    Value *g = value_mul(f, create_tracked_value(0.5, NULL, 0, "0.5"));
    g = value_add(g, value_mul(create_tracked_value(10.0, NULL, 0, "10"), value_pow(f, create_tracked_value(-1.0, NULL, 0, "-1.0"))));

    value_backward(g);

    // Forward pass check
    assert(fabs(g->data - 24.704082) < TOL);
    // Backward pass check
    assert(fabs(a->grad - 138.833819) < TOL);
    assert(fabs(b->grad - 645.577259) < TOL);

    // Clean up
    free_tracked_values();
    printf("test_more_ops passed\n");
}

// ----------------------------------------------------------------------------
// neural network tests

void test_neuron(void)
{
    RNG rng;
    rng_init(&rng, 42);
    Neuron *n = neuron_new(&rng, 2, 1);
    Value *x1 = create_tracked_value(2.0, NULL, 0, "x1");
    Value *x2 = create_tracked_value(3.0, NULL, 0, "x2");
    Value *result = neuron_call(n, (Value *[]){x1, x2});
    assert(result != NULL);
    
    neuron_free(n);
    free_tracked_values();
    printf("test_neuron passed\n");
}

void test_layer(void)
{
    RNG rng;
    rng_init(&rng, 42);
    Layer *l = layer_new(&rng, 2, 3, 1);
    Value *x1 = create_tracked_value(2.0, NULL, 0, "x1");
    Value *x2 = create_tracked_value(3.0, NULL, 0, "x2");
    Value **result = layer_call(l, (Value *[]){x1, x2});
    assert(result != NULL);
    for (int i = 0; i < 3; i++)
    {
        assert(result[i] != NULL);
    }

    free(result);
    layer_free(l);
    free_tracked_values();
    printf("test_layer passed\n");
}

void test_mlp(void)
{
    RNG rng;
    rng_init(&rng, 42);
    MLP *m = mlp_new(&rng, 2, (int[]){3, 1}, 2);
    Value *x1 = create_tracked_value(2.0, NULL, 0, "x1");
    Value *x2 = create_tracked_value(3.0, NULL, 0, "x2");
    Value **result = mlp_call(m, (Value *[]){x1, x2});
    assert(result != NULL);
    assert(result[0] != NULL);
    
    mlp_free(m);
    free(result);
    free_tracked_values();
    printf("test_mlp passed\n");
}

int main(void)
{
    value_tracker_init();
    // RNG tests
    test_rng_initialization();
    test_random_u32();
    test_random();
    test_uniform();
    test_gen_data_determinism();
    test_gen_data_splits();
    test_gen_data_labels();
    test_random_hardcoded();
    test_uniform_hardcoded();
    test_gen_data_hardcoded();
    // Value tests
    test_value_creation();
    test_value_add();
    test_value_mul();
    test_value_pow();
    test_value_relu();
    test_value_tanh();
    test_sanity_check();
    test_more_ops();
    // Neural network tests
    test_neuron();
    test_layer();
    test_mlp();

    free(g_value_tracker.values);
    printf("All tests passed!\n");
    return 0;
}
