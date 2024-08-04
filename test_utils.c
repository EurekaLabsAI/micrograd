#define TESTING
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.c"

void test_rng_initialization()
{
    RNG rng;
    rng_init(&rng, 42);
    assert(rng.state == 42);
    printf("test_rng_initialization passed\n");
}

void test_random_u32()
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

void test_random()
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

void test_uniform()
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

void test_gen_data_determinism()
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

void test_gen_data_splits()
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

void test_gen_data_labels()
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

void test_random_hardcoded()
{
    RNG rng;
    rng_init(&rng, 42);

    assert(rng_random(&rng) == 0.3390852212905884f);
    assert(rng_random(&rng) == 0.7822558283805847f);
    assert(rng_random(&rng) == 0.790136992931366f);

    printf("test_random_hardcoded passed\n");
}

void test_uniform_hardcoded()
{
    RNG rng;
    rng_init(&rng, 42);

    assert(rng_uniform(&rng, -1, 1) == -0.32182955741882324f);
    assert(rng_uniform(&rng, -1, 1) == 0.5645116567611694f);
    assert(rng_uniform(&rng, -1, 1) == 0.5802739858627319f);

    printf("test_uniform_hardcoded passed\n");
}

void test_gen_data_hardcoded()
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

int main()
{
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

    printf("All tests passed!\n");
    return 0;
}
