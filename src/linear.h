#ifndef LINEAR_H
#define LINEAR_H
#include "utils.h"

typedef struct {
    int input_channel;
    int output_channel;
    float input_thres; 
    union {
        int *weights_b;    // For BNN and TBN layer
        struct {
            int *weights_t0;
            int *weights_t1;
        };              // For TNN layer
        float *weights_f;
    };
    quant_type quant;
} linear_layer;

typedef union {
    ttype *t;
    int *b;
} linear_input;

linear_layer* create_linear_layer(int input_channel, int output_channel, quant_type quant);
float* linear_forward(linear_layer* layer, float* input);
#endif // LINEAR_H