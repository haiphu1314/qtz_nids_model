#ifndef CONV_H
#define CONV_H
#include "utils.h"

typedef struct {
    int input_channel;
    int output_channel;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    float input_thres; 
    quant_type quant;
    union {
        int ****weights_b;    // For BNN and TBN layer
        struct {
            int ****weights_t0; //(output channel, input channel, kernelsize, kernelsize)
            int ****weights_t1;
        };              // For TNN layer
        float ****weights_f;
    };
} conv_layer;

typedef union {
    ttype ***t;
    int ***b;
} conv_input;

conv_layer* create_conv_layer(int input_channel, int output_channel, int kernel_size, int stride, int padding, int dilation, quant_type quant);
float*** conv_forward(conv_layer* layer, float ***input, int input_height, int input_width);
#endif // CONV_H