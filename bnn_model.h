#ifndef BNN_MODEL_H
#define BNN_MODEL_H
#include "utils.h" 

typedef struct {
    int input_channel;
    int output_channel;
    int *weights;
} BNN_Layer;

BNN_Layer* bnn_read_model(const char* filename, int* num_layers);
int bnn_forward(BNN_Layer* layers, int num_layers, int* input, int* output);

#endif // BNN_MODEL_H