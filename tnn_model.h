#ifndef TNN_MODEL_H
#define TNN_MODEL_H
#include "utils.h" 
#define MAX_TXT_LINES 10240

typedef struct {
    int input_channel;
    int output_channel;
    float thres;
    int *weights_0;
    int *weights_1;
} TNN_Layer;

void binary_act(int *input_array, int *sign_array, int size);
TNN_Layer* tnn_read_model(const char* filename, int* num_layers);
int tnn_forward(TNN_Layer* layers, int num_layers, ttype* input, ttype* output);

#endif // TNN_MODEL_H