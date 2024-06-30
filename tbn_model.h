#ifndef TBN_MODEL_H
#define TBN_MODEL_H
#include "utils.h" 

typedef struct {
    int input_channel;
    int output_channel;
    float thres;
    int *weights;
} TBN_Layer;

TBN_Layer* tbn_read_model(const char* filename, int* num_layers);
int tbn_forward(TBN_Layer* layers, int num_layers, ttype* input, ttype* output);

#endif // TBN_MODEL_H