#ifndef FP_MODEL_H
#define FP_MODEL_H
#define MAX_TXT_LINES 10240

typedef struct {
    int input_channel;
    int output_channel;
    float *weights;
} FP_Layer;

FP_Layer* fp_read_model(const char* filename, int* num_layers);
void fp_forward(FP_Layer* layers, int num_layers, const float* input, float* output, int input_size, int output_size);


#endif // FP_MODEL_H