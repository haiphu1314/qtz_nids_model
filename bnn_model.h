#ifndef BNN_MODEL_H
#define BNN_MODEL_H
#define MAX_TXT_LINES 10240

typedef struct {
    int input_channel;
    int output_channel;
    int *weights;
} BNN_Layer;

int bitCount(int n);
int sign(int x);
void binary_act(int *input_array, int *sign_array, int size);
int bnn_count_layers(const char* filename);
BNN_Layer* bnn_read_model(const char* filename, int* num_layers);
int bnn_forward(BNN_Layer* layers, int num_layers, int* input, int* output);

#endif // BNN_MODEL_H