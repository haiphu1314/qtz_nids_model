#include "bnn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#define USE_MSSE
#ifdef USE_MSSE
    #include <nmmintrin.h>
    #define bitCount _mm_popcnt_u32
#else
    int bitCount(int n) {
        n = (n & 0x55555555u) + ((n >> 1) & 0x55555555u);
        n = (n & 0x33333333u) + ((n >> 2) & 0x33333333u);
        n = (n & 0x0f0f0f0fu) + ((n >> 4) & 0x0f0f0f0fu);
        n = (n & 0x00ff00ffu) + ((n >> 8) & 0x00ff00ffu);
        n = (n & 0x0000ffffu) + ((n >>16) & 0x0000ffffu);
        return n;
    }
#endif

// int bitCount(int n) {
//     int count = 0;
//     while (n) {
//         n &= (n - 1); 
//         count++;
//     }
//     return count;
// }

int sign(int x) {
    return (x > 0) - (x < 0);
}

void binary_act(int *input_array, int *sign_array, int size) {
    for (int i = 0; i < size; i++) {
        sign_array[i] = sign(input_array[i]);
    }
}

int bnn_count_layers(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    char line[MAX_TXT_LINES];
    int layer_count = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            layer_count++;
        }
    }

    fclose(file);
    return layer_count;
}

BNN_Layer* bnn_read_model(const char* filename, int* num_layers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_TXT_LINES];
    int layer_count = 0;
    int max_layers = bnn_count_layers(filename); // Số lớp tối đa
    BNN_Layer *layers = (BNN_Layer *)malloc(max_layers * sizeof(BNN_Layer));

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            BNN_Layer *layer = &layers[layer_count++];
            fscanf(file, "input_channel: %d\n", &layer->input_channel);
            fscanf(file, "output_channel: %d\n", &layer->output_channel);
            int weight_size = 0;
            if((layer->input_channel % 32) == 0){
                weight_size = (layer->input_channel/32) * layer->output_channel;
            }
            else{
                weight_size = (layer->input_channel/32 + 1) * layer->output_channel;
            }
            layer->weights = (int *)malloc(weight_size * sizeof(int));
            for (int i = 0; i < weight_size; ++i) {
                fscanf(file, "0x%x\n", &layer->weights[i]);
            }
        }
    }

    fclose(file);
    *num_layers = layer_count;
    // printf("Layer count %d\n", layer_count);
    return layers;
}

int bnn_forward(BNN_Layer* layers, int num_layers, int* input, int* output) {
    char input_size = layers[0].input_channel;
    char output_size = layers[num_layers - 1].output_channel;
    char size32_input = (input_size%32) ? (input_size/32 + 1)  : (input_size/32);
    int* curr_input = (int *)malloc(size32_input * sizeof(int));
    memcpy(curr_input, input, size32_input * sizeof(int));
    
    int *out_pd = (int *)malloc(output_size * sizeof(int));
    for (int l = 0; l < num_layers; ++l) {
        BNN_Layer *layer = &layers[l];
        char size32_curr_input = (layer->input_channel%32) ? (layer->input_channel/32 + 1) : (layer->input_channel/32); //2
        char size32_next_input = (layer->output_channel%32) ? (layer->output_channel/32 + 1) : (layer->output_channel/32);
        // int mat_cnt = 0;
        int* next_input = (int *)calloc(size32_next_input, sizeof(int));
        for (int i = 0; i < layer->output_channel; ++i) {
            int cnt_minus_one = 0;
            for (int j = 0; j < size32_curr_input; ++j) {
                int ixorw = (curr_input[j]) ^ (layer->weights[i*size32_curr_input+j]);
                cnt_minus_one += bitCount(ixorw);
                // mat_cnt+=1;
            }
            int cnt_one = layer->input_channel - cnt_minus_one;
            
            if(cnt_minus_one > cnt_one){
                next_input[i/32] |= 1<<(i%32);         //If number of bit 1 higher then number of bit 0 => bit 1 to result
            }
            if(l == num_layers-1){
                out_pd[i/32] = cnt_one-cnt_minus_one;
                // printf("Output %d\n", out_pd[i/32]);
            }
        }
        // printf("mat_cnt %d\n", mat_cnt);
        free(curr_input);
        curr_input = next_input;
    }

    char size32_output = (output_size%32) ? (output_size/32 + 1)  : (output_size/32);
    memcpy(output, curr_input, size32_output * sizeof(int));
    free(curr_input);
    return 0;
    // int max_index = 0;
    // for (int i = 1; i < output_size; i++) {
    //     if (out_pd[i] > out_pd[max_index]) {
    //         max_index = i;
    //     }
    // }
    // return max_index;
}