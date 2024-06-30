#include "bnn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"


BNN_Layer* bnn_read_model(const char* filename, int* num_layers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    int sizeint = 8 * sizeof(int);
    char line[MAX_CHARS_LINE];
    int layer_count = 0;
    int max_layers = count_layers(filename); // Số lớp tối đa
    BNN_Layer *layers = (BNN_Layer *)malloc(max_layers * sizeof(BNN_Layer));

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            BNN_Layer *layer = &layers[layer_count++];
            fscanf(file, "input_channel: %d\n", &layer->input_channel);
            fscanf(file, "output_channel: %d\n", &layer->output_channel);
            int weight_size = 0;
            if((layer->input_channel % sizeint) == 0){
                weight_size = (layer->input_channel/sizeint) * layer->output_channel;
            }
            else{
                weight_size = (layer->input_channel/sizeint + 1) * layer->output_channel;
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
    int input_size = layers[0].input_channel;
    int output_size = layers[num_layers - 1].output_channel;
    int sizeint = 8 * sizeof(int);
    int sizeint_input = (input_size%sizeint) ? (input_size/sizeint + 1)  : (input_size/sizeint);
    int* curr_input = (int *)malloc(sizeint_input * sizeof(int));
    memcpy(curr_input, input, sizeint_input * sizeof(int));
    int *out_pd = (int *)malloc(output_size * sizeof(int));
    for (int l = 0; l < num_layers; ++l) {
        BNN_Layer *layer = &layers[l];
        int sizeint_curr_input = (layer->input_channel%sizeint) ? (layer->input_channel/sizeint + 1) : (layer->input_channel/sizeint); //2
        int sizeint_next_input = (layer->output_channel%sizeint) ? (layer->output_channel/sizeint + 1) : (layer->output_channel/sizeint);
        // int mat_cnt = 0;
        int* next_input = (int *)calloc(sizeint_next_input, sizeof(int));
        for (int i = 0; i < layer->output_channel; ++i) {
            int cnt_minus_one = 0;
            for (int j = 0; j < sizeint_curr_input; ++j) {
                int ixorw = (curr_input[j]) ^ (layer->weights[i*sizeint_curr_input+j]);
                cnt_minus_one += bitCount(ixorw);
                // mat_cnt+=1;
            }
            int cnt_one = layer->input_channel - cnt_minus_one;
            
            if(cnt_minus_one > cnt_one){
                next_input[i/sizeint] |= 1<<(i%sizeint);         //If number of bit 1 higher then number of bit 0 => bit 1 to result
            }
            if(l == num_layers-1){
            // if(l == 3){
                out_pd[i] = cnt_one-cnt_minus_one;
                // printf("Output %d\n", out_pd[i]);
            }
            // if(l == 1){
            //     printf("Output %d\n", out_pd[i/32]);
            // }
        }
        // printf("mat_cnt %d\n", mat_cnt);
        free(curr_input);
        curr_input = next_input;
    }

    int sizeint_output = (output_size%sizeint) ? (output_size/sizeint + 1)  : (output_size/sizeint);
    memcpy(output, curr_input, sizeint_output * sizeof(int));
    free(curr_input);
    // return 0;
    // // like sofmax, Find best value for prediction
    int max_index = 0;
    for (int i = 0; i < output_size; i++) {
        if (out_pd[i] > out_pd[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}