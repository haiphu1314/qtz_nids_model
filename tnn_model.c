#include "tnn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"


TNN_Layer* tnn_read_model(const char* filename, int* num_layers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    int sizeint = 8 * sizeof(int);
    char line[MAX_TXT_LINES];
    int layer_count = 0;
    int max_layers = count_layers(filename);
    TNN_Layer *layers = (TNN_Layer *)malloc(max_layers * sizeof(TNN_Layer));

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            TNN_Layer *layer = &layers[layer_count++];
            fscanf(file, "input_channel: %d\n", &layer->input_channel);
            fscanf(file, "output_channel: %d\n", &layer->output_channel);
            fscanf(file, "input_thres: %f\n", &layer->thres);
            int weight_size = 0;
            if((layer->input_channel % sizeint) == 0){
                weight_size = (layer->input_channel/sizeint) * layer->output_channel;
            }
            else{
                weight_size = (layer->input_channel/sizeint + 1) * layer->output_channel;
            }

            layer->weights_0 = (int *)malloc(weight_size * sizeof(int));
            layer->weights_1 = (int *)malloc(weight_size * sizeof(int));

            for (int i = 0; i < weight_size; ++i) {
                fscanf(file, "0x%x\n", &layer->weights_0[i]);
                fscanf(file, "0x%x\n", &layer->weights_1[i]);
            }
        }
    }
    fclose(file);
    *num_layers = layer_count;
    // printf("Layer count %d\n", layer_count);
    return layers;
}

int tnn_forward(TNN_Layer* layers, int num_layers, ttype* input, ttype* output) {
    int input_size = layers[0].input_channel;
    int output_size = layers[num_layers - 1].output_channel;
    int sizeint = 8 * sizeof(int);
    int sizeint_input = (input_size%sizeint) ? (input_size/sizeint + 1)  : (input_size/sizeint);
    ttype* curr_input = (ttype *)malloc(sizeint_input * sizeof(ttype));
    memcpy(curr_input, input, sizeint_input * sizeof(ttype));
    int *out_pd = (int *)malloc(output_size * sizeof(int));
    for (int l = 0; l < num_layers; ++l) {
        TNN_Layer *layer = &layers[l];
        int sizeint_curr_input = (layer->input_channel%sizeint) ? (layer->input_channel/sizeint + 1) : (layer->input_channel/sizeint); //2
        int sizeint_next_input = (layer->output_channel%sizeint) ? (layer->output_channel/sizeint + 1) : (layer->output_channel/sizeint);
        // int mat_cnt = 0;
        ttype* next_input = (ttype *)calloc(sizeint_next_input, sizeof(ttype));

        float thres = layers[l+1].thres;
        // printf("layer %d\n", l+1);
        for (int i = 0; i < layer->output_channel; ++i) {
            int cnt_minus_one = 0;
            int cnt_one = 0;
            for (int j = 0; j < sizeint_curr_input; ++j) {
                int result_bit0 = (curr_input[j].bit_1 & layer->weights_0[i*sizeint_curr_input+j]) | (curr_input[j].bit_0 & layer->weights_1[i*sizeint_curr_input+j]);
                int result_bit1 = (curr_input[j].bit_1 & layer->weights_1[i*sizeint_curr_input+j]) | (curr_input[j].bit_0 & layer->weights_0[i*sizeint_curr_input+j]);
                cnt_minus_one += bitCount(result_bit0);
                cnt_one += bitCount(result_bit1);
            }     
            float result_mul = (float)(cnt_one - cnt_minus_one);
        
            if(result_mul>=thres){
                next_input[i/sizeint].bit_1 |= 1<<(i%sizeint);
            }
            else if (result_mul<=(-thres))
            {
                next_input[i/sizeint].bit_0 |= 1<<(i%sizeint);
            }
            if(l == num_layers-1){
                out_pd[i/sizeint] = cnt_one-cnt_minus_one;
                // printf("Output %d\n", out_pd[i/32]);
            }
        }
        // printf("mat_cnt %d\n", mat_cnt);
        
        free(curr_input);
        curr_input = next_input;
    }

    int sizeint_output = (output_size%sizeint) ? (output_size/sizeint + 1)  : (output_size/sizeint);
    memcpy(output, curr_input, sizeint_output * sizeof(ttype));
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