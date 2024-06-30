#include "fp_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"

FP_Layer* fp_read_model(const char* filename, int* num_layers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_CHARS_LINE];
    int layer_count = 0;
    int max_layers = count_layers(filename); 
    FP_Layer *layers = (FP_Layer *)malloc(max_layers * sizeof(FP_Layer));

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            FP_Layer *layer = &layers[layer_count++];
            fscanf(file, "input_channel: %d\n", &layer->input_channel);
            fscanf(file, "output_channel: %d\n", &layer->output_channel);

            int weight_size = layer->input_channel * layer->output_channel;
            if(strstr(line, "weight")){
                layer->weights = (float *)malloc(weight_size * sizeof(float));

                for (int i = 0; i < weight_size; ++i) {
                    fscanf(file, "%f", &layer->weights[i]);
                }
            }
        }
    }

    fclose(file);
    *num_layers = layer_count;
    return layers;
}

void fp_forward(FP_Layer* layers, int num_layers, const float* input, float* output, int input_size, int output_size) {
    float* curr_input = (float *)malloc(input_size * sizeof(float));
    memcpy(curr_input, input, input_size * sizeof(float));

    for (int l = 0; l < num_layers; ++l) {
        FP_Layer *layer = &layers[l];
        float* next_input = (float *)calloc(layer->output_channel, sizeof(float));
        // int mul_cnt = 0;
        for (int i = 0; i < layer->output_channel; ++i) {
            for (int j = 0; j < layer->input_channel; ++j) {
                next_input[i] += curr_input[j] * layer->weights[i * layer->input_channel + j];
                // mul_cnt += 1;
            }
        }
        // printf("mul_cnt: %d\n", mul_cnt);
        free(curr_input);
        curr_input = next_input;
    }

    memcpy(output, curr_input, output_size * sizeof(float));
    free(curr_input);
}