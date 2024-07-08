/**************************************************************************
 * Author: Hai Phu
 * Email: haiphu@hcmut.edu.vn
 * Date: July 2, 2024
 **************************************************************************/
 
#include "utils.h"
#include "linear.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

linear_layer* create_linear_layer(int input_channel, int output_channel, quant_type quant) {
    linear_layer *layer = (linear_layer *)malloc(sizeof(linear_layer));
    layer->input_channel = input_channel;
    layer->output_channel = output_channel;
    layer->quant = quant;
    layer->input_thres = 0.0;

    int weight_size = (input_channel % SIZEINT) == 0 ? (input_channel/SIZEINT)* output_channel : (input_channel/SIZEINT+1)* output_channel;
    switch(quant){
        case BNN:
        case TBN:
            layer->weights = (int *)malloc(weight_size* sizeof(int));
            if (layer->weights == NULL) {
                fprintf(stderr, "Memory allocation failed for weights\n");
                exit(1);
            }
            break;
        case TNN:
            layer->weights_0 = (int *)malloc(weight_size* sizeof(int));
            if (layer->weights_0 == NULL) {
                fprintf(stderr, "Memory allocation failed for weights_0\n");
                exit(1);
            }
            layer->weights_1 = (int *)malloc(weight_size* sizeof(int));
            if (layer->weights_1 == NULL) {
                fprintf(stderr, "Memory allocation failed for weights_1\n");
                exit(1);
            }
            break;
        default:
            fprintf(stderr, "create_linear_layer: Unknown quantization type \n");
            exit(1);        

    }
    return layer;
}

void free_linear_layer(linear_layer *layer) {
    switch (layer->quant) {
        case BNN:
        case TBN:
            free(layer->weights);
            break;
        case TNN:
            free(layer->weights_0);
            free(layer->weights_1);
            break;
    }
}

float* linear_forward(linear_layer* layer, float* input) {
    int input_channel = layer->input_channel;
    int output_channel = layer->output_channel;
    float input_thres = layer->input_thres;
    quant_type quant = layer->quant;
    int sizeint_input = (input_channel%SIZEINT)?(input_channel/SIZEINT + 1):(input_channel/SIZEINT);
    // int sizeint_output = (output_channel%SIZEINT)?(output_channel/SIZEINT + 1):(output_channel/SIZEINT);
    float* output = (float*)malloc(output_channel * sizeof(float));

    linear_input input_quant;
    switch (quant){
        case BNN:
            input_quant.b = (int *)calloc(sizeint_input, sizeof(int));
            for (int k = 0; k < input_channel; k++){
                if(input[k] < input_thres){
                    input_quant.b[k/SIZEINT] |= 1 << (k%SIZEINT);
                }
            }
            break;
        case TBN:
        case TNN:
            input_quant.t = (ttype *)calloc(sizeint_input, sizeof(ttype));
            for (int k = 0; k < input_channel; k++){
                if(input[k] > input_thres){
                    input_quant.t[k/SIZEINT].bit_1 |= 1 << (k%SIZEINT);
                }
                else if(input[k] < -input_thres){
                    input_quant.t[k/SIZEINT].bit_0 |= 1 << (k%SIZEINT);
                }
            }            
            break;
        default:
            fprintf(stderr, "linear_forward: Unknown quantization type\n");
            exit(1);     
    }

    switch (quant){
        case BNN:
            for (int i = 0; i < output_channel; ++i) {
                int cnt_minus_one = 0;
                for (int j = 0; j < sizeint_input; ++j) {
                    int result_bit = (input_quant.b[j]) ^ (layer->weights[i*sizeint_input+j]);
                    cnt_minus_one += bitCount(result_bit);
                }
                int cnt_one = input_channel - cnt_minus_one;
                output[i] = (float)(cnt_one-cnt_minus_one);
            }
            // printf("%f %f %f\n", output[0], output[1], output[2]);
            return output;
            break;
        case TBN:
            for (int i = 0; i < output_channel; ++i) {
                int cnt_minus_one = 0;
                int cnt_one = 0;
                for (int j = 0; j < sizeint_input; ++j) {
                    int weight = layer->weights[i*sizeint_input+j];
                    int i_weight = ~weight;
                    int result_bit0 = (input_quant.t[j].bit_1 & i_weight) | (input_quant.t[j].bit_0 & weight);
                    int result_bit1 = (input_quant.t[j].bit_1 & weight) | (input_quant.t[j].bit_0 & i_weight);
                    cnt_minus_one += bitCount(result_bit0);
                    cnt_one += bitCount(result_bit1);
                }
                output[i] = (float)(cnt_one-cnt_minus_one);
            }
            return output;
            break;
        case TNN:
            for (int i = 0; i < output_channel; ++i) {
                int cnt_minus_one = 0;
                int cnt_one = 0;
                for (int j = 0; j < sizeint_input; ++j) {
                    int result_bit0 = (input_quant.t[j].bit_1 & layer->weights_0[i*sizeint_input+j]) | (input_quant.t[j].bit_0 & layer->weights_1[i*sizeint_input+j]);
                    int result_bit1 = (input_quant.t[j].bit_1 & layer->weights_1[i*sizeint_input+j]) | (input_quant.t[j].bit_0 & layer->weights_0[i*sizeint_input+j]);
                    cnt_minus_one += bitCount(result_bit0);
                    cnt_one += bitCount(result_bit1);
                }
                output[i] = (float)(cnt_one-cnt_minus_one);
            }
            return output;
            break;
    }
    return output;
}