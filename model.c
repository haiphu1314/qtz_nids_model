/**************************************************************************
 * Author: Hai Phu
 * Email: haiphu@hcmut.edu.vn
 * Date: July 2, 2024
 **************************************************************************/

#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"
#include "linear.h"
#include "conv.h"

layer_node* add_layer(layer_node *model, layer_type type, char* layer_name, void *layer) {
    layer_node *new_node = (layer_node*) malloc(sizeof(layer_node));
    new_node->layer_type = type;
    strcpy(new_node->layer_name, layer_name);
    if (type == LINEAR) {
        new_node->linear = (linear_layer*) layer;
        new_node->conv = NULL;
    } else if (type == CONV) {
        new_node->conv = (conv_layer*) layer;
        new_node->linear = NULL;
    }
    if (model == NULL) {
        return new_node;
    }
    layer_node *current = model;
    while (current->next != NULL) {
        current = current->next;
    }
    current->next = new_node;
    new_node->next = NULL;
    return model;
}

// float* model_forward(layer_node* network, float* input) {
//     layer_node* current;
//     current = network;
//     float* current_input = input;
//     while (current != NULL) {
//         if (current->layer_type == LINEAR) {
//             linear_layer *l_layer = current->linear;
//             float* current_output = linear_forward(l_layer, current_input);
//             current_input = current_output;
//         }
//         // } else if (current->type == CONV) {
//         //     conv_forward(current->cnn, (float**)current_input, (float**)current_output, current_height, current_width);
//         //     current_input = current_output;
//         // }
//         current = current->next;
//     }
//     return current_input;
// }

void* get_layer(layer_node *model, char* layer_name){
    layer_node *current = model;
    while(current!=NULL){
        if(strcmp(current->layer_name, layer_name) == 0){
            if(current->conv != NULL){
                return current->conv;
            }
            else if (current->linear != NULL){
                return current->linear;
            }
        }
        current = current->next;
    }
    fprintf(stderr, "layer %s does not exist!\n",layer_name);
    exit(EXIT_FAILURE);       
}


void load_weight_from_txt(layer_node *model, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    char line[MAX_CHARS_LINE];
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            int input_channel;
            int output_channel;
            int quant_i;
            quant_type quant;
            char layer_name[20];
            float thres;
            fscanf(file, "layer_name: %s\n", layer_name);
            fscanf(file, "input_channel: %d\n", &input_channel);
            fscanf(file, "output_channel: %d\n", &output_channel);
            fscanf(file, "quant_type: %d\n", &quant_i);
            fscanf(file, "input_thres: %f\n", &thres);
            switch(quant_i){
                case 0:
                    quant = BNN;
                    break;
                case 1:
                    quant = TBN;
                    break;
                case 2:
                    quant = TNN;
                    break;
            }
            int found_layer = 0; 
            layer_node *current = model;
            printf("load layer %s ... \n", layer_name);
            while (current != NULL) {
                if(strcmp(current->layer_name, layer_name) == 0){
                    found_layer = 1;                    
                    if(input_channel != current->linear->input_channel){
                        fprintf(stderr, "input channel of %s mismatch!\n",current->layer_name);
                        exit(EXIT_FAILURE);                        
                    }
                    if(output_channel != current->linear->output_channel){
                        fprintf(stderr, "output channel of %s mismatch!\n",current->layer_name);
                        exit(EXIT_FAILURE);                        
                    }
                    if(quant != current->linear->quant){
                        fprintf(stderr, "quantization method of %s mismatch!\n",current->layer_name);
                        exit(EXIT_FAILURE);                        
                    }
                    if(thres != current->linear->input_thres){
                        fprintf(stderr, "input threshold of %s mismatch!\n",current->layer_name);
                        exit(EXIT_FAILURE);                        
                    }                    
                    break;
                }
                current = current->next;
                
            }
            if (!found_layer) {
                printf("Layer %s does not exist", layer_name);
                exit(EXIT_FAILURE);
            }
            int weight_size = (input_channel % SIZEINT) == 0 ? (input_channel/SIZEINT)*output_channel : (input_channel/SIZEINT+1)*output_channel;
            if (quant == TNN){
                for (int i = 0; i < weight_size; ++i) {
                    fscanf(file, "0x%x\n", &current->linear->weights_0[i]);
                    fscanf(file, "0x%x\n", &current->linear->weights_1[i]);
                }
            }
            else{
                for (int i = 0; i < weight_size; ++i) {
                    fscanf(file, "0x%x\n", &current->linear->weights[i]);
                }
            }

        }
        else if (strstr(line, "conv")) {
            char layer_name[20];
            int input_channel;
            int output_channel;
            int kernel_size;
            int stride;
            int padding;
            int dilation;
            int quant_i;
            float thres;
            quant_type quant;
            
            fscanf(file, "layer_name: %s\n", layer_name);
            fscanf(file, "input_channel: %d\n", &input_channel);
            fscanf(file, "output_channel: %d\n", &output_channel);
            fscanf(file, "kernel_size: %d\n", &kernel_size);
            fscanf(file, "stride: %d\n", &stride);
            fscanf(file, "padding: %d\n", &padding);
            fscanf(file, "dilation: %d\n", &dilation);
            fscanf(file, "quant_type: %d\n", &quant_i);
            fscanf(file, "input_thres: %f\n", &thres);
            switch(quant_i){
                case 0:
                    quant = BNN;
                    break;
                case 1:
                    quant = TBN;
                    break;
                case 2:
                    quant = TNN;
                    break;
            }
            int found_layer = 0; 
            layer_node *current = model;
            printf("load layer %s ... \n", layer_name);
            while (current != NULL) {
                if(strcmp(current->layer_name, layer_name) == 0){
                    found_layer = 1;
                    if(input_channel != current->conv->input_channel){
                        fprintf(stderr, "Input channel of layer %s is %d, but in the .txt file, it is %d.\n",current->layer_name, current->conv->input_channel, input_channel);
                        exit(EXIT_FAILURE);                        
                    }
                    if(output_channel != current->conv->output_channel){
                        fprintf(stderr, "Output channel of layer %s is %d, but in the .txt file, it is %d.\n",current->layer_name, current->conv->output_channel, output_channel);
                        exit(EXIT_FAILURE);                        
                    }
                    if(quant != current->conv->quant){
                        fprintf(stderr, "quantization method of %s mismatch!\n",current->layer_name);
                        exit(EXIT_FAILURE);                        
                    }
                    if(thres != current->conv->input_thres){
                        fprintf(stderr, "Input threshold of layer %s is %f, but in the .txt file, it is %f.\n",current->layer_name, current->conv->input_thres, thres);
                        exit(EXIT_FAILURE);                        
                    }                    
                    break;
                }
                
                current = current->next;
            }
            if (!found_layer) {
                printf("Layer %s does not exist", layer_name);
                exit(EXIT_FAILURE);
            }
            int weight_size = (input_channel % SIZEINT) == 0 ? (input_channel/SIZEINT) : (input_channel/SIZEINT+1);
            // if(quant == TNN){
            //     weight_size = weight_size*2;
            // }
            int qb = (quant == TNN) ? 2 : 1;
            for (int oc = 0; oc < output_channel; oc++) {
                for (int q = 0; q < qb; q++){
                    for(int w = 0; w < weight_size; w++){
                        for (int kh = 0; kh < kernel_size; kh++) {
                            int values[10];
                            switch (kernel_size){
                                case 1:
                                    fscanf(file, "0x%x\n", &values[0]);
                                    break;
                                case 2:
                                    fscanf(file, "0x%x, 0x%x\n", &values[0], &values[1]);
                                    break;
                                case 3:
                                    fscanf(file, "0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2]);
                                    break;
                                case 4:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3]);
                                    break;
                                case 5:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3], &values[4]);
                                    break;
                                case 6:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5]);
                                    break;
                                case 7:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6]);
                                    break;
                                case 8:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7]);
                                    break;
                                case 9:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8]);
                                    break;
                                case 10:
                                    fscanf(file, "0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9]);
                                    break;
                                default:
                                    printf("Unsupported kernel_size: %d\n", kernel_size);
                                    break;
                            }
                            if(quant != TNN){
                                for(int kw = 0; kw<kernel_size; kw++){
                                    current->conv->weights[oc][w][kh][kw] = values[kw];
                                }
                            }
                            else{
                                if(q == 0){
                                    for(int kw = 0; kw<kernel_size; kw++){
                                        current->conv->weights_0[oc][w][kh][kw] = values[kw];
                                    }
                                }
                                else if(q == 1){
                                    for(int kw = 0; kw<kernel_size; kw++){
                                        current->conv->weights_1[oc][w][kh][kw] = values[kw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}