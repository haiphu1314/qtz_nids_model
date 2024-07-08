/**************************************************************************
 * Author: Hai Phu
 * Email: haiphu@hcmut.edu.vn
 * Date: July 2, 2024
 **************************************************************************/

#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"

conv_layer* create_conv_layer(int input_channel, int output_channel, int kernel_size, int stride, int padding, int dilation, quant_type quant) {
    conv_layer *layer = (conv_layer *)malloc(sizeof(conv_layer));
    layer->input_channel = input_channel;
    layer->output_channel = output_channel;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->dilation = dilation;
    
    layer->input_thres = 0.0;
    layer->quant = quant;
    int intput_sizeint = (input_channel % SIZEINT) == 0 ? (input_channel/SIZEINT): (input_channel/SIZEINT+1);
    int dim1 = output_channel;
    int dim2 = intput_sizeint;
    int dim3 = kernel_size;
    int dim4 = kernel_size;
    switch(quant){
        case BNN:
        case TBN:
            layer->weights = allocate_4d_int_array(dim1, dim2, dim3, dim4);
            if (layer->weights == NULL) {
                fprintf(stderr, "Memory allocation failed for weights\n");
                exit(1);
            }
            break;
        case TNN:
            layer->weights_0 = allocate_4d_int_array(dim1, dim2, dim3, dim4);
            if (layer->weights_0 == NULL) {
                fprintf(stderr, "Memory allocation failed for weights_0\n");
                exit(1);
            }
            layer->weights_1 = allocate_4d_int_array(dim1, dim2, dim3, dim4);
            if (layer->weights_1 == NULL) {
                fprintf(stderr, "Memory allocation failed for weights_1\n");
                exit(1);
            }
            break;
        default:
            fprintf(stderr, "create_conv_layer: Unknown quantization type \n");
            exit(1);        

    }
    return layer;
}

int get_pixel(int ***input, int width, int height, int x, int y, int c, int padding, int dilation) {
    int padded_x = (x - padding) * dilation;
    int padded_y = (y - padding) * dilation;
    if (padded_x < 0 || padded_y < 0 || padded_x >= width || padded_y >= height) {
        return 0;
    }
    return input[c][padded_y][padded_x];
}

ttype get_pixel_ttype(ttype ***input, int width, int height, int x, int y, int c, int padding, int dilation) {
    int padded_x = x - padding * dilation;
    int padded_y = y - padding * dilation; 
    if (padded_x < 0 || padded_y < 0 || padded_x >= width || padded_y >= height) {
        ttype pixel;
        pixel.bit_0 = 0;
        pixel.bit_1 = 0;
        return pixel; // Padding zero
    }
    return input[c][padded_y][padded_x];
}

float*** conv_forward(conv_layer* layer, float ***input, int input_height, int input_width) {
    int input_channel = layer->input_channel;
    int output_channel = layer->output_channel;
    int kernel_size = layer->kernel_size;
    int stride = layer->stride;
    int padding = layer->padding;
    int dilation = layer->dilation;
    float input_thres = layer->input_thres;
    quant_type quant = layer->quant;

    int sizeint_input = (input_channel%SIZEINT)?(input_channel/SIZEINT + 1):(input_channel/SIZEINT);
    int output_height = (int)((input_height+2*padding-dilation*(kernel_size-1)-1)/stride)+1; //height
    int output_width = (int)((input_width +2*padding-dilation*(kernel_size-1)-1)/stride)+1; //width
    float*** output = allocate_3d_float_array(output_channel, output_height, output_width);
    conv_input input_quant;
    int dim1_input = sizeint_input;
    int dim2_input = input_height; //height
    int dim3_input = input_width; //width

    switch (quant){
        case BNN:
            input_quant.b = allocate_3d_int_array(dim1_input, dim2_input, dim3_input);
            for (int c = 0; c < input_channel; c++){
                for (int h = 0; h < dim2_input; h++){
                    for (int w = 0; w < dim3_input; w++){
                        if(input[c][h][w] < input_thres){
                            input_quant.b[c/SIZEINT][h][w] |= 1 << (c%SIZEINT);
                        }
                    }
                }
            }
            break;
        case TBN:
        case TNN:
            input_quant.t = allocate_3d_ttype_array(dim1_input, dim2_input, dim3_input);
            
            for (int c = 0; c < input_channel; c++){
                for (int h = 0; h < dim2_input; h++){
                    for (int w = 0; w < dim3_input; w++){
                        if(input[c][h][w] >= input_thres){
                            input_quant.t[c/SIZEINT][h][w].bit_1 |= 1 << (c%SIZEINT);
                        }
                        else if(input[c][h][w] <= -input_thres){
                            input_quant.t[c/SIZEINT][h][w].bit_0 |= 1 << (c%SIZEINT);
                        }
                    }
                }
            }
            // if(input_channel==32){
            //     printf("i_raw: %f\n", input[6][0][1]);
            // }
            // printf("i0: %.08x %.08x %.08x\n",input_quant.t[0][0][0].bit_0, input_quant.t[0][0][1].bit_0, input_quant.t[0][0][2].bit_0);
            // printf("i1: %.08x %.08x %.08x\n",input_quant.t[0][0][0].bit_1, input_quant.t[0][0][1].bit_1, input_quant.t[0][0][2].bit_1);

            
            break;
        default:
            fprintf(stderr, "conv_forward: Unknown quantization type\n");
            exit(1);     
    }

    switch (quant){
        case BNN:
            for (int co = 0; co < output_channel; co++){
                for (int y = 0; y < output_height; y++){
                    for (int x = 0; x < output_width; x++) {
                        int cnt_minus_one = 0;
                        int cnt_zero = 0;
                        for (int kc =0; kc < sizeint_input; kc++) {
                            for (int ky = 0; ky < kernel_size; ky++) {
                                for (int kx = 0; kx < kernel_size; kx++){
                                    int padded_x = (x * stride + kx * dilation - padding) * dilation;
                                    int padded_y = (y * stride + ky * dilation - padding) * dilation;
                                    if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height) {
                                        cnt_zero +=1;
                                    }
                                    else{
                                        int result_bit = input_quant.b[kc][padded_y][padded_x] ^ layer->weights[co][kc][ky][kx];
                                        cnt_minus_one += bitCount(result_bit);
                                    }
                                }
                            }
                        }
                        int cnt_one = (kernel_size*kernel_size-cnt_zero)*input_channel - cnt_minus_one;
                        output[co][y][x] = (float)(cnt_one-cnt_minus_one);
                    }
                }
            }
            return output;
            break;
        case TBN:
            for (int co = 0; co < output_channel; co++){
                for (int y = 0; y < output_height; y++){
                    for (int x = 0; x < output_width; x++) {
                        int cnt_minus_one = 0;
                        int cnt_one = 0;
                        int cnt_zero = 0;
                        for (int kc =0; kc < sizeint_input; kc++) {
                            for (int ky = 0; ky < kernel_size; ky++) {
                                for (int kx = 0; kx < kernel_size; kx++){
                                    int padded_x = (x * stride + kx * dilation - padding) * dilation;
                                    int padded_y = (y * stride + ky * dilation - padding) * dilation;
                                    if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height) {
                                        cnt_zero +=1;
                                    }
                                    else{
                                        int weight = layer->weights[co][kc][ky][kx];
                                        int i_weight = ~weight;
                                        int result_bit0 = (input_quant.t[kc][padded_y][padded_x].bit_1 & i_weight) | (input_quant.t[kc][padded_y][padded_x].bit_0 & weight);
                                        int result_bit1 = (input_quant.t[kc][padded_y][padded_x].bit_1 & weight) | (input_quant.t[kc][padded_y][padded_x].bit_0 & i_weight);
                                        cnt_minus_one += bitCount(result_bit0);
                                        cnt_one += bitCount(result_bit1);
                                    }
                                }
                            }
                        }
                        output[co][y][x] = (float)(cnt_one-cnt_minus_one);
                    }
                }
            }
            return output;
            break;

        case TNN:
            // int check = 0;
            for (int co = 0; co < output_channel; co++){
                for (int y = 0; y < output_height; y++){
                    for (int x = 0; x < output_width; x++) {
                        int cnt_minus_one = 0;
                        int cnt_one = 0;
                        int cnt_zero = 0;
                        // if(check == 0) {
                        //     printf("sizeint_input: %d\n", sizeint_input);
                        //     check = 1;
                        // }
                        for (int kc = 0; kc < sizeint_input; kc++) {
                            for (int ky = 0; ky < kernel_size; ky++) {
                                for (int kx = 0; kx < kernel_size; kx++){
                                    int padded_x = (x * stride + kx * dilation - padding) * dilation;
                                    int padded_y = (y * stride + ky * dilation - padding) * dilation;
                                    if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height) {
                                        cnt_zero +=1;
                                    }
                                    else{
                                        // if(check == 0 && kc == 1){
                                        //     printf("layer->weights_0: %.08x\n",layer->weights_0[0][0][0][0]);
                                        //     printf("layer->weights_1: %.08x\n\n",layer->weights_1[0][0][0][0]);

                                        //     printf("layer->weights_0: %.08x\n",layer->weights_0[0][1][0][0]);
                                        //     printf("layer->weights_1: %.08x\n\n",layer->weights_1[0][1][0][0]);
                                        //     check = 1;
                                        // }
                                        int result_bit0 = (input_quant.t[kc][padded_y][padded_x].bit_1 & layer->weights_0[co][kc][ky][kx]) | (input_quant.t[kc][padded_y][padded_x].bit_0 & layer->weights_1[co][kc][ky][kx]);
                                        int result_bit1 = (input_quant.t[kc][padded_y][padded_x].bit_1 & layer->weights_1[co][kc][ky][kx]) | (input_quant.t[kc][padded_y][padded_x].bit_0 & layer->weights_0[co][kc][ky][kx]);
                                        cnt_minus_one += bitCount(result_bit0);
                                        cnt_one += bitCount(result_bit1);
                                    }
                                // printf("%d %d %d %d\n",co,kc,ky,kx);
                                }
                            }
                        }
                        output[co][y][x] = (float)(cnt_one-cnt_minus_one);
                    }
                }
            }
            // if (check<1){
            //     printf("w07: %.08x %.08x %.08x\n",layer->weights_0[7][0][0][0], layer->weights_0[7][0][0][1], layer->weights_0[7][0][0][2]);
            //     printf("w07: %.08x %.08x %.08x\n",layer->weights_0[7][0][1][0], layer->weights_0[7][0][1][1], layer->weights_0[7][0][1][2]);
            //     printf("w07: %.08x %.08x %.08x\n",layer->weights_0[7][0][2][0], layer->weights_0[7][0][2][1], layer->weights_0[7][0][2][2]);

            //     printf("w17: %.08x %.08x %.08x\n",layer->weights_1[7][0][0][0], layer->weights_1[7][0][0][1], layer->weights_1[7][0][0][2]);
            //     printf("w17: %.08x %.08x %.08x\n",layer->weights_1[7][0][1][0], layer->weights_1[7][0][1][1], layer->weights_1[7][0][1][2]);
            //     printf("w17: %.08x %.08x %.08x\n",layer->weights_1[7][0][2][0], layer->weights_1[7][0][2][1], layer->weights_1[7][0][2][2]);
            //     check = 1;
            // }
            return output;
            break;
    }
    return output;
}