#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include "model.h"
#include "testcase.h"
#include <time.h>

#define NUM_TESTCASES 84000
// #define NO_TESTS 1
#define NO_TESTS 100000

int power(int base, int exponent) {
    int result = 1;
    for(int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}
float getRandomNumber() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}
int main() {
    
    layer_node* model = NULL;
    
    conv_layer *conv1 = create_conv_layer(1, 64, 3, 1, 1, 1, TNN);
    model = add_layer(model, CONV, "conv1", conv1);
    conv_layer *conv2 = create_conv_layer(64, 128, 3, 1, 1, 1, TNN);
    model = add_layer(model, CONV, "conv2", conv2);
    linear_layer *linear1 = create_linear_layer(128*7*7, 128, TNN);
    model = add_layer(model, LINEAR, "linear1", linear1);
    linear_layer *linear2 = create_linear_layer(128, 3, TNN);
    model = add_layer(model, LINEAR, "linear2", linear2);

    load_weight_from_txt(model, "train/tcnn_model_parameters_128.txt");

    // conv_layer *conv1 = create_conv_layer(1, 32, 3, 1, 1, 1, 0.0, BNN);
    // model = add_layer(model, CONV, "conv1", conv1);
    // conv_layer *conv2 = create_conv_layer(32, 64, 3, 1, 1, 1, 0.0, BNN);
    // model = add_layer(model, CONV, "conv2", conv2);
    // linear_layer *linear1 = create_linear_layer(3136, 128, 0.0, BNN);
    // model = add_layer(model, LINEAR, "linear1", linear1);
    // linear_layer *linear2 = create_linear_layer(128, 3, 0.0, BNN);
    // model = add_layer(model, LINEAR, "linear2", linear2);

    // load_weight_from_txt(model, "bcnn_model_parameters.txt");

    int input_height = 7;
    int input_width = 7;

    float*** input = allocate_3d_float_array(1, input_height, input_width);
    for (int c=0; c<1; c++){
        for (int h=0; h<input_height; h++){
            for (int w=0; w<input_width; w++){
                input[c][h][w] = power(-1,w);
            }
        }
    }
    conv_input input_quant;
    input_quant.b = allocate_3d_int_array(1, input_height, input_width);
    for (int c = 0; c < 1; c++){
        for (int h = 0; h < input_height; h++){
            for (int w = 0; w < input_width; w++){
                if(input[c][h][w] < 0.0){
                    input_quant.b[c/SIZEINT][h][w] |= 1 << (c%SIZEINT);
                }
            }
        }
    }
    srand(time(NULL));
    clock_t start, end;
    start = clock();

    for (int ct =0; ct<NO_TESTS; ct++){
        // float*** input = allocate_3d_float_array(1, input_height, input_width);
        input = conv_forward(conv1, input, input_height, input_width);
        // printf("\nCONV1: \n");
        // for(int c =0; c<64; c++){
        //     printf("CHANNEL: %d\n",c);
        //     for(int h = 0; h<7;h++) {
        //         for(int w = 0; w<7;w++){
        //             printf("%.4f, ", input[c][h][w]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        input = conv_forward(conv2, input, input_height, input_width);
        // printf("\nCONV2: \n");
        // for(int c =0; c<64; c++){
        //     printf("CHANNEL: %d\n",c);
        //     for(int h = 0; h<7;h++) {
        //         for(int w = 0; w<7;w++){
        //             printf("%.4f, ", input[c][h][w]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        float *input_linear = (float *)malloc(input_height*input_width*128*sizeof(float));
        int i=0;
        for (int c=0; c<128; c++){
            for (int h=0; h<7; h++){
                for (int w=0; w<7; w++){
                    input_linear[i] = input[c][h][w];
                    i+=1;
                }
            }
        }
        // printf("\nINPUT__LINEAR: \n");
        // for (int i = 0; i <128; i++){
        //     printf("%f ", input_linear[i]);
        // }
        // printf("\n");
        float *output_l1 = (float *)malloc(128*sizeof(float));
        output_l1 = linear_forward(linear1, input_linear);
        // free(input_linear);
        // printf("\nLINEAR: \n");
        // for (int i = 0; i <128; i++){
        //     printf("%f ", output_l1[i]);
        // }
        // printf("\n");
        float *output = (float *)malloc(3*sizeof(float));
        output = linear_forward(linear2, output_l1);
        free(output_l1);
        // printf("%f, %f, %f \n", output[0], output[1], output[2]);
    }
    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Thời gian thực thi mô hình TBN: %f giây\n\n", time_taken);
}
