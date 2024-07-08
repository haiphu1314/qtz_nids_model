#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include "model.h"
#include "testcase.h"
#include <time.h>

#define NUM_TESTCASES 84000
#define NO_TESTS 1
// #define NO_TESTS 100000

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
    
    conv_layer *conv1 = create_conv_layer(1, 32, 3, 1, 1, 1, 0.2948492467403412, TNN);
    model = add_layer(model, CONV, "conv1", conv1);
    conv_layer *conv2 = create_conv_layer(32, 64, 3, 1, 1, 1, 0.7563087940216064, TNN);
    model = add_layer(model, CONV, "conv2", conv2);
    linear_layer *linear1 = create_linear_layer(3136, 128, 6.522428512573242, TNN);
    model = add_layer(model, LINEAR, "linear1", linear1);
    linear_layer *linear2 = create_linear_layer(128, 3, 22.42291831970215, TNN);
    model = add_layer(model, LINEAR, "linear2", linear2);

    load_weight_from_txt(model, "tcnn_model_parameters.txt");

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
        // printf("%f, %f, %f \n", input[0][0][0], input[0][0][1], input[0][0][2]);
        input = conv_forward(conv2, input, input_height, input_width);
        float input_linear[input_height*input_width*64];
        int i=0;
        for (int c=0; c<64; c++){
            for (int h=0; h<7; h++){
                for (int w=0; w<7; w++){
                    input_linear[i] = input[c][h][w];
                    i+=1;
                }
            }
        }
        input = linear_forward(linear1, input_linear);
        float *output = linear_forward(linear2, input);
        printf("%f, %f, %f \n", output[0], output[1], output[2]);
    }
    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Thời gian thực thi mô hình TBN: %f giây\n\n", time_taken);
}
