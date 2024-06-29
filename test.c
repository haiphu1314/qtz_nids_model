#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include "bnn_model.h"
#include "tnn_model.h"
#include "tbn_model.h"

#include "fp_model.h"

#define DEBUG 2
#define NO_TESTS 100000

void generateRandomArray(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100; 
    }
}

int createBitmaskFromArray(float arr[], int size) {
    int bitmask = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0) {
            bitmask |= (1 << i);
        }
    }
    return bitmask;
}

int main() {
    #if DEBUG == 1
        int num_layers;
        BNN_Layer* layers_bnn = bnn_read_model("bnn_model_parameters.txt", &num_layers);
        srand(time(NULL));
        float input_raw[13];
        int output[1];
        int input[1];

        printf("====================START=======================\n\n");
        srand(time(NULL));

        clock_t start, end;
        start = clock();
        generateRandomArray(input_raw, 13);
        for(int i = 0; i < NO_TESTS; i++){
            input[1] = createBitmaskFromArray(input_raw, 13);
            // printf("Generating %.8x\n",input[1]);
            bnn_forward(layers_bnn, num_layers, input, output);
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("Thời gian thực thi mô hình BNN: %f giây\n\n", time_taken);
        printf("====================END=======================\n");

    #elif DEBUG == 2
        int num_layers;
        ttype output_tn[1];
        ttype input_tn[1];
        input_tn->bit_0 = 0x0390;
        input_tn->bit_1 = 0x0020;
        printf("====================START=======================\n\n");

        TNN_Layer* layers_tnn = tnn_read_model("tnn_model_parameters.txt", &num_layers);
        srand(time(NULL));
        clock_t start, end;
        start = clock();
        for(int i = 0; i < NO_TESTS; i++){
            tnn_forward(layers_tnn, num_layers, input_tn, output_tn);
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("Thời gian thực thi mô hình TNN: %f giây\n\n", time_taken);
        printf("====================END=======================\n");
    
    #elif DEBUG == 3
        int num_layers;
        ttype output_tn[1];
        ttype input_tn[1];
        input_tn->bit_0 = 0x0390;
        input_tn->bit_1 = 0x0020;
        printf("====================START=======================\n\n");

        TBN_Layer* layers_tbn = tbn_read_model("tbn_model_parameters.txt", &num_layers);
        srand(time(NULL));
        clock_t start, end;
        start = clock();
        for(int i = 0; i < NO_TESTS; i++){
            tbn_forward(layers_tbn, num_layers, input_tn, output_tn);
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        // tbn_forward(layers_tbn, num_layers, input_tn, output_tn);
        printf("Thời gian thực thi mô hình TBN: %f giây\n\n", time_taken);
        printf("====================END=======================\n");

    #elif DEBUG == 4
        float input_raw[13];
        float input_fp[1];
        float output_fp[1];
        int num_layers;
        printf("====================START=======================\n\n");
        FP_Layer* layers_fp = fp_read_model("fp_model_parameters.txt", &num_layers);
        srand(time(NULL));
        clock_t start, end;
        start = clock();
        generateRandomArray(input_raw, 13);
        for(int i = 0; i < NO_TESTS; i++){
            fp_forward(layers_fp, num_layers, input_fp, output_fp, 1, 1);
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("Thời gian thực thi mô hình FP: %f giây\n\n", time_taken);
        printf("====================END=======================\n");        
    #elif DEBUG == 99
    // int num_layers;
    // BNN_Layer* layers_bnn = bnn_read_model("bnn_model_parameters.txt", &num_layers);

    // clock_t start, end;
    // int c = 1;
    // int d = 111;
    // start = clock();
    // printf("====================START=======================\n\n");
    // for (int i = 0; i <100000; i++){
    //     for (int j = 0; j < 1164; j++){
    //         int a = c^d;
    //         bitCount(a);
    //     }
    // }
    // end = clock();
    // double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("Thời gian bitcount: %f giây\n\n", time_taken);
    // return 0;
    #endif


}
