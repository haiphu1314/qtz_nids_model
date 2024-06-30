#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include "bnn_model.h"
#include "tnn_model.h"
#include "tbn_model.h"
#include "fp_model.h"
#include "linear.h"

#include "testcase.h"

#define DEBUG 99
#define NO_TESTS 100000
#define NUM_TESTCASES 84000
#define DATA_SIZE 13
#define DATA_SIZE_IN_INT 1


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
        // Latency
        int num_layers;
        BNN_Layer* layers_bnn = bnn_read_model("bnn_model_parameters.txt", &num_layers);
        int output[1];

        srand(time(NULL));
        float input_raw[DATA_SIZE];
        int input[1];

        printf("====================START=======================\n\n");
        srand(time(NULL));

        clock_t start, end;
        start = clock();
        generateRandomArray(input_raw, DATA_SIZE);
        for(int i = 0; i < NO_TESTS; i++){
            input[1] = createBitmaskFromArray(input_raw, DATA_SIZE);
            // printf("Generating %.8x\n",input[1]);
            bnn_forward(layers_bnn, num_layers, input, output);
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("Thời gian thực thi mô hình BNN: %f giây\n\n", time_taken);

        // Acc
        Testcase testcases[NUM_TESTCASES];
        int num_testcases = 0;
        read_testcases("testcase.txt", testcases, &num_testcases);
        int sizeint = 8 * sizeof(int);
        int correct = 0;
        start = clock();
        for (int i = 0; i < num_testcases; i++) {
            int input_acc[DATA_SIZE_IN_INT] = {0};
            for(int k = 0; k < DATA_SIZE; k++){
                if (testcases[i].data[k] < 0){
                    input_acc[k/sizeint] |= 1<<(k%sizeint);
                }
            }
            int predict = bnn_forward(layers_bnn, num_layers, input_acc, output);
            if (predict == testcases[i].label) {
                correct +=1;
            }
            // else{
            //     printf("Failed at testcase %d\n", i);                
            // }
        }
        end = clock();
        float test_accuracy = 100.0 * (float)correct / (float)NUM_TESTCASES;
        printf("Test Accuracy: %f\n",test_accuracy);
        time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Accuracy test time: %f giây\n\n", time_taken);
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

        // Acc
        Testcase testcases[NUM_TESTCASES];
        int num_testcases = 0;
        read_testcases("testcase.txt", testcases, &num_testcases);
        int sizeint = 8 * sizeof(int);
        int correct = 0;
        start = clock();
        for (int i = 0; i < num_testcases; i++) {
            ttype input_acc[DATA_SIZE_IN_INT] = {0};
            for(int k = 0; k < DATA_SIZE; k++){
                float thres = layers_tnn[0].thres;
                if (testcases[i].data[k] >=thres){
                    input_acc[k/sizeint].bit_1 |= 1<<(k%sizeint);
                }
                else if (testcases[i].data[k] <= -(thres)){
                    input_acc[k/sizeint].bit_0 |= 1<<(k%sizeint);                   
                }
            }
            int predict = tnn_forward(layers_tnn, num_layers, input_acc, output_tn);
            if (predict == testcases[i].label) {
                correct +=1;
            }
            // else{
            //     printf("Failed at testcase %d\n", i);                
            // }
        }
        end = clock();
        float test_accuracy = 100.0 * (float)correct / (float)NUM_TESTCASES;
        printf("Test Accuracy: %f\n",test_accuracy);
        time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Accuracy test time: %f giây\n\n", time_taken);
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
        // Acc
        Testcase testcases[NUM_TESTCASES];
        int num_testcases = 0;
        read_testcases("testcase.txt", testcases, &num_testcases);
        int sizeint = 8 * sizeof(int);
        int correct = 0;
        start = clock();
        for (int i = 0; i < num_testcases; i++) {
            ttype input_acc[DATA_SIZE_IN_INT] = {0};
            for(int k = 0; k < DATA_SIZE; k++){
                float thres = layers_tbn[0].thres;
                if (testcases[i].data[k] >=thres){
                    input_acc[k/sizeint].bit_1 |= 1<<(k%sizeint);
                }
                else if (testcases[i].data[k] <= -(thres)){
                    input_acc[k/sizeint].bit_0 |= 1<<(k%sizeint);                   
                }
            }
            int predict = tbn_forward(layers_tbn, num_layers, input_acc, output_tn);
            if (predict == testcases[i].label) {
                correct +=1;
            }
            // else{
            //     printf("Failed at testcase %d\n", i);                
            // }
        }
        end = clock();
        float test_accuracy = 100.0 * (float)correct / (float)NUM_TESTCASES;
        printf("Test Accuracy: %f\n",test_accuracy);
        time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Accuracy test time: %f giây\n\n", time_taken);
        printf("====================END=======================\n");
    #elif DEBUG == 4
        float input_raw[DATA_SIZE];
        float input_fp[1];
        float output_fp[1];
        int num_layers;
        printf("====================START=======================\n\n");
        FP_Layer* layers_fp = fp_read_model("fp_model_parameters.txt", &num_layers);
        srand(time(NULL));
        clock_t start, end;
        start = clock();
        generateRandomArray(input_raw, DATA_SIZE);
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
    int num_layers;
    linear_tnn_layer* layers_tnn = read_linear_model("tnn_model_parameters.txt", &num_layers, 0);
    #endif

}
