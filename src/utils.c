/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-29 13:56:10
 * @ Modified time: 2024-07-10 19:41:31
 * @ Description:
 */

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

// #define USE_MSSE
#ifdef USE_MSSE
    #include <nmmintrin.h>
    int bitCount(int n){
        return _mm_popcnt_u32(n);
    } 
#else
    int bitCount(int n) {
        n = (n & 0x55555555u) + ((n >> 1) & 0x55555555u);
        n = (n & 0x33333333u) + ((n >> 2) & 0x33333333u);
        n = (n & 0x0f0f0f0fu) + ((n >> 4) & 0x0f0f0f0fu);
        n = (n & 0x00ff00ffu) + ((n >> 8) & 0x00ff00ffu);
        n = (n & 0x0000ffffu) + ((n >>16) & 0x0000ffffu);
        return n;
    }
#endif

int sign(int x) {
    return (x > 0) - (x < 0);
}

int count_layers(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    char line[MAX_CHARS_LINE];
    int layer_count = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            layer_count++;
        }
    }

    fclose(file);
    return layer_count;
}

int**** allocate_4d_int_array(int dim1, int dim2, int dim3, int dim4) {
    int ****array = (int****)malloc(dim1 * sizeof(int***));
    for (int i = 0; i < dim1; ++i) {
        array[i] = (int***)malloc(dim2 * sizeof(int**));
        for (int j = 0; j < dim2; ++j) {
            array[i][j] = (int**)malloc(dim3 * sizeof(int*));
            for (int k = 0; k < dim3; ++k) {
                array[i][j][k] = (int*)malloc(dim4 * sizeof(int));
            }
        }
    }
    return array;
}

float**** allocate_4d_float_array(int dim1, int dim2, int dim3, int dim4) {
    float ****array = (float****)malloc(dim1 * sizeof(float***));
    for (int i = 0; i < dim1; ++i) {
        array[i] = (float***)malloc(dim2 * sizeof(float**));
        for (int j = 0; j < dim2; ++j) {
            array[i][j] = (float**)malloc(dim3 * sizeof(float*));
            for (int k = 0; k < dim3; ++k) {
                array[i][j][k] = (float*)malloc(dim4 * sizeof(float));
            }
        }
    }
    return array;
}


int*** allocate_3d_int_array(int dim1, int dim2, int dim3) {
    int ***array = (int***)malloc(dim1 * sizeof(int**));
    for (int i = 0; i < dim1; ++i) {
        array[i] = (int**)malloc(dim2 * sizeof(int*));
        for (int j = 0; j < dim2; ++j) {
            array[i][j] = (int*)malloc(dim3 * sizeof(int));
        }
    }
    return array;
}

float*** allocate_3d_float_array(int dim1, int dim2, int dim3) {
    float ***array = (float***)malloc(dim1 * sizeof(float**));
    for (int i = 0; i < dim1; ++i) {
        array[i] = (float**)malloc(dim2 * sizeof(float*));
        for (int j = 0; j < dim2; ++j) {
            array[i][j] = (float*)malloc(dim3 * sizeof(float));
        }
    }
    return array;
}

ttype*** allocate_3d_ttype_array(int dim1, int dim2, int dim3) {
    ttype ***array = (ttype***)malloc(dim1 * sizeof(ttype**));
    for (int i = 0; i < dim1; ++i) {
        array[i] = (ttype**)malloc(dim2 * sizeof(ttype*));
        for (int j = 0; j < dim2; ++j) {
            array[i][j] = (ttype*)malloc(dim3 * sizeof(ttype));
        }
    }
    return array;
}

float** allocate_2d_float_array(int dim1, int dim2) {
    float **array = (float**)malloc(dim1 * sizeof(float*));
    for (int i = 0; i < dim1; ++i) {
        array[i] = (float*)malloc(dim2 * sizeof(float));
    }
    return array;
}