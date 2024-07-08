#include "conv.h"
#ifndef UTILS_H
#define UTILS_H
#define MAX_CHARS_LINE 1024
#define SIZEINT 32

typedef struct {
    int bit_0;
    int bit_1;
} ttype;

typedef enum {
    BNN,
    TBN,
    TNN
} quant_type;

int bitCount(int n);
int sign(int x);
int count_layers(const char* filename);
int**** allocate_4d_int_array(int dim1, int dim2, int dim3, int dim4);
float**** allocate_4d_float_array(int dim1, int dim2, int dim3, int dim4);
int*** allocate_3d_int_array(int dim1, int dim2, int dim3);
float*** allocate_3d_float_array(int dim1, int dim2, int dim3);
ttype*** allocate_3d_ttype_array(int dim1, int dim2, int dim3);
float** allocate_2d_float_array(int dim1, int dim2);

#endif // UTILS_H