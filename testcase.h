#ifndef TESECASE_H
#define TESECASE_H
#include "utils.h" 
#define DATA_SIZE 13

typedef struct {
    float data[DATA_SIZE];
    int label;
} Testcase;

void read_testcases(const char *filename, Testcase testcases[], int *num_testcases);
#endif // TESECASE_H