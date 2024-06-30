#include "testcase.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"

void read_testcases(const char *filename, Testcase testcases[], int *num_testcases) {
    FILE *fp;
    char line[MAX_CHARS_LINE];  
    int index = 0;  

    fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Không thể mở file");
        return;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "testcase: ") == line) {
            sscanf(line, "testcase: %*d");  
        } else if (strstr(line, "data:tensor([") != NULL) {
            sscanf(line, "data:tensor([%f, %f, %f, %f, %f, %f, %f, %f",
                   &testcases[index].data[0], &testcases[index].data[1],
                   &testcases[index].data[2], &testcases[index].data[3],
                   &testcases[index].data[4], &testcases[index].data[5],
                   &testcases[index].data[6], &testcases[index].data[7]);

        } else if (strstr(line, "device") != NULL) {
            // Đọc dòng data
            sscanf(line, "%f, %f, %f, %f, %f",
                   &testcases[index].data[8], &testcases[index].data[9],
                   &testcases[index].data[10], &testcases[index].data[11],
                   &testcases[index].data[12]);

        } else if (strstr(line, "label: ") == line) {
            // Đọc dòng label
            sscanf(line, "label: %d", &testcases[index].label);
            index++;
        }
    }

    *num_testcases = index-1;  
    fclose(fp);
}