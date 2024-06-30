#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#define USE_MSSE
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