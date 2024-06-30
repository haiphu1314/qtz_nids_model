#ifndef UTILS_H
#define UTILS_H
#define MAX_CHARS_LINE 1024
typedef struct {
    int bit_0;
    int bit_1;
} ttype;

int bitCount(int n);
int sign(int x);
int count_layers(const char* filename);

#endif // UTILS_H