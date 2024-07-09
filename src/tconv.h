#ifndef TCONV_H
#define TCONV_H

void conv_transpose_2d(float *input, float *output, float *weights, float *bias, 
                       int in_channels, int out_channels, 
                       int in_height, int in_width, 
                       int kernel_size, int stride, int padding, int output_padding);

#endif // TCONV_H