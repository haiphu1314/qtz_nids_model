/**
 * @ Author: Hai Phu, Email: haiphu@hcmut.edu.vn
 * @ Create Time: 2024-07-09 11:10:58
 * @ Modified by: Hai Phu
 * @ Modified time: 2024-07-09 16:00:41
 * @ Description:
 */

void conv_transpose_2d(float *input, float *output, float *weights, float *bias, 
                       int in_channels, int out_channels, 
                       int in_height, int in_width, 
                       int kernel_size, int stride, int padding, int output_padding) {
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int i = 0; i < out_height * out_width; ++i) {
            output[oc * out_height * out_width + i] = bias[oc];
        }
    }


    for (int ic = 0; ic < in_channels; ++ic) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ih = 0; ih < in_height; ++ih) {
                for (int iw = 0; iw < in_width; ++iw) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int oh = ih * stride + kh - padding;
                            int ow = iw * stride + kw - padding;
                            if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                output[oc * out_height * out_width + oh * out_width + ow] += 
                                    input[ic * in_height * in_width + ih * in_width + iw] * 
                                    weights[(oc * in_channels + ic) * kernel_size * kernel_size + kh * kernel_size + kw];
                            }
                        }
                    }
                }
            }
        }
    }
}