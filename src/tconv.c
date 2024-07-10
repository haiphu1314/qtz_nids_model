/**
 * @ Author: Hai Phu, Email: haiphu@hcmut.edu.vn
 * @ Create Time: 2024-07-09 11:10:58
 * @ Modified by: Hai Phu
 * @ Modified time: 2024-07-09 18:43:59
 * @ Description:
 */

tconv_layer* create_tconv_layer(int input_channel, int output_channel, int kernel_size, int stride, int padding, int dilation, quant_type quant) {
    conv_layer *layer = (conv_layer *)malloc(sizeof(conv_layer));
    layer->input_channel = input_channel;
    layer->output_channel = output_channel;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->dilation = dilation;
    
    layer->input_thres = 0.0;
    layer->quant = quant;
    int intput_sizeint = (input_channel % SIZEINT) == 0 ? (input_channel/SIZEINT): (input_channel/SIZEINT+1);
    int dim1 = output_channel;
    int dim2 = intput_sizeint;
    int dim3 = kernel_size;
    int dim4 = kernel_size;
    switch(quant){
        case BNN:
        case TBN:
            layer->weights = allocate_4d_int_array(dim1, dim2, dim3, dim4);
            if (layer->weights == NULL) {
                fprintf(stderr, "Memory allocation failed for weights\n");
                exit(1);
            }
            break;
        case TNN:
            layer->weights_0 = allocate_4d_int_array(dim1, dim2, dim3, dim4);
            if (layer->weights_0 == NULL) {
                fprintf(stderr, "Memory allocation failed for weights_0\n");
                exit(1);
            }
            layer->weights_1 = allocate_4d_int_array(dim1, dim2, dim3, dim4);
            if (layer->weights_1 == NULL) {
                fprintf(stderr, "Memory allocation failed for weights_1\n");
                exit(1);
            }
            break;
        default:
            fprintf(stderr, "create_conv_layer: Unknown quantization type \n");
            exit(1);        

    }
    return layer;
}

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