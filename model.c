#include <stdio.h>
#include <stdlib.h>
#include <math.h>



int bitCount(int n) {
    int count = 0;
    while (n) {
        n &= (n - 1); 
        count++;
    }
    return count;
}

int main() {
    return 0;
}


typedef struct {
    int input_channel;
    int output_channel;
    float *weights;
} Layer;



// Hàm đọc weight từ file txt
Layer* read_model(const char* filename, int* num_layers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[1024];
    int layer_count = 0;
    int max_layers = count_layers(filename); // Số lớp tối đa
    Layer *layers = (Layer *)malloc(max_layers * sizeof(Layer));

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "linear")) {
            // if (layer_count == max_layers) {
            //     max_layers *= 2;
            //     layers = (Layer *)realloc(layers, max_layers * sizeof(Layer));
            // }
            Layer *layer = &layers[layer_count++];
            fscanf(file, "input_channel: %d\n", &layer->input_channel);
            fscanf(file, "output_channel: %d\n", &layer->output_channel);

            int weight_size = layer->input_channel * layer->output_channel;
            layer->weights = (float *)malloc(weight_size * sizeof(float));

            for (int i = 0; i < weight_size; ++i) {
                fscanf(file, "%f", &layer->weights[i]);
            }
        }
    }

    fclose(file);
    *num_layers = layer_count;
    return layers;
}

// Hàm tính toán forward
void forward(Layer* layers, int num_layers, const float* input, float* output, int input_size, int output_size) {
    float* curr_input = (float *)malloc(input_size * sizeof(float));
    memcpy(curr_input, input, input_size * sizeof(float));

    for (int l = 0; l < num_layers; ++l) {
        Layer *layer = &layers[l];
        float* next_input = (float *)calloc(layer->output_channel, sizeof(float));

        for (int i = 0; i < layer->output_channel; ++i) {
            for (int j = 0; j < layer->input_channel; ++j) {
                next_input[i] += curr_input[j] * layer->weights[i * layer->input_channel + j];
            }
        }

        free(curr_input);
        curr_input = next_input;
    }

    memcpy(output, curr_input, output_size * sizeof(float));
    free(curr_input);
}

void free_model(Layer* layers, int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        free(layers[i].weights);
    }
    free(layers);
}

int main() {
    int num_layers;
    Layer* layers = read_model("model_parameters.txt", &num_layers);

    int input_size = layers[0].input_channel;
    int output_size = layers[num_layers - 1].output_channel;

    float input[input_size];
    for (int i = 0; i < input_size; ++i) {
        input[i] = 1.0; // Giá trị đầu vào ví dụ
    }

    float output[output_size];
    forward(layers, num_layers, input, output, input_size, output_size);

    printf("Output: ");
    for (int i = 0; i < output_size; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");

    free_model(layers, num_layers);
    return 0;
}