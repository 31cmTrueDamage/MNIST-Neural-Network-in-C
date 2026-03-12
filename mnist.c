#include <stdio.h>
#include "mnist.h"

void load_mnist_image(FILE* img_file, double* target_array) {
    unsigned char pixel;
    for (int i = 0; i < 784; i++) {
        if (fread(&pixel, sizeof(unsigned char), 1, img_file) > 0) {
            target_array[i] = (double)pixel / 255.0;
        }
    }
}

void load_mnist_label(FILE* label_file, double* target_output_array) {
    unsigned char label;
    if (fread(&label, sizeof(unsigned char), 1, label_file) > 0) {
        for (int i = 0; i < 10; i++) target_output_array[i] = 0.0;
        target_output_array[label] = 1.0;
    }
}