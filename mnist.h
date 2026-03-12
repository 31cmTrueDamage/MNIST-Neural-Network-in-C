#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>

void load_mnist_image(FILE* img_file, double* target_array);
void load_mnist_label(FILE* label_file, double* target_output_array);

#endif