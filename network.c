#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"

double apply_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void init_layer(Layer* l, int input_count, int neuron_count) {
    l->neuron_count = neuron_count;
    l->input_count  = input_count;

    l->weights = malloc(neuron_count * input_count * sizeof(double));
    l->biases  = malloc(neuron_count * sizeof(double));
    l->outputs = malloc(neuron_count * sizeof(double));
    l->deltas  = malloc(neuron_count * sizeof(double));

    for (int i = 0; i < neuron_count * input_count; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        l->weights[i] = -1.0 + (r * 2.0);
    }
    for (int i = 0; i < neuron_count; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        l->biases[i] = -1.0 + (r * 2.0);
    }
}

void forward_pass(Layer* l, double* input_values) {
    for (int i = 0; i < l->neuron_count; i++) {
        double sum = 0.0;
        for (int j = 0; j < l->input_count; j++) {
            sum += l->weights[i * l->input_count + j] * input_values[j];
        }
        sum += l->biases[i];
        l->outputs[i] = apply_sigmoid(sum);
    }
}

void free_layer(Layer* l) {
    free(l->weights);
    free(l->biases);
    free(l->outputs);
    free(l->deltas);
}

void back_propagation(Network* net, double* target_output, double* original_inputs, double lr) {
    int L = net->layer_count - 1;
    Layer* last_layer = &net->layers[L];

    // Output layer deltas
    for (int i = 0; i < last_layer->neuron_count; i++) {
        double diff  = last_layer->outputs[i] - target_output[i];
        double slope = last_layer->outputs[i] * (1.0 - last_layer->outputs[i]);
        last_layer->deltas[i] = diff * slope;
    }

    // Hidden layer deltas
    for (int i = L - 1; i >= 0; i--) {
        Layer* cur  = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        for (int j = 0; j < cur->neuron_count; j++) {
            double sum = 0.0;
            for (int k = 0; k < next->neuron_count; k++) {
                sum += next->weights[k * next->input_count + j] * next->deltas[k];
            }
            cur->deltas[j] = sum * (cur->outputs[j] * (1.0 - cur->outputs[j]));
        }
    }

    // Update weights and biases
    for (int i = L; i >= 0; i--) {
        Layer* cur = &net->layers[i];
        double* inputs = (i == 0) ? original_inputs : net->layers[i - 1].outputs;

        for (int j = 0; j < cur->neuron_count; j++) {
            cur->biases[j] -= lr * cur->deltas[j];
            for (int k = 0; k < cur->input_count; k++) {
                cur->weights[j * cur->input_count + k] -= lr * cur->deltas[j] * inputs[k];
            }
        }
    }
}

void calculate_network_cost(Network* net, double* target_output) {
    Layer* last = &net->layers[net->layer_count - 1];
    double sum  = 0.0;
    for (int i = 0; i < last->neuron_count; i++) {
        double diff = last->outputs[i] - target_output[i];
        sum += diff * diff;
    }
    printf("MSE: %f\n", sum / last->neuron_count);
}