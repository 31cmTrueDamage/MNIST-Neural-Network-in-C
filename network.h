#ifndef NETWORK_H
#define NETWORK_H

typedef struct Layer {
    int neuron_count;
    int input_count;

    double* weights;
    double* biases;
    double* outputs;
    double* deltas;
} Layer;

typedef struct Network {
    int layer_count;
    Layer* layers;
} Network;

void   init_layer(Layer* l, int input_count, int neuron_count);
void   forward_pass(Layer* l, double* input_values);
void   free_layer(Layer* l);

void   back_propagation(Network* net, double* target_output, double* original_inputs, double lr);
void   calculate_network_cost(Network* net, double* target_output);

double apply_sigmoid(double x);

#endif