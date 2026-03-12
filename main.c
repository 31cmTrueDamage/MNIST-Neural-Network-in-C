#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "mnist.h"

// ── helpers ──────────────────────────────────────────────────────────────────

static int run_forward(Network* net, double* input_pixels) {
    double* current_input = input_pixels;
    for (int j = 0; j < net->layer_count; j++) {
        forward_pass(&net->layers[j], current_input);
        current_input = net->layers[j].outputs;
    }
    Layer* out = &net->layers[net->layer_count - 1];
    int prediction = 0;
    double max_val = out->outputs[0];
    for (int j = 1; j < 10; j++) {
        if (out->outputs[j] > max_val) {
            max_val    = out->outputs[j];
            prediction = j;
        }
    }
    return prediction;
}

// ── phases ───────────────────────────────────────────────────────────────────

static void phase_train(Network* net, int epochs, double lr) {
    FILE* img_file = fopen("train-images.idx3-ubyte", "rb");
    FILE* lbl_file = fopen("train-labels.idx1-ubyte", "rb");
    if (!img_file || !lbl_file) { printf("Error: Could not open training files.\n"); return; }

    double input_pixels[784];
    double target_outputs[10];

    printf("--- Phase 1: Training (%d epochs, lr=%.3f) ---\n", epochs, lr);
    for (int epoch = 0; epoch < epochs; epoch++) {
        fseek(img_file, 16, SEEK_SET);
        fseek(lbl_file,  8, SEEK_SET);
        for (int i = 0; i < 60000; i++) {
            load_mnist_image(img_file, input_pixels);
            load_mnist_label(lbl_file, target_outputs);
            run_forward(net, input_pixels);
            back_propagation(net, target_outputs, input_pixels, lr);
        }
        printf("  Epoch %d / %d complete\n", epoch + 1, epochs);
    }

    fclose(img_file);
    fclose(lbl_file);
}

static void phase_test(Network* net) {
    FILE* img_file = fopen("t10k-images.idx3-ubyte", "rb");
    FILE* lbl_file = fopen("t10k-labels.idx1-ubyte", "rb");
    if (!img_file || !lbl_file) { printf("Error: Could not open test files.\n"); return; }

    fseek(img_file, 16, SEEK_SET);
    fseek(lbl_file,  8, SEEK_SET);

    double input_pixels[784];
    int correct = 0;
    int total   = 10000;

    printf("\n--- Phase 2: Testing Accuracy ---\n");
    for (int i = 0; i < total; i++) {
        load_mnist_image(img_file, input_pixels);
        unsigned char actual;
        fread(&actual, sizeof(unsigned char), 1, lbl_file);
        if (run_forward(net, input_pixels) == (int)actual) correct++;
    }

    printf("  Correct: %d / %d\n", correct, total);
    printf("  Accuracy: %.2f%%\n", ((double)correct / total) * 100.0);

    fclose(img_file);
    fclose(lbl_file);
}

static void phase_interactive(Network* net) {
    FILE* img_file = fopen("t10k-images.idx3-ubyte", "rb");
    FILE* lbl_file = fopen("t10k-labels.idx1-ubyte", "rb");
    if (!img_file || !lbl_file) { printf("Error: Could not open test files.\n"); return; }

    printf("\n--- Phase 3: Interactive Test ---\n");
    printf("Press ENTER for a random digit, 'q' to quit.\n\n");

    double input_pixels[784];
    char   buf[8];

    while (1) {
        printf("[ ENTER = random digit | q = quit ]: ");
        fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin)) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;

        int pick = rand() % 10000;
        fseek(img_file, 16 + pick * 784, SEEK_SET);
        fseek(lbl_file,  8 + pick,       SEEK_SET);

        load_mnist_image(img_file, input_pixels);
        unsigned char actual;
        fread(&actual, sizeof(unsigned char), 1, lbl_file);

        int prediction = run_forward(net, input_pixels);

        // ASCII art
        printf("\n");
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double p = input_pixels[row * 28 + col];
                if      (p > 0.75) printf("##");
                else if (p > 0.50) printf("**");
                else if (p > 0.25) printf("..");
                else               printf("  ");
            }
            printf("\n");
        }

        // Confidence bars
        Layer* out = &net->layers[net->layer_count - 1];
        printf("\nConfidence:\n");
        for (int j = 0; j < 10; j++) {
            int bar = (int)(out->outputs[j] * 20);
            printf("  %d [", j);
            for (int b = 0; b < 20; b++) printf(b < bar ? "=" : " ");
            printf("] %5.1f%%%s\n",
                   out->outputs[j] * 100.0,
                   j == prediction ? " <-- predicted" : "");
        }

        printf("\nActual: %d  |  Predicted: %d  |  %s\n\n",
               (int)actual, prediction,
               prediction == (int)actual ? "CORRECT :)" : "WRONG :(");
    }

    fclose(img_file);
    fclose(lbl_file);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    srand(time(NULL));

    // Network architecture: 784 -> 128 -> 10
    Network net;
    net.layer_count = 2;
    net.layers      = malloc(net.layer_count * sizeof(Layer));
    init_layer(&net.layers[0], 784, 128);
    init_layer(&net.layers[1], 128,  10);

    phase_train(&net, /*epochs=*/5, /*lr=*/0.1);
    phase_test(&net);
    phase_interactive(&net);

    // Cleanup
    for (int i = 0; i < net.layer_count; i++) free_layer(&net.layers[i]);
    free(net.layers);

    return 0;
}