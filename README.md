# MNIST Neural Network in C

A handwritten digit classifier built from scratch in C, no ML libraries, just raw math. Trains on the MNIST dataset and reaches ~95% accuracy.

## How it works

Example of a simple 2-layer fully connected neural network:

```
Input (784)  →  Hidden (128, sigmoid)  →  Output (10, sigmoid)
```

- **Forward pass** — weighted sum + sigmoid activation at each layer
- **Backpropagation** — chain rule to compute gradients, SGD to update weights
- **Loss** — Mean Squared Error (MSE)

## Project structure

```
├── main.c      # Training loop, testing, interactive mode
├── network.c   # Layer init, forward pass, backprop
├── network.h
├── mnist.c     # MNIST binary file loading
├── mnist.h
└── Makefile
```

## Getting started

### 1. Get the MNIST dataset

Download the 4 binary files from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download) and place them in the project root:

```
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

### 2. Build

```bash
make
```

Or manually:

```bash
gcc -Wall -O2 -o nn main.c network.c mnist.c -lm
```

### 3. Run

```bash
./nn
```

Training takes a minute or two. After it finishes you get:

```
--- Phase 1: Training (5 epochs, lr=0.100) ---
  Epoch 1 / 5 complete
  ...

--- Phase 2: Testing Accuracy ---
  Correct: 9512 / 10000
  Accuracy: 95.12%

--- Phase 3: Interactive Test ---
Press ENTER for a random digit, 'q' to quit.
```

In interactive mode, random test images are rendered as ASCII art with per-digit confidence scores

## Configuration

Edit the top of `main.c` to tweak:

| Parameter | Default | Effect |
|-----------|---------|--------|
| Hidden neurons | 128 | More = smarter but slower |
| Epochs | 5 | More = better accuracy (watch for overfitting) |
| Learning rate | 0.1 | Lower = more stable, higher = faster but unstable |
