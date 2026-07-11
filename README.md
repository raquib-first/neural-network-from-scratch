# Neural Network from Scratch in C++

![C++](https://img.shields.io/badge/C++-17-blue)
![Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-95.71%25-brightgreen)

A fully functional feedforward neural network built from the ground up in pure C++ — 
no ML libraries, no PyTorch, no TensorFlow. Every component including matrix math, 
backpropagation, Adam optimizer, and dropout is implemented by hand.

Trained on 60,000 MNIST images and achieves **95.71% accuracy** on 10,000 test samples.

---

## Why This Project

Most ML practitioners use frameworks like PyTorch without understanding what happens 
underneath. This project was built to understand every single operation — from how 
a matrix multiply works in memory, to why the Softmax + Cross Entropy gradient 
simplifies to `ŷ - y`, to why Adam outperforms vanilla SGD.

Every line of code was written from scratch. No external dependencies except the C++ 
standard library.

---

## Architecture

```
Input Layer      Hidden Layer 1     Hidden Layer 2     Output Layer
   (784)     →     (128, ReLU)   →    (64, ReLU)    →   (10, Softmax)
```

- **Input Layer**: 784 pixel values from a 28×28 grayscale image
- **Hidden Layer 1**: 128 neurons, ReLU activation
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons, Softmax activation (one per digit, 0–9)

### Forward Pass

1. **Weighted sum + bias**:  
   `z = W·x + b`

2. **ReLU activation** (applied at each hidden layer):  
   `a = max(0, z)`

3. **Repeat** for each hidden layer

4. **Softmax output**:  
   Produces 10 probabilities, one per digit (0–9)

5. **Prediction**:  
   `Predicted digit = argmax(output)`

---

## Features

### Core
- **Custom Matrix class** — flat `vector<double>` with row-major storage (`data[i*cols+j]`) for cache efficiency. Supports `+`, `-`, `*`, `/`, transpose, Hadamard product, scalar ops, `applyFunc()`
- **He initialization** — `scale = sqrt(2/input_size)` prevents vanishing/exploding gradients with ReLU
- **Activations** — ReLU (hidden layers), numerically stable Softmax with max subtraction (output layer), Sigmoid
- **Forward pass** — `z = W·x + b` then `a = activation(z)`, storing both for backprop

### Training
- **Backpropagation** — full chain rule implementation. Output gradient `ŷ - y` (Softmax + Cross Entropy combined), hidden gradients via `Wᵀ · δ`, weight gradients via `δ · xᵀ`
- **Adam optimizer** — first moment `m`, second moment `v`, bias correction terms, per-weight adaptive learning rates
- **SGD optimizer** — simple gradient descent as alternative
- **Cross Entropy loss** — `L = -Σ y·log(ŷ)` with natural log

### Practical
- **Inverted Dropout** — random neuron deactivation during training, scaled by `1/(1-p)` to keep expected values consistent. Disabled during inference
- **Training/Inference mode** — `setTraining(true/false)` switches dropout on/off across all layers
- **Binary model save/load** — weights and biases serialized to `.bin` file using raw byte I/O with `reinterpret_cast`
- **MNIST parser** — IDX binary format reader with big-endian to little-endian byte reversal
- **Epoch shuffling** — training indices shuffled each epoch using `std::shuffle` + `mt19937`

---

## Project Structure

```
src/
├── matrix.hpp      # Matrix class — flat storage, all math operators, He init,
│                   #   applyFunc(), binary save/load
├── layer.hpp       # Layer class — forward pass, backprop, activations,
│                   #   dropout, weight update, save/load
├── optimizer.hpp   # Adam and SGD — moment tracking, bias correction,
│                   #   adaptive learning rates
├── network.hpp     # Network class — layer management, full training pipeline,
│                   #   loss computation, binary save/load, mode switching
├── mnist.hpp       # MNIST IDX binary parser — magic number validation,
│                   #   big-endian reversal, pixel normalization
└── main.cpp        # Training loop, epoch management, shuffling
```

---

## How to Build and Run

**Requirements:** g++ with C++17 support

### Download MNIST Dataset
Download all 4 files from http://yann.lecun.com/exdb/mnist/ and place them in the 
same directory as the compiled binary:
```
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

### Compile and Run
```bash
# standard build
g++ -std=c++17 -o main src/main.cpp

# optimized build (recommended)
g++ -O3 -std=c++17 -o main src/main.cpp

# run
./main
```

### Expected Output
```
epoch 0 avg loss: 0.230234
epoch 1 avg loss: 0.155757
epoch 2 avg loss: 0.140046
accuracy: 95.71%
21134s
```

---

## Results

All experiments run for 3 epochs · 60,000 training samples · 10,000 test samples  
Architecture: 784 → 128 → 64 → 10

| Version                        | Optimizer | Compile Flag | Training Time | Test Accuracy |
|--------------------------------|-----------|--------------|---------------|---------------|
| Baseline                       | Adam      | default      | 21134s        | 95.71%        |
| Loop order optimization        | Adam      | default      |  7982s        | 95.84%        |
| Compiler optimization          | Adam      | `-O3`        |   829s        | 96.40%        |
| Loop + Compiler optimization   | Adam      | `-O3`        |   950s        | 96.65%        |
| Dropout 0.2                    | Adam      | `-O3`        |  1244s        | 96.20%        |
| SGD lr=0.01                    | SGD       | `-O3`        |   173s        | 96.31%        |

**Key takeaways:**
- `-O3` compiler flag alone gives **25x speedup** with zero code changes
- Loop order optimization gives **2.6x speedup** from cache-friendly memory access
- SGD with tuned learning rate is **5x faster than Adam** and competitive on accuracy
- Adam is more robust — converges reliably without learning rate tuning
- Dropout needs more than 3 epochs to show its benefit
- Best accuracy: **96.65%**
- Fastest training: **173s**

---

## Key Concepts and Implementation Details

### Why flat vector<double> instead of vector<vector<double>>?
Each row in a `vector<vector<double>>` is a separate heap allocation at a random 
memory address. When the CPU reads memory it fetches 64-byte cache lines — with 
flat storage, the next 8 doubles are already in cache. With nested vectors, each 
row access causes a cache miss. In a 1000×1000 matrix multiply with 1 billion 
memory accesses, this difference is 0.5s vs 50s.

### Why does Softmax + Cross Entropy simplify to ŷ - y?
The Jacobian of Softmax combined with the derivative of Cross Entropy loss 
cancels out the exponential terms — the `i==j` and `i!=j` cases of the Jacobian 
merge with the log derivative, leaving just `ŷ - y`. This is why Softmax and 
Cross Entropy are always used together at the output layer.

### Why He initialization?
ReLU zeros out roughly half of all neurons (negative values). This halves the 
variance of the signal passing through each layer. He initialization compensates 
by scaling weights by `sqrt(2/input_size)` — the factor of 2 restores the 
original variance, keeping signals from shrinking to zero or exploding across 
deep networks.

### Why Adam over SGD?
SGD uses the same learning rate for every weight. Adam tracks a running average 
of gradients (`m`) and squared gradients (`v`) per weight. Weights with large, 
consistent gradients get smaller updates; weights with small or noisy gradients 
get larger updates. This adaptive per-weight learning rate makes Adam converge 
faster and more reliably, especially early in training.

### Why inverted dropout?
During training with dropout rate 0.2, only 80% of neurons are active. At 
inference all 100% are active — the next layer suddenly sees 25% more signal 
than it was trained on. Scaling active neurons by `1/(1-0.2) = 1.25` during 
training compensates, keeping expected signal magnitude identical between 
training and inference.

---

## Roadmap

- [ ] ikj loop order — fix cache-unfriendly access pattern in matrix multiply
- [ ] Mini-batch training — accumulate gradients over N samples before updating
- [ ] Real image input via `stb_image.h` — predict from any `.jpg` or `.png`
- [ ] CUDA acceleration — port matrix multiply to GPU
- [ ] Convolutional layers — extend to image-specific architecture

---

## What This Project Demonstrates

This project is the conceptual foundation of every modern ML framework:

| This project       | Production equivalent              |
|--------------------|------------------------------------|
| `matrix.hpp`       | NumPy ndarray / PyTorch Tensor     |
| `layer.hpp`        | `torch.nn.Linear`                  |
| `relu()`, `softmax()` | `torch.nn.functional`           |
| `backward()`       | `loss.backward()`                  |
| Adam optimizer     | `torch.optim.Adam`                 |
| MNIST training loop| Any supervised training script     |

The difference is PyTorch runs on GPU with automatic differentiation. 
The math is identical.

---
