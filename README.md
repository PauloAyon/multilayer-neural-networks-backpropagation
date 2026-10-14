# Multilayer Neural Networks and Backpropagation

This project implements multilayer perceptrons (MLP) with hand-written backpropagation for two tasks:

1. **XOR / NOT-XOR Classification** — Learning non-linearly separable logic functions using MLP with sigmoid activation. Tests three architectures: 2-2-2, 2-4-2, and 2-4-4-2 (extra hidden layer).

2. **Function Approximation** — Approximating two non-linear functions using MLP with both sigmoid and tanh activations in hidden layers (linear output):
   - f(x) = 2·cos(2x) - sin(x) for x in [0, 2π]
   - f(x) = 2·cos(2x) - sin(x) + 0.9·exp(sin(3x)) for x in [-2π, 2π]

## Requirements

- Python 3.10+
- numpy
- matplotlib

Install dependencies:
```bash
python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## How to Run

```bash
python main.py
```

All required figures are saved to the `output/` directory. Regression inputs and targets are standardized during training. The program creates the output directory if it does not exist.

## Project Structure

```
Filename/
├── main.py                 # Entry point - runs all tasks
├── README.md               
├── requirements.txt        
├── .gitignore              
├── output/                 # Generated figures (auto-created)
└── processing/
    ├── mlp.py              # Core MLP implementation with backpropagation
    ├── xor.py              # Task 1: XOR/NOT-XOR classification
    ├── regression.py       # Task 2: Function approximation
    ├── metrics.py          # MSE and accuracy metrics
    └── plots.py            # Plotting utilities
```

## Parameters to Tweak

Edit `processing/xor.py` and `processing/regression.py` to adjust:
- Network architecture (hidden layer sizes)
- Learning rate
- Number of epochs
- Activation function (sigmoid/tanh for regression)

## Generated Figures

The program produces comparison figures for each task:
- **XOR**: Loss curves for all architectures, target-versus-prediction plots for both outputs, and weight evolution for the lowest-MSE model
- **Function Approximation**: Sigmoid vs tanh approximation plots and MSE curves, plus weight and bias evolution for the lower-MSE activation for each function.
