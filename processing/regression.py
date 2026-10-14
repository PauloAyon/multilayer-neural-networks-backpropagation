import numpy as np
from processing.mlp import MLP
from processing.metrics import mse
from processing.plots import plot_weights, plot_comparison, save_figure
import matplotlib.pyplot as plt

def generate_data():
    # Function 1: f(x) = 2*cos(2x) - sin(x) for 0 <= x <= 2*pi
    x1 = np.append(np.arange(0, 2 * np.pi, 0.1), 2 * np.pi).reshape(-1, 1)
    y1 = 2 * np.cos(2 * x1) - np.sin(x1)
    
    # Function 2: f(x) = 2*cos(2x) - sin(x) + 0.9*exp(sin(3x)) for -2*pi <= x <= 2*pi
    x2 = np.append(np.arange(-2 * np.pi, 2 * np.pi, 0.1), 2 * np.pi).reshape(-1, 1)
    y2 = 2 * np.cos(2 * x2) - np.sin(x2) + 0.9 * np.exp(np.sin(3 * x2))
    
    return (x1, y1), (x2, y2)

def train_model(X, y, hidden_sizes, activation, epochs, lr, name, scaled_initialization=False):
    if not isinstance(hidden_sizes, list):
        hidden_sizes = [hidden_sizes]
    architecture = '-'.join(str(size) for size in hidden_sizes)
    print(f"  {name} ({activation}, hidden layers: {architecture})")

    x_mean = np.mean(X, axis=0)
    x_scale = np.std(X, axis=0)
    y_mean = np.mean(y, axis=0)
    y_scale = np.std(y, axis=0)
    x_scale[x_scale == 0] = 1
    y_scale[y_scale == 0] = 1

    X_scaled = (X - x_mean) / x_scale
    y_scaled = (y - y_mean) / y_scale

    nn = MLP(input_size=X.shape[1], hidden_sizes=hidden_sizes, output_size=1, activation=activation)
    if scaled_initialization:
        nn.weights[0] *= 0.5
        for layer_idx in range(1, len(nn.weights)):
            fan_in = hidden_sizes[layer_idx - 1]
            nn.weights[layer_idx] *= np.sqrt(2 / fan_in)

    nn.train(X_scaled, y_scaled, epochs=epochs, learning_rate=lr, log_interval=100)
    y_pred = nn.predict(X_scaled) * y_scale + y_mean
    final_mse = mse(y, y_pred)
    print(f"    Final MSE: {final_mse:.6f}")
    return nn, y_pred, final_mse

def run_regression():
    print()
    print("Running Task 2: Function Approximation...")
    
    (x1, y1), (x2, y2) = generate_data()
    
    results = {}
    
    # Function 1 experiments
    print("  Function 1: f(x) = 2*cos(2x) - sin(x), x in [0, 2*pi]")
    # Sigmoid
    nn1_sig, y_pred1_sig, mse1_sig = train_model(x1, y1, 30, 'sigmoid', 50000, 0.01, 'Sigmoid')
    results['func1_sigmoid'] = {'model': nn1_sig, 'mse': mse1_sig, 'y_pred': y_pred1_sig, 'X': x1, 'y': y1}
    # Tanh
    nn1_tanh, y_pred1_tanh, mse1_tanh = train_model(x1, y1, 30, 'tanh', 50000, 0.01, 'Tanh')
    results['func1_tanh'] = {'model': nn1_tanh, 'mse': mse1_tanh, 'y_pred': y_pred1_tanh, 'X': x1, 'y': y1}
    
    # Function 2 experiments
    print("  Function 2: f(x) = 2*cos(2x) - sin(x) + 0.9*exp(sin(3x)), x in [-2*pi, 2*pi]")
    # Sigmoid
    nn2_sig, y_pred2_sig, mse2_sig = train_model(
        x2, y2, 80, 'sigmoid', 120000, 0.03, 'Sigmoid',
        scaled_initialization=True
    )
    results['func2_sigmoid'] = {'model': nn2_sig, 'mse': mse2_sig, 'y_pred': y_pred2_sig, 'X': x2, 'y': y2}
    # Tanh
    nn2_tanh, y_pred2_tanh, mse2_tanh = train_model(
        x2, y2, [40, 40], 'tanh', 200000, 0.02, 'Tanh',
        scaled_initialization=True
    )
    results['func2_tanh'] = {'model': nn2_tanh, 'mse': mse2_tanh, 'y_pred': y_pred2_tanh, 'X': x2, 'y': y2}
    
    # Combined comparison figures
    print()
    print("  Generating comparison figures...")
    
    # Function 1: Loss curves comparison
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, act in zip(axes, ['sigmoid', 'tanh']):
        key = f'func1_{act}'
        epochs = range(0, len(results[key]['model'].loss_history) * 100, 100)
        ax.plot(epochs, results[key]['model'].loss_history, color='red', linewidth=2)
        ax.set_title(f'Function 1 - {act.capitalize()} (raw MSE: {results[key]["mse"]:.4f})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Standardized MSE')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_figure('function1_loss_comparison.png')
    plt.close()
    
    # Function 2: Loss curves comparison
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, act in zip(axes, ['sigmoid', 'tanh']):
        key = f'func2_{act}'
        epochs = range(0, len(results[key]['model'].loss_history) * 100, 100)
        ax.plot(epochs, results[key]['model'].loss_history, color='blue', linewidth=2)
        ax.set_title(f'Function 2 - {act.capitalize()} (raw MSE: {results[key]["mse"]:.4f})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Standardized MSE')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_figure('function2_loss_comparison.png')
    plt.close()
    
    # Function 1: Best sigmoid vs tanh approximation
    plot_comparison(
        x1, y1,
        results['func1_sigmoid']['y_pred'], results['func1_tanh']['y_pred'],
        mse1_sig, mse1_tanh,
        'Function 1 - Sigmoid vs Tanh Approximation', 'function1_approximation.png'
    )
    
    # Function 2: Best sigmoid vs tanh approximation
    plot_comparison(
        x2, y2,
        results['func2_sigmoid']['y_pred'], results['func2_tanh']['y_pred'],
        mse2_sig, mse2_tanh,
        'Function 2 - Sigmoid vs Tanh Approximation', 'function2_approximation.png'
    )
    
    # Select the lower-MSE activation for each function
    best_func1 = min(['func1_sigmoid', 'func1_tanh'], key=lambda key: results[key]['mse'])
    best_func2 = min(['func2_sigmoid', 'func2_tanh'], key=lambda key: results[key]['mse'])
    for best_key, function_name, filename in [
        (best_func1, 'Function 1', 'function1_weights_biases.png'),
        (best_func2, 'Function 2', 'function2_weights_biases.png')
    ]:
        best_result = results[best_key]
        activation = best_key.rsplit('_', 1)[1]
        plot_weights(
            best_result['model'].weight_history,
            f'{function_name} Best ({activation.capitalize()}) - Weights and Biases',
            filename,
            log_interval=100,
            bias_history=best_result['model'].bias_history
        )

    # Summary
    print()
    print("  Best Results Summary:")
    print(f"    Function 1: {best_func1.rsplit('_', 1)[1]} (MSE = {results[best_func1]['mse']:.6f})")
    print(f"    Function 2: {best_func2.rsplit('_', 1)[1]} (MSE = {results[best_func2]['mse']:.6f})")

    return results