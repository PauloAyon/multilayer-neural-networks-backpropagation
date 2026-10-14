import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

def save_figure(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')

def plot_loss(loss_history, title, filename, log_interval=10):
    plt.figure(figsize=(8, 5))
    epochs = range(0, len(loss_history) * log_interval, log_interval)
    plt.plot(epochs, loss_history, color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    save_figure(filename)
    plt.close()

def plot_weights(weight_history, title, filename, log_interval=10, bias_history=None):
    if not weight_history:
        return
    if bias_history is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    epochs = range(0, len(weight_history) * log_interval, log_interval)
    n_layers = len(weight_history[0])
    
    for layer_idx in range(n_layers):
        layer_weights_list = [w[layer_idx] for w in weight_history]
        layer_weights_np = np.array(layer_weights_list)
        n_weights = layer_weights_np.shape[1] * layer_weights_np.shape[2]
        
        if n_weights <= 8:
            for i in range(layer_weights_np.shape[1]):
                for j in range(layer_weights_np.shape[2]):
                    axes[0].plot(epochs, layer_weights_np[:, i, j],
                                 label=f'L{layer_idx+1} W[{i},{j}]', alpha=0.7)
        else:
            axes[0].plot(epochs, layer_weights_np[:, 0, 0], label=f'L{layer_idx+1} W[0,0]')
            axes[0].plot(epochs, layer_weights_np[:, -1, -1], label=f'L{layer_idx+1} W[-1,-1]')

    axes[0].set_title('Weight evolution')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Weight value')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    if bias_history is not None:
        for layer_idx in range(n_layers):
            layer_biases = np.array([biases[layer_idx] for biases in bias_history])
            n_biases = layer_biases.shape[2]
            selected_biases = range(n_biases) if n_biases <= 4 else [0, n_biases // 2, n_biases - 1]
            for bias_idx in selected_biases:
                axes[1].plot(
                    epochs, layer_biases[:, 0, bias_idx],
                    label=f'L{layer_idx+1} b[{bias_idx}]', alpha=0.8
                )

        axes[1].set_title('Bias evolution')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Bias value')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(filename)
    plt.close(fig)

def plot_xor_results(X, y_true, predictions, names, filename):
    x_labels = [f'({int(row[0])},{int(row[1])})' for row in X]
    x_pos = np.arange(len(x_labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['s', '^', 'D']

    for column, output_name in enumerate(['XOR', 'NOT-XOR']):
        ax = axes[column]
        ax.plot(x_pos, y_true[:, column], color='black', linestyle='--', marker='o',
                markerfacecolor='white', markersize=8,
                label='Target', linewidth=1.5, zorder=5)

        for index, (y_pred, name) in enumerate(zip(predictions, names)):
            offset = (index - (len(predictions) - 1) / 2) * 0.12
            ax.plot(
                x_pos + offset, y_pred[:, column],
                color=colors[index], marker=markers[index], linestyle=':',
                label=name, linewidth=1.5, markersize=5, zorder=6
            )

        ax.set_title(output_name)
        ax.set_xlabel('Input')
        ax.set_xticks(x_pos, x_labels)
        ax.set_xlim(-0.4, len(x_labels) - 0.6)
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')

    axes[0].set_ylabel('Output')
    fig.suptitle('XOR and NOT-XOR: Targets and MLP Predictions')
    fig.tight_layout()
    save_figure(filename)
    plt.close(fig)

def plot_regression(X, y_true, y_pred, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(X.flatten(), y_true.flatten(), color='blue', alpha=0.5, s=10, label='True Function')
    plt.plot(X.flatten(), y_pred.flatten(), color='red', linewidth=2, label='MLP Prediction')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure(filename)
    plt.close()

def plot_comparison(X, y_true, y_pred_sigmoid, y_pred_tanh, mse_sigmoid, mse_tanh, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(X.flatten(), y_true.flatten(), color='black', alpha=0.3, s=10, label='True Function')
    plt.plot(X.flatten(), y_pred_sigmoid.flatten(), color='blue', linewidth=2, label=f'Sigmoid (MSE: {mse_sigmoid:.4f})')
    plt.plot(X.flatten(), y_pred_tanh.flatten(), color='red', linewidth=2, label=f'Tanh (MSE: {mse_tanh:.4f})')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure(filename)
    plt.close()
