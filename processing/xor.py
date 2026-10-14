import numpy as np
from processing.mlp import MLP
from processing.metrics import mse, accuracy
from processing.plots import plot_weights, plot_xor_results, save_figure
import matplotlib.pyplot as plt

def run_xor():
    print("Multilayer Neural Networks and Backpropagation")
    print()
    print("Running Task 1: XOR / NOT-XOR Classification...")
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    
    print("  Experiment 1: 2-2-2 Architecture (2 hidden neurons)")
    nn1 = MLP(input_size=2, hidden_sizes=[2], output_size=2, activation='sigmoid')
    nn1.train(X, y, epochs=30000, learning_rate=0.5, log_interval=10)
    y_pred1 = nn1.predict(X)
    mse1 = mse(y, y_pred1)
    acc1 = accuracy(y, y_pred1)
    print(f"    Final MSE: {mse1:.6f}")
    print(f"    Accuracy: {acc1:.2%}")
    
    print("  Experiment 2: 2-4-2 Architecture (4 hidden neurons)")
    nn2 = MLP(input_size=2, hidden_sizes=[4], output_size=2, activation='sigmoid')
    nn2.train(X, y, epochs=30000, learning_rate=0.5, log_interval=10)
    y_pred2 = nn2.predict(X)
    mse2 = mse(y, y_pred2)
    acc2 = accuracy(y, y_pred2)
    print(f"    Final MSE: {mse2:.6f}")
    print(f"    Accuracy: {acc2:.2%}")
    
    print("  Experiment 3: 2-4-4-2 Architecture (Two hidden layers: 4, 4 neurons)")
    nn3 = MLP(input_size=2, hidden_sizes=[4, 4], output_size=2, activation='sigmoid')
    nn3.train(X, y, epochs=30000, learning_rate=0.1, log_interval=10)
    y_pred3 = nn3.predict(X)
    mse3 = mse(y, y_pred3)
    acc3 = accuracy(y, y_pred3)
    print(f"    Final MSE: {mse3:.6f}")
    print(f"    Accuracy: {acc3:.2%}")
    
    # Combined loss curve for all architectures
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, nn, name, mse_val in zip(axes, [nn1, nn2, nn3], 
                                      ['2-2-2', '2-4-2', '2-4-4-2'], 
                                      [mse1, mse2, mse3]):
        epochs = range(0, len(nn.loss_history) * 10, 10)
        ax.plot(epochs, nn.loss_history, color='red', linewidth=2)
        ax.set_title(f'{name} (MSE: {mse_val:.4f})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_figure('xor_all_loss_curves.png')
    plt.close()
    
    # Predictions for all architectures
    models = [nn1, nn2, nn3]
    model_names = ['2-2-2', '2-4-2', '2-4-4-2']
    predictions = [y_pred1, y_pred2, y_pred3]
    plot_xor_results(X, y, predictions, model_names, 'xor_all_predictions.png')
    
    best_index = int(np.argmin([mse1, mse2, mse3]))
    best_model = models[best_index]
    best_name = model_names[best_index]
    plot_weights(best_model.weight_history, f'XOR Best Model ({best_name}) - Weight Evolution', 'xor_best_weights.png')
    print(f"  Best architecture: {best_name} (MSE: {[mse1, mse2, mse3][best_index]:.6f})")
    
    print()
    return {'2-2-2': mse1, '2-4-2': mse2, '2-4-4-2': mse3}