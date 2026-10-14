import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def accuracy(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(float)
    return np.mean(y_pred_binary == y_true)