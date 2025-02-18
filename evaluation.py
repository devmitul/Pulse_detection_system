import numpy as np
from sklearn.metrics import r2_score
import torch
from sklearn.metrics import mean_absolute_error


def evaluate_model(model, data_loader, device):
    """Evaluate model performace using MAE and R^2 score"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            predictions.append(outputs.cpu().numpy())
            targets.append(y.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    mse = np.mean((predictions - targets)**2)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {'MSE': mse, 'MAE': mae, 'R^2 Score': r2}
