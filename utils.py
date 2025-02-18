import torch

def monte_carlo_dropout(model,x,iterations= 50):
    """Predict uncertainty using Monte Carlo Dropout"""
    model.train() #Activate dropout layers
    predictions = []    
    for _ in range(iterations):
        with torch.no_grad():
            predictions.append(model(x).unsqueeze(0))
    predictions = torch.cat(predictions)
    mean = predictions.mean(0)
    std = predictions.std(0)
    return mean, std
        