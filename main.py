import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pulse_detection import HybridCNNLSTM
from data_generation import save_dataset
from evaluation import evaluate_model
from visualization import plot_pulse

class PulseDataset(Dataset):
    def __init__(self, h5_filename):
        self.h5_file = h5py.File(h5_filename, "r")
        self.pulses = self.h5_file["pulses"]
        self.mus = self.h5_file["mus"][:]
        self.lefts = self.h5_file["lefts"][:]
        self.rights = self.h5_file["rights"][:]
        self.length = len(self.mus)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pulse = self.pulses[f"pulse_{idx}"][:]
        target = np.array([self.mus[idx], self.lefts[idx], self.rights[idx]])
        return torch.tensor(pulse, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def train_model(model, train_loader, test_loader, device, num_epochs=15):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Test Loss={avg_test_loss:.6f}")

    return train_losses, test_losses

if __name__ == "__main__":
    save_dataset("train_data.h5", n_samples=8000, window_length=400, T=2.0)
    save_dataset("test_data.h5", n_samples=2000, window_length=400, T=2.0)

    train_dataset = PulseDataset("train_data.h5")
    test_dataset = PulseDataset("test_data.h5")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNLSTM(cnn_channels=64, lstm_hidden=128).to(device)
    train_losses, test_losses = train_model(model, train_loader, test_loader, device, num_epochs=15)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.show()

    #Adding evaluation metrics and visualizations
    metrics = evaluate_model(model, test_loader, device)
    print("Evaluation Metrics:{metrics}") 

    #Visualoze a sample pulse
    sample_pulse, sample_target = test_dataset[0]
    sample_pulse = sample_pulse.unsqueeze(0).to(device)
    with torch.no_grad():
        sample_prediction = model(sample_pulse).cpu().numpy()[0]
    plot_pulse(test_dataset.h5_file["times"]["time_0"][:], sample_pulse.cpu().numpy()[0], sample_prediction)