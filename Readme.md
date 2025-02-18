# Pulse Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Introduction
The Pulse Detection System is a hybrid CNN-LSTM model designed to detect peaks and thresholds in noisy time-series data. This project provides a robust solution for applications such as industrial IoT, healthcare (ECG/PPG analysis), and scientific research. By combining the strengths of CNNs for feature extraction with LSTMs for modeling temporal dynamics, the system is highly effective even in noisy environments.

## Key Features
- **Hybrid Architecture:**
  - CNN layers extract local features from time-series data.
  - LSTM layers model temporal dependencies and long-range patterns.
- **Attention Mechanism:**
  - Multi-head self-attention focuses on critical signal regions, improving peak detection accuracy.
- **Data Augmentation:**
  - Generates synthetic pulses with controlled noise and variability for robust training.
- **Performance Metrics:**
  - Evaluates model performance using MSE, MAE, and R² score.
- **Visualization:**
  - Provides interactive pulse visualization with predicted thresholds using Plotly for enhanced interpretability.

## Installation

### Clone the Repository
```bash
git clone https://github.com/devmitul/Pulse_detection_system.git
cd Pulse_detection_system
```
## Set Up Virtual Environment

### For Linux/macOS:
```bash
python -m venv pulse_env
source pulse_env/bin/activate
```
## For Windows:
```bash
python -m venv pulse_env
pulse_env\Scripts\activate
```

## Install Dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

Run the data generation script to create training and test datasets:

```bash
python data_generation.py
```
### Train the Model

Train the model using the generated data:

```bash
python main.py
```

### Evaluate the Model

After training, the model’s performance metrics (MSE, MAE, R²) will be printed, and a sample pulse prediction will be visualized using Plotly.

# Contributing  

We welcome contributions to enhance this project. To contribute:  

1. **Fork the repository.**  
2. **Create a new branch:**  
   ```bash
   git checkout -b feature/new-feature
    git commit -m "Add new feature"
   git push origin feature/new-feature
    ```

# Troubleshooting  

### Q: Runtime Error: Unmet CUDA requirements  
**A:** Ensure PyTorch is installed with the correct CUDA version. Use the CPU-only version if CUDA is unavailable:  
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
### Q: Runtime Error: Unmet CUDA requirements
**A:** Install h5py using pip:
```bash
pip install h5py
```

# License  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

# Contact  

For questions or support, please contact:  
**Email:** [devmitul@gmail.com](mailto:mitulnaliyadhara2000@gmail.com)  


