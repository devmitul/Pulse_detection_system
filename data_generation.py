import numpy as np
import h5py

def skewed_gaussian(x, mu, sigma_l, sigma_r, A):
    return np.where(
        x < mu,
        A * np.exp(-((x - mu)**2) / (2 * sigma_l**2)),
        A * np.exp(-((x - mu)**2) / (2 * sigma_r**2))
    )

def generate_augmented_pulse(window_length, T):
    mu = np.random.uniform(0.2, 0.8 * T)
    sigma_l = np.random.uniform(0.01, 0.05)
    sigma_r = np.random.uniform(0.01, 0.05)
    x = np.linspace(0, T, window_length)
    A = np.random.uniform(0.8, 1.2)
    clean = skewed_gaussian(x, mu, sigma_l, sigma_r, A)
    amplified = clean * np.random.uniform(0.9, 1.1)
    noise = np.random.normal(0, np.random.uniform(0.01, 0.1), window_length)
    return x, amplified + noise, mu, sigma_l, sigma_r

def save_dataset(filename, n_samples=10000, window_length=400, T=2.0):
    with h5py.File(filename, 'w') as hf:
        pulses = hf.create_group('pulses')
        times = hf.create_group('times')
        mus = []
        lefts = []
        rights = []
        for i in range(n_samples):
            x, pulse, mu, sigma_l, sigma_r = generate_augmented_pulse(window_length, T)
            pulses.create_dataset(f'pulse_{i}', data=pulse)
            times.create_dataset(f'time_{i}', data=x)
            mus.append(mu)
            lefts.append(mu - np.sqrt(2 * sigma_l**2 * np.log(10)))
            rights.append(mu + np.sqrt(2 * sigma_r**2 * np.log(10)))
        hf.create_dataset('mus', data=mus)
        hf.create_dataset('lefts', data=lefts)
        hf.create_dataset('rights', data=rights)