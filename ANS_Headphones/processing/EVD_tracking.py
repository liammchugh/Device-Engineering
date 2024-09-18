import numpy as np

def evd_predict(signal_package, prediction_horizon=150, sampling_rate=5000):
    # Assuming signal_package has 500 samples (100ms of data at 5000 Hz)
    
    # Step 1: Compute the autocovariance matrix of the signal
    autocov_matrix = np.cov(signal_package, rowvar=False)
    
    # Step 2: Perform Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(autocov_matrix)
    
    # Step 3: Sort eigenvalues and eigenvectors (for signal dynamics analysis)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 4: Predict the next state using dominant eigenvector
    # Use the largest eigenvalue's corresponding eigenvector to predict future samples
    dominant_eigenvector = eigenvectors[:, 0]
    
    # Step 5: Estimate next signal values over the prediction horizon
    time_steps = int(prediction_horizon * (sampling_rate / 1000))
    predicted_signal = np.dot(signal_package[-time_steps:], dominant_eigenvector)
    
    return predicted_signal

if __name__ == "__main__":
    t = np.linspace(0, 0.1, 500)
    signal_package = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
    
    predicted_signal = evd_predict(signal_package)
    print("Predicted Signal:", predicted_signal)