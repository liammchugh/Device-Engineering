import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_noise_std, measurement_noise_std):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

    def predict(self, control_input, control_matrix):
        process_noise = np.random.randn(self.num_particles, control_matrix.shape[1]) * self.process_noise_std
        self.particles += np.dot(control_matrix, control_input) + process_noise

    def update(self, measurement, measurement_matrix):
        for i, particle in enumerate(self.particles):
            prediction = np.dot(measurement_matrix, particle)
            error = np.linalg.norm(measurement - prediction)
            self.weights[i] *= np.exp(-error ** 2 / (2 * self.measurement_noise_std ** 2))

        self.weights /= np.sum(self.weights)
        self.resample()

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

class BayesianLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=25, output_size=5):
        super(BayesianLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logvar = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mu = self.fc_mu(lstm_out)
        logvar = self.fc_logvar(lstm_out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def train(self, x, y, n_iter):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
    
if __name__ == '__main__':
    method = 'ParticleFilter'  # Change this to 'BayesianLSTM' to test the BayesianLSTM class
    if method == 'ParticleFilter':
        # Example setup for Particle Filter
        num_particles = 100
        state_dim = 2
        process_noise_std = 0.1
        measurement_noise_std = 0.1

        pf = ParticleFilter(num_particles, state_dim, process_noise_std, measurement_noise_std)

        control_input = np.array([[0.1]])
        control_matrix = np.array([[1, 0], [0, 1]])
        measurement = np.array([[1.2]])
        measurement_matrix = np.array([[1, 0]])

        pf.predict(control_input, control_matrix)
        pf.update(measurement, measurement_matrix)
