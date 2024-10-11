import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Build the RNN model
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1))  # Output layer for predicting the next sample
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to predict signal using trained RNN
def rnn_predict(signal_package, model, time_horizon=150, sampling_rate=5000):
    # Reshape signal package for model input [samples, time steps, features]
    signal_package = signal_package.reshape((1, signal_package.shape[0], 1))
    
    # Predict the next signal values over the time horizon
    time_steps = int(time_horizon * (sampling_rate / 1000))
    predictions = []

    # Use the model to predict one step ahead repeatedly for the time horizon
    for _ in range(time_steps):
        next_sample = model.predict(signal_package)
        predictions.append(next_sample)
        # Shift the input window to simulate real-time package processing
        signal_package = np.roll(signal_package, -1)
        signal_package[0, -1, 0] = next_sample  # Append the prediction to the input
    
    return np.array(predictions).flatten()

if __name__ == "__main__":
    signal_package = np.random.randn(500)  # Example signal data
    model = build_rnn_model((500, 1))  # Assuming 500 samples input
    # Train the model before using it
    # model.fit(X_train, y_train, epochs=10)
    predicted_signal = rnn_predict(signal_package, model)
    print("Predicted Signal:", predicted_signal)