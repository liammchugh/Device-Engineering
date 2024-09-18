import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path, history_steps=1000, prediction_steps=500, overlap=1.0):
    data = pd.read_csv(file_path)
    values = data.values.flatten()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))
    
    X, y = [], []
    step_size = int(prediction_steps * (overlap))
    
    # Create overlapping windows of historical data for training. 
    # Each window overlaps with the previous by 'step_size' samples.
    for i in range(0, len(scaled_values) - history_steps - prediction_steps, step_size):
        X.append(scaled_values[i:i + history_steps])
        y.append(scaled_values[i + history_steps:i + history_steps + prediction_steps])
    
    return np.array(X), np.array(y), scaler

def train_model(X_train, y_train, X_test, y_test, model_save_path):
    model = dl.build_rnn_model((X_train.shape[1], 1))
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])
    return model

if __name__ == "__main__":
    file_path = 'your_timeseries_data.csv'
    model_save_path = 'rnn_model.h5'
    from ANS_Headphones.processing import DL_RNN as dl
    X, y, scaler = load_data(file_path, history_steps=1000, prediction_steps=500, overlap=1.0)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, X_test, y_test, model_save_path)
    np.save('scaler.npy', scaler)
    print(f"Model trained and saved to {model_save_path}")