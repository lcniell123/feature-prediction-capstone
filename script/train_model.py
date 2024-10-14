import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_data(file_path):
    """Load data from an Excel file."""
    data = pd.read_excel(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data: handle numeric columns and fill NaN values."""
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.fillna(0, inplace=True)
    return data

def build_model(input_shape):
    """Build a neural network model."""
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)  # Single output for regression
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(data, target_column):
    """Train a neural network model on the provided data for a specific target column."""
    X = data.drop(columns=[target_column])  # Drop the target column
    y = data[target_column]  # Target variable

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)  # Scale all features

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = build_model(X_train.shape[1])  # Input shape for the model

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Save the model and scaler
    model.save(f'models/{target_column}_model.h5')
    joblib.dump(scaler, f'models/{target_column}_scaler.pkl')  # Save the scaler for later use

def main():
    data = load_data('data/input_data.xlsx')
    processed_data = preprocess_data(data)

    target_columns = [
        'Vibration Frequency',
        'Vibration Amplitude',
        'Bearing Temperature',
        'Motor Temperature',
        'Belt Load',
        'Torque',
        'Noise Levels',
        'Current and Voltage',
        'Hydraulic Pressure',
        'Belt Thickness',
        'Roller Condition'
    ]

    for target_column in target_columns:
        print(f"Training model for {target_column}...")
        train_model(processed_data, target_column)
        print(f"Model for {target_column} trained and saved.")

if __name__ == "__main__":
    main()
