import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from an Excel file."""
    data = pd.read_excel(file_path)
    print("Data loaded successfully.")
    return data

def preprocess_data(data):
    """Preprocess the data: handle numeric columns and fill NaN values."""
    # Convert all columns to numeric, forcing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)
    return data

def split_data(data, target_column):
    """Split the data into features and target, and scale the features."""
    # Ensure only numeric columns are used
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # Drop the target column from features
    numeric_columns.remove(target_column)

    X = data[numeric_columns]  # Features only
    y = data[target_column]  # Target variable

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)  # Scale all features

    return X_scaled, y, scaler

if __name__ == "__main__":
    data = load_data('data/input_data.xlsx')
    processed_data = preprocess_data(data)
