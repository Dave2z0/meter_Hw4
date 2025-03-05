import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_preprocess import DataPreprocessing 

class ANNModel:
    def __init__(self, file_path, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200):
        self.file_path = file_path
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, 
                                   learning_rate_init=self.learning_rate_init, 
                                   max_iter=self.max_iter, 
                                   random_state=42)
        self.data_processor = DataPreprocessing()  

    def load_and_preprocess_data(self):
        """Loads and preprocesses the dataset."""
        data = self.data_processor.load_data(self.file_path)  
        
        # Splitting features and target
        X, y = data[:, :-1], data[:, -1]  

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_model(self):
        """Trains the ANN model using the dataset."""
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def test_model(self):
        """Tests the trained model and prints accuracy."""
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"Model Accuracy: {accuracy:.4f}")
        return accuracy
