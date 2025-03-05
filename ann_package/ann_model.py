from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from .dataset_handler import DatasetHandler  # Import parent class

class ANNModel(DatasetHandler):
    def __init__(self, file_path, target_column, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200):
        super().__init__(file_path)  # Call parent constructor
        self.target_column = target_column
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                   learning_rate_init=learning_rate_init, 
                                   max_iter=max_iter, 
                                   random_state=42)

    def train_model(self):
        """Trains the ANN model using the dataset."""
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def test_model(self):
        """Tests the trained model and prints accuracy."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        return accuracy
