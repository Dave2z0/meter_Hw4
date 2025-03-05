# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# class DatasetHandler:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.data = None
#         self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
#         self.scaler = StandardScaler()

#     def load_data(self):
#         """Loads dataset from a CSV file."""
#         self.data = pd.read_csv(self.file_path,sep="\t", header=None).dropna()
#         return self.data

#     def preprocess_data(self, target_column):
#         """Splits data into training and test sets, then scales features."""
    
#         print("Dataset columns:", self.data.columns)  # Debugging line
    
#         if target_column not in self.data.columns:
#             raise KeyError(f"Column '{target_column}' not found in dataset. Available columns: {self.data.columns.tolist()}")
        
#         X = self.data.drop(columns=[target_column])
#         y = self.data[target_column]
    
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#         # Standardize the data
#         X_train = self.scaler.fit_transform(X_train)
#         X_test = self.scaler.transform(X_test)
    
#         self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
#         return X_train, X_test, y_train, y_test
    

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DatasetHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()

    def load_data(self):
        """Loads dataset from a TXT file, assuming no headers."""
        column_count = 36  # Adjust based on the number of feature columns
        column_names = [f"feature_{i}" for i in range(column_count)] + ["target"]  # Assign "target" as last column
        
        # Load the dataset and manually assign column names
        self.data = pd.read_csv(self.file_path, sep="\t", header=None, names=column_names)

        print("Dataset loaded successfully!")
        print(self.data.head())  # Debugging: Check first rows
        print("Columns in dataset:", self.data.columns.tolist())  # Verify column names

        return self.data

    def preprocess_data(self, target_column="target"):
        """Splits data into training and test sets, then scales features."""
        print("Dataset columns:", self.data.columns)  # Debugging

        if target_column not in self.data.columns:
            raise KeyError(f"Column '{target_column}' not found in dataset. Available columns: {self.data.columns.tolist()}")

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test
