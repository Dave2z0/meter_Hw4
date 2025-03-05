import pandas as pd

class DataPreprocessing:
    def __init__(self):
        pass

    def load_data(self, path):
        """Loads dataset from a TXT file, assumes tab-separated values with no headers."""
        data = pd.read_csv(path, sep="\t", header=None).dropna()

        print("First 5 rows of dataset:")
        print(data.head())  # Debugging: Show first few rows
        
        return data.to_numpy()  
