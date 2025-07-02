import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        print("Loading data...")
        data = pd.read_excel(self.filepath)
        print(f"Data shape: {data.shape}")
        print(data.info())
        print(data.describe())
        return data