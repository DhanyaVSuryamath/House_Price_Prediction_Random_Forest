class DataCleaner:
    def __init__(self, data):
        self.data = data

    def display_dtypes(self):
        print("\nData types:")
        for dtype in ['object', 'int64', 'float64']:
            cols = self.data.select_dtypes(include=dtype).columns.tolist()
            print(f"{dtype} columns: {cols}\n")

    def missing_summary(self):
        print("\nMissing values:\n", self.data.isnull().sum())

    def show_uniques(self, columns):
        print("\nUnique values:")
        for col in columns:
            print(f"{col}: {self.data[col].dropna().unique().tolist()}" if col in self.data.columns else f"{col}: Column not found!")
