from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self, data):
        self.train_data = data.iloc[:1460, :].copy()
        self.test_data = data.iloc[1460:, :12].copy()

    def fill_missing(self):
        fill_mode = ['Exterior1st', 'MSZoning']
        fill_mean = ['BsmtFinSF2', 'TotalBsmtSF']
        for col in fill_mode:
            self.train_data[col] = self.train_data[col].fillna(self.train_data[col].mode()[0])
        for col in fill_mean:
            self.train_data[col] = self.train_data[col].fillna(self.train_data[col].mean())

    def drop_id_column(self):
        self.train_data.drop(columns=['Id'], inplace=True, errors='ignore')

    def encode_columns(self):
        le = LabelEncoder()
        for col in ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']:
            if col in self.train_data.columns:
                self.train_data[col] = le.fit_transform(self.train_data[col])

    def split_X_y(self):
        X = self.train_data.drop(columns=['SalePrice'])
        y = self.train_data['SalePrice']
        return X, y
