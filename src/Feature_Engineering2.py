from sklearn.preprocessing import LabelEncoder
class FeatureEngineer:
    def __init__(self, data):
        self.train_data = data.copy()
        print("TRAINING DATA",self.train_data,"\n")
        self.test_data = data.copy()
        print("TESTING DATA",self.test_data)

    def fill_missing(self):
        fill_mode = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
        fill_mean = ['Id','MSSubClass','LotArea','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF2','TotalBsmtSF']

        for col in fill_mode:
            self.train_data[col] = self.train_data[col].fillna(self.train_data[col].mode()[0])
            self.test_data[col] = self.test_data[col].fillna(self.test_data[col].mode()[0])

        for col in fill_mean:
            self.train_data[col] = self.train_data[col].fillna(self.train_data[col].mean())
            self.test_data[col] = self.test_data[col].fillna(self.test_data[col].mean())

    def drop_id_column(self):
        self.train_data.drop(columns=['Id'], inplace=True, errors='ignore')
        self.test_data.drop(columns=['Id'], inplace=True, errors='ignore')

    def encode_columns(self):
        le = LabelEncoder()
        for col in ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']:
            if col in self.train_data.columns:
                self.train_data[col] = le.fit_transform(self.train_data[col])

        LE = LabelEncoder()
        for col in ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']:
            if col in self.test_data.columns:
                self.test_data[col] = LE.fit_transform(self.test_data[col])

    def split_X_y(self):
        X = self.train_data.drop(columns=['SalePrice'])
        y = self.train_data['SalePrice']
        return X, y
