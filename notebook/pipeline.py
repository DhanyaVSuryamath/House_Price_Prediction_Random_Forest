
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error




class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        print(" Loading data...")
        data = pd.read_excel(self.filepath)
        print(f"Data shape: {data.shape}")
        print(data.info())
        print(data.describe())
        return data


class DataCleaner:
    def __init__(self, data):
        self.data = data

    def display_dtypes(self):
        obj = self.data.dtypes == "object"
        ints = self.data.dtypes == "int64"
        floats = self.data.dtypes == "float64"
        print(" Object columns:", self.data.columns[obj].tolist(), "\n")
        print(" Integer columns:", self.data.columns[ints].tolist(), "\n")
        print(" Float columns:", self.data.columns[floats].tolist(), "\n")

    def missing_summary(self):
        print(" Missing values:\n", self.data.isnull().sum())

    def show_uniques(self, columns):
        print("\n Unique values:")
        for col in columns:
            if col in self.data.columns:
                print(f"{col}: {self.data[col].dropna().unique().tolist()}")
            else:
                print(f"{col}: Column not found!")


class FeatureEngineer:
    def __init__(self, data):
        self.train_data = data.iloc[:1460, :].copy()
        self.test_data = data.iloc[1460:, :12].copy()

    def fill_missing(self):
        self.train_data['Exterior1st'] = self.train_data['Exterior1st'].fillna(self.train_data['Exterior1st'].mode()[0])
        self.train_data['MSZoning'] = self.train_data['MSZoning'].fillna(self.train_data['MSZoning'].mode()[0])
        self.train_data['BsmtFinSF2'] = self.train_data['BsmtFinSF2'].fillna(self.train_data['BsmtFinSF2'].mean())
        self.train_data['TotalBsmtSF'] = self.train_data['TotalBsmtSF'].fillna(self.train_data['TotalBsmtSF'].mean())

    def drop_id_column(self):
        if 'Id' in self.train_data.columns:
            self.train_data.drop(columns=['Id'], inplace=True)

    def encode_columns(self):
        encoding = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
        le = LabelEncoder()
        for col in encoding:
            if col in self.train_data.columns:
                self.train_data[col] = le.fit_transform(self.train_data[col])

    def split_X_y(self):
        X = self.train_data.drop(columns=['SalePrice'], axis=1)
        y = self.train_data['SalePrice']
        return X, y


class ModelTrainer:
    def __init__(self, X, y):
        self.X_TRAIN, self.X_TEST, self.Y_TRAIN, self.Y_TEST = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_linear_regression(self):
        print("\n Training Linear Regression...")
        ln = LinearRegression()
        ln.fit(self.X_TRAIN, self.Y_TRAIN)
        y_pred_LR = ln.predict(self.X_TEST)
        mape = mean_absolute_percentage_error(self.Y_TEST, y_pred_LR)
        r2 = r2_score(self.Y_TEST, y_pred_LR)
        print("Linear Regression MAPE:", mape)
        print("Linear Regression R² Score:", r2)

    def train_random_forest(self):
        print("\n Training Random Forest Regressor...")
        rf = RandomForestRegressor(n_estimators=10)
        rf.fit(self.X_TRAIN, self.Y_TRAIN)
        y_pred_RF = rf.predict(self.X_TEST)
        mape = mean_absolute_percentage_error(self.Y_TEST, y_pred_RF)
        r2 = r2_score(self.Y_TEST, y_pred_RF)
        print("Random Forest MAPE:", mape)
        print("Random Forest R² Score:", r2)
        
         # Calculate MSE
        mse = mean_squared_error(self.Y_TEST, y_pred_RF)
        print(f"MSE: {mse}")

        # Calculate RMSE
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")
        
        # Calculate MAE
        mae = mean_absolute_error(self.Y_TEST, y_pred_RF)
        print(f"MAE: {mae}")


# MASTER PIPELINE RUNNER FUNCTION

def run_full_pipeline(filepath):
    # Load data
    loader = DataLoader(filepath)
    data = loader.load_data()

    # Clean and inspect data
    cleaner = DataCleaner(data)
    cleaner.display_dtypes()
    cleaner.missing_summary()
    cleaner.show_uniques(['MSSubClass', 'MSZoning', 'LotConfig', 'BldgType', 'OverallCond', 'Exterior1st'])

    # Feature engineering
    fe = FeatureEngineer(data)
    fe.fill_missing()
    fe.drop_id_column()
    fe.encode_columns()
    X, y = fe.split_X_y()

    # Train and evaluate models
    trainer = ModelTrainer(X, y)
    trainer.train_linear_regression()
    trainer.train_random_forest()


# 🟢 RUN THE PIPELINE (Pass your dataset path here)
run_full_pipeline(r"C:\Users\CHINMAYI SURYAMATH\Desktop\ML PROJECT ECOMMERCE\Final_House_Price_prediction\HousePricePrediction.xlsx")

 