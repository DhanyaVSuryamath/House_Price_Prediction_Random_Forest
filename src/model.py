from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

class ModelTrainer:
    def __init__(self, X, y):
        self.X_TRAIN, self.X_TEST, self.Y_TRAIN, self.Y_TEST = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_linear_regression(self):
        print("\nTraining Linear Regression...")
        model = LinearRegression()
        model.fit(self.X_TRAIN, self.Y_TRAIN)
        return model, model.predict(self.X_TEST)

    def train_random_forest(self):
        print("\nTraining Random Forest...")
        model_RFR = RandomForestRegressor(n_estimators=10)
        model_RFR.fit(self.X_TRAIN, self.Y_TRAIN)
        os.makedirs('model', exist_ok=True)
        with open('model/random_forest_model.pkl', 'wb') as f:
            pickle.dump(model_RFR, f)
        print("Random Forest model saved to 'model/random_forest_model.pkl'")
        return model_RFR, model_RFR.predict(self.X_TEST)
    