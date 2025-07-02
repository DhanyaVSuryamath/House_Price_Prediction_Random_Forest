from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
        model = RandomForestRegressor(n_estimators=10)
        model.fit(self.X_TRAIN, self.Y_TRAIN)
        return model, model.predict(self.X_TEST)