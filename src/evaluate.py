from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\nEvaluation - {model_name}")
    print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
    print("R² Score:", r2_score(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    print("MSE:", mse)
    print("RMSE:", np.sqrt(mse))
    print("MAE:", mean_absolute_error(y_true, y_pred))

