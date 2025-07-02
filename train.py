from src.config import RAW_DATA_PATH
from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer
from src.evaluate import evaluate_model

def run_pipeline():
    # Load
    data = DataLoader(RAW_DATA_PATH).load_data()

    # Clean
    cleaner = DataCleaner(data)
    cleaner.display_dtypes()
    cleaner.missing_summary()
    cleaner.show_uniques(['MSSubClass', 'MSZoning', 'LotConfig', 'BldgType', 'OverallCond', 'Exterior1st'])

    # Feature Engineering
    fe = FeatureEngineer(data)
    fe.fill_missing()
    fe.drop_id_column()
    fe.encode_columns()
    X, y = fe.split_X_y()

    # Model training
    trainer = ModelTrainer(X, y)
    lr_model, lr_preds = trainer.train_linear_regression()
    rf_model, rf_preds = trainer.train_random_forest()

    # Evaluation
    evaluate_model(trainer.Y_TEST, lr_preds, "Linear Regression")
    evaluate_model(trainer.Y_TEST, rf_preds, "Random Forest")

if __name__ == "__main__":
    run_pipeline()