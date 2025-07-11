import pickle
import pandas as pd
from src.Feature_Engineering2 import FeatureEngineer

def predict_on_test(test_filepath):
    # Load test data
    test_data = pd.read_csv(test_filepath) 
    
    # Preprocess test data same as training
    data_transform = FeatureEngineer(test_data)
    data_transform.fill_missing()
    data_transform.drop_id_column()
    data_transform.encode_columns()
    
    X_test = data_transform.test_data  
    
    # Load the saved model
    with open('model/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Save predictions
    output = pd.DataFrame({'Id': test_data['Id'], 'PredictedSalePrice': predictions})
    output.to_csv('output/test_predictions.csv', index=False)
    print("Predictions saved to 'output/test_predictions.csv'")

    output2 = pd.DataFrame({'Id' : test_data['Id'],   'MSSubClass': test_data['MSSubClass'], 'MSZoning': test_data['MSZoning'], 'LotArea': test_data['LotArea'], 
         'LotConfig': test_data['LotConfig'],  'BldgType': test_data['BldgType'],'OverallCond': test_data['OverallCond'], 'YearBuilt': test_data['YearBuilt'], 
         'YearRemodAdd': test_data['YearRemodAdd'], 'Exterior1st': test_data['Exterior1st'], 'BsmtFinSF2': test_data['BsmtFinSF2'], 'TotalBsmtSF': test_data['TotalBsmtSF'],
          'PredictedSalePrice': predictions })
    
    output2.to_csv('output/test_predictions2.csv', index=False)
    print("Predictions saved to 'output/test_predictions2.csv'")

if __name__ == "__main__":
    predict_on_test(r"C:\Users\dsuryamath\DHANYA\ML_PRACTICE_PROJECTS\House_Price_Prediction_Random_Forest\data\Test_data.csv")
