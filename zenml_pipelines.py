# zenml_pipeline.py
from zenml import pipeline, step
from preprocessing import preprocess_data, train_model, evaluate_model, load_data
import pandas as pd
from typing import Tuple 
from typing_extensions import Annotated

@step
def load_data_step() -> pd.DataFrame:
    file_path = 'C:/Users/DELL/Downloads/heart/heart_disease_uci.csv'
    data = load_data(file_path)
    print(data.shape)
    return data

@step
def preprocess_data_step(data: pd.DataFrame) -> pd.DataFrame:
    preprocessed_data = preprocess_data(data['fbs'])
    return {'preprocessed_data': preprocessed_data}

@step
def train_model_step(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    model, y_test, y_pred = train_model(preprocessed_data['data'])
    return {'model': model, 'y_test': y_test, 'y_pred': y_pred}

@step 
def evaluate_model_step(model_data: dict) -> dict:
    mse, r2, rmse = evaluate_model(model_data['y_test'], model_data['y_pred'])
    return {'mse': mse, 'r2': r2, 'rmse': rmse}

@pipeline 
def heart_pipeline():
    data = load_data_step()
    preprocessed_data = preprocess_data_step(data)
    model_data = train_model_step(preprocessed_data)
    evaluate_model_step(model_data)
    print(data['fbs'].isnull().sum())
    return data
if __name__ == "__main__":
    run = heart_pipeline()
    # Access the results if needed
    print("Results:", run.evaluate_model_step.outputs)
