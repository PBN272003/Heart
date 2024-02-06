# zenml_pipeline.py
from zenml import pipeline, step
from preprocessing import load_data, preprocess_data, train_model,evaluate_model
import os 
import sys

# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# sys.path.append(root_path)

@step
def load_data_step() -> dict:
    file_path = 'heart_disease_uci.csv'
    data = load_data(file_path)
    return {'data': data}

@step
def preprocess_data_step(data: dict) -> dict:
    preprocessed_data = preprocess_data(data['data'])
    return {'preprocessed_data': preprocessed_data}

@step
def train_model_step(data: dict) -> None:
    model = train_model(data['preprocessed_data'])
    return model

@step 
def evaluate_model_step(data: dict) -> None:
    # Assuming y_test and y_pred are available in data['preprocessed_data']
    y_test = data['preprocessed_data']['y_test']
    y_pred = data['model'].predict(data['preprocessed_data']['X_test'])
    
    mse, r2, rmse = evaluate_model(y_test, y_pred)
    return mse,r2,rmse 



@pipeline
def heart_pipeline():
    data = load_data_step()
    preprocessed_data = preprocess_data_step(data)
    model = train_model_step(preprocessed_data)
    evaluate_model({'model': model, 'preprocessed_data': preprocessed_data})


if __name__ == "__main__":
    run = heart_pipeline()