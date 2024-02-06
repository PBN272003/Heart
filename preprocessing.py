import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,mean_absolute_error,precision_score,r2_score,mean_squared_error
from sklearn.pipeline import Pipeline 
#from heart import impute_categorical_data,impute_continuous_data, scale_data, encode_data
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error,r2_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data 

def preprocess_data(data):
    # Perform your preprocessing here
    categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']
    bool_cols = ['fbs','exang']
    numeric_cols = ['oldpeak','thalch','chol','trestbps','age']
    missing_data_cols = data.isnull().sum()[data.isnull().sum() > 0].index.tolist()
    data = impute_categorical_data(data, categorical_cols, bool_cols,missing_data_cols)
    data = impute_continuous_data(data, numeric_cols, bool_cols,missing_data_cols)
    data = scale_data(data)
    data = encode_data(data)
    model, y_test, y_pred = train_model(data)
    mse, r2, rmse = evaluate_model(y_test, y_pred)
    return data, model, mse, r2, rmse


def impute_categorical_data(data, categorical_cols, bool_cols,missing_data_cols):
    heart_null = data[data[categorical_cols].isnull()]
    heart_not_null = data[data[categorical_cols].notnull()]
    X = heart_not_null.drop(categorical_cols,axis=1)
    y = heart_not_null[categorical_cols]
    
    other_missing_cols =  [col for col in missing_data_cols if col != categorical_cols]
    
    label_encoder = LabelEncoder()
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
            
    if categorical_cols in bool_cols:
        y = label_encoder.fit_transform(y)
        
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)
    
    for col in other_missing_cols:
        if X[col].isnull().sum()>0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train,y_train)
    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("The feature '"+ categorical_cols+ "' has been imputed with", round((accuracy * 100), 2), "accuracy\n")
    
    X = heart_null.drop(categorical_cols, axis=1)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
        
    if len(heart_null) > 0:
        heart_null[categorical_cols] = random_forest.predict(X)
        # Map predicted boolean values back to True/False if the target variable is boolean
        if categorical_cols in bool_cols:
            heart_null[categorical_cols] = heart_null[categorical_cols].map({0: False, 1: True})
        else:
            pass
    else:
        pass
    
    heart_combined = pd.concat([heart_not_null, heart_null])
    return heart_combined[categorical_cols]

def impute_continuous_data(data, numeric_cols, bool_cols,missing_data_cols):
    heart_null = data[data[numeric_cols].isnull()]
    heart_not_null = data[data[numeric_cols].notnull()]
    X = heart_not_null.drop(numeric_cols,axis=1)
    y = heart_not_null[numeric_cols]
    other_missing_cols = [col for col in missing_data_cols if col != numeric_cols]
    
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
            
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor on the non-missing data
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)

    # Predict the target variable for the missing data
    y_pred = random_forest.predict(X_test)

    # Print regression performance metrics
    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    # Prepare the missing data for imputation
    X = heart_null.drop(numeric_cols, axis=1)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
            
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(heart_null) > 0: 
        heart_null[numeric_cols] = random_forest.predict(X)
    else:
        pass

    df_combined = pd.concat([heart_not_null, heart_null])
    
    return df_combined[numeric_cols]

def scale_data(data):
    columns_to_scale = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']
    minmax_scaler={}
    for col in columns_to_scale:
        minmax = MinMaxScaler()
        data[col] = minmax.fit_transform(data[[col]])
        minmax_scaler[col] = minmax
    return data, minmax_scaler

def encode_data(data):
    encoded_columns = ['sex','dataset','cp','fbs','restecg','exang','slope','ca','thal']
    label_encoder = {}
    for col in encoded_columns:
        encode = LabelEncoder()
        data[col] = encode.fit_transform(data[[col]])
        label_encoder[col] = encode
    return data, label_encoder

def train_model(data):
    # Split the data into features and target variable
    X = data.drop(columns=['num'])
    y = data['num']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    # Train the model (RandomForestClassifier as an example)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_balanced, y_balanced)

    # Evaluate the model
    y_pred = model.predict(X_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion_mat)

    # Return the trained model (you can save it if needed)
    return model,y_test,y_pred

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    rmse = mean_squared_error(y_test,y_pred,squared=False)
    return mse,r2,rmse
    