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
from heart import impute_categorical_data,impute_continuous_data, scale_data, encode_data

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data 

def preprocess_data(data):
    # Perform your preprocessing here
    data = impute_categorical_data(data)
    data = impute_continuous_data(data)
    data = scale_data(data)
    data = encode_data(data)
    return data

def impute_categorical_data(data):
    