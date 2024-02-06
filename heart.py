#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


heart = pd.read_csv('heart_disease_uci.csv')
heart.head()


# In[3]:


heart.info()


# In[4]:


print(heart.isnull().sum().sort_values(ascending=False))


# In[5]:


heart['age'].describe()


# In[6]:


sns.histplot(heart['age'],kde=True)
plt.axvline(heart['age'].mean(),color='blue')
plt.axvline(heart['age'].median(),color = 'yellow')
plt.axvline(heart['age'].mode()[0],color='violet')
print('Mean:',heart['age'].mean())
print('Median:', heart['age'].median())
print('Mode:', heart['age'].mode()[0])


# In[7]:


fig = px.histogram(data_frame=heart, x='age', color='sex')
fig.show()


# In[8]:


grouped_stats = heart.groupby('dataset')['age'].agg(['mean', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else None])


grouped_stats.columns = ['mean_age', 'median_age', 'mode_age']


print(grouped_stats.round())


# In[9]:


grouped_stats.plot(kind='bar', figsize=(10, 6))
plt.title('Mean, Median, and Mode Ages by Dataset')
plt.xlabel('Dataset')
plt.ylabel('Age')
plt.legend(loc='upper right')
plt.show()


# In[10]:


print(heart['sex'].value_counts())


# In[11]:


male = 726 
female = 194 
total = male + female 
male_percentage = (male/total) * 100
female_percentage = (female/total) * 100
print(f"female percentage is:{female_percentage:.2f}%")
print(f"male percentage is:{male_percentage:.2f}%")
difference_percentage = ((male - female) / female) * 100
print(f"Males are {difference_percentage:.2f}% more than females in the data.")


# In[12]:


(heart['dataset'].unique())


# In[13]:


heart['dataset'].value_counts()


# In[14]:


print(heart.groupby('sex')['dataset'].value_counts())


# In[15]:


fig = px.histogram(data_frame=heart, x='age', color='dataset')
fig.show()

# print the mean median and mode of age column grouped by dataset column
print(f"Mean of Data Set: {heart.groupby('dataset')['age'].mean()}")
print("-------------------------------------")
print(f"Median of Data Set: {heart.groupby('dataset')['age'].median()}")
print("-------------------------------------")
print(f"Mode of Data Set: {heart.groupby('dataset')['age'].agg(pd.Series.mode)}")
print("-------------------------------------")


# In[16]:


print(heart['cp'].unique())


# In[17]:


print(heart['cp'].value_counts())


# In[18]:


print(sns.countplot(heart, x='cp', hue='sex'))


# In[19]:


print(heart.groupby('cp')['sex'].value_counts())


# In[20]:


fig = px.histogram(data_frame=heart, x='age', color='cp')
fig.show()

# print the mean median and mode of age column grouped by dataset column
print(f"Mean of Data Set: {heart.groupby('cp')['age'].mean()}")
print("-------------------------------------")
print(f"Median of Data Set: {heart.groupby('cp')['age'].median()}")
print("-------------------------------------")
print(f"Mode of Data Set: {heart.groupby('cp')['age'].agg(pd.Series.mode)}")
print("-------------------------------------")


# In[21]:


heart['trestbps'].describe()


# In[22]:


print(sns.histplot(heart['trestbps'], kde=True))


# In[23]:


fig = px.histogram(data_frame=heart, x='trestbps', color='dataset')
fig.show()


# In[24]:


heart.groupby(heart['sex'])[['trestbps']].describe()


# In[25]:


sns.histplot(heart['chol'], kde=True)
plt.axvline(heart['chol'].mean(), color='red')
plt.axvline(heart['chol'].median(), color='green')
plt.axvline(heart['chol'].mode()[0], color='blue')


# In[26]:


heart.groupby(heart['sex'])[['chol']].describe()


# In[27]:


heart['fbs'].value_counts()


# In[28]:


print(heart.groupby('fbs')['sex'].value_counts())


# In[29]:


fig = px.histogram(data_frame=heart, y='age',x = 'sex', color='fbs')
fig.show()


# In[30]:


heart['restecg'].value_counts()


# In[31]:


print(heart.groupby('restecg')['sex'].value_counts())


# In[32]:


print(heart.groupby('restecg')['dataset'].value_counts())


# In[33]:


counts = heart.groupby(heart['restecg'])[['dataset','sex']].value_counts().unstack()
print(counts.plot.bar())


# In[34]:


counts = heart.groupby(heart['restecg'])[['dataset','sex']].value_counts().unstack()
print(counts)


# In[35]:


counts = heart.groupby(heart['restecg'])[['sex']].value_counts()
print(counts)


# In[36]:


print(sns.histplot(heart['thalch'],kde = True))


# In[37]:


sns.histplot(heart['thalch'], kde=True)
plt.axvline(heart['thalch'].mean(), color='red')
plt.axvline(heart['thalch'].median(), color='green')
plt.axvline(heart['thalch'].mode()[0], color='blue')


# In[38]:


grouped_stats = heart.groupby('dataset')['thalch'].agg(['mean', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else None])

# Renaming the columns for clarity
grouped_stats.columns = ['mean_thalch', 'median_thalch', 'mode_thalch']

# Displaying the observations
print(grouped_stats.round())


# In[39]:


heart['exang'].value_counts()


# In[40]:


heart.groupby(heart['exang'])[['sex']].value_counts()


# In[41]:


import plotly.express as px
import pandas as pd

# Assuming df is your DataFrame
grouped_counts = heart.groupby(['restecg', 'exang']).size().reset_index(name='count')

fig = px.bar(grouped_counts, x='exang', y='count', color='restecg',
             text='count', facet_col='restecg', facet_col_wrap=3,
             labels={'exang': 'Exang', 'count': 'Count'})

fig.update_layout(title='Exang Counts by Restecg',
                  xaxis_title='Exang',
                  yaxis_title='Count')

fig.show()


# In[42]:


sns.histplot(heart['oldpeak'], kde=True)
plt.axvline(heart['oldpeak'].mean(), color='red')
plt.axvline(heart['oldpeak'].median(), color='green')
plt.axvline(heart['oldpeak'].mode()[0], color='blue')


# In[43]:


heart['slope'].value_counts()


# In[44]:


heart.groupby(heart['slope'])['restecg'].value_counts()


# In[45]:


heart['ca'].value_counts()


# In[46]:


print(heart['thal'].unique())


# In[47]:


print(heart['thal'].value_counts())


# In[48]:


heart.groupby(heart['thal'])['sex'].value_counts()


# In[49]:


heart.groupby(heart['thal'])[['sex','dataset']].value_counts()


# In[50]:


heart.groupby(heart['thal'])['fbs'].value_counts()


# In[51]:


import plotly.express as px
import pandas as pd

# Assuming df is your DataFrame
grouped_counts = heart.groupby(['thal', 'sex', 'dataset']).size().reset_index(name='count')

fig = px.bar(grouped_counts, x='sex', y='count', color='dataset',
             facet_col='thal', facet_col_wrap=3,
             labels={'sex': 'Sex', 'count': 'Count', 'dataset': 'Dataset'},
             title='Counts by Thal and Sex')

fig.show()


# In[52]:


print(heart['num'].unique())


# In[53]:


heart['num'].value_counts()


# In[54]:


heart.groupby(heart['num'])[['sex']].value_counts()


# In[55]:


heart.groupby(heart['num'])[['dataset']].value_counts()


# In[56]:


heart.groupby(heart['num'])[['dataset','sex']].value_counts()


# In[57]:


import plotly.express as px
import pandas as pd

# Assuming df is your DataFrame
grouped_counts = heart.groupby(['num', 'sex', 'dataset']).size().reset_index(name='count')

fig = px.bar(grouped_counts, x='sex', y='count', color='dataset',
             facet_col='num', facet_col_wrap=3,
             labels={'sex': 'Sex', 'count': 'Count', 'dataset': 'Dataset'},
             title='Counts by num and Sex')

fig.show()


# In[58]:


heart.head()


# In[59]:


heart_corr = ['age','trestbps','chol','thalch','oldpeak']
plt.figure(figsize=(8,6))
sns.heatmap(heart[heart_corr].corr(),annot=True)
plt.title("Correlation Plot")
plt.show()


# In[60]:


categorical_variables = ['sex','fbs','restecg','exang','slope','ca','thal']
numerical_variables = ['age','trestbps','chol','thalch','oldpeak']
target_variable = 'num'

for cat_var in categorical_variables:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=cat_var, hue=target_variable, data=heart)
    plt.title(f"{cat_var} vs {target_variable}")
    plt.show()
    
for num_var in numerical_variables:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target_variable, y=num_var, data=heart)
    plt.title(f"{num_var} vs {target_variable}")
    plt.show()


# In[61]:


heart.head()


# In[62]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'num' is the target variable
target_variable = 'num'

# Independent variables (features)
independent_variables = ['age','sex','cp','fbs','restecg','trestbps', 'chol', 'thalch','exang','slope','ca','thal', 'oldpeak']

# Combine the target variable and independent variables
variables_to_plot = [target_variable] + independent_variables

# Select the relevant columns from the DataFrame
df_subset = heart[variables_to_plot]

# Calculate the correlation matrix
correlation_matrix = df_subset.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[63]:


print(heart.isnull().sum().sort_values(ascending=False))


# In[64]:


heart.isnull().sum()[heart.isnull().sum() > 0].sort_values(ascending=False)
missing_data_cols = heart.isnull().sum()[heart.isnull().sum() > 0].index.tolist()
missing_data_cols


# In[65]:


categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']
bool_cols = ['fbs','exang']
numeric_cols = ['oldpeak','thalch','chol','trestbps','age']


# In[66]:


def impute_categorical_data(passed_col):
    heart_null = heart[heart[passed_col].isnull()]
    heart_not_null = heart[heart[passed_col].notnull()]
    X = heart_not_null.drop(passed_col,axis=1)
    y = heart_not_null[passed_col]
    
    other_missing_cols =  [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
            
    if passed_col in bool_cols:
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
    print("The feature '"+ passed_col+ "' has been imputed with", round((accuracy * 100), 2), "accuracy\n")
    
    X = heart_null.drop(passed_col, axis=1)
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
        heart_null[passed_col] = random_forest.predict(X)
        # Map predicted boolean values back to True/False if the target variable is boolean
        if passed_col in bool_cols:
            heart_null[passed_col] = heart_null[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass
    
    heart_combined = pd.concat([heart_not_null, heart_null])
    return heart_combined[passed_col]

def impute_continuous_data(passed_col):
    heart_null = heart[heart[passed_col].isnull()]
    heart_not_null = heart[heart[passed_col].notnull()]
    X = heart_not_null.drop(passed_col,axis=1)
    y = heart_not_null[passed_col]
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
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
    X = heart_null.drop(passed_col, axis=1)
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
        heart_null[passed_col] = random_forest.predict(X)
    else:
        pass

    df_combined = pd.concat([heart_not_null, heart_null])
    
    return df_combined[passed_col]


# In[67]:


import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((heart[col].isnull().sum() / len(heart)) * 100, 2))+"%")
    if col in categorical_cols:
        heart[col] = impute_categorical_data(col)
    elif col in numeric_cols:
        heart[col] = impute_continuous_data(col)
    else:
        pass


# In[68]:


print(heart.isnull().sum())


# In[69]:


heart.head()


# In[70]:

def scale_data(data):
    columns_to_scale = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']
    minmax_scaler={}
    for col in columns_to_scale:
        minmax = MinMaxScaler()
        heart[col] = minmax.fit_transform(heart[[col]])
        minmax_scaler[col] = minmax
    return data, minmax_scaler


# In[71]:

def encode_data(data):
    encoded_columns = ['sex','dataset','cp','fbs','restecg','exang','slope','ca','thal']
    label_encoder = {}
    for col in encoded_columns:
        encode = LabelEncoder()
        heart[col] = encode.fit_transform(heart[[col]])
        label_encoder[col] = encode
    return data, label_encoder
        


# In[72]:


heart.head(8)


# In[73]:


from imblearn.over_sampling import SMOTE
X = heart.drop(columns=['num'])  # Features
y = heart['num']  # Target variable

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Print the shapes of the new splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[74]:


heart[heart['trestbps'] == 0]
# remove this row:
heart = heart.drop(heart[heart['trestbps'] == 0].index)


# In[75]:


print("So The number of row after removing 0 from the column trestbps are:)",heart.shape)


# In[76]:


import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
xgboost = xgb.XGBClassifier(random_state = 42)
xgboost.fit(X_train,y_train)
xgb_predict = xgboost.predict(X_test)
#xgb_auc = roc_auc_score(y_test, xgb_predict)
#fpr,tpr = roc_curve(y_test,xgb_predict)

accuracy = accuracy_score(y_test,xgb_predict)
print(f"XGBoost Accuracy: {accuracy:.2f}")
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_predict))


# In[77]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create and train a Random Forest classifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, random_forest_predictions)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Print classification report for detailed metrics
print("Random Forest Classification Report:")
print(classification_report(y_test, random_forest_predictions))


# In[78]:


X.columns


# In[79]:





# In[80]:


from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Identify numerical and categorical columns
columns_to_scale = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']
encoded_columns = ['thal', 'ca', 'dataset', 'slope', 'exang', 'restecg', 'fbs', 'cp', 'sex']

# Create transformers for scaling
numeric_transformer = ('scale', MinMaxScaler(), columns_to_scale)
#categorical_transformer = ('encode',LabelEncoder(),encoded_columns)

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[numeric_transformer],
    remainder='passthrough'
)

# Create the RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=42)

# Create the pipeline with SMOTE
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', random_forest_model)
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test set
accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline Accuracy: {accuracy:.2f}")
y_pred=pipeline.predict(X_test)
MAE = mean_absolute_error(y_test,y_pred)
print("MEAN_ABSOLUTE_ERROR:",MAE)



# In[81]:


from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(random_forest_model,X_balanced,y_balanced,cv=5)
print("Cross-validation Scores:",cv_score)
import numpy as np

mean_score = np.mean(cv_score)
std_score = np.std(cv_score)

print(f"Mean Cross-Validation Score: {mean_score:.2f}")
print(f"Standard Deviation of Cross-Validation Scores: {std_score:.2f}")


# In[82]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Plot learning curves
train_sizes, train_scores, test_scores = learning_curve(random_forest_model, X_balanced, y_balanced, cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Testing Score')
plt.legend()
plt.show()


# In[83]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# Assuming you have X_train, X_test, y_train, and y_test from your dataset

# Create and train a Random Forest classifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set and get predicted probabilities for each class
random_forest_probabilities = random_forest_model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
plt.figure(figsize=(8, 6))
for i in range(random_forest_probabilities.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test == i, random_forest_probabilities[:, i])
    roc_auc = roc_auc_score(y_test == i, random_forest_probabilities[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# Plot random guessing line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest (Multiclass)')
plt.legend(loc='lower right')
plt.show()



# In[84]:


cm = confusion_matrix(y_test, random_forest_predictions)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# In[85]:
heart,minmax_scaler = scale_data(heart)
heart,label_encoder = encode_data(heart)

for col, scaler in minmax_scaler.items():
    # Use inverse_transform to get the original values
    heart[col] = scaler.inverse_transform(heart[[col]])


# In[86]:


encoded_columns


# In[87]:


for col in encoded_columns:
    # Retrieve the corresponding LabelEncoder for the column
    encode = label_encoder[col]

    # Inverse transform the data
    heart[col] = encode.inverse_transform(heart[col])


# In[88]:


print(heart['sex'].unique())


# In[89]:


classification_dummy_data = {
    'id': [4],  # Assuming 'id' is a feature in your dataset
    'age': [45.0],
    'sex': ['Female'],
    'dataset': ['Cleveland'],
    'cp': ['atypical angina'],
    'trestbps': [130.0],
    'chol': [240.0],
    'fbs': [False],
    'restecg': ['normal'],
    'thalch': [145.0],
    'exang': [True],
    'oldpeak': [1.5],
    'slope': ['flat'],
    'ca': [1.0],
    'thal': ['reversable defect'],
}



# Convert the dictionary to a pandas DataFrame
new_X = pd.DataFrame(classification_dummy_data)

# Encode categorical or string features in the DataFrame
for col in new_X:
  if new_X[col].dtype == 'category' or new_X[col].dtype == 'object':
    new_X[col] = LabelEncoder().fit_transform(new_X[col])


# In[90]:


new_predictions = random_forest_model.predict(new_X)

# Display the predictions
print("Predictions:", new_predictions)


# In[91]:


import pickle 
pickle.dump(heart,open('heart.pkl','wb'))
pickle.dump(pipeline,open('pipe.pkl','wb'))


# In[92]:


heart.head(5)
