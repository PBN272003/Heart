# import streamlit as st 
# import pickle 
# import numpy as np 
# import pandas as pd 
# from sklearn.preprocessing import LabelEncoder
# pipe = pickle.load(open('pipe.pkl','rb'))
# heartt = pickle.load(open('heart.pkl','rb'))
# st.title("HEART DISEASE PREDICTOR")
# st.sidebar.header("User Input Features")
# id = st.number_input('Id of the Patient')
# age = st.number_input('Age of the Patient')
# sex = st.selectbox('Sex',['Male','Female'])
# dataset = st.selectbox('Dataset',heartt['dataset'].unique())
# cp = st.selectbox('Chest Pain Type',heartt['cp'].unique())
# trestbps = st.number_input('Enter the value of RestingBlood Pressure')
# chol = st.number_input("Enter the Cholestrol value")
# fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (FBS)',['True','False'])
# restecg = st.selectbox('Resting Electrocardiographic Results',heartt['restecg'].unique())
# thalch = st.number_input('Enter the value of Thalch(Max Heart Rate)')
# exang = st.selectbox('Exercise Induced Angina (Exang)',['True','False'])
# oldpeak = st.number_input('Enter the value of Oldpeak')
# slope = st.selectbox('Slope of the Peak Excercise ST Segment',heartt['slope'].unique())
# ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (CA)',heartt['ca'].unique())
# thal = st.selectbox('Thalassemia',heartt['thal'].unique())
# if st.button('Predict Level'):
#      user_input = pd.DataFrame({
#     'id': [id],
#     'age': [age],
#     'sex': [sex],
#     'dataset': [dataset],
#     'cp': [cp],
#     'trestbps': [trestbps],
#     'chol': [chol],
#     'fbs': [fbs],
#     'restecg': [restecg],
#     'thalch': [thalch],
#     'exang': [exang],
#     'oldpeak': [oldpeak],
#     'slope': [slope],
#     'ca': [ca],
#     'thal': [thal]
#      })

     
#      for col in user_input:
#         if user_input[col].dtype == 'category' or user_input[col].dtype == 'object':
#             encoder = LabelEncoder()
#             user_input[col] = encoder.fit_transform(user_input[col])

#     # Make predictions using the loaded pipeline
#      prediction = pipe.predict(user_input)
#      disease_levels = {
#         0: 'No Heart Disease',
#         1: 'Mild Heart Disease',
#         2: 'Moderate Heart Disease',
#         3: 'Severe Heart Disease',
#         4: 'Critical Heart Disease'
#     }
    
#      predicted_level = disease_levels[prediction[0]]
    
#      st.title(f"The predicted Heart Disease Level is {prediction[0]} ({predicted_level})")
import streamlit as st 
import pickle 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from PIL import Image
# Load the trained model and additional data
pipe = pickle.load(open('pipe.pkl','rb'))
heartt = pickle.load(open('heart.pkl','rb'))
heart_image = Image.open('cadio.png')
# Streamlit app layout
st.title("HEART DISEASE PREDICTOR")

st.sidebar.header("User Input Features")
st.image(heart_image,use_column_width=True,width=1000)

# User input form
with st.sidebar.form("user_input_form"):
    st.write("### Enter Patient Details:")
    id = st.number_input('ID of the Patient', min_value=1)
    age = st.number_input('Age of the Patient', min_value=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    dataset = st.selectbox('Dataset', heartt['dataset'].unique())
    chest_pain_type = st.selectbox('Chest Pain Type', heartt['cp'].unique())
    resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=1)
    cholesterol = st.number_input('Cholesterol', min_value=1)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (FBS)', ['True', 'False'])
    resting_ecg = st.selectbox('Resting Electrocardiographic Results', heartt['restecg'].unique())
    max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=1)
    exercise_angina = st.selectbox('Exercise Induced Angina (Exang)', ['True', 'False'])
    st_depression = st.number_input('ST Depression induced by exercise relative to rest', min_value=0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', heartt['slope'].unique())
    num_major_vessels = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (CA)', heartt['ca'].unique())
    thalassemia = st.selectbox('Thalassemia', heartt['thal'].unique())
    
    # Submit button
    submitted = st.form_submit_button("Predict Level")

# Display prediction result
if submitted:
    # Create a DataFrame with user input features
    user_input = pd.DataFrame({
        'id': [id],
        'age': [age],
        'sex': [sex],
        'dataset': [dataset],
        'cp': [chest_pain_type],
        'trestbps': [resting_blood_pressure],
        'chol': [cholesterol],
        'fbs': [fbs],
        'restecg': [resting_ecg],
        'thalch': [max_heart_rate],
        'exang': [exercise_angina],
        'oldpeak': [st_depression],
        'slope': [slope],
        'ca': [num_major_vessels],
        'thal': [thalassemia]
    })

    # Encode categorical or string features in the DataFrame
    for col in user_input:
        if user_input[col].dtype == 'category' or user_input[col].dtype == 'object':
            encoder = LabelEncoder()
            user_input[col] = encoder.fit_transform(user_input[col])

    # Make predictions using the loaded pipeline
    prediction = pipe.predict(user_input)
    
    # Map numeric prediction to corresponding levels
    disease_levels = {
        0: 'No Heart Disease',
        1: 'Mild Heart Disease',
        2: 'Moderate Heart Disease',
        3: 'Severe Heart Disease',
        4: 'Critical Heart Disease'
    }
    
    predicted_level = disease_levels[prediction[0]]
    
    # Display prediction result with improved UI
    st.write("### Prediction Result:")
    st.success(f"The predicted Heart Disease Level is {predicted_level}")

     


