import streamlit as st
import pandas as pd
import joblib
# Load the trained model
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
expectations = joblib.load('columns.pkl')


st.title("Heart Disease Prediction by indrajeet")

st.markdown("provide the following details to predict the likelihood of heart disease:")
    # Collect user input
age = st.slider("Age", 18,100,40)
sex = st.selectbox("Gender", options=["Male", "Female","Transgender"])
cp = st.selectbox("Chest Pain Type", ["ATA","NAP",'TA',"ASY"])
restbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
max_hr = st.slider("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])

    # Predict button
if st.button("Predict"):
    raw_input = {
        'age': age, 
        'sex': sex,
        'cp': cp,
        'restbps': restbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'max_hr': max_hr,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope
    }
    input_df = pd.DataFrame([raw_input])
    

    for col in expectations:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expectations]
    input_data = scaler.transform(input_df)
    predict_heart_disease = model.predict(input_data)[0]



    
    if predict_heart_disease == 1:
        st.error("The model predicts that you have heart disease.")
    else:
        st.success("The model predicts that you do not have heart disease.")
   
