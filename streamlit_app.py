import streamlit as st
import pandas as pd
import joblib

# Load the trained logistic regression model
model = joblib.load('model.pkl')

# App title
st.title("🚢 Titanic Survival Prediction App")

st.markdown("Enter the passenger details below:")

# User inputs
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["female", "male"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
Parch = st.slider("Parents/Children Aboard", 0, 6, 0)
Fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
Embarked_Q = st.checkbox("Embarked at Queenstown (Q)?")
Embarked_S = st.checkbox("Embarked at Southampton (S)?")

# Encode Sex to 0/1 (same as your training)
Sex = 1 if Sex == "male" else 0

# Prepare input data as DataFrame with correct columns
input_data = pd.DataFrame([{
    'Pclass': Pclass,
    'Sex': Sex,
    'Age': Age,
    'SibSp': SibSp,
    'Parch': Parch,
    'Fare': Fare,
    'Embarked_Q': int(Embarked_Q),
    'Embarked_S': int(Embarked_S)
}])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = "🎉 Survived" if prediction == 1 else "❌ Did Not Survive"
    st.subheader(f"Prediction: {result}")
