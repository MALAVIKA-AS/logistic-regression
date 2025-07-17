import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("Titanic Survival Prediction")

# User inputs
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "Q", "C"])

# Encode inputs
Sex = 1 if Sex == "male" else 0
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

# Final input order based on training
input_data = pd.DataFrame([[
    Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S
]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 Survived")
    else:
        st.error("❌ Did not survive")
