import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('model.pkl')

st.title("Titanic Survival Prediction App 🚢")

st.write("Enter passenger details below to predict survival:")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
Age = st.slider("Age", 0, 100, 30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 30.0)
Embarked_Q = st.selectbox("Embarked at Q? (0 = No, 1 = Yes)", [0, 1])
Embarked_S = st.selectbox("Embarked at S? (0 = No, 1 = Yes)", [0, 1])

# Predict button
if st.button("Predict"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'])

    # Make prediction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to SURVIVE!")
    else:
        st.error("❌ The passenger is likely to NOT survive.")
