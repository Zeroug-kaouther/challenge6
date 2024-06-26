import streamlit as st

import pandas as pd

import joblib

# Load the trained model

model = joblib.load("trained_model.pkl")

# Define function to make predictions

def predict(input_features):

    # Perform any necessary preprocessing on the input_features

    # Make predictions using the loaded model

    prediction = model.predict(input_features)

    return prediction

# Create the web interface

def main():

    st.title('Your Machine Learning Model Deployment')

    st.write('Enter the features below to get predictions:')
    # Create input fields for user to enter data
    feature1 = st.number_input("SepalLengthCm")

    feature2 = st.number_input("SepalWidthCm")
    feature3 = st.number_input("PetalLengthCm")
    feature4 = st.number_input("PetalWidthCm")


    # Add more input fields as needed

    # Combine input features into a DataFrame

    input_data = pd.DataFrame({"SepalLengthCm": [feature1], "SepalWidthCm": [feature2],"PetalLengthCm": [feature3],"PetalWidthCm": [feature4] })

    # Add more features as needed

    if st.button('Predict'):

        prediction = predict(input_data)

        st.write('Prediction:', prediction)

if __name__ == '__main__':
    
    main()