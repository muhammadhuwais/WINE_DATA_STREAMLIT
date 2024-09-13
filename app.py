import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open(r'model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function for prediction
def predict_wine_quality(input_data):
    """Predict wine quality based on user inputs."""
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  # Reshape input for prediction
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit app
def main():
    """Main function to run the Wine Quality Prediction App."""
    
    # Setting up the page layout
    st.set_page_config(page_title="Wine Quality Prediction App", layout="wide")
    
    # Title and description
    st.markdown("<h1 style='text-align: center; color: #8B0000;'>Wine Quality Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #808080;'>Fill in the wine details below to predict its quality</h3>", unsafe_allow_html=True)
    st.write("---")
    
    # Create columns for a clean layout
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
        citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)

    with col2:
        chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, step=1)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, step=1)
        density = st.number_input("Density", min_value=0.0, step=0.0001)

    with col3:
        pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.01)
        sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
        alcohol = st.number_input("Alcohol Percentage", min_value=0.0, step=0.1)
    
    # Spacing for better UI
    st.write(" ")
    
    # Predict button
    if st.button("Predict Wine Quality"):
        # Prepare input for prediction
        input_data = [
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH,
            sulphates, alcohol
        ]

        # Make prediction
        result = predict_wine_quality(input_data)

        # Display result
        if result == 1:
            st.success("🍷 Good Quality Wine!")
        else:
            st.error("⚠️ Bad Quality Wine.")
    
    # Footer
    st.write("---")
    st.markdown("<p style='text-align: center; color: #808080;'>Developed with ❤️ by [Uwais]</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
