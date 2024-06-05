from pathlib import Path
import streamlit as st
import pandas as pd
import pickle
import sys


# Loading up the Regression model we created
model_file = "/home/wladyka/Study/NUM_project/NUM/data/models/model.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Caching the model for faster loading
@st.cache_data

# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Mapping categorical values to numerical
    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

    cut = cut_map[cut]
    color = color_map[color]
    clarity = clarity_map[clarity]

    prediction = model.predict(pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], 
                                            columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']))
    return prediction

# Streamlit App
st.title('Diamond Price Predictor')
st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnRoc2F5bWkwaDZ5YWl4ZzF4M2dqbjJvdDB3djY5bWtudjM2N25naiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/46rXT61bNDxAc/giphy.webp", use_column_width=True, width=200)
st.header('Enter the characteristics of the diamond:')

with st.form("diamond_form"):
    st.subheader("Main Parameters:")
    carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0, step=0.01, help="The weight of the diamond in carats")
    cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], help="The quality of the diamond's cut")
    color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'], help="The color of the diamond (from J to D, where D is the clearest)")
    clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], help="The clarity of the diamond")

    st.subheader("Physical Parameters:")
    depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=60.0, step=0.1, help="The depth of the diamond as a percentage of its diameter")
    table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=55.0, step=0.1, help="The width of the diamond's table as a percentage of its diameter")
    x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=5.0, step=0.1, help="The length of the diamond in millimeters")
    y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=5.0, step=0.1, help="The width of the diamond in millimeters")
    z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=3.0, step=0.1, help="The height of the diamond in millimeters")

    submit = st.form_submit_button("Predict Price")

if submit:
    price = predict(carat, cut, color, clarity, depth, table, x, y, z)
    st.subheader(f'The predicted price of the diamond is ${price[0]:.2f} USD')
