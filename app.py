import streamlit as st
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open("data/RidgeModel.pkl", 'rb'))
data = pd.read_csv('data/Cleaned_data.csv')

city_list = sorted(data['city'].unique())

st.set_page_config(page_title="Car Price Predictor", page_icon="🏡")
st.title("Home Price Predictor App")

st.write("This app predicts the price of a home you want to sell. Try filling the details below:")
city_name = st.selectbox("Select the City:", city_list)
bhk_name = st.text_input("How many BHK you want?:")
bath_name = st.text_input("How many Bathroom you want?:")
size_name = st.text_input("Area(sqft):")


def predict():
    prediction = pipe.predict(pd.DataFrame(columns=['bed', 'bath', 'city', 'house_size'],
                                           data=np.array([bhk_name, bath_name, city_name, size_name]).reshape(1, 4)))

    return str(np.round(prediction[0], 2))


if st.button("Predict Price"):
    predicted_price = predict()
    st.balloons()
    st.write("Our Prediction")
    st.success("Model is Predicting it's a: - 💲{}".format(predicted_price))
