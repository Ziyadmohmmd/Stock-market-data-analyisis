

import numpy as np
import pandas as pd
import streamlit as st
import pickle
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from datetime import timedelta, date
Standard = pickle.load(open("Standard","rb"))
LSTM = load_model('LSTM.h5')


def Prediction(x):
    test1 = Standard.transform(np.array(x).reshape(-1,1))
    test = test1.reshape(1,13,1)
    pred = LSTM.predict(test)
    predict = Standard.inverse_transform(pred)
    return predict

def main():
    st.title("Group 3 Stock Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-align:center;>Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    x1 = st.number_input('Enter 1st value:',411.35)
    x2 = st.number_input('Enter 2 value:',415.70)
    x3 = st.number_input('Enter 3 value:',419.00)
    x4 = st.number_input('Enter 4 value:',410.75)
    x5 = st.number_input('Enter 5 value:',412.54)
    x6 = st.number_input('Enter 6 value:',416.10)
    x7 = st.number_input('Enter 7 value:',419.15)
    x8 = st.number_input('Enter 8 value:',416.50)
    x9 = st.number_input('Enter 9 value:',412.04)
    x10 = st.number_input('Enter 10 value:',401.70)
    x11 = st.number_input('Enter 11 value:',409.23)
    x12 = st.number_input('Enter 12 value:',420.79)
    x13 = st.number_input('Enter 13 value:',424.25)
    x = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]
    result= ""
    if st.button("Predict"):
        result=Prediction(x)
    st.success('The predicted values is {}'.format(result))

if __name__ == '__main__':
    main()



