import pandas as pd
import numpy as np
import pickle
import streamlit as st

pickle_in = open('model.pkl', 'rb')
nb = pickle.load(pickle_in)

def welcome():
    return 'Selamat Datang'

def prediction(sepal_length, sepal_width, petal_length, petal_width):

    prediction = nb.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

def main():
    st.title("Aplikasi Prediksi Bunga Iris Algoritma Naive Bayes")
    st.markdown('Oleh : Agung Gunawan (2019230012) | Universitas Darma Persada | UAS Datamining')
    st.write('\n')
    st.markdown('Silakan isi form berikut terlebih dahulu :')
    
    st.write('\n')
    
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    
    sepal_length = st.text_input("Sepal Length", "0")
    sepal_width = st.text_input("Sepal Width", "0")
    petal_length = st.text_input("Petal Length", "0")
    petal_width = st.text_input("Petal Width", "0")
    result =""
    
    if st.button("PREDIKSI"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    st.success('Hasil Prediksi {}'.format(result))
    
if __name__=='__main__':
    main()
