import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo SVM entrenado
model = joblib.load('best_pipeline_svm.sav')

# Título de la aplicación
st.title("Predicción de Subida o Bajada del Precio de Solana")

# Descripción de la app
st.write("""
    Esta aplicación utiliza un modelo SVM para predecir si el precio de Solana subirá o bajará
    en función de indicadores técnicos.
""")

# Definir los campos de entrada para los indicadores técnicos
EMA_15 = st.number_input('Ingrese el valor de EMA_15')
EMA_20 = st.number_input('Ingrese el valor de EMA_20')
EMA_50 = st.number_input('Ingrese el valor de EMA_50')
RSI = st.number_input('Ingrese el valor del RSI')
MACD = st.number_input('Ingrese el valor del MACD')
MACD_Signal = st.number_input('Ingrese el valor del MACD_Signal')
SAR = st.number_input('Ingrese el valor del SAR')

# Crear el DataFrame con los valores ingresados
input_data = pd.DataFrame({
    'EMA_15': [EMA_15],
    'EMA_20': [EMA_20],
    'EMA_50': [EMA_50],
    'RSI': [RSI],
    'MACD': [MACD],
    'MACD_Signal': [MACD_Signal],
    'SAR': [SAR]
})

# Botón para realizar la predicción
if st.button('Predecir'):
    # Realizar la predicción con el modelo cargado
    prediction = model.predict(input_data)
    
    # Mostrar el resultado de la predicción
    if prediction[0] == 1:
        st.success('El modelo predice que el precio subirá en las próximas 5 horas.')
    else:
        st.warning('El modelo predice que el precio bajará en las próximas 5 horas.')

