import streamlit as st
import pickle
import numpy as np
import pandas as pd


# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi_val = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create a DataFrame with ONE row
    query_df = pd.DataFrame([[
        company,
        type,
        resolution,
        ram,
        weight,
        touchscreen_val,
        ips_val,
        ppi_val,
        cpu,
        hdd,
        ssd,
        gpu,
        os
    ]], columns=[
        'Company', 'TypeName', 'ScreenResolution', 'Ram', 'Weight',
        'Touchscreen', 'IPS', 'ppi', 'Cpu Brand', 'HDD', 'SSD', 'Gpu Brand', 'os'
    ])

    pred = pipe.predict(query_df)[0]
    st.title("The predicted price of this configuration is " + str(int(np.exp(pred))))
