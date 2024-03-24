import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pickle

# Load the SARIMA model
with open('sarima_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

# Set a title
st.title('Real Estate Value Prediction')

# Set background image
bg_image = image.imread('appimage.jpg')
fig, ax = plt.subplots()
ax.imshow(bg_image)
ax.axis('off')  # Hide axes
st.pyplot(fig)

# Creating sliders for input
forecast_period = st.slider('Select forecast period (months):', min_value=1, max_value=24, value=12)

# Predicting future values
future_dates = model.make_future_dataframe(periods=forecast_period, freq='M')
forecast = model.predict(future_dates)

# Assuming forecast contains predicted median, and we simulate best & worst case
# For simplicity, let's assume a fixed percentage above and below the forecast as best and worst cases
best_case = forecast * 1.1  # 10% increase
worst_case = forecast * 0.9  # 10% decrease
median_case = forecast

# Creating a DataFrame for display
df_display = pd.DataFrame({
    'Date': future_dates[-forecast_period:]['ds'],
    'Best Case': best_case[-forecast_period:],
    'Median Case': median_case[-forecast_period:],
    'Worst Case': worst_case[-forecast_period:]
})

# Displaying the table
st.write('Forecasted Real Estate Values:', df_display)

# Ensure you have matplotlib installed for plotting
st.line_chart(df_display.set_index('Date'))

