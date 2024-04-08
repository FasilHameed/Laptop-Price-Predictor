from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai 
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Load environment variables
load_dotenv()

# Configure Google API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the Generative Model
model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Set page title and description
st.title("🚀 Laptop Price Predictor & Component Chat Bot 🤖")
st.markdown("### This app predicts the price of a laptop based on its configuration. ✨")

# Input fields for laptop price prediction
with st.sidebar.expander("Configuration", expanded=True):
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of the Laptop', min_value=0.0, max_value=10.0, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size', min_value=10.0, max_value=30.0, step=0.1)
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())
    os = st.selectbox('OS', df['os'].unique())
    predict_button = st.button('🔮 Predict Price 📈')

# Predict function
def predict_price(company, type, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)
    predicted_price_usd = int(np.exp(pipe.predict(query)[0]))  # Price in dollars
    return predicted_price_usd

# Display prediction
if predict_button:
    predicted_price = predict_price(company, type, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os)
    st.success(f"💰 The predicted price of this configuration is ₹{predicted_price}")

# Chat bot section
st.markdown("---")
st.subheader("💬 Component Chat Bot")

# Input textbox for chat bot
component_name = st.selectbox("Select a Component:", ["RAM", "CPU", "GPU", "Screen", "Battery", "Storage"])
submit_button = st.button("Ask")

# Handle question submission
if submit_button and component_name:
    response = get_gemini_response(component_name)
    st.write("**Bot:**", response)
