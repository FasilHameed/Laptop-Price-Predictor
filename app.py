import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page title and description
st.title("🚀 Laptop Price Predictor 🎮")
st.markdown("### This app predicts the price of a laptop based on its configuration. ✨")

# Input fields
with st.sidebar.expander("Configuration", expanded=True):  # Set expanded=True to keep the sidebar open by default
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


# Footer HTML code
footer_with_image_light_blue = """
<style>
    .footer {
        padding: 20px;
        text-align: center;
        background-color: #f0f0f0;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
    }
    .line {
        border-top: 1px solid #ddd;
        margin: 10px 0;
    }
    .connect-text {
        font-size: 18px;
        margin-bottom: 10px;
    }
    .footer img {
        margin: 0 10px;
    }
    .powered-by {
        font-size: 14px;
        color: #888;
    }
</style>

<div class="footer">
    <div class="line"></div>
    <div class="connect-text">Connect with me at</div>
    <a href="https://github.com/FasilHameed" target="_blank"><img src="https://img.icons8.com/plasticine/30/000000/github.png" alt="GitHub"></a>
    <a href="https://www.linkedin.com/in/faisal--hameed/" target="_blank"><img src="https://img.icons8.com/plasticine/30/000000/linkedin.png" alt="LinkedIn"></a>
    <a href="tel:+917006862681"><img src="https://img.icons8.com/plasticine/30/000000/phone.png" alt="Phone"></a>
    <a href="mailto:faisalhameed763@gmail.com"><img src="https://img.icons8.com/plasticine/30/000000/gmail.png" alt="Gmail"></a>
    <div class="line"></div>
    <div class="powered-by">Powered By <img src="https://img.icons8.com/clouds/30/000000/gemini.png" alt="Gemini"> Gemini 💫 and Streamlit 🚀</div>
</div>
"""

# Render Footer
st.markdown(footer_with_image_light_blue, unsafe_allow_html=True)
