# Laptop Price Predictor & Component Chat Bot
-------------------------------------------

## Overview
--------
This Streamlit web application predicts the price of a laptop based on its configuration and also provides information about various laptop components using a chatbot AI. It integrates Google's Gemini AI model for natural language processing.

The application includes the following features:
- Predicting laptop price based on configuration inputs such as brand, type, RAM, CPU, GPU, etc.
- Providing information about specific laptop components (RAM, CPU, GPU, screen, battery, storage) through a chatbot interface.

## Installation
------------
1. Clone the repository:

```git clone https://github.com/your_username/laptop-price-predictor.git```

Navigate to the project directory:

```cd laptop-price-predictor```


2. Install the required dependencies:

```pip install -r requirements.txt```


3. Download the pre-trained model files (pipe.pkl and df.pkl) and place them in the project directory.
4. Set up environment variables:
- Create a .env file in the project directory.
- Add your Google API key to the .env file:


```GOOGLE_API_KEY=your_api_key_here```



## Usage
-----
1. Run the Streamlit app:

```streamlit run app.py```



2. Once the app is running, you can:
- Input laptop configuration details in the sidebar to predict the laptop price.
- Use the chatbot interface to ask questions about specific laptop components.

## Technologies Used
-----------------
- Python
- Streamlit
- Google GenerativeAI (Gemini)
- scikit-learn
- NumPy
- pandas

## Contributors
------------
- Fasil Hameeed

## License
-------
This project is licensed under the MIT License.

## Working Demo
[Click here](https://huggingface.co/spaces/lisaf/Laptop-Price-Predictor-With-Gemini-AI-Bot) for a working demo.
