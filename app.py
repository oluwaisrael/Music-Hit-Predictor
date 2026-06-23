import streamlit as st
import pandas as pd
import joblib
import numpy as np


@st.cache_resource
def load_models():
    logreg = joblib.load('logreg_afrobeats_model.pkl')
    scaler = joblib.load('afrobeats_scaler.pkl')
    return logreg, scaler

logreg, scaler = load_models()

st.set_page_config(page_title="Hit Predictor")
st.title("Afrobeats Hit Predictor")

st.markdown("""
### The "Hit Song Science" Reality Check
Can we predict an Afrobeats hit just by looking at its danceability or energy? 
We tested 1,000 real Spotify tracks. The truth? Audio features alone are weak predictors. 
Our cross-validated Logistic Regression model hits about **57.3% accuracy** — basically a coin toss. 

*Why?* Because vibes, marketing, TikTok trends, and artist reputation matter way more than the exact BPM. 

**Play with the sliders below and see if you can engineer the perfect track based purely on the data!**
""")

st.sidebar.header("Tweak the Audio Features")

duration_ms = st.sidebar.slider("Duration (ms)", 100000, 500000, 210000) 
explicit = st.sidebar.selectbox("Explicit Content?", [0, 1])
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.75)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.65)
key = st.sidebar.slider("Key (0-11)", 0, 11, 5)
loudness = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -5.0)
mode = st.sidebar.selectbox("Mode (0=Minor, 1=Major)", [0, 1])
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.10)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.20)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.00)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.15)
valence = st.sidebar.slider("Valence (Vibe/Happiness)", 0.0, 1.0, 0.70)
tempo = st.sidebar.slider("Tempo (BPM)", 50.0, 200.0, 105.0)
time_signature = st.sidebar.slider("Time Signature", 1, 5, 4)


if st.button("Predict Hit Potential", type="primary"):
    
    input_data = pd.DataFrame({
        'duration_ms': [duration_ms],
        'explicit': [explicit],
        'danceability': [danceability],
        'energy': [energy],
        'key': [key],
        'loudness': [loudness],
        'mode': [mode],
        'speechiness': [speechiness],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'liveness': [liveness],
        'valence': [valence],
        'tempo': [tempo],
        'time_signature': [time_signature]
    })


    scaled_input = scaler.transform(input_data)
    
    prediction = logreg.predict(scaled_input)[0]
    probabilities = logreg.predict_proba(scaled_input)[0]
    hit_prob = probabilities[1] * 100  

    st.divider()
    
    if prediction == 1:
        st.success(f"The model thinks it's a Hit! (Probability: {hit_prob:.1f}%)")
    else:
        st.error(f"Likely a Flop. (Hit Probability: {hit_prob:.1f}%)")
    

    st.progress(int(hit_prob))