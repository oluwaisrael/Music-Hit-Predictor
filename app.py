import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. LOAD THE SAVED MODEL & SCALER
# ==========================================
# @st.cache_resource ensures the app only loads these into RAM once when it boots up,
# rather than reloading them every single time you move a slider.
@st.cache_resource
def load_models():
    logreg = joblib.load('logreg_afrobeats_model.pkl')
    scaler = joblib.load('afrobeats_scaler.pkl')
    return logreg, scaler

logreg, scaler = load_models()

# ==========================================
# 2. APP HEADER & NARRATIVE
# ==========================================
st.set_page_config(page_title="Hit Predictor", page_icon="🎧")
st.title("🎧 Afrobeats Hit Predictor")

st.markdown("""
### The "Hit Song Science" Reality Check
Can we predict an Afrobeats hit just by looking at its danceability or energy? 
We tested 1,000 real Spotify tracks. The truth? Audio features alone are weak predictors. 
Our cross-validated Logistic Regression model hits about **57.3% accuracy** — basically a coin toss. 

*Why?* Because vibes, marketing, TikTok trends, and artist reputation matter way more than the exact BPM. 

**Play with the sliders below and see if you can engineer the perfect track based purely on the data!**
""")

# ==========================================
# 3. USER INPUTS (SIDEBAR)
# ==========================================
st.sidebar.header("Tweak the Audio Features")

# These inputs are mapped exactly to the Spotify API features
duration_ms = st.sidebar.slider("Duration (ms)", 100000, 500000, 210000) # Default ~3.5 mins
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

# ==========================================
# 4. PREDICTION ENGINE
# ==========================================
if st.button("Predict Hit Potential", type="primary"):
    
    # 1. Package inputs into a DataFrame. 
    # CRITICAL: The column order must exactly match the 'X' dataframe from your model.py
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

    # 2. Scale the data using the exact same math applied to the training data
    scaled_input = scaler.transform(input_data)
    
    # 3. Make prediction and grab the probability
    prediction = logreg.predict(scaled_input)[0]
    probabilities = logreg.predict_proba(scaled_input)[0]
    hit_prob = probabilities[1] * 100  # Probability of class 1 (Hit)

    # ==========================================
    # 5. DISPLAY RESULTS
    # ==========================================
    st.divider()
    
    if prediction == 1:
        st.success(f"🔥 The model thinks it's a Hit! (Probability: {hit_prob:.1f}%)")
    else:
        st.error(f"📉 Likely a Flop. (Hit Probability: {hit_prob:.1f}%)")
    
    # Show a visual progress bar for the probability
    st.progress(int(hit_prob))