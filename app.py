import streamlit as st
import numpy as np
import librosa
import joblib

st.title("🎙️ Emotion Detection")

model = joblib.load("model.pkl")

file = st.file_uploader("Upload WAV", type=["wav"])

if file:
    st.audio(file)

    audio, sr = librosa.load(file, sr=22050)

    audio = audio * 50
    audio = librosa.util.normalize(audio)

    # SAME 80 FEATURES
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.concatenate((
        np.mean(mfcc.T, axis=0),
        np.std(mfcc.T, axis=0)
    ))

    probs = model.predict_proba([mfcc])[0]
    prediction = model.classes_[np.argmax(probs)]

    st.success(f"Emotion: {prediction.upper()}")

    for e, p in zip(model.classes_, probs):
        st.write(f"{e}: {round(p*100,2)}%")