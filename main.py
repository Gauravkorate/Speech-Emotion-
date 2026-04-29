import os
import numpy as np
import librosa
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
import joblib
import sys

# ==========================================
# PHASE 1: LOAD DATA AND TRAIN THE MODEL
# ==========================================
DATASET_PATH = "TESS Toronto emotional speech set data"

X = []
y = []

print("Loading dataset...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # ✅ correct label
            label = file.split("_")[2].lower()

            audio, sr = librosa.load(file_path, sr=22050)
            audio = librosa.util.normalize(audio)

            # ✅ 80 FEATURES (mean + std)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc = np.concatenate((
                np.mean(mfcc.T, axis=0),
                np.std(mfcc.T, axis=0)
            ))

            X.append(mfcc)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Emotion distribution:", Counter(y))

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")
print("-" * 40)

# ==========================================
# PHASE 2: INPUT AUDIO
# ==========================================

fs = 22050

if len(sys.argv) > 1:
    file_path = sys.argv[1]
    print("Testing file:", file_path)

    live_audio, sr = librosa.load(file_path, sr=22050)
    live_audio = librosa.util.normalize(live_audio)

else:
    print("Recording...")
    seconds = 3

    live_audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    live_audio = live_audio.flatten()

    # boost
    live_audio = live_audio * 50
    live_audio = librosa.util.normalize(live_audio)

print("Processing...")

# ✅ SAME FEATURE (80)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

mfcc = np.concatenate((
    np.mean(mfcc.T, axis=0),
    np.std(mfcc.T, axis=0)
))

features = [mfcc]

probs = model.predict_proba(features)[0]
prediction = model.classes_[np.argmax(probs)]

print("Detected Emotion:", prediction.upper())

for e, p in zip(model.classes_, probs):
    print(f"{e}: {round(p*100,2)}%")