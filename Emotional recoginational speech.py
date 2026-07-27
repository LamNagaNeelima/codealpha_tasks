import os
import numpy as np
from scipy.io import wavfile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

DATASET_PATH = "dataset"

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    print("Dataset folder created. Add .wav files and run again.")
    exit()

def extract_features(file_path):
    sample_rate, audio = wavfile.read(file_path)
    audio = audio.astype(float)

    mean = np.mean(audio)
    std = np.std(audio)
    energy = np.sum(audio ** 2) / len(audio)
    zero_crossings = np.mean(np.abs(np.diff(np.sign(audio))))

    return np.array([mean, std, energy, zero_crossings])

X = []
y = []

files = os.listdir(DATASET_PATH)

if len(files) == 0:
    print("No audio files found in dataset folder.")
    exit()

for file in files:
    if file.endswith(".wav"):
        path = os.path.join(DATASET_PATH, file)

        features = extract_features(path)
        X.append(features)

        label = file.split("_")[0]
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)

print("Training...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")