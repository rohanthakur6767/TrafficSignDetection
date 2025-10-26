import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pyttsx3
import time

# --------------------------
# 1️⃣ Load Model & Labels
# --------------------------
model = load_model('model/traffic_classifier_cnn.h5')

# Load label names
# If your CSV has headers ClassId,Name
labels = pd.read_csv('data/label_names.csv')
labels_dict = dict(zip(labels['ClassId'], labels['SignName']))

# --------------------------
# 2️⃣ Initialize Voice Engine
# --------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speaking speed

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --------------------------
# 3️⃣ Webcam Capture
# --------------------------
cap = cv2.VideoCapture(0)
last_spoken = ""
last_time = 0
cooldown = 2  # seconds between voice alerts

print("🚦 Traffic Sign Detection Started... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------------
    # 4️⃣ Preprocess Frame
    # --------------------------
    img = cv2.resize(frame, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # --------------------------
    # 5️⃣ Prediction
    # --------------------------
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    if confidence > 0.8:
        label = labels_dict.get(class_id, "Unknown")
        # Draw rectangle and text
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (20, 20), (620, 460), (0, 255, 0), 2)

        # --------------------------
        # 6️⃣ Voice Alert (with cooldown)
        # --------------------------
        current_time = time.time()
        if label != last_spoken or current_time - last_time > cooldown:
            speak(f"{label} detected")
            last_spoken = label
            last_time = current_time

    # --------------------------
    # 7️⃣ Display Frame
    # --------------------------
    cv2.imshow("AI Traffic Sign Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
