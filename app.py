from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

# ✅ Load model once at startup (not inside a route)
MODEL_PATH = 'model/traffic_classifier_cnn.h5'
model = load_model(MODEL_PATH)

# ✅ Define class labels
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# ✅ Only use webcam locally — disable for Render (no physical camera)
USE_CAMERA = os.environ.get("USE_CAMERA", "False").lower() == "true"

if USE_CAMERA:
    camera = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                img = cv2.resize(frame, (32, 32))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0) / 255.0

                preds = model.predict(img)
                pred_class = np.argmax(preds)
                label = classes.get(pred_class, "Unknown")

                cv2.putText(frame, label, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if USE_CAMERA:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not available in this environment"


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return "No file uploaded"

    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(32, 32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    preds = model.predict(img)
    pred_class = np.argmax(preds)
    result = classes.get(pred_class, "Unknown Sign")

    return render_template('result.html', prediction=result, img_path=file_path)


if __name__ == '__main__':
    # ✅ Set host and port explicitly for Render
    app.run(host='0.0.0.0', port=10000, debug=False)
