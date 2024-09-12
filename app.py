# from flask import Flask, render_template, Response, request, jsonify
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# # Initialize Mediapipe Holistic Model
# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# # Load the trained CNN model
# model = tf.keras.models.load_model('sign_language_cnn_model150.h5')

# # Load the LabelEncoder used during training
# label_encoder = LabelEncoder()
# label_encoder.classes_ = np.load('classes150.npy')

# # Define image size (same as during model training)
# IMG_SIZE = 64

# # Initialize camera
# cap = None

# def preprocess_image(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
#     image_normalized = image_resized / 255.0
#     image_expanded = np.expand_dims(image_normalized, axis=0)
#     return image_expanded

# def predict_sign(image):
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     predicted_class = np.argmax(prediction)
#     predicted_label = label_encoder.inverse_transform([predicted_class])[0]
#     return predicted_label

# def gen_frames():
#     global cap
#     if cap is None or not cap.isOpened():
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             raise RuntimeError("Cannot open camera")
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(frame_rgb)
#             if results.left_hand_landmarks or results.right_hand_landmarks:
#                 predicted_sign = predict_sign(frame)
#                 cv2.putText(frame, f"Sign: {predicted_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/current_prediction')
# def current_prediction():
#     global cap
#     if cap and cap.isOpened():
#         success, frame = cap.read()
#         if success:
#             predicted_sign = predict_sign(frame)
#             return jsonify({'prediction': predicted_sign})
#     return jsonify({'prediction': 'None'})

# @app.route('/start_camera', methods=['POST'])
# def start_camera():
#     global cap
#     if cap is None or not cap.isOpened():
#         cap = cv2.VideoCapture(0)
#         if cap.isOpened():
#             return jsonify({'message': 'Camera started'})
#         else:
#             return jsonify({'message': 'Cannot open camera'}), 500
#     return jsonify({'message': 'Camera already running'})

# @app.route('/stop_camera', methods=['POST'])
# def stop_camera():
#     global cap
#     if cap and cap.isOpened():
#         cap.release()
#         cap = None
#         return jsonify({'message': 'Camera stopped'})
#     return jsonify({'message': 'Camera is not running'})

# if __name__ == '__main__':
#     app.run(debug=True)









from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import time

app = Flask(__name__)

# Initialize Mediapipe Holistic Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load the trained CNN model
model = tf.keras.models.load_model('sign_language_cnn_model_word11150.h5')

# Load the LabelEncoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes11150.npy')

# Define image size (same as during model training)
IMG_SIZE = 64

# Initialize camera
cap = None

# Variables for sentence making
sentence = ""
last_sentence = ""
current_word = ""
start_time = None
SENTENCE_TIMEOUT = 0.75  # 0.75 seconds to consider a word continuous
running = False

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

def predict_sign(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

def gen_frames():
    global cap, current_word, start_time, sentence, last_sentence, running
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            predicted_sign = predict_sign(frame)
            current_time = time.time()

            if predicted_sign == current_word:
                if start_time and (current_time - start_time) >= SENTENCE_TIMEOUT:
                    if len(sentence) == 0 or sentence.split()[-1] != predicted_sign:
                        sentence += f"{predicted_sign} "
                    current_word = None
            else:
                current_word = predicted_sign
                start_time = current_time
            
            # Update the sentence
            sentence_display = sentence.strip()
            last_sentence_display = last_sentence

            # Update the GUI
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if sentence:
                last_sentence = sentence.strip()
                sentence = ""
            
            last_sentence_display = last_sentence
            sentence_display = sentence.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction():
    global last_sentence, sentence
    return jsonify({'prediction': current_word, 'sentence': sentence, 'last_sentence': last_sentence})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, running
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        running = True
        if cap.isOpened():
            return jsonify({'message': 'Camera started'})
        else:
            return jsonify({'message': 'Cannot open camera'}), 500
    return jsonify({'message': 'Camera already running'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, running
    if cap and cap.isOpened():
        cap.release()
        cap = None
        running = False
        return jsonify({'message': 'Camera stopped'})
    return jsonify({'message': 'Camera is not running'})

if __name__ == '__main__':
    app.run(debug=True)






