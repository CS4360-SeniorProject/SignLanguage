from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import mediapipe as mp
from keras.models import load_model

app = Flask(__name__)


models_folder = os.path.join('C:\\Final Project\\SignLanguage\\Model Trainer\\Models')


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_selfie_segmentation = mp.solutions.selfie_segmentation # Segmentation masking

# potential TODOs: 
# -> add hand specific segmentation for better detections
# -> apply joint bilateral filter to results.segmentation_mask w/ image

def draw_styled_landmarks(image, results):
     
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def mediapipe_segmentation(image):
    bg_image = None                                             # Can set color or image as bg if desired
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                               # Image is no longer writeable
    results = selfie_segmentation.process(image)                # Apply segmentation mask
    image.flags.writeable = True                                # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)              # COLOR COVERSION RGB 2 BGR
    
    # referenced nicolai nielsen segmentation tutorial #
    # Draw segmentation on background of video
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1 #was 0.15
    
    # Filter background
    # Can apply an image or flat color instead of blur, but would need implimentation atm
    if bg_image is None:
        bg_image = cv2.GaussianBlur(image, (55,55),0)

    output_image = np.where(condition, image, bg_image)
    return output_image

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results




# Load the alphabet model
alphabet_model_filename = 'alphabet.h5'
alphabet_model_filepath = os.path.join(models_folder, alphabet_model_filename)

try:
    alphabet_model = load_model(alphabet_model_filepath)
    print(f"Alphabet model {alphabet_model_filename} loaded successfully.")
except Exception as e:
    print(f"Error loading alphabet model {alphabet_model_filename}: {str(e)}")


alphabets = (['A', 'B', 'C'])

                                            # Testing One Model Only
# Define the prob_viz function
def prob_viz(prediction_labels, input_frame):
    output_frame = input_frame.copy()
    y_offset = 60
    for i, label in enumerate(prediction_labels):
        cv2.putText(output_frame, label, (20, y_offset + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    return output_frame

sequence = []
sentence = []
predictions = []
threshold = 0.5  # Example threshold, adjust based on your model's performance

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Mock-up of a function for making predictions with your model

def make_prediction(sequence):
    # Ensure the sequence is the correct shape for your model
    sequence = np.array(sequence).reshape(1, 30, 126)  # Reshape to (1, 30, 126)

    # Make prediction
    prediction = alphabet_model.predict(sequence)
    
    # Convert the prediction to a label
    predicted_label_index = np.argmax(prediction)
    predicted_label = alphabets[predicted_label_index]
    
    return predicted_label


def update_sentence(predictions, predicted_label):
    # Mock updating sentence for demonstration
    if len(predictions) < 5:
        predictions.append(predicted_label)
    else:
        predictions.pop(0)
        predictions.append(predicted_label)
    # For simplicity, just return the most frequent label in predictions
    return max(set(predictions), key=predictions.count)



def gen_frames():
    global sequence, sentence, predictions  # Ensure these variables are accessible
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Draw landmarks with custom styling
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keeps last 30
            
            if len(sequence) == 30:
                predicted_label = make_prediction(sequence)
                sentence = update_sentence(predictions, predicted_label)
                
                # Use prob_viz function to overlay the predicted label on the frame
                image = prob_viz([sentence], image)  # Assuming prob_viz expects a list of labels
            
            # Convert back to BGR for encoding
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue  # Skip this frame if encoding failed
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
