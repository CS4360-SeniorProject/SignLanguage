                                            # Testing One Model Only
import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
import time
import mediapipe as mp
from keras.models import load_model

models_folder = os.path.join('C:\\Final Project\\SignLanguage\\Model Trainer\\Models')


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_selfie_segmentation = mp.solutions.selfie_segmentation # Segmentation masking

# potential TODOs: 
# -> add hand specific segmentation for better detections
# -> apply joint bilateral filter to results.segmentation_mask w/ image

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



def extract_keypoints(results):
   # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
   # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])



# Load the alphabet model
alphabet_model_filename = 'alphabet.h5'
alphabet_model_filepath = os.path.join(models_folder, alphabet_model_filename)
try:
    alphabet_model = load_model(alphabet_model_filepath)
    print(f"Alphabet model {alphabet_model_filename} loaded successfully.")
except Exception as e:
    print(f"Error loading alphabet model {alphabet_model_filename}: {str(e)}")

print(os.getcwd())

alphabets = (['A', 'B', 'C'])

                                            # Testing One Model Only
                                            # Testing One Model Only
# Define the prob_viz function
def prob_viz(prediction_labels, input_frame):
    output_frame = input_frame.copy()
    y_offset = 60
    for i, label in enumerate(prediction_labels):
        cv2.putText(output_frame, label, (20, y_offset + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame
sequence = []
sentence = []
predictions = []
threshold = 1

# Define the gesture recognition function
def gesture_recognition_function(frame):
    # Your gesture recognition code here
    # Process the frame and return the predicted label
    predicted_label = "A"  # Example label
    return predicted_label

cap = cv2.VideoCapture(0)

# Set mediapipe model 
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic, \
    mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:

    # Inside the loop
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Segment video background
        frame = mediapipe_segmentation(frame)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = alphabet_model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_label = alphabets[np.argmax(res)]
            predictions.append(predicted_label)
            
            if np.unique(predictions[-5:])[0] == predicted_label and res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if predicted_label != sentence[-1]:
                        sentence.append(predicted_label)
                else:
                    sentence.append(predicted_label)

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Call gesture recognition function
            predicted_label = gesture_recognition_function(frame)

            # Display the label text only
            image = prob_viz(predicted_label, image)

        # Show the image with the label text
        #cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
