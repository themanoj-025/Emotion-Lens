import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define emotion categories
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_emotion_model(model_path='emotion_model.h5'):
    """Loads the pre-trained emotion detection model."""
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model (Ensure {model_path} is downloaded!): {e}")
        return None

def main():
    print("Initializing Emotion Detection via WebCam...")
    
    # Load the trained CNN model
    model = load_emotion_model()
    if model is None:
        return

    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the WebCam
    cap = cv2.VideoCapture(0)
    
    print("\nPress 'q' to quit the webcam view.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break
            
        # Convert frame to grayscale for the face detector and our CNN
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Crop the face region of interest (ROI)
            roi_gray = gray_frame[y:y+h, x:x+w]
            
            # Resize ROI to 48x48 (which is what our model expects)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Preprocess the image
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Make a prediction on the ROI
            prediction = model.predict(roi, verbose=0)[0]
            max_index = int(np.argmax(prediction))
            predicted_emotion = EMOTIONS[max_index]
            confidence = prediction[max_index]
            
            # Put the predicted emotion text on the bounding box
            label = f"{predicted_emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the video feed with the bounding box
        cv2.imshow('Real-time Emotion Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
