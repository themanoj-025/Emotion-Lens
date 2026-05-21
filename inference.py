import sys
import numpy as np
import cv2
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
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_emotion(model, image_path):
    """Predicts the emotion of a given image."""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image at {image_path}")
            return
            
        # Preprocess the image for the model
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # In a real scenario you would use a face detector (like Haar Cascades or MTCNN) 
        # to crop the face first. Here we assume the image is mostly the face.
        # Resize to match model input shape (48x48)
        resized_img = cv2.resize(gray_img, (48, 48))
        
        # Convert to array and rescale
        img_array = img_to_array(resized_img)
        img_array = img_array / 255.0
        
        # Expand dimensions to match batch format (1, 48, 48, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)[0]
        
        # Get the index of the highest probability
        max_index = np.argmax(predictions)
        predicted_emotion = EMOTIONS[max_index]
        confidence = predictions[max_index]
        
        print(f"\nResults for {image_path}:")
        print(f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2%})")
        
        print("\nAll Probabilities:")
        for i, emotion in enumerate(EMOTIONS):
            print(f" - {emotion}: {predictions[i]:.2%}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # Load model and predict
    model = load_emotion_model()
    predict_emotion(model, image_path)
