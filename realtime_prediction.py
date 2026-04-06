import cv2
import numpy as np
from tensorflow.keras.models import load_model

def main():
    # Load pre-trained model
    try:
        print("Loading pre-trained model...")
        model = load_model('models/gender_age_model.h5', compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error: Could not load the model. Please train the model and generate it inside models/.")
        print(f"Exception message: {e}")
        return

    # Load OpenCV's Haar Cascade for front face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start getting video input from default camera
    cap = cv2.VideoCapture(0)

    print("Starting real-time webcam prediction. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            # Extract the actual face bounding box from the frame
            face_img = frame[y:y+h, x:x+w]
            
            # Preprocess the face for the model inputs
            face_img = cv2.resize(face_img, (64, 64))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = face_img.astype(np.float32) / 255.0
            face_img = np.expand_dims(face_img, axis=0) # Add batch dimension (1, 64, 64, 3)
            
            # Predict age and gender
            predictions = model.predict(face_img, verbose=0)
            gender_pred = predictions[0][0][0] # Sigmoid output (0 to 1)
            age_pred_probs = predictions[1][0] # Array of 5 probabilities
            
            # Interpret the predictions
            gender_label = "Female" if gender_pred > 0.5 else "Male"
            
            # Decode Age Group
            age_groups = ["Child (0-12)", "Teen (13-19)", "Young Adult (20-35)", "Adult (36-55)", "Senior (56+)"]
            age_class = np.argmax(age_pred_probs)
            age_label = age_groups[age_class]
            
            # Draw bounding box and display text around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{gender_label}, {age_label}"
            
            # Add background rectangle for better text visibility
            cv2.rectangle(frame, (x, y-30), (x+w+40, y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Show the output frame with overlays
        cv2.imshow("Gender and Age Prediction Live", frame)
        
        # Exit the loop execution when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup on exit
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
