import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create dataset directory if it doesn't exist
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Get user name
user_name = input("Enter your name: ").strip()
user_dir = os.path.join(dataset_path, user_name)

# Create user directory if it doesn't exist
if not os.path.exists(user_dir):
    os.makedirs(user_dir)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("\nFace capture instructions:")
print("1. Make sure you're in good lighting")
print("2. Keep a neutral expression first")
print("3. Then make slight variations (smile, turn head slightly)")
print("4. Press 'q' when you've captured enough images (100-200 recommended)\n")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with more sensitive parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100) )  # Larger minimum size to ensure quality
    
    for (x, y, w, h) in faces:
        # Save the face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to consistent size for better training
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Save the image
        img_path = os.path.join(user_dir, f"{user_name}_{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        count += 1
        
        # Draw rectangle and show count
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Face Collection - Press Q to stop', frame)
    
    # Stop when 'q' is pressed or enough images collected
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCollected {count} face samples. Now training the model...")

# Prepare training data
x_train = []
y_labels = []
label_ids = {}
current_id = 0

# Walk through dataset directory
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            
            # Create ID for each new person
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]
            
            # Convert to grayscale and numpy array
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            
            # Detect faces in the saved images
            faces = face_cascade.detectMultiScale(
                image_array,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100))
            
            # Add to training data if face is detected
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
            else:
                print(f"No face detected in {path} - skipping")

# Check if we have enough training data
if len(x_train) < 1:
    print("\nERROR: Not enough valid face images for training!")
    print("Please try again with better lighting and more face variations.")
    exit()

# Train the recognizer
try:
    recognizer.train(x_train, np.array(y_labels))
    
    # Save the trained model
    recognizer.save("face-trainer.yml")
    
    # Save the label mappings
    with open("face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    
    print("\nTraining successful! Model saved to:")
    print(f"- face-trainer.yml (model)")
    print(f"- face-labels.pickle (labels)")
    print(f"\nTotal faces trained: {len(x_train)}")
    print(f"Number of people: {len(label_ids)}")
    
except Exception as e:
    print(f"\nError during training: {str(e)}")
    print("Possible causes:")
    print("- Not enough face samples (need at least 10-20 good ones)")
    print("- All images must contain detectable faces")
    print("- Try again with better lighting conditions")
