import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

# Load the MTCNN model for face detection
detector = MTCNN()

# Load the FaceNet model as a library
embedder = FaceNet()

# Function to detect and extract faces from image/frame
def extract_faces(image, required_size=(160, 160)):
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        print("No face detected.")
        return []
    
    face_data = []
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_img = image[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, required_size)
        face_data.append((face_img, (x1, y1, x2, y2)))
    
    return face_data

# Function to get face embeddings from the FaceNet model
def get_face_embedding(face):
    face = face.astype('float32')
    embedding = embedder.embeddings([face])[0]
    return embedding

# Function to save multiple embeddings to a file
def save_embeddings(embeddings_dict, filename):
    np.save(filename, embeddings_dict)
    print(f"Embeddings saved to {filename}.npy")

# Function to load multiple embeddings from a file
def load_embeddings(filename):
    return np.load(filename, allow_pickle=True).item()

# Function to recognize face by comparing embeddings
def recognize_face(new_face_embedding, saved_embeddings, threshold=0.8):
    min_distance = float('inf')
    best_match = None

    for name, embedding in saved_embeddings.items():
        distance = norm(new_face_embedding - embedding)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = name

    if best_match:
        return True, best_match, min_distance
    else:
        return False, "Unknown", min_distance

# Step 1: Register multiple persons' face embeddings
embeddings_dict = {}

# Register person 1 (Aleena)
img = cv2.imread('aleena.jpg')
faces = extract_faces(img)
if faces:
    face, _ = faces[0]  # Take the first face
    embedding = get_face_embedding(face)
    embeddings_dict['AleenaAjith'] = embedding

# Register person 2 (Salman)
img = cv2.imread('image.jpeg')
faces = extract_faces(img)
if faces:
    face, _ = faces[0]  # Take the first face
    embedding = get_face_embedding(face)
    embeddings_dict['SALMAN'] = embedding

# Save embeddings of multiple persons to a file
save_embeddings(embeddings_dict, 'multiple_face_embeddings')

# Step 2: Load saved embeddings (from the image step)
saved_embeddings = load_embeddings('multiple_face_embeddings.npy')

# Step 3: Open the video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output video
output_video_path = 'recognized_faces_video.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Step 4: Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Detect faces in the current frame
    faces = extract_faces(frame)
    
    for face, bbox in faces:
        if face is not None:
            new_face_embedding = get_face_embedding(face)
            
            # Compare the face embedding with all saved embeddings
            match, best_match_name, distance = recognize_face(new_face_embedding, saved_embeddings)
            
            # Draw the result on the frame
            x1, y1, x2, y2 = bbox
            color = (0, 0, 255)  # Red for not recognized
            if match:
                result_text = f"Face recognized"
                color = (0, 255, 0)  # Green for recognized
            else:
                result_text = f"Face not recognized."

            # Draw the bounding box on the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add the label with the name just above the bounding box
            cv2.putText(frame, best_match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Add the result text on the frame (optional)
            cv2.putText(frame, result_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Step 5: Save the frame with the result to the output video
    out.write(frame)

    # Display the frame with the result (optional)
    cv2.imshow('Face Recognition Result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to stop the video processing

# Step 6: Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video_path}")
