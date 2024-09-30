import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

# Load the MTCNN model for face detection
detector = MTCNN()

# Load the FaceNet model as a library
embedder = FaceNet()

# Function to detect and extract face from image
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

# Register person 1
img = cv2.imread('aleena.jpg')
faces_data = extract_faces(img)
if faces_data:
    for face_img, _ in faces_data:
        embedding = get_face_embedding(face_img)
        embeddings_dict['Aleena'] = embedding

# Register person 2
img = cv2.imread('ponnu.jpeg')
faces_data = extract_faces(img)
if faces_data:
    for face_img, _ in faces_data:
        embedding = get_face_embedding(face_img)
        embeddings_dict['Aleeshya'] = embedding

img = cv2.imread('amma.jpeg')
faces_data = extract_faces(img)
if faces_data:
    for face_img, _ in faces_data:
        embedding = get_face_embedding(face_img)
        embeddings_dict['Shija'] = embedding        

img = cv2.imread('achan.jpeg')
faces_data = extract_faces(img)
if faces_data:
    for face_img, _ in faces_data:
        embedding = get_face_embedding(face_img)
        embeddings_dict['Ajith'] = embedding        

# Save embeddings of multiple persons to a file
save_embeddings(embeddings_dict, 'multiple_face_embeddings')

# Step 2: Load saved embeddings
saved_embeddings = load_embeddings('multiple_face_embeddings.npy')

# Step 3: Detect faces from a new input image and get their embeddings
new_img = cv2.imread('family.jpeg')
faces_data = extract_faces(new_img)
if faces_data:
    for face_img, bbox in faces_data:
        new_face_embedding = get_face_embedding(face_img)
        
        # Step 4: Compare the new face embedding with all saved embeddings
        match, best_match_name, distance = recognize_face(new_face_embedding, saved_embeddings)
        
        # Display the result on the new image
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255)  # Red for not recognized
        if match:
            color = (0, 255, 0)  # Green for recognized

        # Draw the bounding box on the face
        cv2.rectangle(new_img, (x1, y1), (x2, y2), color, 2)
        
        # Add the label with the name just above the bounding box
        cv2.putText(new_img, best_match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Step 5: Save the final image with the result to a file
output_image_path = 'recognized_face_result.jpg'
cv2.imwrite(output_image_path, new_img)
print(f"Result saved as {output_image_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
