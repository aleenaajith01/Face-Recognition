# Face-Recognition

Face recognition using FaceNet and MTCNN involves a combination of face detection and face embedding extraction, followed by comparing the embeddings to recognize faces. 

####1. MTCNN (Multi-Task Cascaded Convolutional Networks) for Face Detection
MTCNN is used to detect faces in an image or video frame. It is a highly efficient face detector that not only locates the position of the face but also provides key facial landmarks like eyes, nose, and mouth. The process of face detection using MTCNN includes:

Face Localization: Detects the position of the face within an image, providing bounding box coordinates.
Landmark Detection: Identifies key facial points (such as eyes, nose, and mouth) for further alignment and preprocessing.
MTCNN is often used because of its accuracy and speed, making it suitable for real-time applications.

####2. FaceNet for Face Embedding
FaceNet is a deep learning model used to generate a numerical representation (embedding) of a face. The key concept in FaceNet is to map faces to a vector space where the distance between vectors corresponds to the similarity of the faces:

Face Embedding: FaceNet converts the detected face (image) into a fixed-size embedding, usually a 128-dimensional vector. These embeddings are unique for each face and can be compared directly to identify similarities.

Training Objective: FaceNet uses a triplet loss during training, where the model learns to minimize the distance between an anchor face and a positive match (same person), while maximizing the distance between the anchor face and a negative example (different person).
 ![image](https://github.com/user-attachments/assets/7d0e47e1-328c-4700-8fae-b39d38a1c45d)

