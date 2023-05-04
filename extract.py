'''import tensorflow as tf
import cv2
import dlib
import os
import numpy as np

# Load pre-trained dlib facial landmark detector
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Load pre-trained dlib face recognition model
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Load an example image
dir_path = "database"

# Create an empty list to store face images and corresponding labels
faces = []
labels = []
names = {}
id = 0

# Loop through the images in the directory
for subdir in os.listdir(dir_path):
    names[id] = subdir
    subjectpath = os.path.join(dir_path, subdir)

    # Create a new list to store the faces detected in this image
    detected_faces = []

    for img_name in os.listdir(subjectpath):
        path = os.path.join(subjectpath, img_name)
        label = id
        image = cv2.imread(path)
        image = cv2.resize(image, (96, 96))

        # Convert the image to grayscale
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        detector = dlib.get_frontal_face_detector()
        faces_rect = detector(image)

        # Loop through each detected face
        for rect in faces_rect:
            # Detect facial landmarks
            landmarks = predictor(image, rect)

            # Convert landmarks to numpy array
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Extract facial features
            face_descriptor = facerec.compute_face_descriptor(image, landmarks)

            # Append the face descriptor to the list of detected faces
            detected_faces.append(face_descriptor)
            print('.',end=' ')

        # Append the list of detected faces to the faces list
        if detected_faces:
            faces.append(detected_faces)
            labels.append(label)
    print()
    # Increment the label id
    if detected_faces:
        id += 1

# Determine the minimum number of face images among all persons
min_faces_per_person = min(len(person_faces) for person_faces in faces)

# Create new lists to store the faces and labels with the same number of rows
new_faces = []
new_labels = []

# Loop through the faces and labels
for person_faces, person_label in zip(faces, labels):
    # Take only the first min_faces_per_person face images
    new_faces.append(person_faces[:min_faces_per_person])
    new_labels.extend([person_label]*min_faces_per_person)

# Flatten the new_faces list and convert it to a numpy array
X = np.array([face_descriptor for person_faces in new_faces for face_descriptor in person_faces])
# Convert new_labels to a numpy array
y = np.array(new_labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Create a directory to store the numpy arrays
dir_path = "data"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Save X and y as numpy arrays in the directory
np.save(os.path.join(dir_path, 'X.npy'), X)
np.save(os.path.join(dir_path, 'y.npy'), y)

# Print the names and labels
for i, name in names.items():
    print("Name {}: Label {}".format(name, i))

# Print the number of faces and labels
print("Number of faces: {}".format(len(faces)))
print("Number of labels: {}".format(len(labels)))'''


import tensorflow as tf
import cv2
import dlib
import os
import numpy as np

# Load pre-trained dlib facial landmark detector
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Load pre-trained dlib face recognition model
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Load an example image
dir_path = "database"

# Create an empty list to store face images and corresponding labels
faces = []
labels = []
names = {}
id = 0

# Loop through the images in the directory
for subdir in os.listdir(dir_path):
    names[id] = subdir
    subjectpath = os.path.join(dir_path, subdir)

    # Create a new list to store the faces detected in this image
    detected_faces = []

    for img_name in os.listdir(subjectpath):
        path = os.path.join(subjectpath, img_name)
        label = id
        image = cv2.imread(path)
        image = cv2.resize(image, (128, 128))

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        detector = dlib.get_frontal_face_detector()
        faces_rect = detector(image)

        # Loop through each detected face
        for rect in faces_rect:
            # Detect facial landmarks
            landmarks = predictor(image, rect)

            # Convert landmarks to numpy array
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Extract facial features
            face_descriptor = facerec.compute_face_descriptor(image, landmarks)

            # Append the face descriptor to the list of detected faces
            detected_faces.append(face_descriptor)
            print('.',end=' ')

        # Append the list of detected faces to the faces list
        if detected_faces:
            faces.append(detected_faces)
            labels.append(label)
    print()
    # Increment the label id
    if detected_faces:
        id += 1

# Determine the minimum number of face images among all persons
min_faces_per_person = min(len(person_faces) for person_faces in faces)

# Create new lists to store the faces and labels with the same number of rows
new_faces = []
new_labels = []

# Loop through the faces and labels
for person_faces, person_label in zip(faces, labels):
    # Take only the first min_faces_per_person face images
    new_faces.append(person_faces[:min_faces_per_person])
    new_labels.extend([person_label]*min_faces_per_person)

# Flatten the new_faces list and convert it to a numpy array
X = np.array([face_descriptor for person_faces in new_faces for face_descriptor in person_faces])
# Convert new_labels to a numpy array
y = np.array(new_labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Create a directory to store the numpy arrays
dir_path = "data"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Save X and y as numpy arrays in the directory
np.save(os.path.join(dir_path, 'X.npy'), X)
np.save(os.path.join(dir_path, 'y.npy'), y)

# Print the names and labels
for i, name in names.items():
    print("Name {}: Label {}".format(name, i))

# Print the number of faces and labels
print("Number of faces: {}".format(len(faces)))
print("Number of labels: {}".format(len(labels)))
