
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import os

dir_path = os.path.abspath('data')

# Load the extracted facial features and labels from the previous program
# Load X and y from the saved numpy arrays
X = np.load(os.path.join(dir_path, 'X.npy'), allow_pickle=True)
y = np.load(os.path.join(dir_path, 'y.npy'), allow_pickle=True)
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# Check for and convert any list elements in X_train and y_train
for i in range(len(X_train)):
    if type(X_train[i]) == list:
        try:
            X_train[i] = np.array(X_train[i]).astype('float32')
        except Exception as e:
            print(f"Error converting X_train element {i} to numpy array: {e}")

for i in range(len(y_train)):
    if type(y_train[i]) == list:
        try:
            y_train[i] = np.array(y_train[i]).astype('float32')
        except Exception as e:
            print(f"Error converting y_train element {i} to numpy array: {e}")

# Define the model architecture
def create_face_recognition_model(hidden_layer_sizes=(128,64,32,16,8,4,2)):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape=(128,)))  # Input layer with 128-dimensional facial features
    for layer_size in hidden_layer_sizes[1:]:
        model.add(Dense(layer_size, activation='relu'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the face recognition model
face_recognition_model = KerasClassifier(build_fn=create_face_recognition_model)

# Define the hyperparameters to tune with GridSearchCV
param_grid = {'hidden_layer_sizes': [(64,32,16), (128,64,32,16,8), (64,64,64,64), (16,16,16,16,16,16,16)],
              'batch_size': [32, 64, 128],
              'epochs': [5, 10, 15]}

# Define the GridSearchCV object with the face recognition model, hyperparameters, and cross-validation
grid_search = GridSearchCV(face_recognition_model, param_grid, cv=3, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and validation score found by GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)
print("Best validation score:", grid_search.best_score_)

# Retrieve the best model
best_model = grid_search.best_estimator_.model

# Save the best model
model_dir = os.path.abspath('models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
best_model.save(os.path.join(model_dir, 'best_model.h5'))

X_test = np.load(os.path.join(dir_path, 'X.npy'), allow_pickle=True)
y_test = np.load(os.path.join(dir_path, 'y.npy'), allow_pickle=True)

# Save the test data
np.save(os.path.join(dir_path, 'X_test.npy'), X_test)
np.save(os.path.join(dir_path, 'y_test.npy'), y_test)
