import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from sklearn.svm import SVC
from keras.models import load_model

best_model = load_model('models/best_model.h5')

dir_path = os.path.abspath('data')

# Load the best trained model
#best_model = tf.keras.models.load_model('best_model.h5')

# Load the test data
X_test = np.load(os.path.join(dir_path, 'X.npy'), allow_pickle=True)
y_test = np.load(os.path.join(dir_path, 'y.npy'), allow_pickle=True)
X_train = np.load(os.path.join(dir_path, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(dir_path, 'y_train.npy'), allow_pickle=True)

# Convert data into compatible format
X_test = np.array([np.array(x) for x in X_test])
y_test = np.array([int(y) for y in y_test])

svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

y_test_categorical = to_categorical(y_test)
# Evaluate the model on the test data
test_loss, test_acc = best_model.evaluate(X_test,y_test, batch_size=32)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Predict on test data
y_pred = best_model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert predicted probabilities to binary labels (0 or 1) using a threshold of 0.5

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute classification report
target_names = ['Not Face', 'Face']  # Specify target class names
cr = classification_report(y_test, y_pred, target_names=target_names,labels=[0, 1, 2])
print("Classification Report:")
print(cr)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

def predict(model, x):
    y_pred = model.predict(x)
    y_pred_binary = (y_pred > 0.5).astype(int)
    return np.hstack([1 - y_pred, y_pred])

y_pred = predict(best_model, X_test)
fpr, tpr, thresholds= roc_curve(y_test_categorical[:, 1], y_pred[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.show()


# Testing the SVM model
svm_predictions = svm_classifier.predict(X_test)

# Evaluating the SVM model's performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1_score = f1_score(y_test, svm_predictions, average='weighted')
print('svm_accuracy= '+ str(svm_accuracy))
print('svm_precision= '+ str(svm_precision))
print('svm_recall= '+str(svm_recall))
print('svm_f1_score= '+str(svm_f1_score))
