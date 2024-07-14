import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load test data
x_test_final = pd.read_csv('x_test_final.csv')  # Adjust path if necessary
y_test = pd.read_csv('y_test.csv')  # Assuming you saved y_test similarly

# Load saved models
lr = pickle.load(open('lr_liver_analysis.pkl', 'rb'))
svm = pickle.load(open('svm_liver_analysis.pkl', 'rb'))
rfc = pickle.load(open('rfc_liver_analysis.pkl', 'rb'))
knn = pickle.load(open('knn_liver_analysis.pkl', 'rb'))

# Predict using loaded models
y_pred_lr = lr.predict(x_test_final)
y_pred_svm = svm.predict(x_test_final)
y_pred_rfc = rfc.predict(x_test_final)
y_pred_knn = knn.predict(x_test_final)

# Evaluate models
lr_acc = accuracy_score(y_test, y_pred_lr)
svm_acc = accuracy_score(y_test, y_pred_svm)
rfc_acc = accuracy_score(y_test, y_pred_rfc)
knn_acc = accuracy_score(y_test, y_pred_knn)

lr_cm = confusion_matrix(y_test, y_pred_lr)
svm_cm = confusion_matrix(y_test, y_pred_svm)
rfc_cm = confusion_matrix(y_test, y_pred_rfc)
knn_cm = confusion_matrix(y_test, y_pred_knn)

# Print or visualize evaluation results as needed
print("Logistic Regression Accuracy:", lr_acc)
print("SVM Accuracy:", svm_acc)
print("Random Forest Accuracy:", rfc_acc)
print("KNN Accuracy:", knn_acc)

print("Logistic Regression Confusion Matrix:\n", lr_cm)
print("SVM Confusion Matrix:\n", svm_cm)
print("Random Forest Confusion Matrix:\n", rfc_cm)
print("KNN Confusion Matrix:\n", knn_cm)
