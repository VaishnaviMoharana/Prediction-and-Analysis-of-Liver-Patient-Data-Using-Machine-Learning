import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from scipy.stats import uniform
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
#data = pd.read_csv(r"https://www.kaggle.com/datasets/uciml/indian-liver-patient-records")
# download the dataset using the above link and copy paste the link here
data = pd.read_csv(r"C:\Users\VAISHNAVI\OneDrive\Desktop\ML datasets\indian_liver_patient.csv")
# download the dataset using the above link and copy paste the link here
print(data)
print(data.info())
print(data.describe())
print(data.columns)
print(data.isnull().sum())
print(data.shape)
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mode()[0])
print(data.isnull().sum())
sns.countplot(x=data['Gender'], data=data)
m, f=data['Gender'].value_counts()
print("No. of Males: ", m)
print("No. of Females: ", f)
plt.show()
sns.countplot(x=data['Dataset'], data=data)
LD, NLD=data['Dataset'].value_counts()
print("Liver disease patients: ", LD)
print("Non-Liver disease patients: ", NLD)
plt.show()
sns.set_style('darkgrid')
plt.figure(figsize=(25,10))
data['Age'].value_counts().plot.bar(color='darkviolet')
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio",color='mediumspringgreen',data=data);
plt.show()
sns.pairplot(data)
plt.show()
sns.boxplot(data['Albumin_and_Globulin_Ratio'])
plt.show()
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
le = LabelEncoder()
x_train_gender = le.fit_transform(x_train['Gender'])
x_test_gender = le.transform(x_test['Gender'])
print(le.classes_)
x_train_gender =x_train_gender.reshape(-1, 1)
x_test_gender = x_test_gender.reshape(-1, 1)
x_train = x_train.drop('Gender', axis=1)
x_test = x_test.drop('Gender', axis=1)
x_train_combined = np.concatenate((x_train.values, x_train_gender), axis=1)
x_test_combined = np.concatenate((x_test.values, x_test_gender), axis=1)
column_names = list(x_train.columns) + ['Gender']
x_train_final = pd.DataFrame(x_train_combined, columns=column_names)
x_test_final = pd.DataFrame(x_test_combined, columns=column_names)
print("Shape of x_train_combined:", x_train_final.shape)
print("Shape of x_test_combined:", x_test_final.shape)
print(x_train_final.sample(5))
x_train_final.to_csv('x_train_final.csv', index=False)
x_test_final.to_csv('x_test_final.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


print("Logistic Regression...")
lr_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'solver': ['liblinear', 'saga']
}
lr_s = LogisticRegression(max_iter=1000)
lr_grid_search = GridSearchCV(estimator=lr_s, param_grid=lr_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
lr_grid_search.fit(x_train_final, y_train)
# Get best parameters
lr_best_params = lr_grid_search.best_params_
print("Best parameters for Logistic Regression:", lr_best_params)
lr = LogisticRegression(**lr_best_params, max_iter=1000)
lr.fit(x_train_final, y_train)
y_pred_lr = lr.predict(x_test_final)
lr_acc = accuracy_score(y_pred_lr, y_test)
print("Accuracy of Logistic Regression: ", lr_acc)
print("Classification Report of Logistic Regression: ", classification_report(y_test, y_pred_lr))
lr_cross = cross_val_score(lr, x_train_final, y_train, scoring='accuracy', cv = 6)
print("Cross Validation Score of Logistic Regression: ", lr_cross.mean())
lr_cm = confusion_matrix(y_pred_lr,y_test)
print("Confusion Matrix of Logistic Regression: ", lr_cm)


print("Support Vector Classifier (SVC)...")
svm_param_dist = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_s = SVC()
# Initialize GridSearchCV
svm_random_search = RandomizedSearchCV(estimator=svm_s, param_distributions=svm_param_dist, n_iter=10, cv=3, scoring='accuracy', verbose=1, n_jobs=2, random_state=42)
# Fit GridSearchCV
svm_random_search.fit(x_train_final, y_train)
svm_best_params = svm_random_search.best_params_
print("Best parameters for SVC:", svm_best_params)
svm = SVC(**svm_best_params)
svm.fit(x_train_final, y_train)
y_pred_svm = svm.predict(x_test_final)
svm_acc = accuracy_score(y_pred_svm, y_test)
print("Accuracy of SVC: ", svm_acc)
print("Classification Report of SVC", classification_report(y_test, y_pred_svm))
svm_cross = cross_val_score(svm, x_train_final, y_train, scoring='accuracy', cv = 6)
print("Cross Validation Score of SVC: ", svm_cross.mean())
svm_cm = confusion_matrix(y_pred_svm,y_test)
print("Confusion Matrix of SVC: ", svm_cm)


print("Random Forest Classifier...")
rfc_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rfc_s = RandomForestClassifier()
rfc_grid_search = GridSearchCV(estimator=rfc_s, param_grid=rfc_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
rfc_grid_search.fit(x_train_final, y_train)
rfc_best_params = rfc_grid_search.best_params_
print("Best parameters for Random Forest Classifier:", rfc_best_params)
rfc = RandomForestClassifier(**rfc_best_params)
rfc.fit(x_train_final, y_train)
ypred_rfc = rfc.predict(x_test_final)
rfc_acc = accuracy_score(ypred_rfc, y_test)
print("Accuracy of Random Forest Classifier: ", rfc_acc)
print("Classification Report of Random Forest Classifier: ", classification_report(y_test, ypred_rfc))
rfc_cross = cross_val_score(rfc, x_train_final, y_train, scoring='accuracy', cv = 6)
print("Cross Validation Score of Random Forest Classifier: ", rfc_cross.mean())
rfc_cm = confusion_matrix(ypred_rfc,y_test)
print("Confusion Matrix of Random Forest Classifier: ", rfc_cm)


print("K Neighbors Classifier...")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_s = KNeighborsClassifier()
knn_grid_search = GridSearchCV(estimator=knn_s, param_grid=knn_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
# Fit GridSearchCV
knn_grid_search.fit(x_train_final, y_train)
knn_best_params = knn_grid_search.best_params_
print("Best parameters for K Neighbors Classifier:", knn_best_params)
knn = KNeighborsClassifier(**knn_best_params)
knn.fit(x_train_final, y_train)
ypred_knn = knn.predict(x_test_final)
knn_acc = accuracy_score(ypred_knn, y_test)
print("Accuracy of KNN: ", knn_acc)
print("Classification Report of KNN: ", classification_report(y_test, ypred_knn))
knn_cross = cross_val_score(knn, x_train_final, y_train, scoring='accuracy', cv = 6)
print("Cross Validation Score of KNN: ", knn_cross.mean())
knn_cm = confusion_matrix(ypred_knn,y_test)
print("Confusion Matrix of KNN: ", knn_cm)
pickle.dump(svm, open('svm_liver_analysis.pkl', 'wb'))
pickle.dump(rfc, open('rfc_liver_analysis.pkl', 'wb'))
pickle.dump(knn, open('knn_liver_analysis.pkl', 'wb'))
pickle.dump(lr, open('lr_liver_analysis.pkl', 'wb'))