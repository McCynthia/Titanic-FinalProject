import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Import SVM class
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Read the training and test datasets from CSV files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Function to fill missing values in a dataframe
def fill_missing_values(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    return df

# Apply the function to both train and test dataframes
train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)

# Drop the 'Cabin' column as it has too many missing values so it's not useful
train_df.drop(['Cabin'], axis=1, inplace=True)
test_df.drop(['Cabin'], axis=1, inplace=True)

# Function to convert categorical variables to numerical
def encode_categorical(df):
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    return df

# Apply the function to both train and test dataframes
train_df = encode_categorical(train_df)
test_df = encode_categorical(test_df)

# Function to perform feature engineering
def feature_engineering(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7,
        'Major': 7, 'Mlle': 8, 'Countess': 9, 'Ms': 8, 'Lady': 9, 'Jonkheer': 10,
        'Don': 11, 'Dona': 11, 'Mme': 8, 'Capt': 7, 'Sir': 9
    }
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'].fillna(0, inplace=True)
    df.drop(['Name', 'Ticket'], axis=1, inplace=True)
    return df

# Apply the feature engineering function to both train and test dataframes
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Ensure test dataframe has the same columns as train dataframe (except 'Survived')
test_df['Survived'] = 0  # Add a dummy 'Survived' column to align columns
test_df = test_df[train_df.columns]

# Normalizing numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'FamilySize']
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

# Feature selection
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

###### Random Forest ######

# Parameter tuning for Random Forest
param_grid_rf = {'n_estimators': [100, 200, 300],
                 'max_depth': [None, 10, 20, 30]}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Get the best parameters
best_params_rf = grid_search_rf.best_params_

# Train Random Forest with the best parameters
rf = RandomForestClassifier(n_estimators=best_params_rf['n_estimators'], max_depth=best_params_rf['max_depth'], random_state=42)
rf.fit(X_train, y_train)

# Evaluate Random Forest model performance
y_pred_rf = rf.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest model accuracy: {accuracy_rf}")
print(classification_report(y_val, y_pred_rf))

###### K-Nearest Neighbor ######

# Parameter tuning for KNN
param_grid_knn = {'n_neighbors': np.arange(1, 30)}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

# Get the best parameter
best_k = grid_search_knn.best_params_['n_neighbors']

# Train KNN with the best parameter
knn = KNeighborsClassifier(n_neighbors=best_k)

# Normalizing numerical features for KNN
scaler_knn = StandardScaler()
numerical_features = ['Age', 'Fare', 'FamilySize']  # Specify numerical features
X_train_scaled = scaler_knn.fit_transform(X_train[numerical_features])  # Apply scaler to training data
X_val_scaled = scaler_knn.transform(X_val[numerical_features])  # Apply scaler to validation data
test_df_scaled = scaler_knn.transform(test_df[numerical_features])  # Apply scaler to test data

# Train KNN with the best parameter using scaled features
knn.fit(X_train_scaled, y_train)

# Evaluate KNN model performance on scaled validation data
y_pred_knn = knn.predict(X_val_scaled)
accuracy_knn = accuracy_score(y_val, y_pred_knn)
print(f"KNN model accuracy: {accuracy_knn}")
print(classification_report(y_val, y_pred_knn))

###### Logistic Regression ######

# Parameter tuning for Logistic Regression (if needed)
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)

# Get the best parameter
best_c = grid_search_lr.best_params_['C']

# Train Logistic Regression with the best parameter
lr = LogisticRegression(C=best_c, max_iter=1000)
lr.fit(X_train, y_train)

# Evaluate Logistic Regression model performance
y_pred_lr = lr.predict(X_val)
accuracy_lr = accuracy_score(y_val, y_pred_lr)
print(f"Logistic Regression model accuracy: {accuracy_lr}")
print(classification_report(y_val, y_pred_lr))

###### Support Vector Machine ######

# Parameter tuning for SVM
param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
svm = SVC(class_weight='balanced')
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)

# Get the best parameters
best_params_svm = grid_search_svm.best_params_

# Train SVM with the best parameters
svm = SVC(C=best_params_svm['C'], gamma=best_params_svm['gamma'], kernel=best_params_svm['kernel'], class_weight='balanced')
svm.fit(X_train, y_train)

# Evaluate SVM model performance
y_pred_svm = svm.predict(X_val)
accuracy_svm = accuracy_score(y_val, y_pred_svm)
print(f"SVM model accuracy with further tuning: {accuracy_svm}")
print(classification_report(y_val, y_pred_svm))

# Predict survival on test data using all models
rf_pred_test = rf.predict(test_df.drop('Survived', axis=1))
knn_pred_test = knn.predict(test_df_scaled)  # Use scaled test data for KNN prediction
lr_pred_test = lr.predict(test_df.drop('Survived', axis=1))
svm_pred_test = svm.predict(test_df.drop('Survived', axis=1))

# Save predictions to CSV files
test_df['Survived_RF'] = rf_pred_test
test_df['Survived_KNN'] = knn_pred_test
test_df['Survived_LR'] = lr_pred_test
test_df['Survived_SVM'] = svm_pred_test

# Save predictions to CSV
predictions_df = test_df[['PassengerId', 'Survived_RF', 'Survived_KNN', 'Survived_LR', 'Survived_SVM']]
predictions_df.to_csv('predictions.csv', index=False)


###### Visualization ######

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, title, filename=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    if filename:
        plt.savefig(filename)
    plt.show()

# Plot Confusion Matrix for each model and save them
plot_confusion_matrix(y_val, y_pred_rf, 'Random Forest Confusion Matrix', 'rf_confusion_matrix.png')
plot_confusion_matrix(y_val, y_pred_knn, 'KNN Confusion Matrix', 'knn_confusion_matrix.png')
plot_confusion_matrix(y_val, y_pred_lr, 'Logistic Regression Confusion Matrix', 'lr_confusion_matrix.png')
plot_confusion_matrix(y_val, y_pred_svm, 'SVM Confusion Matrix', 'svm_confusion_matrix.png')

# Accuracy Comparison Plot
models = ['Random Forest', 'KNN', 'Logistic Regression', 'SVM']
accuracies = [accuracy_rf, accuracy_knn, accuracy_lr, accuracy_svm]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.ylim(0.0, 1.0)
plt.savefig('accuracy_comparison.png')
plt.show()
