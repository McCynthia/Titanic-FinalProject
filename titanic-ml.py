import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Read the training and test datasets from CSV files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Function to fill missing values in a dataframe
def fill_missing_values(df):
    # Fill missing values in 'Age' with the median age
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Fill missing values in 'Embarked' with the most common embarkation port
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # Fill missing values in 'Fare' with the median fare
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
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()
    # Convert 'Sex' column to numerical values (0 or 1)
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    # Convert 'Embarked' column to numerical values (0, 1, 2)
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    return df

# Apply the function to both train and test dataframes
train_df = encode_categorical(train_df)
test_df = encode_categorical(test_df)

# Function to perform feature engineering (create new features and modify existing ones)
def feature_engineering(df):
    # Create a new feature 'FamilySize' by adding 'SibSp' (siblings/spouses) and 'Parch'(parents/children) plus 1 (the passanger themselves)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Extract titles from names
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # Map titles to numerical categories
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7,
        'Major': 7, 'Mlle': 8, 'Countess': 9, 'Ms': 8, 'Lady': 9, 'Jonkheer': 10,
        'Don': 11, 'Dona': 11, 'Mme': 8, 'Capt': 7, 'Sir': 9
    }
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'].fillna(0, inplace=True)
    
    # Drop columns that are not needed for the model
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    
    return df

# Apply the feature engineering function to both train and test dataframes
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Ensure test dataframe has the same columns as train dataframe (except 'Survived')
test_df['Survived'] = 0  # Add a dummy 'Survived' column to align columns
test_df = test_df[train_df.columns]

# Display the first few rows of the cleaned and processed data
print(train_df.head())
print(test_df.head())

# Save the cleaned and processed data to new CSV files
train_df.to_csv('cleaned_train.csv', index=False)
test_df.to_csv('cleaned_test.csv', index=False)
