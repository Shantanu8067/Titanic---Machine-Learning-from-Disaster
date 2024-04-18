#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

# Data preprocessing
titanic_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

# Handling missing values
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Feature engineering
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']

# Split the data into features and target
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[4]:





# In[ ]:


import subprocess

# Install TensorFlow using pip
subprocess.call(['pip', 'install', 'tensorflow'])




# In[ ]:


# Import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the horses_or_humans dataset from TensorFlow datasets
dataset, info = tfds.load('horses_or_humans', split='train', with_info=True)

# Define input image dimensions
IMG_HEIGHT, IMG_WIDTH = 300, 300
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Preprocess and normalize images
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32)
    image /= 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_image)

# Shuffle and batch the dataset
BATCH_SIZE = 32
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=IMG_SHAPE),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(dataset, epochs=10)

