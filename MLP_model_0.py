"""
MLP_model_0.py
----------------------------------------------------
Author: Thanasis Tsakoulis
PhD Candidate, University of Patras
Date: 25 June 2025, 14:37

Description:
This script trains a baseline Multilayer Perceptron (MLP) model 
on the MIT-BIH Arrhythmia dataset. It evaluates the model on 
accuracy, precision, recall, F1-score, and AUC, and visualizes 
training history (loss/accuracy curves) and the ROC curve.

The trained model is saved as 'MLP_model_0.h5'.
----------------------------------------------------
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# =============================
# Load the data
# =============================
train_X = np.load('Arryth_MITBIH_train_X.npy')
train_y = np.load('Arryth_MITBIH_train_y.npy')

val_X = np.load('Arryth_MITBIH_val_X.npy')
val_y = np.load('Arryth_MITBIH_val_y.npy')

test_X = np.load('Arryth_MITBIH_test_X.npy')
test_y = np.load('Arryth_MITBIH_test_y.npy')

# =============================
# Model Definition
# =============================
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(32,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Set optimizer
optimizer = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# =============================
# Callbacks
# =============================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,  
    ),
    ModelCheckpoint("MLP_model_0.h5", save_best_only=True)
]

# Summary
model.summary()

# =============================
# Training
# =============================
history = model.fit(
    train_X, train_y, 
    batch_size=64, 
    epochs=100, 
    validation_data=(val_X, val_y), 
    callbacks=callbacks
)

# =============================
# Training Curves
# =============================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# =============================
# Evaluation
# =============================
predictions_float = model.predict(test_X)
predictions = (predictions_float > 0.5).astype("int32")

# Metrics
accuracy = accuracy_score(test_y, predictions)
print(f"Accuracy: {accuracy}")

precision = precision_score(test_y, predictions)
print(f"Precision: {precision}")

recall = recall_score(test_y, predictions)
print(f"Recall: {recall}")

f1 = f1_score(test_y, predictions)
print(f"F1 Score: {f1}")

auc = roc_auc_score(test_y, predictions_float)
fpr, tpr, thresholds = roc_curve(test_y, predictions_float)

# ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
