"""
TF_pruning.py
----------------------------------------------------
Author: Thanasis Tsakoulis
PhD Candidate, University of Patras
Date: 25 June 2025, 15:12

Description:
This script applies structured pruning on a trained MLP model 
(using TensorFlow Model Optimization Toolkit - TFMOT). 
A polynomial decay schedule gradually increases sparsity from 
25% to 75% over training steps. 

The script evaluates both the original and the pruned model on 
the MIT-BIH Arrhythmia dataset and saves the pruned model as 
'tf_pruned_model.h5'.
----------------------------------------------------
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow_model_optimization as tfmot

# =============================
# Load base model and dataset
# =============================
model = load_model('MLP_model_0.h5')

train_X = np.load('Arryth_MITBIH_train_X.npy')
train_y = np.load('Arryth_MITBIH_train_y.npy')
val_X = np.load('Arryth_MITBIH_val_X.npy')
val_y = np.load('Arryth_MITBIH_val_y.npy')
test_X = np.load('Arryth_MITBIH_test_X.npy')
test_y = np.load('Arryth_MITBIH_test_y.npy')

# =============================
# Define pruning schedule
# =============================
b_size = 32
num_epochs = 20
e_step = np.ceil(len(train_X) / b_size).astype(np.int32) * num_epochs
print(f'End step: {e_step}')

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.25,
    final_sparsity=0.75,
    begin_step=0,
    end_step=e_step
)

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule=pruning_schedule
)

# =============================
# Compile pruned model
# =============================
pruned_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='logs')
]

# =============================
# Train pruned model
# =============================
pruned_model.fit(
    train_X, train_y,
    batch_size=b_size,
    epochs=num_epochs,
    validation_data=(val_X, val_y),
    callbacks=callbacks
)

# =============================
# Evaluation
# =============================
original_eval = model.evaluate(test_X, test_y, verbose=0)
pruned_eval = pruned_model.evaluate(test_X, test_y, verbose=0)

print(f'Original Model Accuracy: {original_eval[1]:.4f}')
print(f'Pruned Model Accuracy: {pruned_eval[1]:.4f}')

# =============================
# Strip pruning & save model
# =============================
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
pruned_model.save('tf_pruned_model.h5')
print("Pruned model saved as 'tf_pruned_model.h5'")
