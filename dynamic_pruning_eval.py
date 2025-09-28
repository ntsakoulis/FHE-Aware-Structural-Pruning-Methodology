"""
dynamic_pruning_eval.py
----------------------------------------------------
Author: Thanasis Tsakoulis
PhD Candidate, University of Patras
Date: 25 June 2025, 15:45

Description:
This script implements a custom dynamic pruning methodology 
for MLP models trained on the MIT-BIH Arrhythmia dataset.

Workflow:
1. Loads a previously pruned model ('tf_pruned_model.h5').
2. Iteratively prunes neurons based on zero-weight statistics.
3. Evaluates multiple pruning rates (25% → 90%).
4. Benchmarks prediction latency across multiple cycles.
5. Saves the top-10 pruned models by accuracy.
6. Visualizes pruning rate vs accuracy and latency.

The purpose of this script is to explore efficient model 
architectures that balance accuracy and inference latency, 
with potential application to FHE (Fully Homomorphic Encryption).
----------------------------------------------------
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, InputLayer
import time
import matplotlib.pyplot as plt

# =============================
# Load baseline pruned model & dataset
# =============================
original_model = load_model('tf_pruned_model.h5')

train_X = np.load('Arryth_MITBIH_train_X.npy')
train_y = np.load('Arryth_MITBIH_train_y.npy')
val_X = np.load('Arryth_MITBIH_val_X.npy')
val_y = np.load('Arryth_MITBIH_val_y.npy')
test_X = np.load('Arryth_MITBIH_test_X.npy')
test_y = np.load('Arryth_MITBIH_test_y.npy')

# Compile and baseline evaluation
original_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
baseline_loss, baseline_accuracy = original_model.evaluate(test_X, test_y, verbose=0)
print("=" * 50)
print(f"Baseline Accuracy (TF-pruned model): {baseline_accuracy:.4f}")
print("=" * 50)


# =============================
# Dynamic pruning functions
# =============================
def count_zero_weights(weights, axis):
    """Count the number of zero (or near-zero) weights along a specified axis."""
    return np.sum(np.isclose(weights, 0), axis=axis)


def prune_neurons_dynamically(weights, prune_rate):
    """Prune neurons dynamically based on zero-weight statistics."""
    pruned_weights, pruned_biases, neuron_indices_to_keep = [], [], []

    for i in range(0, len(weights), 2):
        if i < len(weights) - 2:
            incoming_zero_counts = count_zero_weights(weights[i], axis=0)
            outgoing_zero_counts = count_zero_weights(weights[i + 2], axis=1)
            total_zero_counts = incoming_zero_counts + outgoing_zero_counts
        else:
            total_zero_counts = count_zero_weights(weights[i], axis=0)

        total_neurons = len(total_zero_counts)
        neurons_to_prune = int(total_neurons * prune_rate)

        if total_neurons - neurons_to_prune < 1:
            neurons_to_prune = total_neurons - 1

        neurons_to_keep = np.argsort(total_zero_counts)[:-neurons_to_prune]
        neurons_to_keep = np.sort(neurons_to_keep)
        neuron_indices_to_keep.append(neurons_to_keep)

        pruned_weights.append(weights[i][:, neurons_to_keep])
        pruned_biases.append(weights[i + 1][neurons_to_keep])

        print(f"Layer {i//2} → Pruned shape: {pruned_weights[-1].shape}")

        if i >= len(weights) - 2:
            if pruned_weights[-1].shape[1] == 0:
                pruned_weights[-1] = weights[i][:, :1]
                pruned_biases[-1] = weights[i + 1][:1]
            break

        next_layer_weights = weights[i + 2]
        weights[i + 2] = next_layer_weights[neurons_to_keep, :]

    return pruned_weights, pruned_biases, neuron_indices_to_keep


def build_pruned_model(input_shape, original_model, prune_rate):
    """Build and compile a pruned model."""
    pruned_weights, pruned_biases, _ = prune_neurons_dynamically(
        original_model.get_weights(), prune_rate
    )

    pruned_model = Sequential()
    pruned_model.add(InputLayer(input_shape=input_shape))

    for i in range(len(pruned_weights)):
        if i < len(pruned_weights) - 1:
            pruned_model.add(Dense(pruned_weights[i].shape[1], activation='relu'))
        else:
            pruned_model.add(Dense(1, activation='sigmoid'))

    pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    pruned_layer_index = 0
    for layer in pruned_model.layers:
        if isinstance(layer, Dense):
            layer.set_weights([
                pruned_weights[pruned_layer_index],
                pruned_biases[pruned_layer_index]
            ])
            pruned_layer_index += 1

    return pruned_model


def shuffle_arrays_in_unison(a, b):
    """Shuffle two arrays in unison."""
    p = np.random.permutation(len(a))
    return a[p], b[p]


# =============================
# Experiment setup
# =============================
prune_rates = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
num_cycles = 20

total_times, average_cycle_times, average_sample_times, accuracies = [], [], [], []
best_pruned_models = []

# =============================
# Pruning evaluation loop
# =============================
for rate in prune_rates:
    print("\n" + "-" * 50)
    print(f"Pruning rate: {rate}")
    reduced_model = build_pruned_model(
        input_shape=(train_X.shape[1],),
        original_model=original_model,
        prune_rate=rate
    )

    _, accuracy = reduced_model.evaluate(test_X, test_y, verbose=0)
    accuracies.append(accuracy)
    print(f"Accuracy of model pruned by {int(rate * 100)}%: {accuracy:.4f}")

    best_pruned_models.append((reduced_model, accuracy, rate))

    total_cycle_time = 0
    for cycle in range(num_cycles):
        shuffled_test_X, shuffled_test_y = shuffle_arrays_in_unison(test_X, test_y)
        start_time = time.time()
        _ = reduced_model.predict(shuffled_test_X, verbose=0)
        end_time = time.time()
        cycle_time = (end_time - start_time) * 1e3
        total_cycle_time += cycle_time
        print(f"Cycle {cycle + 1} prediction time: {cycle_time:.2f} ms")

    total_time = total_cycle_time
    average_time_per_cycle = total_time / num_cycles
    average_time_per_sample = total_time / (num_cycles * test_X.shape[0])

    total_times.append(total_time)
    average_cycle_times.append(average_time_per_cycle)
    average_sample_times.append(average_time_per_sample)

    print(f"Total time for {num_cycles} cycles: {total_time:.2f} ms")
    print(f"Average time per cycle: {average_time_per_cycle:.2f} ms")
    print(f"Average time per sample: {average_time_per_sample:.6f} ms")


# =============================
# Save top models
# =============================
best_pruned_models.sort(key=lambda x: x[1], reverse=True)
top_n = 10
for i in range(min(top_n, len(best_pruned_models))):
    model, accuracy, rate = best_pruned_models[i]
    model_name = f"pruned_model_top_{i+1}_prune_{int(rate*100)}_acc_{accuracy:.4f}.h5"
    model.save(model_name)
    print(f"Saved: {model_name}")


# =============================
# Visualization
# =============================
plt.figure(figsize=(6, 12))

plt.subplot(2, 1, 1)
plt.plot(prune_rates, total_times, marker='o')
plt.xlabel('Pruning Rate')
plt.ylabel('Total Prediction Time (ms)')
plt.title('Pruning Rate vs Total Prediction Time')

plt.subplot(2, 1, 2)
plt.plot(prune_rates, average_cycle_times, marker='o')
plt.xlabel('Pruning Rate')
plt.ylabel('Average Cycle Time (ms)')
plt.title('Pruning Rate vs Average Cycle Time')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 12))

plt.subplot(2, 1, 1)
plt.plot(prune_rates, accuracies, marker='o')
plt.xlabel('Pruning Rate')
plt.ylabel('Accuracy')
plt.title('Pruning Rate vs Accuracy')

plt.subplot(2, 1, 2)
plt.plot(prune_rates, average_cycle_times, marker='o')
plt.xlabel('Pruning Rate')
plt.ylabel('Average Cycle Time (ms)')
plt.title('Pruning Rate vs Average Cycle Time')

plt.tight_layout()
plt.show()
