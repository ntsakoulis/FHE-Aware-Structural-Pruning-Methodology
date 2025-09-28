import onnx
import numpy as np
import time
import pandas as pd
import gc
from concrete.ml.torch.compile import compile_onnx_model

# Parameters
batch_size = 50  # Size of each batch (from which mini batches are sampled)
mini_batch_size = 10  # Size of each mini batch
quant_bits = [2, 3, 5, 6, 7]  # Quantization bits
run_fhe = 1  # 0: just evaluate, 1: run FHE model
random_seed = 42  # Random seed for reproducibility
start_batch = 2  # Batch to start from

# Set random seed for reproducibility
np.random.seed(random_seed)

# Load input
input_set = np.load('./anomaly_detection/test_X.npy')
output_set = np.load('./anomaly_detection/test_y.npy')

print(f"Length of dataset: {len(output_set)}")

# Load models
model_files = [
    "./anomaly_detection/pruned_model_2.onnx",
    "./anomaly_detection/pruned_model_35_percent_acc_9635.onnx",
    "./anomaly_detection/pruned_model_50_percent_acc_9525.onnx"
]

# Function to compile and evaluate the model
def compile_and_evaluate(onnx_model, model_name, mini_testX, mini_testY, bits, batch, selected_indices):
    # Compile
    print("Compiling model with quantization bits:", bits)
    start_time = time.time()
    quantized_module = compile_onnx_model(onnx_model, mini_testX, n_bits=bits)
    compile_time = time.time() - start_time
    print(f"Compiled model in --- {compile_time} seconds ---")

    # Run model in clear
    print("Running clear model")
    start_time = time.time()
    y_clear = quantized_module.forward(mini_testX, fhe="disable")
    clear_time = (time.time() - start_time) / mini_batch_size
    quant_y_clear = (y_clear > 0.5).astype(int).flatten()
    accuracy_clear = np.mean(quant_y_clear == mini_testY.astype(int)) * 100
    print(f"Execution in clear --- {clear_time} seconds ---")
    print(f"Accuracy (clear): {accuracy_clear} %")

    # Print debug information
    print(f"Ground truth: {mini_testY}")
    print(f"Predictions (clear): {quant_y_clear}")
    print(f"Raw outputs (clear): {y_clear}")

    # Calculate the percentage of 1s and 0s in the ground truth and the outputs
    perc_ones_gt = np.mean(mini_testY) * 100
    perc_zeros_gt = 100 - perc_ones_gt
    perc_ones_pred_clear = np.mean(quant_y_clear) * 100
    perc_zeros_pred_clear = 100 - perc_ones_pred_clear

    print(f"Ground Truth - Percentage of 1s: {perc_ones_gt:.2f}%, Percentage of 0s: {perc_zeros_gt:.2f}%")
    print(f"Predictions (clear) - Percentage of 1s: {perc_ones_pred_clear:.2f}%, Percentage of 0s: {perc_zeros_pred_clear:.2f}%")

    result = {
        "model_name": model_name,
        "num_items": mini_batch_size,
        "batch": batch,
        "q_bits": bits,
        "compile_time": compile_time,
        "clear_time": clear_time,
        "accuracy_clear": accuracy_clear,
        "perc_ones_gt": perc_ones_gt,
        "perc_zeros_gt": perc_zeros_gt,
        "perc_ones_pred_clear": perc_ones_pred_clear,
        "perc_zeros_pred_clear": perc_zeros_pred_clear,
        "y_clear": y_clear.tolist(),
        "quant_y_clear": quant_y_clear.tolist(),
        "ground_truth_x": mini_testX.tolist(),
        "ground_truth_y": mini_testY.tolist(),
        "selected_indices": selected_indices.tolist()
    }

    # Run model in FHE
    if run_fhe:
        print("Running FHE model")
        start_time = time.time()
        y_fhe = quantized_module.forward(mini_testX, fhe="execute")
        fhe_time = (time.time() - start_time) / mini_batch_size
        quant_y_fhe = (y_fhe > 0.5).astype(int).flatten()
        accuracy_fhe = np.mean(quant_y_fhe == mini_testY.astype(int)) * 100
        print(f"Execution in FHE --- {fhe_time} seconds ---")
        print(f"Accuracy (FHE): {accuracy_fhe} %")

        # Print debug information
        print(f"Predictions (FHE): {quant_y_fhe}")
        print(f"Raw outputs (FHE): {y_fhe}")

        result.update({
            "fhe_time": fhe_time,
            "accuracy_fhe": accuracy_fhe,
            "y_fhe": y_fhe.tolist(),
            "quant_y_fhe": quant_y_fhe.tolist(),
            "perc_ones_pred_fhe": np.mean(quant_y_fhe) * 100,
            "perc_zeros_pred_fhe": 100 - np.mean(quant_y_fhe) * 100,
        })
    else:
        result.update({
            "fhe_time": None,
            "accuracy_fhe": None,
            "y_fhe": None,
            "quant_y_fhe": None,
            "perc_ones_pred_fhe": None,
            "perc_zeros_pred_fhe": None,
        })

    return result

# Loop to process the dataset in batches 
num_batches = len(output_set) // batch_size
print(f"Total number of batches: {num_batches}")

for batch in range(start_batch, num_batches):
    print(f"===================================================================")
    print(f"Processing batch {batch + 1}/{num_batches}")

    # Calculate indices for the current batch
    batch_start_index = batch * batch_size
    batch_end_index = batch_start_index + batch_size
    if batch_end_index > len(output_set):
        batch_end_index = len(output_set)
        batch_start_index = batch_end_index - batch_size  # Ensure we get the last `batch_size` samples if we exceed the length
    indices = list(range(batch_start_index, batch_end_index))

    # Select mini batch (10 samples from the current batch)
    selected_indices = np.random.choice(indices, mini_batch_size, replace=False)
    selected_indices.sort()  # Ensure indices are in ascending order
    mini_testX = input_set[selected_indices]
    mini_testY = output_set[selected_indices]

    # Print percentage of 1s and 0s in mini batch
    ones_count = np.sum(mini_testY)
    zeros_count = len(mini_testY) - ones_count
    print(f"Mini batch {batch + 1} has {ones_count} ones and {zeros_count} zeros")

    # DataFrame to store results for this batch
    results_df = pd.DataFrame()

    for model_file in model_files:
        model_start = time.time()
        print(f"-------------------------------------------------------------------")
        model_name = model_file.split('/')[-1]  # Extract model name from file path
        print(f"Processing model: {model_name}")

        # Load model
        onnx_model = onnx.load(model_file)
        onnx.checker.check_model(onnx_model)
        print("Loaded ONNX model")

        for bits in quant_bits:
            print("      ")
            result = compile_and_evaluate(onnx_model, model_name, mini_testX, mini_testY, bits, batch, selected_indices)
            results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
            print("Waiting for 10 seconds to cool down...")
            time.sleep(10)  # Wait for 10 seconds to cool down

        model_time = time.time() - model_start
        print(f"Model {model_name} took {model_time:.2f} seconds")

    # Free up memory
    gc.collect()

    # Save results to an Excel file for this batch
    results_df.to_excel(f"results_batch_{batch + 1}.xlsx", index=False, header=True)
    print(f"Results saved to results_batch_{batch + 1}.xlsx")

    print("Waiting for 60 seconds to cool down until next batch")
    time.sleep(60)  # Wait for 60 seconds to cool down

