# FHE-Aware MLP Methodology

A framework methodology for pruning and quantizing neural networks tailored for Fully Homomorphic Encryption (FHE) inference, enabling reduced model size, lower latency, and efficient encrypted computations.

---

## Dataset Preparation

The MIT-BIH Arrhythmia dataset contains more than 15 heartbeat annotations, 
grouped into five main categories (N, S, V, F, Q).  
To address class imbalance (~90% normal beats), the **N class was downsampled** 
and all abnormal beats were merged into a single class, resulting in a **binary 
classification task (Normal vs Abnormal)**.

Each sample is represented by **32 features**, derived from ECG R-peak analysis 
(time-domain intervals, wave amplitudes, and morphology features).  
The final dataset contains **24,148 balanced samples** from two ECG channels.

---

## Workflow

1. **Baseline Training** → Train and evaluate an MLP model on the MIT-BIH dataset.  
2. **Structured Pruning (TFMOT)** → Apply polynomial decay pruning (25% → 75%).  
3. **Dynamic Pruning** → Explore multiple pruning rates, benchmark accuracy & latency, and keep the Top-10 models.  
4. **ONNX Conversion** → Convert selected models to `.onnx` format.  
5. **FHE Experiments** → Quantize & evaluate the best models in clear and FHE mode.

---

## Usage

- **Step 1 – Train baseline**
  - Run:
    ```bash
    python MLP_model_0.py
    ```
  - Trains the baseline MLP model on the balanced MIT-BIH dataset.  
  - Saves the model as `MLP_model_0.h5`.

- **Step 2 – Apply pruning (TFMOT)**
  - Run:
    ```bash
    python TF_pruning.py
    ```
  - Applies structured pruning with polynomial decay (25% → 75%).  
  - Saves the pruned model as `tf_pruned_model.h5`.

- **Step 3 – Dynamic pruning & selection of best models**
  - Run:
    ```bash
    python dynamic_pruning_eval.py
    ```
  - Iteratively prunes neurons based on zero-weight statistics.  
  - Benchmarks accuracy & latency.  
  - Saves the **Top-10 best pruned models** as `.h5`.

- **Step 4 – Convert best models to ONNX**
  - Run:
    ```bash
    python h5_onnx_converter.py
    ```
  - Converts `.h5` models into `.onnx` format.  
  - Validates with ONNX checker.

- **Step 5 – Run FHE experiments**
  - Run:
    ```bash
    python fhe_experiments.py
    ```
  - Loads ONNX models from Step 3.  
  - Runs quantization (2–7 bits).  
  - Evaluates in both **clear** and **FHE mode** using Concrete-ML.  
  - Saves batch results as Excel files (`results_batch_X.xlsx`).

