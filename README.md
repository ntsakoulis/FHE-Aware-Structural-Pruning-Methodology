# FHE-Aware-Structural-Pruning-Methodology
A framework methodology for pruning and quantizing neural networks tailored for Fully Homomorphic Encryption (FHE) inference, enabling reduced model size, lower latency, and efficient encrypted computations.


# FHE-Aware MLP Methodology

This repository contains a complete workflow for training, pruning, and evaluating 
Multilayer Perceptron (MLP) models for arrhythmia classification, with a focus on 
efficient deployment under Fully Homomorphic Encryption (FHE).

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

The full workflow is organized into the following steps:

1. **Train baseline MLP** (`MLP_model_0.py`)  
   - Trains and evaluates a baseline MLP.  
   - Output: `MLP_model_0.h5`.

2. **Structured pruning** (`TF_pruning.py`)  
   - Applies polynomial decay pruning (25% → 75%) with TFMOT.  
   - Output: `tf_pruned_model.h5`.

3. **Dynamic pruning & latency evaluation** (`dynamic_pruning_eval.py`)  
   - Iteratively prunes neurons with zero weights.  
   - Evaluates pruning rates (25% → 90%).  
   - Benchmarks accuracy & latency (ms per sample).  
   - **Saves the Top-10 pruned models**.  

4. **Convert to ONNX** (`h52onnx.py`)  
   - Converts selected `.h5` models (baseline or pruned) to `.onnx`.  
   - Validates with ONNX checker.  

5. **FHE Experiments** (`fhe_experiments.py`)  
   - Takes the **best models from Step 3** (saved `.onnx` versions).  
   - Runs quantization (2–7 bits).  
   - Evaluates in both **clear** and **FHE mode** using Concrete-ML.  
   - Records compile time, accuracy, and per-sample latency.  
   - Saves results to `.xlsx` files per batch.

---

## Scripts Overview

- **MLP_model_0.py** → Baseline training & evaluation.  
- **TF_pruning.py** → Structured pruning with TFMOT.  
- **dynamic_pruning_eval.py** → Custom pruning + latency benchmarks, outputs top models.  
- **h52onnx.py** → Convert Keras `.h5` to ONNX.  
- **fhe_experiments.py** → Compile/evaluate ONNX models in clear & FHE mode.  

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Model Optimization Toolkit (`tfmot`)
- NumPy
- Matplotlib
- scikit-learn
- pandas
- tf2onnx
- onnx
- [Concrete-ML](https://github.com/zama-ai/concrete-ml)

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
    python h52onnx.py
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

