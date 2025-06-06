# Matrix Factorization With Comparison Data

This repository contains the codebase for a semester project at **EPFL**, conducted at the **Indy Lab**, under the supervision of **Dr. Suryanarayana Sankagiri**.

## 🎯 Project Goal

The project investigates **matrix factorization models** trained on **comparative data** — i.e., user-item **triplet comparisons** rather than ratings. 

We analyze the impact of several parameters on learning dynamics and reconstruction performance, including:

- **Embedding dimension** `d`
- **Sparsity** of comparisons `p`
- **Noise** in the data `s` (scaling factor)
- **Redundancy** `k` (number of times a preference is repeated)
- **Triplet sampling strategies** (random, margin, SVD-based, popularity-biased, etc.)

The analysis includes both **quantitative metrics** (accuracy, Pearson/Spearman correlations, reconstruction error) and **visual comparisons**.

## 📁 Structure

| File / Folder       | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `generation_data.py`| Code for generating synthetic matrices and sampling triplet comparisons     |
| `structure.py`        | Core logic: model training, evaluation, reconstruction error, correlations  |
| `visualization.py`  | Full suite of plotting functions (heatmaps, loss curves, grouped plots...)  |
| `Runs.ipynb`        | Notebook used to **launch experiments and save results**                    |
| `Plots.ipynb`       | Notebook used to **visualize results and generate plots for the report**    |
| `pdf/Matrix_Factorization_with_Comparison_Data.pdf` | Final **report** detailing the methodology, experiments and insights  |
| `requirements.txt`  | Minimal dependency list to run the code                                     |

## 📊 How to Use

To reproduce the results:

- Use `Runs.ipynb` to run experiments with different parameters.
- Use `Plots.ipynb` to generate visualizations from `.pkl` result files.
- All source code is organized by functionality. For deeper understanding of each method, refer directly to the notebooks.

## 📄 Final Report

📄 The final PDF report is available in the [`pdf/`](./pdf) folder:  
[`Matrix_Factorization_with_Comparison_Data.pdf`](./pdf/Matrix_Factorization_with_Comparison_Data.pdf)

It presents detailed experiments, methodology, key results, and conclusions.

## 💾 Installation

Install dependencies:
```bash
pip install -r requirements.txt
````

## 👤 Author

**Mayeul Cassier**
Semester Project @ EPFL – Indy Lab
Supervised by Dr. Suryanarayana Sankagiri


---

### 📁 Files management

```
Matrix-Factorization-With-Comparison-Data/
├── generation_data.py
├── helpers.py
├── visualization.py
├── Runs.ipynb
├── Plots.ipynb
├── requirements.txt
├── README.md
└── pdf/
    └── Matrix_Factorization_with_Comparison_Data.pdf

