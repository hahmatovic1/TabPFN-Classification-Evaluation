# Evaluating the Effectiveness of TabPFN on Tabular Classification Tasks

This project evaluates the effectiveness of **TabPFN**, a transformer-based model designed for classification on tabular data. The aim is to assess its performance on real-world data and compare it with traditional classifiers such as Random Forest.

## Overview

Tabular data is a dominant format in various industries, playing a central role in machine learning workflows. Unlike deep learning models that require retraining, **TabPFN uses in-context learning**, enabling fast inference without any fine-tuning. This makes it a promising solution for scenarios requiring low-latency predictions and minimal compute resources.

## Why TabPFN?

- Zero retraining: predictions are made using prior meta-learned knowledge  
- Good performance on small to medium datasets  
- Supports both numerical and categorical features  
- Limitations with imbalanced data and larger datasets

## Dataset Selection: Adult Income

The **Adult Income dataset** was selected based on:

1. **Real-world relevance and complexity**  
   Derived from the U.S. Census, it reflects actual socio-economic conditions and has applications in finance, marketing, and policy-making.

2. **Mixed-type attributes**  
   It contains both numerical and categorical features, making it well-suited for testing TabPFN’s ability to handle realistic tabular structures.

3. **Size and availability**  
   With around 49,000 instances, it enables controlled experiments across multiple input sizes.

We tested TabPFN on the following sample sizes:
- 10,000 rows *(Colab memory limit reached here)*
- 5,000 rows  
- 2,500 rows  
- 500 rows *(edge-case, low-data scenario)*

> Note: At 10,000 instances, Google Colab failed due to GPU memory constraints, reflecting the upper bound of feasible usage for TabPFN v1 in limited environments.

## Additional Datasets (Reviewed Only)

As part of the context-setting and analysis, the following datasets were reviewed but **not used for model training or evaluation**:

- Breast Cancer Wisconsin Dataset  
- PRIMA Indian Diabetes Dataset  
- Wine Quality Dataset  
- Student Performance Dataset

## Research Papers (Reviewed)

The following papers were reviewed to better understand TabPFN’s capabilities and improvements:

1. **“TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second”**  
   *Noah Hollmann, Samuel Muller, Katharina Eggensperger, Frank Hutter*

2. **“Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks”**  
   *Benjamin Feuer, Chinmay Hegde, Niv Cohen*

3. **“TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems”**  
   *Si-Yang Liu, Han-Jia Ye*

## Tools and Environment

- Google Colab (T4 GPU)  
- Python 3.10  
- Libraries: `tabpfn`, `torch`, `pandas`, `scikit-learn`, `matplotlib`


# Results

## Results

### Dataset: Adult Income

The Adult Income dataset consists of 48,841 instances and is known for its class imbalance:
- Class `<=50K`: 24,720 instances (75.9%)
- Class `>50K`: 7,841 instances (24.1%)

We evaluated TabPFN’s performance on three stratified subsets:
- 10,000 instances
- 5,000 instances
- 500 instances

Each subset preserves the original class distribution.

---

### Evaluation Summary

| Subset Size | Accuracy | Precision (<=50K / >50K) | Recall (<=50K / >50K) | F1-score (<=50K / >50K) | Macro F1 | Weighted F1 |
|-------------|----------|---------------------------|------------------------|--------------------------|-----------|--------------|
| **10,000**  | 86.25%   | 0.88 / 0.77               | 0.94 / 0.61            | 0.91 / 0.68              | 0.80      | 0.86         |
| **5,000**   | 85.30%   | 0.88 / 0.74               | 0.93 / 0.61            | 0.91 / 0.67              | 0.79      | 0.85         |
| **500**     | 88.00%   | 0.89 / 0.83               | 0.96 / 0.62            | 0.92 / 0.71              | 0.82      | 0.87         |

---

### Key Observations

- On **10,000 instances**, TabPFN achieved its highest F1-score for the dominant class (0.91), but significantly lower for the minority class (0.68). Google Colab GPU memory was fully utilized at this size.

- On **5,000 instances**, performance remained stable, with balanced precision and recall. This was the best trade-off between runtime and result quality.

- On **500 instances**, overall accuracy was highest (88%), and surprisingly good F1-scores were obtained, especially for the minority class (0.71). This contradicts initial assumptions about weak performance on small, imbalanced samples.

---
## Comparison with Random Forest

To contextualize TabPFN’s performance, we trained a standard Random Forest classifier using identical subsets (500 and 5,000 instances) and evaluated its classification performance.

---

### Evaluation Summary (500 Instances)

| Model        | Accuracy | F1-score (<=50K / >50K) | Macro F1 | Weighted F1 |
|--------------|----------|--------------------------|----------|--------------|
| **TabPFN**   | 88.00%   | 0.92 / 0.71              | 0.82     | 0.87         |
| **RandomForest** | 90.00%   | 0.94 / 0.76              | 0.85     | 0.89         |

**Observation:** On the 500-instance subset, Random Forest outperformed TabPFN in both overall accuracy and F1-score, especially for the minority class `>50K`. This indicates that classical ensemble methods may be more resilient in low-resource settings with imbalanced data.

---

### Evaluation Summary (5,000 Instances)

| Model        | Accuracy | F1-score (<=50K / >50K) | Macro F1 | Weighted F1 |
|--------------|----------|--------------------------|----------|--------------|
| **TabPFN**   | 85.30%   | 0.91 / 0.67              | 0.79     | 0.85         |
| **RandomForest** | 83.30%   | 0.89 / 0.61              | 0.75     | 0.82         |

**Observation:** At 5,000 instances, TabPFN outperformed Random Forest in all F1 metrics and accuracy. This suggests that TabPFN begins to leverage its transformer-based generalization capabilities more effectively with moderately sized datasets.


While Random Forest has the edge in extremely small datasets (500 instances), TabPFN shows superior generalization and robustness at mid-sized scales (5,000+ instances). The transformer model’s advantage becomes more apparent as the dataset grows, though it still struggles slightly with class imbalance.

## Summary

TabPFN shows competitive performance across multiple dataset sizes. Its transformer-based in-context learning enables rapid evaluation without explicit training, but class imbalance remains a limiting factor. The model is well-suited for practical prototyping when working with tabular data.

