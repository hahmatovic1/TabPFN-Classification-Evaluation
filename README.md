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


