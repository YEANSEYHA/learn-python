# Project: Learn Python for AI/ML/Deep Learning

## Goal
Teaching the user Python from scratch, focused **only** on concepts needed for AI/ML/Deep Learning. No web dev, no GUI, no fluff.

## Reference Book
"Deep Learning: A Visual Approach" by Andrew Glassner (771 pages, in `Reference/` folder)

## Learning Plan
- 30-day curriculum across 7 modules (see `PLAN.md` for details, `CURRICULUM.md` for quick reference)
- Each module maps directly to book chapters
- Practice files go in `module1/` through `module7/` folders

## Modules
1. Python Foundations (variables, functions, classes) → Days 1-3
2. NumPy (arrays, linear algebra, vectorization) → Days 4-6
3. Statistics & Probability (distributions, Bayes, metrics) → Days 7-9
4. Visualization with Matplotlib → Days 10-11
5. Classical ML with Scikit-learn → Days 12-15
6. Neural Networks from Scratch → Days 16-21
7. Deep Learning with PyTorch (CNNs, RNNs, Transformers, GANs) → Days 22-30

## Teaching Style
- **Book role:** Conceptual understanding (no code in the book)
- **Claude role:** Python coding teacher — implement what the book teaches
- **Lesson format per topic:**
  1. Definition — what it is, why it matters for ML
  2. Example Visual — ASCII/text diagram showing the concept
  3. Example how it works — runnable code with output
  4. Exercise — generate 1 problem at a time, next after user completes it
- **Practice files:** Keep clean for user to write code. Examples stay as comments only.
- **Flow:** Definition → Visual → Example → solve problem → verify → next problem → next topic

## Problem Difficulty Rules
- **Start beginner-friendly:** Each problem focuses on ONE core concept only
- **Keep problems small:** 1-3 lines of code to solve, not 6+ sub-questions
- **Build up gradually:** Problem 1 = simplest form → Problem 2 = adds one layer → Problem 3 = combines concepts
- **Do NOT overload:** Never put multiple concepts in a single problem. If a topic has 3 formulas, make 3 separate problems (one per formula)
- **Explain before asking:** Always give the summary lesson, key formulas, and a visual BEFORE the first problem

## Current Progress
- Module 1: ✓ Complete
  - Loops: ✓ Complete (for, range, while, enumerate, zip, break/continue, list comprehension)
  - Loop Practice: ✓ 4/4 problems solved
  - Functions: ✓ Complete (def, default params, return multiple values, lambda)
  - Functions Practice: ✓ 4/4 problems solved
  - Conditionals: ✓ Complete (if/elif/else, logical ops, falsy values)
  - Dictionaries: ✓ Complete
  - Classes: ✓ Complete (BankAccount, Dataset, SmartAccount/inheritance)
  - String Methods: ✓ Complete (strip, split, join, replace, f-string formatting)
- Module 2: ✓ Complete
  - Arrays (creation, shapes, dtypes): ✓ Complete (4/4 problems)
  - Indexing, Slicing, Boolean Masks: ✓ Complete (3/3 problems)
  - Reshaping & Broadcasting: ✓ Complete (3/3 problems)
  - Math Operations: ✓ Complete (3/3 problems)
- Module 3: ✓ Complete (Statistics & Probability)
  - Descriptive Stats (mean, median, std, variance, z-score, IQR outliers): ✓ Complete (3/3 problems)
  - Probability Distributions: Skipped for now (1/3 problems done)
  - Conditional & Joint Probability: ✓ Complete (3/3 problems — P(A), P(A AND B), P(A|B))
  - Bayes' Rule: ✓ Complete (2/2 problems — formula + NumPy data)
  - Confusion Matrix, Precision, Recall, F1: ✓ Complete (3/3 problems)
  - ROC Curves & AUC: ✓ Complete (3/3 problems)
  - Information Theory (Entropy, KL Divergence): ✓ Complete (3/3 problems)
- Module 4: ✓ Complete (Visualization with Matplotlib)
  - Line Plots & Scatter Plots: ✓ Complete (3/3 problems)
  - Histograms & Heatmaps: ✓ Complete (3/3 problems)
  - Subplots & Multi-figure Layouts: ✓ Complete (3/3 problems)
  - Image Display & Manipulation: ✓ Complete (3/3 problems)
- Module 5: In Progress (Classical ML with Scikit-learn)
  - Data Pipeline (train_test_split, StandardScaler, cross-validation): ✓ Complete (3/3 problems)
  - Overfitting & Underfitting (bias-variance, gap detection, early stopping): ✓ Complete (3/3 problems)
  - Linear Regression (single & multiple features, MSE, R-squared): ✓ Complete (3/3 problems)
  - Logistic Regression (sigmoid, classification, confusion matrix): ✓ Complete (3/3 problems)
