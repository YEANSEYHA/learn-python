# Python for AI/ML/Deep Learning - Learning Curriculum
### Reference: "Deep Learning: A Visual Approach" by Andrew Glassner

> Only Python concepts that matter for AI/ML/DL. No fluff.

---

## MODULE 1: Python Foundations for AI/ML (Days 1-3)
**Book Chapters: Prep for all chapters**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | Variables, Data Types (int, float, bool, str) | Every ML model uses numeric data | `module1/01_variables.py` |
| 2 | Lists, Tuples, Dictionaries, Sets | Data storage, hyperparameters, configs | `module1/02_collections.py` |
| 3 | List Comprehensions & Generators | Efficient data processing pipelines | `module1/03_comprehensions.py` |
| 4 | Functions & Lambda Functions | Building reusable ML components | `module1/04_functions.py` |
| 5 | Classes & OOP Basics | PyTorch models are classes, custom layers | `module1/05_classes.py` |
| 6 | File I/O & Data Loading | Loading datasets, saving models | `module1/06_file_io.py` |
| 7 | Error Handling | Debugging training loops | `module1/07_errors.py` |
| 8 | Iterators & Decorators | DataLoaders, timing functions | `module1/08_iterators_decorators.py` |

---

## MODULE 2: NumPy - The Foundation of All ML (Days 4-6)
**Book Chapters: Ch 2 (Statistics), Ch 5 (Curves & Surfaces), Ch 7 (Neural Networks)**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | Arrays: Creation, Shapes, dtypes | Tensors ARE multi-dimensional arrays | `module2/01_arrays.py` |
| 2 | Indexing, Slicing, Boolean Masks | Selecting data, filtering samples | `module2/02_indexing.py` |
| 3 | Reshaping, Broadcasting | Matching tensor dimensions in networks | `module2/03_reshape_broadcast.py` |
| 4 | Math Operations (element-wise, dot product) | Forward pass, weight multiplication | `module2/04_math_ops.py` |
| 5 | Linear Algebra (matmul, transpose, inverse) | Neural network core computation | `module2/05_linear_algebra.py` |
| 6 | Random Numbers & Distributions | Weight initialization, data augmentation | `module2/06_random.py` |
| 7 | Vectorization vs Loops | Speed: why NumPy beats Python loops 100x | `module2/07_vectorization.py` |

---

## MODULE 3: Statistics & Probability in Python (Days 7-9)
**Book Chapters: Ch 2 (Essential Statistics), Ch 3 (Measuring Performance), Ch 4 (Bayes' Rule)**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | Mean, Median, Std, Variance | Data normalization, batch norm | `module3/01_descriptive_stats.py` |
| 2 | Probability Distributions | Understanding data, generative models | `module3/02_distributions.py` |
| 3 | Conditional & Joint Probability | Bayes classifiers, probabilistic models | `module3/03_probability.py` |
| 4 | Bayes' Rule Implementation | Naive Bayes classifier from scratch | `module3/04_bayes.py` |
| 5 | Confusion Matrix, Precision, Recall, F1 | Model evaluation (Ch 3) | `module3/05_metrics.py` |
| 6 | ROC Curves & AUC | Comparing model performance | `module3/06_roc_auc.py` |
| 7 | Information Theory (Entropy, KL Divergence) | Loss functions, VAEs (Ch 6, Ch 18) | `module3/07_information_theory.py` |

---

## MODULE 4: Data Visualization with Matplotlib (Days 10-11)
**Book Chapters: Supports ALL chapters (the book is visual!)**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | Line Plots & Scatter Plots | Loss curves, data distributions | `module4/01_basic_plots.py` |
| 2 | Histograms & Heatmaps | Weight distributions, confusion matrices | `module4/02_histograms_heatmaps.py` |
| 3 | Subplots & Multi-figure Layouts | Comparing experiments | `module4/03_subplots.py` |
| 4 | Image Display & Manipulation | Visualizing CNN filters, input data | `module4/04_images.py` |

---

## MODULE 5: Classical ML in Python (Days 12-15)
**Book Chapters: Ch 8-12 (Training/Testing, Overfitting, Ensembles, Classifiers, Data)**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | Train/Test/Validation Split | Ch 8: Foundation of all ML | `module5/01_data_splitting.py` |
| 2 | Cross-Validation (k-fold) | Ch 8: Robust model evaluation | `module5/02_cross_validation.py` |
| 3 | Overfitting & Underfitting | Ch 9: The central challenge of ML | `module5/03_overfitting.py` |
| 4 | Feature Scaling & Normalization | Ch 12: Data preprocessing | `module5/04_preprocessing.py` |
| 5 | k-Nearest Neighbors | Ch 11: Simplest classifier | `module5/05_knn.py` |
| 6 | Decision Trees & Random Forests | Ch 10-11: Ensemble methods | `module5/06_trees_forests.py` |
| 7 | Support Vector Machines | Ch 11: Powerful classifier | `module5/07_svm.py` |
| 8 | Gradient Descent from Scratch | Ch 5, 14: Core optimization algorithm | `module5/08_gradient_descent.py` |

---

## MODULE 6: Neural Networks from Scratch (Days 16-21)
**Book Chapters: Ch 13-15 (Neural Networks, Backpropagation, Optimizers)**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | Perceptron & Neurons | Ch 13: Building block of all DL | `module6/01_perceptron.py` |
| 2 | Activation Functions (ReLU, Sigmoid, Tanh, Softmax) | Ch 13: Non-linearity in networks | `module6/02_activations.py` |
| 3 | Forward Pass (from scratch) | Ch 13: How data flows through a network | `module6/03_forward_pass.py` |
| 4 | Loss Functions (MSE, Cross-Entropy) | Ch 14: Measuring error | `module6/04_loss_functions.py` |
| 5 | Backpropagation (from scratch) | Ch 14: How networks learn | `module6/05_backprop.py` |
| 6 | Optimizers (SGD, Adam) | Ch 15: Making learning efficient | `module6/06_optimizers.py` |
| 7 | Regularization (Dropout, L1/L2) | Ch 15: Preventing overfitting | `module6/07_regularization.py` |
| 8 | Batch Normalization | Ch 15: Stabilizing training | `module6/08_batch_norm.py` |
| 9 | Full Neural Network from Scratch | Putting it all together | `module6/09_full_network.py` |

---

## MODULE 7: Deep Learning with PyTorch (Days 22-30)
**Book Chapters: Ch 16-23 (CNNs, RNNs, Transformers, GANs, RL, Autoencoders)**

| # | Topic | Why It Matters for AI/ML | Practice File |
|---|-------|-------------------------|---------------|
| 1 | PyTorch Tensors & Autograd | Foundation of PyTorch | `module7/01_pytorch_basics.py` |
| 2 | nn.Module & Building Models | Creating custom architectures | `module7/02_nn_module.py` |
| 3 | DataLoader & Datasets | Efficient data pipeline | `module7/03_dataloaders.py` |
| 4 | Training Loop Pattern | The standard DL training recipe | `module7/04_training_loop.py` |
| 5 | CNN for Image Classification | Ch 16-17: Convnets in practice | `module7/05_cnn.py` |
| 6 | Transfer Learning | Ch 17: Using pretrained models | `module7/06_transfer_learning.py` |
| 7 | Autoencoder & VAE | Ch 18: Unsupervised learning | `module7/07_autoencoder.py` |
| 8 | RNN & LSTM | Ch 19: Sequential data | `module7/08_rnn_lstm.py` |
| 9 | Attention & Transformers | Ch 20: The architecture behind LLMs | `module7/09_transformers.py` |
| 10 | GAN (Generative Adversarial Network) | Ch 22: Generating new data | `module7/10_gan.py` |
| 11 | Reinforcement Learning Basics | Ch 21: Agent-based learning | `module7/11_reinforcement.py` |

---

## How to Use This Curriculum

1. **Go module by module** - each builds on the previous
2. **Run every practice file** - reading is not enough, TYPE the code
3. **Read the matching book chapter** alongside each module
4. **Modify the examples** - break things, fix them, understand why
5. **Ask me to explain** any concept you don't understand

## Progress Tracker

- [ ] Module 1: Python Foundations
- [ ] Module 2: NumPy
- [ ] Module 3: Statistics & Probability
- [ ] Module 4: Visualization
- [ ] Module 5: Classical ML
- [ ] Module 6: Neural Networks from Scratch
- [ ] Module 7: Deep Learning with PyTorch
