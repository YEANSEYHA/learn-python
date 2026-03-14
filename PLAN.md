# Python for AI/ML/Deep Learning - Detailed Learning Plan
### Reference: "Deep Learning: A Visual Approach" by Andrew Glassner
### Goal: Go from beginner to expertise in Python for AI/ML/DL

---

## PHASE 1: Python Core (Days 1-3) — "Learn the Language"
> Skip everything that doesn't help you build AI/ML models

### Day 1: Variables, Data Types & Collections
**Book prep:** Read Ch 1 (Overview of Machine Learning) to understand WHY you're learning this

**What to learn:**
- Variables: `int`, `float`, `bool`, `str` — these become your model's data
- Lists: ordered data → think "a row of training samples"
- Tuples: immutable data → think "image dimensions (height, width, channels)"
- Dictionaries: key-value pairs → think "hyperparameters: {'lr': 0.001, 'epochs': 50}"
- Sets: unique values → think "unique class labels"

**Skip:** String formatting tricks, regex, complex string ops — not needed for ML

**Practice:** Build a program that stores a dataset as a list of dictionaries

---

### Day 2: Functions, Lambdas & List Comprehensions
**Why for AI/ML:** Every ML pipeline is built from functions. Lambda functions are used constantly in data processing.

**What to learn:**
- Functions with `*args`, `**kwargs` — ML frameworks use these everywhere
- Lambda functions — `sorted(data, key=lambda x: x['loss'])`
- List comprehensions — `[x**2 for x in range(10)]` → fast data transforms
- Generator expressions — memory-efficient data pipelines for large datasets
- `map()`, `filter()`, `zip()` — batch data processing

**Practice:** Write functions that compute mean, standard deviation, and normalize a list of numbers

---

### Day 3: Classes (OOP) & File I/O
**Why for AI/ML:** PyTorch models = Python classes. Every custom layer, every model you build inherits from `nn.Module`

**What to learn:**
- Classes: `__init__`, methods, attributes
- Inheritance — `class MyModel(nn.Module):` ← this IS deep learning
- `__call__`, `__repr__`, `__len__` — used by DataLoaders
- File I/O: reading CSV, JSON, binary files
- `with` statement (context managers) — safe file handling
- `pickle` / `json` — saving/loading model configs

**Skip:** Multiple inheritance, metaclasses, abstract classes — overkill for now

**Practice:** Build a `Dataset` class that loads data from a CSV file

---

## PHASE 2: NumPy Mastery (Days 4-6) — "Think in Arrays, Not Loops"
> NumPy IS the foundation. PyTorch tensors = GPU-powered NumPy arrays.

### Day 4: Arrays & Operations
**Book chapters:** Ch 2 (Essential Statistics), Ch 5 (Curves & Surfaces)

**What to learn:**
- `np.array()`, `np.zeros()`, `np.ones()`, `np.random.randn()` — weight initialization
- Shapes: `(batch_size, channels, height, width)` — the shape of deep learning
- `dtype`: `float32` is the standard for ML (not float64!)
- Element-wise ops: `+`, `*`, `/` — these are parallel GPU operations
- `np.dot()`, `np.matmul()`, `@` operator — THE core operation of neural networks

**Key insight:** A neural network forward pass is literally: `output = activation(weights @ input + bias)`

**Practice:** Implement matrix multiplication from scratch, then compare with `np.matmul()`

---

### Day 5: Indexing, Reshaping & Broadcasting
**Why for AI/ML:** You'll reshape tensors hundreds of times. Broadcasting errors are the #1 beginner bug.

**What to learn:**
- Fancy indexing: `array[array > 0]` — ReLU activation in one line!
- Slicing: `data[:32]` — grabbing a mini-batch
- `reshape()`, `flatten()`, `squeeze()`, `unsqueeze()` — tensor gymnastics
- Broadcasting rules — how `(3,1) + (1,4)` becomes `(3,4)`
- `np.concatenate()`, `np.stack()` — combining batches

**Practice:** Take a (28,28) image array, flatten it to (784,), reshape to (1,28,28), and understand why

---

### Day 6: Vectorization & Performance
**Why for AI/ML:** If you write a `for` loop over data in ML, you're doing it wrong.

**What to learn:**
- Vectorized operations vs loops — benchmark the 100x speedup
- `np.where()` — conditional operations without loops
- `np.einsum()` — Einstein summation (used in Transformers!)
- Memory layout: C-order vs Fortran-order (affects GPU performance)
- `np.linalg` — eigenvalues, SVD, norms (used in PCA, regularization)

**Practice:** Implement batch normalization using only NumPy (no loops)

---

## PHASE 3: Statistics & Probability (Days 7-9) — "Understand Your Data"
> Maps directly to Book Chapters 2, 3, 4, 6

### Day 7: Descriptive Statistics & Distributions
**Book chapters:** Ch 2 (Essential Statistics)

**What to learn:**
- `np.mean()`, `np.std()`, `np.var()`, `np.median()` — data normalization
- Normal distribution — weight initialization (`np.random.randn() * 0.01`)
- Uniform distribution — random sampling
- Histograms — visualizing weight distributions during training
- Correlation — feature selection

**Practice:** Generate data from different distributions, compute stats, plot histograms

---

### Day 8: Probability & Bayes
**Book chapters:** Ch 3 (Measuring Performance), Ch 4 (Bayes' Rule)

**What to learn:**
- Conditional probability — P(spam | word="free")
- Bayes' theorem — the foundation of probabilistic ML
- Prior, likelihood, posterior — Bayesian thinking
- Confusion matrix, precision, recall, F1-score — HOW you evaluate models
- ROC curve & AUC — comparing classifiers

**Practice:** Build a Naive Bayes spam classifier from scratch

---

### Day 9: Information Theory
**Book chapter:** Ch 6 (Information Theory)

**What to learn:**
- Entropy — measures uncertainty (used in decision trees)
- Cross-entropy — THE loss function for classification
- KL Divergence — used in VAEs (Ch 18), measures distribution difference
- Log-likelihood — foundation of many loss functions

**Key insight:** When you train a classifier with `CrossEntropyLoss`, you're minimizing the KL divergence between predicted and true distributions

**Practice:** Implement cross-entropy loss from scratch, compare with library version

---

## PHASE 4: Visualization (Days 10-11) — "See What Your Model Sees"
> The book is called "A Visual Approach" for a reason!

### Day 10-11: Matplotlib & Data Visualization
**What to learn:**
- `plt.plot()` — training loss curves (you'll plot THOUSANDS of these)
- `plt.scatter()` — data distributions, decision boundaries
- `plt.imshow()` — displaying images, CNN feature maps, attention maps
- `plt.subplot()` — comparing predictions vs ground truth
- Heatmaps — confusion matrices, correlation matrices, attention weights
- Saving figures — for papers and reports

**Practice:** Create a dashboard that shows: loss curve, accuracy curve, confusion matrix, sample predictions

---

## PHASE 5: Classical ML with Scikit-learn (Days 12-15) — "Before Deep, Learn Shallow"
> Maps to Book Chapters 8-12

### Day 12: Data Pipeline
**Book chapters:** Ch 8 (Training & Testing), Ch 12 (Managing Data)

**What to learn:**
- `train_test_split()` — the very first step
- `StandardScaler`, `MinMaxScaler` — feature normalization
- `LabelEncoder`, `OneHotEncoder` — encoding categories
- Cross-validation with `cross_val_score()` — robust evaluation
- Data augmentation concepts — getting more from less data

**Practice:** Build a complete data pipeline: load → clean → split → scale → encode

---

### Day 13: Classifiers
**Book chapter:** Ch 11 (Classifiers)

**What to learn:**
- k-Nearest Neighbors — simplest classifier, understand distance metrics
- Decision Trees — visual, interpretable, understand information gain
- Random Forests — ensemble power (Ch 10)
- SVM — maximum margin classification, kernel trick
- Compare all classifiers on the same dataset

**Practice:** Train all 4 classifiers on Iris dataset, compare accuracies, plot decision boundaries

---

### Day 14: Gradient Descent Deep Dive
**Book chapters:** Ch 5 (Curves & Surfaces), Ch 14 (Backpropagation)

**What to learn:**
- Derivatives and gradients — the math behind learning
- Gradient descent from scratch — see the algorithm work
- Learning rate effects — too high = diverge, too low = slow
- Stochastic vs Mini-batch vs Batch gradient descent
- Visualize the loss landscape

**This is THE most important algorithm in all of deep learning.**

**Practice:** Implement gradient descent to fit a line to data, animate the convergence

---

### Day 15: Model Evaluation
**Book chapter:** Ch 9 (Overfitting & Underfitting)

**What to learn:**
- Overfitting: model memorizes training data (high train acc, low test acc)
- Underfitting: model too simple (low train acc, low test acc)
- Bias-variance tradeoff — the central tension in ML
- Learning curves — diagnose over/underfitting visually
- Regularization preview (L1, L2) — controlling model complexity

**Practice:** Deliberately overfit a model, then fix it with regularization. Plot learning curves.

---

## PHASE 6: Neural Networks from Scratch (Days 16-21) — "Build It to Understand It"
> Maps to Book Chapters 13-15. THIS IS WHERE IT GETS REAL.

### Day 16: The Perceptron & Activation Functions
**Book chapter:** Ch 13 (Neural Networks)

**What to learn:**
- Single neuron: `output = activation(w·x + b)` — code this!
- Activation functions: Sigmoid, Tanh, ReLU, LeakyReLU, Softmax
- Plot each activation and its derivative
- Why ReLU dominates (vanishing gradient problem)

**Practice:** Implement a perceptron that learns AND, OR, XOR gates

---

### Day 17-18: Forward Pass & Loss Functions
**Book chapters:** Ch 13-14

**What to learn:**
- Multi-layer network: stack neurons into layers
- Forward pass: data flows input → hidden → output
- MSE loss (regression), Cross-entropy loss (classification)
- Softmax + Cross-entropy = standard classification setup
- Implement forward pass for a 3-layer network

**Practice:** Build a forward-only network, compute loss on MNIST-like data

---

### Day 19-20: Backpropagation from Scratch
**Book chapter:** Ch 14 (Backpropagation) — THE MOST IMPORTANT CHAPTER

**What to learn:**
- Chain rule — how gradients flow backwards
- Computing gradients for each layer
- Weight updates: `w = w - lr * gradient`
- Implementing backprop step by step
- Numerical gradient checking (verify your math!)

**This is the hardest part. Take your time. Re-read Ch 14 multiple times.**

**Practice:** Full backprop implementation. Train a network to classify handwritten digits.

---

### Day 21: Optimizers, Regularization & Batch Norm
**Book chapter:** Ch 15 (Optimizers)

**What to learn:**
- SGD with momentum — ball rolling downhill
- Adam optimizer — the default choice in modern DL
- L2 regularization (weight decay) — penalize large weights
- Dropout — randomly kill neurons during training
- Batch normalization — normalize layer inputs

**Practice:** Compare SGD vs Adam on the same problem. Add dropout, see the effect.

---

## PHASE 7: Deep Learning with PyTorch (Days 22-30) — "Go Pro"
> Maps to Book Chapters 16-23. Everything comes together here.

### Day 22-23: PyTorch Fundamentals
**What to learn:**
- Tensors (like NumPy but with GPU support)
- `requires_grad=True` — automatic differentiation
- `nn.Module` — base class for ALL models
- `nn.Linear`, `nn.ReLU`, `nn.Sequential` — building blocks
- Training loop: forward → loss → backward → step → zero_grad

**Practice:** Rebuild your from-scratch network in PyTorch. Notice how much easier it is.

---

### Day 24-25: CNNs — Convolutional Neural Networks
**Book chapters:** Ch 16-17 (CNNs, Convnets in Practice)

**What to learn:**
- `nn.Conv2d` — the convolution operation
- Filters, stride, padding — controlling output size
- `nn.MaxPool2d` — downsampling feature maps
- Famous architectures: LeNet → AlexNet → VGG → ResNet
- Transfer learning with pretrained models

**Practice:** Build a CNN for CIFAR-10 image classification. Use transfer learning with ResNet.

---

### Day 26: Autoencoders & VAEs
**Book chapter:** Ch 18 (Autoencoders)

**What to learn:**
- Encoder-decoder architecture — compress then reconstruct
- Latent space — the compressed representation
- Variational Autoencoder — generating new data
- KL divergence loss + reconstruction loss

**Practice:** Build an autoencoder for MNIST. Visualize the latent space.

---

### Day 27: RNNs & LSTMs
**Book chapter:** Ch 19 (Recurrent Neural Networks)

**What to learn:**
- Sequential data processing — text, time series
- Vanishing gradient in RNNs — why plain RNNs fail
- LSTM gates: forget, input, output — long-term memory
- `nn.LSTM`, `nn.GRU` — PyTorch implementations
- Sequence-to-sequence models

**Practice:** Build a character-level text generator with LSTM

---

### Day 28-29: Attention & Transformers
**Book chapter:** Ch 20 (Attention and Transformers) — THE chapter for modern AI

**What to learn:**
- Attention mechanism — "which parts of the input matter?"
- Self-attention — every token attends to every other token
- Multi-head attention — multiple attention perspectives
- Positional encoding — since transformers have no position sense
- The Transformer architecture — encoder + decoder
- This is the architecture behind GPT, BERT, and ALL modern LLMs

**Practice:** Implement self-attention from scratch. Build a small Transformer.

---

### Day 30: GANs & Reinforcement Learning
**Book chapters:** Ch 21 (RL), Ch 22 (GANs)

**What to learn:**
- GAN: Generator vs Discriminator — adversarial training
- Mode collapse — the main challenge in GANs
- RL basics: agent, environment, reward, policy
- Q-learning — learning from rewards
- Policy gradients — optimizing actions directly

**Practice:** Build a simple GAN for generating MNIST digits. Implement Q-learning for CartPole.

---

## QUICK REFERENCE: What to Skip in Python

| Skip This | Why |
|-----------|-----|
| Web frameworks (Flask, Django) | Not ML |
| Database ORMs | Not ML |
| GUI programming (Tkinter) | Not ML |
| Complex regex | Rarely needed |
| Async/await | Not needed for ML |
| Package publishing | Not needed yet |
| Decorators (advanced) | Basics only |
| Threading/multiprocessing details | PyTorch handles this |

---

## TOOLS TO INSTALL

```bash
pip install numpy matplotlib scikit-learn torch torchvision
```

## DAILY ROUTINE

1. **30 min** — Read the matching book chapter
2. **90 min** — Code the practice exercises
3. **30 min** — Modify examples, experiment, break things
4. **10 min** — Write down what you learned (in your own words)

---

## Ready?

Tell me: **"Start Module 1"** and I'll generate the first practice file with detailed explanations and exercises.
