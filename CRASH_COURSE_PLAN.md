# Deep Learning Crash Course — Study Plan

**Book:** "Deep Learning Crash Course" by Volpe et al. (683 pages, No Starch Press)
**Prereqs:** Our current 30-day curriculum (Modules 1-7)
**Format:** Read chapter → code along → complete projects → practice problems
**Estimated pace:** ~45 days (2-3 topics per day, 3 days/week for projects)

---

## Phase 1: Neural Network Foundations (Days 1-8)
> Builds on our Module 6 (neural nets from scratch) — now with PyTorch

### Week 1-2: Dense Networks
| Day | Chapter | Topics | Project |
|-----|---------|--------|---------|
| 1 | Ch 1 | Single neuron, 1D/2D classification, bias, activation functions | — |
| 2 | Ch 1 | Two-layer & three-layer networks, backpropagation | — |
| 3 | Ch 1 | **Project 1A:** Classify MNIST handwritten digits | MNIST |
| 4 | Ch 2 | Regression (single neuron → multi-layer), mini-batch training | — |
| 5 | Ch 2 | Train/validation split, tracking metrics | — |
| 6 | Ch 2 | **Project 2A:** Emulate a physical system (digital twin) | Physics sim |
| 7-8 | — | Review & practice: build your own classifier + regressor from scratch | — |

---

## Phase 2: Computer Vision (Days 9-18)
> CNNs, autoencoders, U-Nets — the core of image-based AI

### Week 3-4: CNNs & Autoencoders
| Day | Chapter | Topics | Project |
|-----|---------|--------|---------|
| 9 | Ch 3 | 1D/2D convolutions, PyTorch layers (conv, pool, upsample) | — |
| 10 | Ch 3 | **Project 3A:** Classify malaria-infected blood smears | Medical imaging |
| 11 | Ch 3 | **Project 3B:** Localize microscopic particles | Object detection |
| 12 | Ch 3 | **Project 3C:** DeepDreams + **Project 3D:** Style transfer | Creative AI |
| 13 | Ch 4 | Encoder-decoders, denoising autoencoders | — |
| 14 | Ch 4 | **Project 4A:** Generate images (VAE) + **4B:** Morph images (WAE) | Generative |
| 15 | Ch 4 | **Project 4C:** Detect anomalies in ECG data | Healthcare |
| 16 | Ch 5 | U-Net architecture, semantic segmentation | — |
| 17 | Ch 5 | **Project 5A:** Detect quantum dots + **5B:** Count cells | Bio imaging |
| 18 | — | Review & practice: build a CNN pipeline for a dataset of your choice | — |

---

## Phase 3: Sequences & Language (Days 19-28)
> RNNs, attention, transformers — the tech behind ChatGPT

### Week 5-6: RNNs & Transformers
| Day | Chapter | Topics | Project |
|-----|---------|--------|---------|
| 19 | Ch 6 | Self-supervised learning (contrastive, non-contrastive, geometric) | — |
| 20 | Ch 6 | **Project 6A:** Localize mouse stem cells with LodeSTAR | Bio |
| 21 | Ch 7 | RNNs, GRUs, LSTMs — understanding recurrence | — |
| 22 | Ch 7 | Temperature prediction with recurrent networks | Time series |
| 23 | Ch 7 | **Project 7A:** Build a text translator (seq2seq) | NLP |
| 24 | Ch 8 | Attention mechanism, dot-product attention, multi-head attention | — |
| 25 | Ch 8 | **Project 8A:** Improve translator with attention | NLP |
| 26 | Ch 8 | **Project 8B:** Sentiment analysis with transformer | NLP |
| 27 | Ch 8 | **Project 8C:** Image classification with Vision Transformer (ViT) | Vision + NLP |
| 28 | — | Review & practice: fine-tune a transformer on your own dataset | — |

---

## Phase 4: Generative AI (Days 29-36)
> GANs & diffusion models — how DALL-E and Stable Diffusion work

### Week 7-8: Generative Models
| Day | Chapter | Topics | Project |
|-----|---------|--------|---------|
| 29 | Ch 9 | GAN architecture — generator vs discriminator, training loop | — |
| 30 | Ch 9 | **Project 9A:** Conditional GAN (generate specific digits) | Generative |
| 31 | Ch 9 | **Project 9B:** Virtual staining + **9C:** CycleGAN image conversion | Medical + creative |
| 32 | Ch 10 | Diffusion models — forward/reverse process, math intuition | — |
| 33 | Ch 10 | Generate MNIST digits with diffusion model | Generative |
| 34 | Ch 10 | **Project 10A:** Bespoke digits + **10B:** Text-to-image generation | Text-to-image |
| 35 | Ch 10 | **Project 10C:** Super-resolution images | Enhancement |
| 36 | — | Review & practice: train a GAN or diffusion model on custom data | — |

---

## Phase 5: Advanced Topics (Days 37-45)
> Graph networks, active learning, reinforcement learning, reservoir computing

### Week 9-10: Cutting Edge
| Day | Chapter | Topics | Project |
|-----|---------|--------|---------|
| 37 | Ch 11 | Graph convolutions, message passing | — |
| 38 | Ch 11 | **Project 11A:** Simulate physical phenomena | Physics |
| 39 | Ch 11 | **Project 11B:** Identify cell trajectories | Bio |
| 40 | Ch 12 | Active learning — uncertainty sampling, query strategies | — |
| 41 | Ch 12 | **Project 12A:** MNIST with active learning | Efficiency |
| 42 | Ch 13 | Reinforcement learning, Q-learning | — |
| 43 | Ch 13 | Deep Q-learning — teach an agent to play Tetris | Gaming AI |
| 44 | Ch 14 | Reservoir computing — predict chaotic systems (Lorenz attractor) | Physics |
| 45 | — | Final review & capstone: pick a real-world problem, build end-to-end | Portfolio |

---

## Summary

| Phase | Days | Chapters | Key Skills |
|-------|------|----------|------------|
| 1. NN Foundations | 1-8 | Ch 1-2 | PyTorch basics, classification, regression |
| 2. Computer Vision | 9-18 | Ch 3-5 | CNNs, autoencoders, U-Nets, medical imaging |
| 3. Sequences & Language | 19-28 | Ch 6-8 | RNNs, transformers, NLP, ViT |
| 4. Generative AI | 29-36 | Ch 9-10 | GANs, diffusion models, text-to-image |
| 5. Advanced Topics | 37-45 | Ch 11-14 | Graphs, active learning, RL, reservoir computing |

**Total projects:** 20+ hands-on projects with real datasets
**Interview-ready skills:** PyTorch, CNNs, Transformers, GANs, Diffusion, RL
