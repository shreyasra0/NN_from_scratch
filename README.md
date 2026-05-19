# nn-from-scratch

A high-performance deep learning framework engineered entirely from first principles in Python and NumPy. This repository contains zero high-level deep learning library dependencies (no PyTorch, TensorFlow, or JAX). Every mathematical operation—from forward-pass tensor gating to backpropagation and adaptive weight decay—is written by hand.

> **Note on Engineering Rigor:** This framework was built on exact mathematical derivations and rigorous optimization mechanics. No vibe coding, no guessing hyperparameters, and no opaque abstractions. Every gradient calculation, moment vector, and weight decay step is derived explicitly from foundational machine learning theory.

---

## Core Architecture Elements

The framework implements a highly specialized, modern optimization pipeline designed to maximize gradient stability and convergence speed on tabular bio-acoustic regression tasks.

* **Layer & Activation Mechanics:** Supports fully connected `Dense` layers using He Initialization to prevent vanishing/exploding gradients, paired with continuous, non-linear `LeakyReLU` activations.
* **Optimization (AdamW):** Features a custom implementation of **AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)**. Unlike standard Adam, it decouples the $L_2$ regularization penalty entirely from the gradient variance updates, preventing the artificial amplification or suppression of structural weight penalties.
* **Inline Validation Engine:** The training loop handles runtime mini-batch stochastic shuffling and dynamically tracks both training and validation Mean Squared Error (MSE) metrics inline per epoch.

---

## Pipeline Architecture

The training pipeline routes data through a highly continuous optimization landscape:

```text
[Data Input] ──> [Dense + LeakyReLU] ──> [MSE Loss + L2 Penalty]
                         │                         │
                         │ (Backward Pass)         │ (Gradient)
                         ▼                         ▼
               [Evaluate dW & db] ────────> [AdamW Engine]
                                                   │
                                                   ▼
                                       (Decoupled Weight Decay)
                                       θ_t+1 = θ_t - α·(m_hat / (sqrt(v_hat) + ε) + λ·θ_t)