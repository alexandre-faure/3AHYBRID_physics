# Tutorial - Hybrid AI for PDE Modeling

## Overview

In this practical session, we study a supervised learning problem at the intersection of numerical simulation and machine learning.

We consider a **2D Poisson problem** on the **unit square**. For each simulation, the geometry and discretization remain fixed, while the **source term varies** from one sample to another. The objective is to learn a model that predicts the solution field from the input forcing term.

More precisely, we work with a family of problems of the form

$$
-\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = f(x,y)
\quad \text{in } \Omega = [0,1]^2
$$

with homogeneous boundary conditions, and with source terms varying across simulations according to parameters such as

$$
f(x,y) = x \sin(a\pi y) + y \sin(b\pi x).
$$

The dataset is generated numerically on a common uniform mesh. Your role is **not** to regenerate the data, but to build and compare machine learning models that map the input fields to the output solution fields.

This TD is part of a broader reflection on **hybrid AI**, from general-purpose architectures to more physically informed models. The course material motivates the distinction between observational, inductive, and learning biases, and highlights approaches such as CNNs, PINNs, and neural operators.

---

## Pedagogical objective

The goal is simple:

> **Design the best possible test-time predictive model for this PDE surrogate task.**

You must begin with **standard, non-informed models**, then move toward **more structured or physics-aware methods**.

This TD is therefore organized as a progression:

1. **Start with generic machine learning baselines**
2. **Assess their strengths and limits**
3. **Introduce more sophisticated models**
4. **Discuss what is gained by adding inductive bias or physical structure**

---

## Expected task

You are given a dataset of input/output pairs:

- **Input:** data describing the forcing term $f(x,y)$ (or additional information...).
- **Output:** the numerical solution $u(x,y)$ on the grid

Your objective is to learn the mapping

$$
\text{inputs} \longmapsto u(x,y)
$$

and to obtain the **best generalization performance on the separate test set**.

---

## Dataset

There are two folders in the repository: `.\training` where :

- `source.npy` contains the forcing term $f$ on the mesh
- `u_full.npy` is the target
- `x_grid.py` and `y_grid.npy` are the mesh coordinates
- `params.npy` contains the parameters $a$ and $b$ for each simulation.

The `.\test` folder is built as the training one. It contains the test data used to evaluate the models. 

---

## Mandatory models

The following two models are **mandatory** and must be implemented, trained, and evaluated:

### 1. MLP baseline (non-informed)

A fully connected neural network that ignores most of the spatial structure of the problem.

Typical possibilities:
- flatten the input grid and predict the flattened output grid,
- or predict from pointwise values $(x,y)\to u(x,y)$.

This model serves as a **generic baseline** to test the universal approximation theorem.

### 2. CNN baseline (non-informed, but spatially structured)

A convolutional neural network operating on the grid.

The CNN is still **not physics-informed**, but it does exploit the fact that the data lives on a regular 2D mesh. It should therefore provide a stronger baseline than the MLP in most cases.

---

## Then: explore more advanced methods

After the two mandatory baselines, you must test at least **one more sophisticated approach**.

Possible directions include:

### Physics-informed or hybrid approaches
- adding a **PDE residual term** to the loss,
- penalizing boundary-condition violations,
- using automatic differentiation when relevant,
- building a **hybrid loss** combining data fitting and physical consistency.

### Operator-learning approaches
- Neural Operator such as the Fourier version (FNO),
- other neural operator variants,
- models designed to learn mappings between fields rather than vectors.

---

## What you should analyze

Your work should not consist only of training models. You are also expected to analyze the results.

In particular, discuss:

- the relative performance of the **MLP** and the **CNN**,
- whether spatial inductive bias helps,
- the nature of prediction errors,
- whether the predictions are smooth, physically plausible, or noisy,
- whether a more informed model improves generalization,
- the trade-off between simplicity, performance, and physical consistency.

Visual comparisons between target and prediction are strongly encouraged.

---

## Suggested evaluation criteria

You may use several complementary metrics, for example:

- **MSE** or **RMSE**
- **relative \(L^2\) error**
- **MAE**
- visual comparison of predicted vs true fields
- error maps
- optional: PDE residual or boundary-condition violation

Be explicit about:
- your train/validation/test split,
- your preprocessing choices,
- your normalization strategy,
- your model-selection protocol.

---

## Deliverables

You are expected to submit:

1. **A notebook or script** containing your experiments
2. **A short report** summarizing:
   - the methods tested,
   - the training setup,
   - the evaluation protocol,
   - the results,
   - your interpretation
3. **Figures** showing representative predictions

---

## Recommended workflow

A good strategy is:

1. Load and understand the dataset
2. Visualize several samples
3. Implement the **MLP baseline**
4. Implement the **CNN baseline**
5. Compare them carefully
6. Propose and test a more advanced model
7. Analyze whether adding structure or physics improves performance

---

## Repository structure

A possible repository organization is:

```text
.
├── README.md
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/
│   └── td_hybrid_ai.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
└── results/
    ├── figures/
    └── logs/
