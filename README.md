# Tutorial - Hybrid AI for PDE Modeling

<img width="664" height="276" alt="image" src="https://github.com/user-attachments/assets/8255a980-ae66-4170-a54a-fdb1b5596697" />


## Overview

In this practical session, we study a supervised learning problem at the intersection of numerical simulation and machine learning.

We consider a **2D Poisson problem** on the **unit square**. For each simulation, the geometry and discretization remain fixed, while the **source term varies** from one sample to another. The objective is to learn a model that predicts the solution field from the input forcing term.

More precisely, we work with a family of problems of the form

$$
-\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = f(x,y)
\quad \text{in } \Omega = [0,1]^2
$$

with homogeneous Dirichlet boundary conditions $u(x,y)=0 \quad \forall x,y\in \partial\Omega$, and with source terms varying across simulations according to parameters such as

$$
f(x,y) = x \sin(a\pi y) + y \sin(b\pi x).
$$

The dataset is generated numerically on a common uniform mesh. Your role is to build and compare machine learning models that map the input fields to the output solution fields.

<img width="1001" height="384" alt="image" src="https://github.com/user-attachments/assets/e6d91d49-3f53-4453-89b4-9870e170b791" />



This tutorial is part of a broader reflection on **hybrid AI**, from general-purpose architectures to more physically informed models. The course material motivates the distinction between observational, inductive, and learning biases, and highlights approaches such as CNNs, PINNs, and neural operators. You can explore the three possible ways to inform models, where inductive and learning biases are the most natural.

---

## Objective

The goal is simple:

> **Design the best possible predictive model for this PDE on the private test data.**

You must begin with **standard, non-informed models**, then move toward **more structured or physics-aware methods**.

This tutorial is therefore organized as a progression:

1. **Start with generic machine learning baselines**
2. **Assess their strengths and limits**
3. **Introduce more sophisticated models**
4. **Discuss what is gained by adding inductive bias or physical structure or more**

---

## Expected task

You are given a dataset of input/output pairs:

- **Input:** data describing the forcing term $f(x,y)$ (or any other useful information provided in the training folder...).
- **Output:** the numerical solution $u(x,y)$ on the grid

Your objective is to learn the mapping

$$
\text{inputs} \longmapsto u(x,y)
$$

and to obtain the **best generalization performance on the separate private test set**.

---

## Dataset

There are two folders in the repository: `.\training`, where :

- `source.npy` contains the forcing term $f$ on the mesh
- `u_full.npy` is the target
- `x_grid.py` and `y_grid.npy` are the mesh coordinates
- the parameters $a$ and $b$ for each simulation are not known !

The `.\test_private` folder is built as the training one. It contains the private test data used to evaluate the models. 

Each dataset contains simulations with distinct pairs of $(a,b)$ parameters, solved numerically. The private test set is generated with a different distribution of $(a,b)$ parameters from the training set, so models should be able to generalize well, thus should understand the underlying physics...

> ⚠️ Models must be trained on the training data! Once trained, they are run on the unseen private test data ⚠️

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

<img width="3738" height="1905" alt="diverse-backbone" src="https://github.com/user-attachments/assets/775c8be4-9917-4118-a672-dcffa5467f5d" />


---

## Then: explore more advanced methods

After the two mandatory baselines, you must test at least **one more sophisticated approach**.

Possible directions include:

### Physics-informed or hybrid approaches
- adding a **PDE residual term** to the loss,
- penalizing boundary-condition violations,
- using automatic differentiation when relevant,
- building a **hybrid loss** combining data fitting and physical consistency.

<img width="750" height="352" alt="image" src="https://github.com/user-attachments/assets/6ff31216-8a22-4b64-92aa-84918afe3f32" />




### Operator-learning approaches
- Neural Operator such as the Fourier version (FNO),
- other neural operator variants,
- models designed to learn mappings between fields rather than vectors.

<img width="3799" height="1251" alt="image" src="https://github.com/user-attachments/assets/da1fa777-9508-44ce-855a-d1e6fde99383" />

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

