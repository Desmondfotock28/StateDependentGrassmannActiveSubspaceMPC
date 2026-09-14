# State-Dependent Grassmann MPC

**Learning state-dependent low-dimensional control subspaces for reduced-dimensional nonlinear model predictive control.**

## Overview

Nonlinear Model Predictive Control (NMPC) provides a powerful framework for controlling nonlinear systems under state and input constraints. However, its online computational cost can become significant when the prediction horizon and control dimension are large.

This project investigates a different approach to reducing the NMPC optimization dimension.

Instead of optimizing the complete open-loop control sequence

$$
U^\star(x) = \arg\min_U J_N(x,U)
$$

we optimize only within a low-dimensional, **state-dependent control subspace**.

The central idea is

$$
\boxed{
U = T_1(x;\theta)v
}
$$

where

$$
T_1(x;\theta)\in\mathbb{R}^{N_u\times n_v},
\qquad
n_v \ll N_u,
$$

is an orthonormal basis generated as a function of the current state, and

$$
v\in\mathbb{R}^{n_v}
$$

contains the reduced NMPC decision variables.

The neural network therefore does **not** directly generate the control action.

Instead,

> **the neural network determines where NMPC should optimize, while NMPC determines the optimal control within that subspace.**

---

## Motivation

Consider the nominal NMPC problem

$$
U^\star(x) = \arg\min_U J_N(x,U)
$$

subject to the nonlinear system dynamics

$$
x_{i+1}=f(x_i,u_i),
$$

state constraints

$$
x_i\in\mathcal X,
$$

and input constraints

$$
u_i\in\mathcal U.
$$

For a prediction horizon $N$ and $n_u$ control inputs,

$$
U =
\begin{bmatrix}
u_0^\top &
u_1^\top &
\cdots &
u_{N-1}^\top
\end{bmatrix}^\top
\in\mathbb R^{Nn_u}.
$$

The optimization dimension therefore grows with both the horizon and number of inputs.

A conventional fixed reduced-order parameterization assumes

$$
U=T_1v,
$$

where $T_1$ is a fixed low-dimensional basis.

However, for nonlinear systems, the important directions in the open-loop control space may depend strongly on the current operating state.

This motivates replacing the global basis $T_1$ with

$$
\boxed{
T_1=T_1(x;\theta).
}
$$

The resulting controller learns a mapping from the system state to an appropriate low-dimensional NMPC optimization subspace.

---

## State-Dependent Control Subspaces

Let

$$
T_1(x;\theta) \in \mathbb R^{N_u\times n_v}
$$

satisfy

$$
T_1(x;\theta)^\top T_1(x;\theta)=I_{n_v}.
$$

The columns of $T_1$ form an orthonormal basis for the active control subspace

$$
\mathcal{S}_a(x) = \mathrm{span}(T_1(x;\theta)).
$$

Computationally,

$$
T_1(x;\theta)\in\mathrm{St}(n_v,N_u),
$$

where $\mathrm{St}(n_v,N_u)$ denotes the Stiefel manifold of $N_u\times n_v$ matrices with orthonormal columns.

However, the physical object of interest is the **subspace**, rather than a particular basis.

For any orthogonal matrix

$$
R\in O(n_v),
$$

the matrices

$$
T_1
\qquad\text{and}\qquad
T_1R
$$

span the same control subspace.

Indeed,

$$
T_1R(R^\top v)=T_1v.
$$

Therefore, the state-dependent active control subspace can be viewed as an element of the Grassmann manifold

$$
\boxed{
\mathcal S_a(x)\in\mathrm{Gr}(n_v,N_u).
}
$$

The corresponding orthogonal projector is

$$
\boxed{
P_a(x)=T_1(x)T_1(x)^\top.
}
$$

This representation is invariant to rotations of the basis.

---

## Neural State-to-Subspace Mapping

The proposed architecture learns

$$
\boxed{
x
\longmapsto
\mathcal S_a(x).
}
$$

A neural network parameterized by $\theta$ first produces an unconstrained matrix

$$
A_\theta(x)\in\mathbb R^{N_u\times n_v}.
$$

An orthonormalization or manifold mapping is then used to obtain

$$
T_1(x;\theta)
=
\mathcal O(A_\theta(x)),
$$

such that

$$
T_1(x;\theta)^\top T_1(x;\theta)=I.
$$

The computational pipeline is therefore

$$
x
\longrightarrow
A_\theta(x)
\longrightarrow
T_1(x;\theta)
\longrightarrow
\mathcal S_a(x).
$$

Possible implementations of $\mathcal O(\cdot)$ include:

- reduced QR decomposition,
- polar decomposition / polar retraction,
- Stiefel-manifold retractions,
- geometry-aware parameterizations.

---

## Reduced NMPC

At each sampling instant $k$, the current state $x_k$ is passed through the learned state-to-subspace mapping:

$$
T_{1,k}=T_1(x_k;\theta).
$$

Instead of optimizing the full control sequence $U$, NMPC solves

$$
\boxed{
v_k^\star
=
\arg\min_{v\in\mathbb R^{n_v}}
J_N(x_k,T_{1,k}v)
}
$$

subject to the original system dynamics and constraints.

The full predicted control trajectory is reconstructed as

$$
\boxed{
U_k^\star=T_{1,k}v_k^\star.
}
$$

Only the first control input is applied to the system,

$$
u_k=u_{0|k}^\star,
$$

and the procedure is repeated at the next sampling instant.

The online architecture is therefore

```text
                 Current state
                      x(k)
                       |
                       v
             +-------------------+
             |  Subspace Network |
             |      f_theta      |
             +-------------------+
                       |
                       v
                   T1(x(k))
                       |
                       v
             +-------------------+
             |   Reduced NMPC    |
             |                   |
             |  optimize v only  |
             +-------------------+
                       |
                       v
              U* = T1(x(k)) v*
                       |
                       v
               Apply first input
```

---

## Reinforcement Learning Formulation

The neural network is not trained to directly approximate the optimal control action.

Instead, reinforcement learning is used to learn the parameters

$$
\theta
$$

of the state-to-subspace mapping

$$
T_1(x;\theta).
$$

The learning problem can be interpreted as

$$
\boxed{
\text{learn which control subspace NMPC should search from each state.}
}
$$

At state $x_k$, the network generates

$$
T_{1,k}=T_1(x_k;\theta),
$$

the reduced NMPC computes

$$
v_k^\star,
$$

and the resulting control is

$$
U_k^\star=T_{1,k}v_k^\star.
$$

After applying the first control input, the system transitions to

$$
x_{k+1}=f(x_k,u_k).
$$

A reward can be constructed from the control objective, for example

$$
r_k=-\ell(x_k,u_k).
$$

A training transition may therefore contain

$$
(x_k,\mathcal S_a(x_k),r_k,x_{k+1}).
$$

The RL objective is to learn

$$
\theta^\star
$$

that maximizes expected closed-loop performance:

$$
\boxed{
\theta^\star
=
\arg\max_\theta
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_k
\right].
}
$$

---

## Actor Interpretation

The actor is fundamentally different from a conventional continuous-control RL policy.

A conventional actor learns

$$
x\longmapsto u.
$$

Here, the actor learns

$$
\boxed{
x\longmapsto\mathcal S_a(x).
}
$$

The actual control is subsequently obtained from model-based optimization:

$$
\mathcal S_a(x)
\longrightarrow
\text{Reduced NMPC}
\longrightarrow
u.
$$

Thus,

```text
Conventional RL:

state ---> neural policy ---> control


Proposed approach:

state ---> neural subspace policy ---> optimization subspace
                                      |
                                      v
                                Reduced NMPC
                                      |
                                      v
                                    control
```

This preserves an explicit model-based optimization layer while using learning to adapt its optimization coordinates.

---

## Geometric Learning

Since the basis satisfies

$$
T_1^\top T_1=I,
$$

learning must respect the geometry of orthonormal matrices.

Given a Euclidean gradient

$$
G=\nabla_{T_1}L,
$$

a tangent-space projection can be constructed as

$$
\Pi_{T_1}(G)
=
G-
T_1\,\mathrm{sym}(T_1^\top G),
$$

where

$$
\mathrm{sym}(A)
=
\frac{1}{2}(A+A^\top).
$$

The resulting direction belongs to the tangent space of the Stiefel manifold.

A retraction can subsequently map the updated point back onto the manifold.

This project will investigate both:

1. differentiable orthonormalization inside the neural network, and
2. explicit Riemannian / manifold-aware optimization.

---

## Grassmann-Invariant Representation

Because

$$
T_1
\quad\text{and}\quad
T_1R
$$

represent the same subspace, quantities used for subspace comparison should ideally be invariant to the choice of basis.

The projector

$$
P_a=T_1T_1^\top
$$

provides such a representation.

For example, changes between consecutive state-dependent subspaces can be measured using

$$
d_k
=
\left\|
P_a(x_{k+1})-P_a(x_k)
\right\|_F.
$$

This avoids interpreting simple rotations of an equivalent basis as changes in the actual control subspace.

A smoothness regularizer may therefore be considered:

$$
L_{\mathrm{smooth}}
=
\left\|
P_a(x_{k+1})-P_a(x_k)
\right\|_F^2.
$$

---

## Sensitivity Structure

The learning architecture introduces the computational chain

$$
\boxed{
\theta
\rightarrow
T_1(x;\theta)
\rightarrow
v^\star
\rightarrow
U=T_1v^\star
\rightarrow
J.
}
$$

At a high level, the sensitivity with respect to the neural-network parameters follows

$$
\nabla_\theta J
=
\left(
\frac{\partial T_1(x;\theta)}
{\partial\theta}
\right)^*
\nabla_{T_1}J.
$$

For

$$
U=T_1v,
$$

a perturbation in the basis produces

$$
dU=dT_1\,v.
$$

Consequently, for a differentiable objective,

$$
\nabla_{T_1}J
=
\nabla_UJ\,v^\top,
$$

subject to the appropriate treatment of the constrained NMPC solution and its Lagrangian sensitivities.

The interaction between reinforcement learning, manifold geometry, and parametric NMPC sensitivity is a central topic of this project.

---

## Research Questions

This repository is intended to investigate several questions:

1. Can a state-dependent control subspace achieve better NMPC performance than a single globally learned subspace at the same reduced dimension $n_v$?

2. How small can $n_v$ become while retaining performance close to full-dimensional NMPC?

3. Can reinforcement learning learn a smooth mapping $x \mapsto \mathcal{S}_a(x)$ over the relevant state space?

4. What is the most effective representation for learning:
   Stiefel bases, Grassmann projectors, or manifold retractions?

5. How should the RL critic represent a subspace while remaining invariant to equivalent basis rotations?

6. How does state-dependent subspace adaptation affect closed-loop stability, constraint satisfaction, and computational cost?

7. How does the method compare with fixed PCA, fixed learned active subspaces, and full NMPC?

---

## Planned Baselines

The initial experimental study will compare:

**Full NMPC**

$$
U^\star=\arg\min_UJ_N(x,U).
$$

**Fixed PCA subspace**

$$
U=T_{\mathrm{PCA}}v.
$$

**Fixed learned active subspace**

$$
U=T_1^\star v.
$$

**State-dependent learned subspace**

$$
\boxed{
U=T_1(x;\theta^\star)v.
}
$$

The comparison will focus on:

- closed-loop cost,
- NMPC solution time,
- constraint satisfaction,
- tracking performance,
- reduced dimension $n_v$,
- convergence behavior,
- variation of the learned subspace over the state space.

---

## Initial Implementation Plan

```text
Phase 1
-------
Implement full NMPC baseline.

Phase 2
-------
Implement fixed reduced NMPC:

    U = T1 v

and verify the reduced optimization.

Phase 3
-------
Implement:

    x -> NN -> A(x) -> orthonormalization -> T1(x)

and verify:

    ||T1(x)^T T1(x) - I||_F ≈ 0.

Phase 4
-------
Integrate state-dependent T1(x) with the NMPC solver.

Phase 5
-------
Implement the RL training architecture.

Phase 6
-------
Introduce Grassmann-invariant subspace metrics and
state-to-state smoothness regularization.

Phase 7
-------
Benchmark against full NMPC, PCA, and fixed learned
active-subspace NMPC.
```

---

## Repository Structure

```text
StateDependentGrassmannMPC/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
│
├── src/
│   ├── models/
│   │   └── subspace_network.py
│   │
│   ├── geometry/
│   │   ├── stiefel.py
│   │   └── grassmann.py
│   │
│   ├── mpc/
│   │   ├── full_nmpc.py
│   │   └── reduced_nmpc.py
│   │
│   ├── rl/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── replay_buffer.py
│   │
│   └── utils/
│
├── experiments/
├── scripts/
├── tests/
└── docs/
```

---

## Current Status

**Research in progress.**

The mathematical formulation, learning architecture, manifold representation, and RL sensitivity analysis are currently under development.

Experimental results will be added as the implementation progresses.

---

## Core Idea

The project can be summarized by three equations:

$$
\boxed{
x
\longmapsto
T_1(x;\theta),
\qquad
T_1^\top T_1=I
}
$$

$$
\boxed{
v^\star
=
\arg\min_v
J_N(x,T_1(x;\theta)v)
}
$$

$$
\boxed{
U^\star
=
T_1(x;\theta)v^\star.
}
$$

In words:

> **Learn the state-dependent optimization subspace; let NMPC optimize the control within it.**

---

## References

Relevant literature on optimization with orthogonality constraints, Stiefel/Grassmann manifolds, nonlinear model predictive control, reinforcement learning, and subspace learning will be added as the project develops.
