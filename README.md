# The Backtracking Dynamical Cavity Method

This is the code accompanying the paper ''The Backtracking Dynamical Cavity Method'' [arxiv](TOLINK) by Freya Behrens, Barbora Hudcová and Lenka Zdeborová.

## Usage
This repository provides a solver for the BDCM for $(p/c)$ backtracking attractors where the update rule is homogenous and independent of the neighbours ordering.
We also provide the functionalities to compute numerical simulations.
All necessary code can be found in the `src` directory.

For the usage of the solver and empirics, see some examples from the paper in
- [Application A - Limiting energy of a quench.ipynb](Application A - Limiting energy of a quench.ipynb)
- [Application B - Dynamical Phase transitions for majority dynamics.ipynb](Application B - Dynamical Phase transitions for majority dynamics.ipynb)
These finish their computation in reasonable time ($p=1$). Note that for larger $p$, the time and memory scaling exponential, so it will take much longer (and quickly forever) to compute.


## Naming Scheme for Update Rules
In addition to the update funtions considered in the paper's examples, this solver is capable of handling all totalistic rules for the state space $S = \{\pm1\}$.
To run the `run_bdcm` function in `src/bdcm/fixed_point_iteartion.py`, the rule needs to be given as a code.
Internally, the states are represented as $S \in \{0,1\}$ as opposed to $S \in  \{\pm 1\}$ from the paper.
In this case an update rule $f(x,k): S \times {0,1,...,d} \to S$, where $x$ is the state of the center and $k$ is the sum of the state of the $d$ neighbors, can be decomposed into functions that take a fixed entry $k\in {0,1,...,d}$.
These functions can be one of four types:
1) *0*: $f_k(x) = 0$
2) *1*: $f_k(x) = 1$
3) *+*: $f_k(x) = x$
4) *-*: $f_k(x) = 1 - x$

The string representation is then a concatanation of the reprentations for the $(f_0,f_1,...,f_d) = $ *0,1,+,-*.

In this context the antiferromagnet fast quench is *11+00*.
The $d=5$ majority rule is *000111*, for $d=4$ the always-stay majority rule is *00+11* and the always-change version is *00-11*.

