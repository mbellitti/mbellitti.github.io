---
title:  "Kernels and Propagators"
date:  2020-06-15
mathjax: true
---

- toc
{:toc}

A linear PDE looks like this

$$
L f(x,t) = J(x,t)
$$

where $$L$$ is a linear differential operator

Inspired by [this](https://physics.stackexchange.com/questions/20797/differentiating-propagator-greens-function-correlation-function-etc) answer on stackexchange.

There are two related but distinct concepts:

- Kernel: used to solve the *homogeneous* equation
- Green function: used to solve the *driven* equation

Given a linear partial differential equation $$L f(x,t)$$
