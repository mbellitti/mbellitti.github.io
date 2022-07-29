---
title:  "Kernels and Propagators"
date:  2020-06-15
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

A linear PDE looks like this

$$
L f(x,t) = J(x,t)
$$

where $$L$$ is a linear differential operator, $f$ is the unknown function and $J$ is a source term. Let's not worry about smoothness issues.

Inspired by [this](https://physics.stackexchange.com/questions/20797/differentiating-propagator-greens-function-correlation-function-etc) answer on stackexchange.

There are two related but distinct concepts:

- Kernel: used to solve the *homogeneous* equation
- Green function: used to solve the *driven* equation

Given a linear partial differential equation $$L f(x,t)$$, its kernel is most easily found by Laplace transform, while the Green function by Fourier transforming the impulse equation
$$
L G(x,t) = \delta(x) \delta(t)
$$
and solving for $G$, the Green function. Here $\delta$ is the Diract delta. Given an arbitrary source $J$ we get the solution $f$ of the original problem thanks to linearity: 
$$
f = G \star J
$$
where $\star$ is the convolution operator.
