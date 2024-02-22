# Stochastic Simulations Methods

## Abstract
This repository provides Julia implementations of various stochastic simulation methods. It focuses on Stochastic Differential Equations and Monte Carlo Methods, offering tools for numerical integration and integration techniques in stochastic contexts. 

## Files Description

### Stochastic_Differential_Equations.jl
This file includes several functions for numerical integration in stochastic differential equations:
- `Euler_Maruyama`: Implements the Euler-Maruyama method for SDEs.
- `Heun`: Uses Heun's method for integrating ordinary differential equations with stochastic terms.
- `Heun_ou`: Adapts Heun's method for Ornstein-Uhlenbeck processes.
- `Heun_ou_trajs`: Generates multiple trajectories using Heun's method for Ornstein-Uhlenbeck processes.
- `nth_order`: Calculates nth order moments of trajectories.

### Monte_Carlo_Methods.jl
This file focuses on various Monte Carlo integration methods:
- `hit_and_miss_multi`: Implements Hit & Miss integration for multivariable functions.
- `uniform_sampling_multi`: Performs uniform sampling integration for multivariable functions.
- `importance_sampling`: Conducts importance sampling integration.
- `Correlations`: Calculates correlations for a given function.
- `calculate_errors`: Assesses errors in correlation time estimation.
