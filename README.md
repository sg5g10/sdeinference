# Differentiable Bayesian inference of SDE parameters using a pathwise series expansion of Brownian motion

This repository contains code that demonstrate the approximation of a SDE by an ODE for inference, supporting the above paper. 

## Dependencies
Generic scientific Python stack: `numpy`, `scipy`, `matplotlib`, `sklearn`, `seaborn`, `joblib`, and `arviz` (0.4.1).

To install `NumPyro` read the following:
http://num.pyro.ai/en/stable/getting_started.html#installation 

## Usage
To run the stochastic Lotka-Volterra model: 
`python lotkavolterra_example.py`

By default the number of particles for PMMH is set to `100`, to change this run with option:
`python lotkavolterra_example.py --pmmh_nparticles 100`





