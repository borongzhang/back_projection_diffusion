# Back-Projection Diffusion
Code for the preprint "Back-Projection Diffusion: Solving the Wideband Inverse Scattering Problem with Diffusion Models", available on ArXiv [here](https://arxiv.org/abs/2408.02866). 

Written by [Borong Zhang](https://github.com/borongzhang), [Martín Guerra](https://sites.google.com/wisc.edu/martinguerra/), [Qin Li](https://sites.google.com/view/qinlimadison/home), and [Leonardo Zepeda-Núñez](https://research.google/people/leonardozepedanez/?&type=google).

## Installation
Project Environment can be installed by 
```
conda create -n jaxflax_isp python=3.11 
conda activate jaxflax_isp
pip install git+https://github.com/google-research/swirl-dynamics.git@main
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install jupyter matplotlib natsort 
```

## Data
After downloading the data, please put the data folder in the root directory of the project.

## Demos
Demos for these models can be found in the `colabs` folder.