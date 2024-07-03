# EquiNet-CNN

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