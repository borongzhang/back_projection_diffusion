# Back-Projection Diffusion
Code for the preprint "Back-Projection Diffusion: Solving the Wideband Inverse Scattering Problem with Diffusion Models", available on ArXiv [here](https://arxiv.org/abs/2408.02866). 

Written by [Borong Zhang](https://borongzhang.github.io/), [Martín Guerra](https://sites.google.com/wisc.edu/martinguerra/), [Qin Li](https://sites.google.com/view/qinlimadison/home), and [Leonardo Zepeda-Núñez](https://research.google/people/leonardozepedanez/?&type=google).

We present Wideband back-projection diffusion, an end-to-end probabilistic framework for approximating the posterior distribution induced by the inverse scattering map from wideband scattering data. 

<img width="1228" alt="prelim" src="https://github.com/user-attachments/assets/e21c10e4-a451-424e-a7c2-a785a207a4d4" />

This framework leverages conditional diffusion models coupled with the underlying physics of wave-propagation and symmetries in the problem, to produce highly accurate reconstructions. The framework introduces a factorization of the score function into a physics-based latent representation inspired by the filtered back-propagation formula and a conditional score function conditioned on this latent representation. 

<img width="1133" alt="diagram" src="https://github.com/user-attachments/assets/a185576e-75b7-42d3-a83d-bf5e4e3fe0a5" />

These two steps are also constrained to obey symmetries in the formulation while being amenable to compression by imposing the rank structure found in the filtered back-projection formula. As a result, empirically, our framework is able to provide sharp reconstructions effortlessly, even recovering sub-Nyquist features in the multiple-scattering regime. It has low-sample and computational complexity, its number of parameters scales sub-linearly with the target resolution, and it has stable training dynamics.

## Enviroment Setup
Project Environment can be installed by 
```
conda create -n jaxflax_isp python=3.11 
conda activate jaxflax_isp
pip install git+https://github.com/google-research/swirl-dynamics.git@main
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install jupyter matplotlib natsort 
```

## Sample Data and Trained Model Parameters
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14745154.svg)](https://doi.org/10.5281/zenodo.14745154)

We make a sample dataset publicly available [via Zenodo](https://doi.org/10.5281/zenodo.14745154). 

After downloading the data, please put the datasets in the data folder, and put tmp folder in the src folder (optional).
The following is the directory structure for the `back_projection_diffusion` project:

```
back_projection_diffusion/
      ├── colab/
      ├── data/
      │   ├── 10hsquares_trainingdata/      # Contains training data for 10h squares
      │   └── 10hsquares_testdata/          # Contains test data for 10h squares
      ├── src/
      │   └── tmp/                          # Contains trained model parameters
      │       ├── equinet_cnn_10hsquares/      
      │       ├── b_equinet_cnn_10hsquares/    
      │       ├── analytical_cnn_10hsquares/  
      │       └── switchnet_cnn_10hsquares/    
```

## Demos
Demos for these models can be found in the `colabs` folder.

## Comments

The following files will be included in a future update:
```
1. Scripts for data generation
2. EquiNet-UViT, WideBNet-CNN, and etc 
3. Brain MRI dataset
```
We are also planning a revision to improve the code running speed.

## Citation

If this code is useful to your research, please cite our preprint:
```
@misc{zhang2024backprojectiondiffusionsolvingwideband,
      title={Back-Projection Diffusion: Solving the Wideband Inverse Scattering Problem with Diffusion Models}, 
      author={Borong Zhang and Martín Guerra and Qin Li and Leonardo Zepeda-Núñez},
      year={2024},
      eprint={2408.02866},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.02866}, 
}
```

