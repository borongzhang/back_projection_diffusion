# Back-Projection Diffusion
This repository contains the code for the main models described in “Back-Projection Diffusion: Solving the Wideband Inverse Scattering Problem with Diffusion Models”, which is available on [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2025.118036). The paper is authored by [Borong Zhang](https://borongzhang.com/), [Martín Guerra](https://sites.google.com/wisc.edu/martinguerra/), [Qin Li](https://sites.google.com/view/qinlimadison/home), and [Leonardo Zepeda-Núñez](https://research.google/people/leonardozepedanez/?&type=google). This repository is written by Borong Zhang.

We present Wideband Back-Projection Diffusion, an end-to-end probabilistic framework for approximating the posterior distribution of the refractive index using the wideband scattering data through the inverse scattering map.

<img width="1228" alt="prelim" src="https://github.com/user-attachments/assets/e21c10e4-a451-424e-a7c2-a785a207a4d4" />

This framework produces highly accurate reconstructions, leveraging conditional diffusion models to draw samples, and also honors the symmetries of the underlying physics of wave-propagation. The procedure is factored into two steps, with the first, inspired by the filtered back-propagation formula, transforms data into a physics-based latent representation, while the second learns a conditional score function conditioned on this latent representation.

<img width="1121" alt="diagram" src="https://github.com/user-attachments/assets/add61dfe-a3cb-4d4f-8cb0-b923293d5134" />

These two steps individually obey their associated symmetries and are amenable to compression by imposing the rank structure found in the filtered back-projection formula. Empirically, our framework has both low sample and computational complexity, with its number of parameters scaling only sub-linearly with the target resolution, and has stable training dynamics. It provides sharp reconstructions effortlessly and is capable of recovering even sub-Nyquist features in the multiple-scattering regime.

## Enviroment Setup
Project Environment can be installed by 
```
conda create -n back_projection_diffusion python=3.11 
conda activate back_projection_diffusion
pip install git+https://github.com/borongzhang/back_projection_diffusion.git@main
pip install --upgrade "jax[cuda12]==0.5.0"
```

## Demos
Demos for `EquiNet-CNN`, `B-EquiNet-CNN`, `Analytical-CNN` and `SwitchNet-CNN` trained on the `10h Overlapping Squares` dataset can be found in the `examples` folder.  

### Sample Dataset and Trained Model Parameters
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14911327.svg)](https://doi.org/10.5281/zenodo.14911327)

We make a sample dataset and corresponding trained model parameters publicly available [via Zenodo](https://doi.org/10.5281/zenodo.14911327). To download the dataset and trained model parameters, simply run:
```
zenodo_get 14911327
```

After downloading the data, by default, please place the datasets in a `data` folder. Optionally, the `tmp` folder—which contains the trained model parameters—can be downloaded and placed in the `examples` folder.  If the trained model parameters in `tmp` are loaded, the training scripts will be automatically skipped.

To change the data directory, modify `training_data_path` and `test_data_path`. To change the directory for the trained model parameters, modify `cond_workdir`.
The following is the default directory structure for the `back_projection_diffusion` project:

```
back_projection_diffusion/
      ├── examples/
      │   └── tmp/                          # Contains trained model parameters
      │       ├── equinet_cnn_10hsquares/      
      │       ├── b_equinet_cnn_10hsquares/    
      │       ├── analytical_cnn_10hsquares/  
      │       └── switchnet_cnn_10hsquares/ 
      ├── data/
      │   ├── 10hsquares_trainingdata/      # Contains training data for 10h squares
      │   └── 10hsquares_testdata/          # Contains test data for 10h squares
      └── src/
```

## Datasets
Datasets can be generated using the MATLAB code in the `data_generation` folder. 

### Synthetic Perturbations
Perturbations `Shepp-Logan`, `3-5-10h Triangles`, and `10h Overlapping Squares` can be generated using the corresponding `eta_generation_?.m` scripts.

![synthetic_media](https://github.com/user-attachments/assets/d4fe637e-bf80-4a40-8678-af30a45ebf3a)

The resolution and number of perturbations should be specified in the `setup scaling parameters` section. The generated perturbations will be stored as an HDF5 file with the following structure:
```
eta.h5/
      ├── /eta
```

### MRI Brain Perturbations 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14760123.svg)](https://doi.org/10.5281/zenodo.14760123) 

The Brain MRI images used as our perturbations were obtained from the [NYU fastMRI Initiative database](https://fastmri.med.nyu.edu/), as described in the works of Florian Knoll et al., fastmri: A publicly available raw k-space and dicom dataset of knee images for accelerated mr image reconstruction using machine learning, 2020, and Jure Zbontar et al., fastmri: An open dataset and benchmarks for accelerated mri, 2019. As such, NYU fastMRI investigators provided the data but did not participate in the analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can be found at fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images.

![MRI Brain Samples](https://github.com/user-attachments/assets/49f9e96b-dfe1-4560-8a9e-9a824a866118) 

We padded, resized, and normalized the perturbations to a native resolution of $n_\eta = 240$ points. Then, we downsampled the perturbations to resolutions of $n_\eta = 60$, $80$, $120$, and $160$.

We make the processed MRI brain perturbations publicly available [via Zenodo](https://doi.org/10.5281/zenodo.14760123). The perturbations are stored as HDF5 files, with filenames in the format `eta-n.h5`, where `n` corresponds to the resolution. To download the dataset, please run:
```
zenodo_get 14760123
```

### Scattering Data Generation
Scattering data can be generated using the `data_generation.m` script.
The dimension, size, and frequencies of the data should be specified in the `setup scaling parameters` section. The generated scattering data will be stored as an HDF5 file with the following structure:
```
scatter.h5/
      ├── /scatter_imag_freq_1
      ├── /scatter_real_freq_1
      ├── /scatter_imag_freq_2
      ├── /scatter_real_freq_2
      ├── /scatter_imag_freq_3
      ├── /scatter_real_freq_3
```

To use, place the `eta.h5` and `scatter.h5` files into a folder, then move that folder to the `data` directory.

## Baseline Models
We have made the baseline deterministic models, and the U-ViT diffusion model publicly available in the [repository](https://github.com/borongzhang/ISP_baseline).

## Credits
This repository makes use of code from the following sources:
1. The code in this repository is largely based on [Swirl-Dynamics](https://github.com/google-research/swirl-dynamics).
2. [Random Shepp-Logan Phantom](https://github.com/matthiaschung/Random-Shepp-Logan-Phantom) by Matthias Chung, which was used for generating the `Shepp-Logan` dataset.
3. Original data generation code was provided by the authors of [Wide-Band Butterfly Network](https://epubs.siam.org/doi/10.1137/20M1383276).
4. The structure of the repository is inspired by Owen Melia and Olivia Tsang‘s [repository](https://github.com/meliao/MFISNets/).

## Citation
If this code is useful to your research, please cite our paper:
```
@article{ZHANG2025118036,
title = {Back-Projection Diffusion: Solving the wideband inverse scattering problem with diffusion models},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {443},
pages = {118036},
year = {2025},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2025.118036},
url = {https://www.sciencedirect.com/science/article/pii/S0045782525003081},
author = {Borong Zhang and Martin Guerra and Qin Li and Leonardo Zepeda-Núñez},
keywords = {Machine learning, Inverse scattering, Generative modeling, Wave propagation, Diffusion models},
abstract = {We present Wideband Back-Projection Diffusion, an end-to-end probabilistic framework for approximating the posterior distribution induced by the inverse scattering map from wideband scattering data. This framework produces highly accurate reconstructions, leveraging conditional diffusion models to draw samples, and also honors the symmetries of the underlying physics of wave-propagation. The procedure is factored into two steps: the first step, inspired by the filtered back-propagation formula, transforms data into a physics-based latent representation, while the second step learns a conditional score function conditioned on this latent representation. These two steps individually obey their associated symmetries and are amenable to compression by imposing the rank structure found in the filtered back-projection formula. Empirically, our framework has both low sample and computational complexity, with its number of parameters scaling only sub-linearly with the target resolution, and has stable training dynamics. It provides sharp reconstructions effortlessly and is capable of recovering even sub-Nyquist features in the multiple-scattering regime.}
}
```

