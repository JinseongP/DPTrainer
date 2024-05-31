

## Overview

Official PyTorch implementation of "In-distribution Public Data Synthesis with Diffusion Models for Differentially Private Image Classification", CVPR 2024.

**[Jinseong Park](https://jinseongp.github.io/) \*, [Yujin Choi](https://scholar.google.com/citations?user=3u0-O2sAAAAJ&hl=en) \*, and Jaewook Lee**   
<sup> * Equal contribution </sup> 

| [paper](https://jinseongp.github.io/assets/files/CVPR2024_In-distribution%20Public%20Data%20Synthesis%20with%20Diffusion%20Models%20for%20Differentially%20Private%20Image%20Classification.pdf) | 



## Step-by-Step Algorithm

0. Environment configuration
1. Training EDM models with 4\% data or Download the generated images 
2. Train warm-up classifiers with standard training
3. DP-SGD



## 0. Environment configuration 

1. Create docker image (or corresponding virtual environment with cuda 11.8 and torch1.13.0)

     ```
   sudo docker run -i -t --ipc=host --name dptrainer--gpus=all anibali/pytorch:1.13.0-cuda11.8 /bin/bash
     ```

2. Install the required packages

     ```
   pip install -r requirements.txt
     ```

   


## 1. Training EDM model with 4\% data
**We provide the generated images with the EDM with 4\% of public data in  [DATADRIVE](https://drive.google.com/drive/folders/14Wsb0Fl7Kp3m12oPQ2-kcr_sf3UB27WE?usp=sharing).**

The number of weight in CIFAR-10 indicates the weight of discriminator in DG.

**Place the synthetic data and the indices for public data at the directory specified below.**

```
${project_page}/DPTrainer/
├── data 
│   ├── cifar-10-edm
│   |   ├── cifar10_data_sampled_index.pt
│   |   ├── cifar10_data_sampled_weight0.npz
│   |   ├── ...
│   ├── cifar-100-edm
├── ...
```

---

Otherwise, you can end-to-end train EDM models with 4\% data. 

### 0) Follow the requirements of EDM

 - Please refer to the official code of EDM: https://github.com/NVlabs/edm

   [Reference] Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." *Advances in Neural Information Processing Systems* 35 (2022): 26565-26577.

### 1) Prepare subsampled dataset
  - For CIFAR-10 dataset, Download **cifar10_data_sampled_4percent.zip** at [DATADRIVE](https://drive.google.com/drive/folders/14Wsb0Fl7Kp3m12oPQ2-kcr_sf3UB27WE?usp=sharing).
   - For CIFAR-100 dataset, Download **cifar100_data_sampled_4percent.zip** at [DATADRIVE](https://drive.google.com/drive/folders/14Wsb0Fl7Kp3m12oPQ2-kcr_sf3UB27WE?usp=sharing).
  - These zip files contain png images with balanced labels, and dataset.json 
  - Place zip files at the directory same as the train.py file of EDM.

### 2) Train EDM model
  - To train the EDM model with 4\% of CIFAR-10 dataset, run: 
  ```
  torchrun --standalone --nproc_per_node=4 train.py --outdir=training-runs --data=cifar10_data_sampled_4percent.zip --cond=1 --arch=ddpmpp 
  ```
  - To train the EDM model with 4\% of CIFAR-100 dataset, run: 
  ```
  torchrun --standalone --nproc_per_node=4 train.py --outdir=training-runs --data=cifar100_data_sampled_4percent.zip --cond=1 --arch=ddpmpp 
  ```

### 3) Generate EDM samples
  - To generate unconditional discriminator-guided 50k samples, run: 
  ```
  torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 --network=./training-runs/PATH/network.pkl
  ```

### 4) (Optionally) Discriminator Guidance 

  - Follow the instructions in https://github.com/alsdudrla10/DG with the trained EDM model and synthetic data.

    **Warning**: You need to train a discriminator (correspondingly classifier) based on the 4\% of public data.

    [Reference] Kim, Dongjun, et al. "Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models." *International Conference on Machine Learning*. PMLR, 2023.

    

## 2. Warm-up Training

Please refer to the `/examples/` folder in this repository.

Follow the instructions of `/examples/cifar10_warmup.ipynb` and  `/examples/cifar100_warmup.ipynb`.



## 3. Training EDM model with 4\% data

  ```
!python main.py --gpu {GPU} --max_grad_norm {MAX_GRAD_NORM} --epsilon {EPSILON} --delta {DELTA}  --data {DATA} --optimizer "{OPTIMIZER}" --epochs {EPOCHS} --batch_size {BATCH_SIZE} --max_physical_batch_size {MAX_PHYSICAL_BATCH_SIZE} --model_name {MODEL_NAME}  --n_class {N_CLASSES} --augmult {N_AUGMULT} --path {PATH} --name {NAME} --memo {MEMO} --public_batch_size {PUBLIC_BATCH_SIZE} --extender {EXTENDER}  --pretrained_dir {WARMUP_PATH}
  ```

For specific usage, follow the instructions of `/examples/cifar10_dpsgd.ipynb` and  `/examples/cifar100_dpsgd.ipynb`.

For details of each parameter, please refer to `main.py`. 



## 4. Citation

```
Will be updated

CVPR 2024
```



- The backbone trainer architecture of this code is based on [adversarial-defenses-pytorch](https://github.com/Harry24k/adversarial-defenses-pytorch). For better usage of the trainer, please refer to adversarial-defenses-pytorch. 

- Furthermore, we use [Opacus](https://github.com/pytorch/opacus) for differentially private training.