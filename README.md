## Docker run

sudo docker run -v /home/tako/disk/sdb/JS:/JS -i -t --ipc=host --name dptrainer --restart always -p 8888:8888 --gpus=all anibali/pytorch:1.13.0-cuda11.8 /bin/bash


## Training EDM model with 4\% data
You can end-to-end train EDM models with 4\% data. 
Otherwise, you can just download the generated data at [DATADIRVE](drive link).

### 1) Prepare subsampled dataset
  - For CIFAR-10 dataset, Download **cifar10_data_sampled_4percent.zip** at [DATADIRVE](drive link).
   - For CIFAR-100 dataset, Download **cifar100_data_sampled_4percent.zip** at [DATADIRVE](drive link).
  - These zip files contain png images with balanced labels, and dataset.json 
  - Place zip files at the directory, which placed train.py file.

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
  torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \
    --network=./training-runs/PATH/network.pkl
  ```

## Reference

 - *Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364.*



## How to run

The two key files are main.py and src/trainer/dptrainer.py
The command for run is 

python main.py --gpu {GPU} --max_grad_norm {MAX_GRAD_NORM} --epsilon {EPSILON} --delta {DELTA}  --data {DATA}\
    --optimizer "{OPTIMIZER}" --epochs {EPOCHS} --batch_size {BATCH_SIZE} --max_physical_batch_size {MAX_PHYSICAL_BATCH_SIZE}\
    --model_name {MODEL_NAME}  --n_class {N_CLASSES} --augmult {N_AUGMULT} --path {PATH} --name {NAME} --memo {MEMO}\
    --public_batch_size {PUBLIC_BATCH_SIZE} --extender {EXTENDER} --public_rho {PUBLIC_RHO} \
    --pretrained_dir {pretrained_dir}

Due to the size limits, we will provide the trained diffusion models and sampled images after submission.

## Environment configuration

The codes are based on python3.8+, CUDA version 11.0+. The specific configuration steps are as follows:

1. Create conda environment
   
   ```shell
   conda create -y -n env_name
   conda activate env_name
   ```

2. Install pytorch (can be different depending on your environments)
   
   ```shell
   conda install pytorch==1.13
   ```

3. Installation requirements (we do not force to download opacus==1.2.0 because it updates torch version to 2.0 sometimes and copy their code in opacus folder).
   
   ```shell
   pip install -r requirements.txt
   ```
