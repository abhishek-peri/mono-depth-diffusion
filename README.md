Download NYU-dataset from https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

Training:

python main.py --base "replace with path to config" -t --gpus 0,1

Config files can be found in the configs/latent-diffusion folder : 
depth-ldm-vq-4-semantic.yaml -> config file for spatial rescaler network
dino-ldm-vq-4-semantic.yaml -> config file for DINO network for conditioning block

Sampling for test:

python semantic_image_sampling.py --resume "path-to-the-trained-ckpt" -n 5000 --batch_size 18 -c 200 -e 1.0

