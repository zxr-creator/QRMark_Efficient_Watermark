import sys
import os
import random
from pathlib import Path

# this script lives in Efficient_Stable_Signature/AquaLoRA/train/…
# so parent.parent == Efficient_Stable_Signature/AquaLoRA
code_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(code_root))

import glob
import PIL
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.checkpoint
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
import lpips

from utils.models_tile import *
from utils.vae_tiling import (
    extract_tile,
    encode_tile,
    decode_tile,
)
import torchsummary

from datetime import datetime

# set seed so that everytime we select the same subset of training set
random.seed(42)

WINDOW_SIZE = 32
KERNEL = torch.ones((1, 1, WINDOW_SIZE, WINDOW_SIZE), dtype=torch.float32) / (WINDOW_SIZE**2)

def PRVL_loss(img1, img2):
    global KERNEL
    diff = torch.abs(img1 - img2)
    diff_combined = torch.mean(diff, dim=1, keepdim=True)
    if KERNEL.device != diff_combined.device:
        KERNEL = KERNEL.to(diff_combined.device)
    diff_sum = F.conv2d(diff_combined, KERNEL, padding=WINDOW_SIZE//2).squeeze(0) # [1, 513, 513]
    max_diff = torch.max(diff_sum)
    return max_diff

def base_augment(image):
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
    image = torch.rot90(image, k=random.randint(0, 3), dims=[-2, -1])
    return image

class testdataset(Dataset):
    def __init__(self, root, random_aug=True):
        self.root = root
        self.image_files = random.sample(glob.glob(root + "/*.png") + glob.glob(root + "/*.jpg"), 100)
        self.random_aug = random_aug

    def __len__(self):
        return len(self.image_files)

    def process(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if self.random_aug and random.random() > 0.5:
            image = base_augment(image)
        return image

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        image = self.process(image)
        return image


def main(args):
    test_loader = torch.utils.data.DataLoader(
        testdataset(args.dataset, args.random_aug),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # define tile num
    tile_num = args.tile_num
    
    # patch util functions for en/decoding tiles
    AutoencoderKL.encode_tile = encode_tile
    AutoencoderKL.decode_tile = decode_tile
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.cuda()
    # Freeze vae and unet
    vae.requires_grad_(False)

    def decode_latents(latents):
        # latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents).sample
        return image

    sec_encoder = SecretEncoder(args.bit_num).cuda()
    sec_decoder = SecretDecoder(tile_num, output_size=args.bit_num).cuda()
    
    loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
    loss_fn_alex.requires_grad_(False)

    # load pretrained result
    models = torch.load(args.resume_from_ckpt)
    sec_encoder.load_state_dict(models['sec_encoder'])
    sec_decoder.load_state_dict(models['sec_decoder'])
    
    sec_encoder.eval()
    sec_decoder.eval()
    average_acc = 0
    for batch_idx, oimage in enumerate(test_loader):
        with torch.no_grad():
            oimage = oimage.cuda()
            latent = vae.encode_tile(oimage).detach()
            msg_val = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
            watermarked_latent, _ = sec_encoder(latent, msg_val.float())
            
            clean_image = vae.decode_tile(latent).detach()
            watermarked_image = vae.decode_tile(watermarked_latent).detach()
            
            
            lpips_loss = loss_fn_alex(clean_image, watermarked_image).mean()
            prvl_loss = PRVL_loss(clean_image, watermarked_image)
            
            tiles = extract_tile(watermarked_image, tile_num)
            
            
            #------------------------------------------------------------------------------------------
            tile_idx = random.randrange(len(tiles))
            decoded_msg = sec_decoder(tiles[tile_idx])
            decoded_msg = torch.argmax(decoded_msg, dim=2)
            acc = 1 - torch.abs(decoded_msg - msg_val).sum().float() / (args.bit_num * args.batch_size)
            
            #------------------------------------------------------------------------------------------
            
            # tiles_batch = torch.cat(tiles, dim=0)
            # decoded_batch = sec_decoder(tiles_batch) # (4*B, bit_num, 2)
            # decoded_labels = torch.argmax(decoded_batch, dim=2) # (4*B, bit_num)
            # decoded_labels = decoded_labels.view(tile_num**2,
            #                                      args.batch_size,
            #                                      args.bit_num)
            # msg_expand = msg_val.unsqueeze(0).expand(tile_num**2, -1, -1)      # (4, B, bits)
            # matches = (decoded_labels == msg_expand)             # (4, B, bits)
            # bit_accs = matches.float().mean(dim=2)               # (4, B)
            # acc = bit_accs.mean()  
            #------------------------------------------------------------------------------------------
            
            average_acc += acc
            print(f"batch_idx {batch_idx}: acc {acc}, lpips_loss {lpips_loss}, prvl_loss {prvl_loss}")
    print(f"average acc: {average_acc / len(test_loader)}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5')
    argparser.add_argument('--batch_size', type=int, default=5)
    argparser.add_argument('--bit_num', type=int, default=48)
    argparser.add_argument('--resume_from_ckpt', type=str, default=None)
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--random_aug', default=True)
    argparser.add_argument('--tile_num', type=int, default=1)
    args = argparser.parse_args()

    main(args)
