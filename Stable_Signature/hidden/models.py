# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from timm.models import vision_transformer

import attenuations

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs) # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1) # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        return x

class ImgEmbed(nn.Module):
    """ Patch to Image Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, num_patches_w, num_patches_h):
        B, S, CKK = x.shape # ckk = embed_dim
        x = self.proj(x.transpose(1,2).reshape(B, CKK, num_patches_h, num_patches_w)) # b s (c k k) -> b (c k k) s -> b (c k k) sh sw -> b c h w
        return x

class VitEncoder(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_bits, last_tanh=True, **kwargs):
        super(VitEncoder, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.msg_linear = nn.Linear(self.embed_dim+num_bits, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=kwargs['patch_size'])

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):

        num_patches = int(self.patch_embed.num_patches**0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        msgs = msgs.unsqueeze(1) # b 1 k
        msgs = msgs.repeat(1, x.shape[1], 1) # b 1 k -> b l k
        for ii, blk in enumerate(self.blocks):
            x = torch.concat([x, msgs], dim=-1) # b l (cpq+k)
            x = self.msg_linear(x)
            x = blk(x)

        x = x[:, 1:, :] # without cls token
        img_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w

class DvmarkEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(DvmarkEncoder, self).__init__()

        transform_layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            transform_layers.append(layer)
        self.transform_layers = nn.Sequential(*transform_layers)

        # conv layers for original scale
        num_blocks_scale1 = 3
        scale1_layers = [ConvBNRelu(channels+num_bits, channels*2)]
        for _ in range(num_blocks_scale1-1):
            layer = ConvBNRelu(channels*2, channels*2)
            scale1_layers.append(layer)
        self.scale1_layers = nn.Sequential(*scale1_layers)

        # downsample x2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # conv layers for downsampled
        num_blocks_scale2 = 3
        scale2_layers = [ConvBNRelu(channels*2+num_bits, channels*4), ConvBNRelu(channels*4, channels*2)]
        for _ in range(num_blocks_scale2-2):
            layer = ConvBNRelu(channels*2, channels*2)
            scale2_layers.append(layer)
        self.scale2_layers = nn.Sequential(*scale2_layers)

        # upsample x2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final_layer = nn.Conv2d(channels*2, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        encoded_image = self.transform_layers(imgs) # b c h w

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1

        scale1 = torch.cat([msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)), encoded_image], dim=1) # b l+c h w
        scale1 = self.scale1_layers(scale1) # b c*2 h w

        scale2 = self.avg_pool(scale1) # b c*2 h/2 w/2
        scale2 = torch.cat([msgs.expand(-1,-1, imgs.size(-2)//2, imgs.size(-1)//2), scale2], dim=1) # b l+c*2 h/2 w/2
        scale2 = self.scale2_layers(scale2) # b c*2 h/2 w/2

        scale1 = scale1 + self.upsample(scale2) # b c*2 h w
        im_w = self.final_layer(scale1) # b 3 h w

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class EncoderDecoder(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        augmentation: nn.Module, 
        decoder:nn.Module,
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float,
        num_bits: int,
        redundancy: int,
        augmentation_tile: nn.Module = nn.Identity(),
        tile: str = 'none',
        tile_size: int = 64,
        tile_type: str = 'random',
        
        
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.augmentation = augmentation
        self.augmentation_tile = augmentation_tile
        self.decoder = decoder
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy
        self.tile = tile
        self.tile_size = tile_size
        self.tile_type = tile_type
    
    @staticmethod
    def get_tile(imgs: torch.Tensor, imgs_aug: torch.Tensor, tile_size: int, tile_type: str = "random"):    
        """
        Extract a (tile_size × tile_size) patch from both `imgs` and `imgs_aug`,
        guaranteeing the slice is in-bounds for *both* tensors.

        Returns
        -------
        tiles       : Tensor  [B, C, tile_size, tile_size]   – patch from `imgs`
        imgs_tiles  : Tensor  [B, C, tile_size, tile_size]   – patch from `imgs_aug`
        coords      : LongTensor [B, 2]                      – (y0, x0) for each sample
        """
        # ----------------------------- sanity checks -----------------------------
        if imgs.ndim != 4 or imgs_aug.ndim != 4:
            raise ValueError("imgs and imgs_aug must have shape (B, C, H, W)")

        B, C, H,  W  = imgs.shape
        _, _, Ha, Wa = imgs_aug.shape

        # Effective spatial extent that works for both tensors
        H_eff, W_eff = min(H, Ha), min(W, Wa)

        if tile_size > H_eff or tile_size > W_eff:
            raise ValueError(
                f"tile_size={tile_size} exceeds available area "
                f"({H_eff}×{W_eff}) after augmentation"
            )

        device, dtype = imgs.device, torch.long

        # --------------------- choose the top-left corner ------------------------
        if tile_type == "random":
            top  = torch.randint(0, H_eff - tile_size + 1, (B,), device=device, dtype=dtype)
            left = torch.randint(0, W_eff - tile_size + 1, (B,), device=device, dtype=dtype)

        elif tile_type == "random_grid":
            n_rows = H_eff // tile_size
            n_cols = W_eff // tile_size
            if n_rows == 0 or n_cols == 0:
                raise ValueError(
                    f"Image ({H_eff}×{W_eff}) cannot be divided into grid "
                    f"of size {tile_size}"
                )
            row_idx = torch.randint(0, n_rows, (B,), device=device, dtype=dtype)
            col_idx = torch.randint(0, n_cols, (B,), device=device, dtype=dtype)
            top  = row_idx * tile_size
            left = col_idx * tile_size

        elif tile_type == "grid":
            top  = torch.zeros(B, device=device, dtype=dtype)
            left = torch.zeros(B, device=device, dtype=dtype)

        else:
            raise ValueError(f"Unsupported tile_type '{tile_type}'")

        # ------------------------- gather the patches ---------------------------
        tiles = torch.empty((B, C, tile_size, tile_size),
                              dtype=imgs.dtype,      device=device)
        imgs_tiles = torch.empty((B, C, tile_size, tile_size),
                              dtype=imgs_aug.dtype,  device=device)
        coords = torch.empty((B, 2), dtype=dtype, device=device)

        for b in range(B):
            y0, x0 = top[b].item(), left[b].item()
            tiles[b]      = imgs[b,     :, y0 : y0 + tile_size, x0 : x0 + tile_size]
            imgs_tiles[b] = imgs_aug[b, :, y0 : y0 + tile_size, x0 : x0 + tile_size]
            coords[b]     = torch.tensor([y0, x0], device=device, dtype=dtype)

        return tiles, imgs_tiles, coords
            
    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
        eval_mode: bool=False,
        eval_aug: nn.Module=nn.Identity(),
    ):
        """
        Does the full forward pass of the encoder-decoder network:
        - encodes the message into the image
        - attenuates the watermark
        - augments the image
        - decodes the watermark

        Args:
            imgs: b c h w
            msgs: b l
        """

        # encoder
        if self.tile == 'both':
            imgs_tile, _, coords = self.get_tile(imgs, imgs, self.tile_size, self.tile_type)
            deltas_w = self.encoder(imgs_tile, msgs)
        else:        
            deltas_w = self.encoder(imgs, msgs) # b c h w
            imgs_tile = imgs
        

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            if self.tile == 'both':
                aa = 1/4.6 # such that aas has mean 1
                aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs_tile.device) 
                deltas_w = deltas_w * aas[None,:,None,None]  
            else:
                aa = 1/4.6 # such that aas has mean 1
                aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
                deltas_w = deltas_w * aas[None,:,None,None]  
            

        # add heatmaps
        if self.attenuation is not None:
            if self.tile == 'both':
                heatmaps = self.attenuation.heatmaps(imgs_tile) # b 1 h w
                deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
            else:    
                heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
                deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        if self.tile == 'both':    
            imgs_w = (self.scaling_i * imgs_tile + self.scaling_w * deltas_w).clone()# b c h w
        else:
            imgs_w = (self.scaling_i * imgs + self.scaling_w * deltas_w).clone()# b c h w

        # data augmentation
        if eval_mode:
            imgs_aug = eval_aug(imgs_w)
            if self.tile == 'decode':
                imgs_tile, imgs_aug, coords = self.get_tile(imgs, imgs_aug, self.tile_size, self.tile_type)
        else:
            if self.tile == 'both':
                imgs_aug = self.augmentation_tile(imgs_w)
            else:    
                imgs_aug = self.augmentation(imgs_w)
            if self.tile == 'decode':
                # print(f"[Debug]:imgs.shape:{imgs.shape}")
                # print(f"[Debug]:imgs_aug.shape:{imgs_aug.shape}")
                imgs_tile, imgs_aug, coords = self.get_tile(imgs, imgs_aug, self.tile_size, self.tile_type)
        fts = self.decoder(imgs_aug) # b c h w -> b d
            
        fts = fts.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        fts = torch.sum(fts, dim=-1)  # b k r -> b k

        return fts, (imgs_w, imgs_aug, coords if self.tile != 'none' else None)

class EncoderWithJND(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
    ):
        """ Does the forward pass of the encoder only """

        # encoder
        deltas_w = self.encoder(imgs, msgs) # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            aa = 1/4.6 # such that aas has mean 1
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
            deltas_w = deltas_w * aas[None,:,None,None]

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
            deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w # b c h w

        return imgs_w