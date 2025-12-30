import torch

def extract_tile(imgs: torch.Tensor, tile_num: int) -> torch.Tensor:
    """
    Extract the **top-left** tile from a batch of images.

    Args:
        imgs: (B, C, H, W) tensor.

    Returns:
        a list of (B, C, tile_h, tile_w) tensor containing the crops.
    """
    
    _, _, H, W = imgs.shape
    tile_h, tile_w = H // tile_num, W // tile_num
    tiles = [
        imgs[:, :, :tile_h, :tile_w],   # top-left
        imgs[:, :, :tile_h, tile_w:],   # top-right
        imgs[:, :, tile_h:, :tile_w],   # bottom-left
        imgs[:, :, tile_h:, tile_w:],   # bottom-right
    ]
    # print(f"[Debug] each tile shape = {tiles[0].shape}")
    
    return tiles

def encode_tile(vae, tile: torch.Tensor):
    """
    
    """
    encode_output = vae.encode(tile)
    vae_latent = encode_output.latent_dist.sample()
    return vae_latent

def decode_tile(vae, latent: torch.Tensor):
    imgs = vae.decode(latent).sample
    # print(f"[Debug] VAE decoded image shape = {imgs.shape}")
    return imgs