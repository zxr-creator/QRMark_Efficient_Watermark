import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from omegaconf import OmegaConf
import utils_model
import utils_img
from bit_reedsolo import ReedSolomonCodec_BW, UncorrectableError_RS

# Assuming these helper functions exist or are imported from your utils
# from your_script import encode_reed_solomon, decode_reed_solomon, _decode_with_tile

class QRMark:
    def __init__(self, params):
        """
        Initializes the QRMark system.
        
        Args:
            params (Namespace): Configuration containing model paths, 
                                bit lengths, loss weights, etc.
        """
        self.params = params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        print(f">>> Initializing QRMark on {self.device}...")

        # 1. Load LDM Autoencoder (The backbone)
        # We keep the encoder frozen.
        config = OmegaConf.load(params.ldm_config)
        self.ldm_ae = utils_model.load_model_from_config(config, params.ldm_ckpt)
        self.ldm_ae = self.ldm_ae.first_stage_model
        self.ldm_ae.eval()
        self.ldm_ae.to(self.device)

        # 2. Load/Prepare Hidden Message Decoder (The extractor)
        # This component extracts the raw bits from the image.
        self.msg_decoder = self._load_message_decoder(params)
        
        # 3. Prepare LDM Decoder for Fine-tuning (The watermarker)
        # We clone the LDM and set the decoder to be trainable.
        self.ldm_decoder = deepcopy(self.ldm_ae)
        self.ldm_decoder.encoder = nn.Identity()
        self.ldm_decoder.quant_conv = nn.Identity()
        self.ldm_decoder.to(self.device)
        
        # Freeze everything except the specific decoder we want to tune
        for param in [*self.msg_decoder.parameters(), *self.ldm_ae.parameters()]:
            param.requires_grad = False
            
        # 4. Setup Optimization (for fine-tuning mode)
        self.optimizer = None
        if hasattr(params, 'optimizer'):
            self.ldm_decoder.train()
            for param in self.ldm_decoder.parameters():
                param.requires_grad = True
            optim_params = utils.parse_params(params.optimizer)
            self.optimizer = utils.build_optimizer(model_params=self.ldm_decoder.parameters(), **optim_params)

        # 5. Define Transforms
        self.vqgan_to_imnet = torch.nn.Sequential(
            utils_img.unnormalize_vqgan, 
            utils_img.normalize_img
        )

    def _load_message_decoder(self, params):
        """Helper to load and optionally whiten the hidden decoder."""
        if 'torchscript' in params.msg_decoder_path:
            return torch.jit.load(params.msg_decoder_path).to(self.device)
        else:
            decoder = utils_model.get_hidden_decoder(
                num_bits=params.num_bits, 
                redundancy=params.redundancy, 
                num_blocks=params.decoder_depth, 
                channels=params.decoder_channels
            ).to(self.device)
            # (Whitening logic omitted for brevity, assumed handled or pre-saved)
            return decoder

    def generate_watermark_key(self):
        """Generates a random key and encodes it using Reed-Solomon."""
        nbit = self.params.num_bits
        if self.params.reed_solomon:
            message_length = nbit - self.params.num_parity_symbols * self.params.m_bits_per_symbol
            msg_key = torch.randint(0, 2, (1, message_length), dtype=torch.float32, device=self.device)
            # Encode using RS
            full_key = self._encode_rs(msg_key) 
            return full_key, msg_key
        else:
            key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=self.device)
            return key, key

    def fine_tune(self, data_loader):
        """
        Fine-tune the LDM decoder to embed the watermark key.
        """
        print(">>> Starting Fine-tuning...")
        self.ldm_decoder.train()
        
        # Generate a fixed key for this session
        target_key, _ = self.generate_watermark_key()
        
        for i, imgs in enumerate(data_loader):
            imgs = imgs.to(self.device)
            keys = target_key.repeat(imgs.size(0), 1)

            # 1. Encode Image -> Latent
            with torch.no_grad():
                z = self.ldm_ae.encode(imgs).mode()
                # Get original reconstruction for perceptual loss
                imgs_recon_orig = self.ldm_ae.decode(z)

            # 2. Decode (Embed Watermark) -> Watermarked Image
            imgs_w = self.ldm_decoder.decode(z)

            # 3. Extract Watermark (Simulation)
            # Transform to ImageNet stats for the extractor
            imgs_w_input = self.vqgan_to_imnet(imgs_w)
            extracted_bits = self.msg_decoder(imgs_w_input) # Or use _decode_with_tile if enabled

            # 4. Compute Losses
            # Watermark Loss (BCE or MSE)
            loss_w = F.binary_cross_entropy_with_logits(extracted_bits * 10.0, keys)
            # Perceptual Loss (between Watermarked and Original reconstruction)
            loss_i = torch.mean((imgs_w - imgs_recon_orig) ** 2) # simplified MSE, actual code uses Watson/SSIM
            
            total_loss = self.params.lambda_w * loss_w + self.params.lambda_i * loss_i

            # 5. Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print(f"Step {i}: Loss={total_loss.item():.4f} (Img={loss_i.item():.4f}, Wmk={loss_w.item():.4f})")

    @torch.no_grad()
    def detect(self, images):
        """
        Detects watermark from images using the Hidden Decoder + Reed-Solomon.
        
        Args:
            images (Tensor): Batch of images [B, C, H, W]
        
        Returns:
            Tensor: Decoded message bits.
        """
        self.msg_decoder.eval()
        images = images.to(self.device)
        
        # Transform (VQGAN -> ImageNet)
        images = self.vqgan_to_imnet(images)
        
        # Extract raw bits
        # (Handles tiling strategies like random/fixed/grid if config requires)
        if self.params.tile != 'none':
             # Simplified tiling logic (e.g., center crop or top-left)
             images = images[:, :, :self.params.tile_size, :self.params.tile_size]
        
        raw_logits = self.msg_decoder(images)

        # Reed-Solomon Decoding
        if self.params.reed_solomon:
            decoded_msg = self._decode_rs(raw_logits)
            return decoded_msg
        
        return (raw_logits > 0).float()

    # --- RS Helpers (Simplified Wrappers) ---
    def _encode_rs(self, msg_key):
        # Calls the function defined in your script
        from bit_reedsolo import ReedSolomonCodec_BW
        # Implementation details hidden for brevity...
        # Returns encoded tensor
        pass 

    def _decode_rs(self, encoded_tensor):
        # Calls the decode_reed_solomon logic
        pass