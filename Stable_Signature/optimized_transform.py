# optimized_transform.py
import torch, triton, triton.language as tl
import torchvision.transforms.functional as TF
from typing import List, Tuple, Union
from PIL import Image

# ===== ImageNet normalization statistics =====
_MEAN    = (0.485, 0.456, 0.406)
_STD     = (0.229, 0.224, 0.225)
_INV_STD = [1.0 / s for s in _STD]


# ---------------------------
# Triton kernel: fused resize+center-crop+toTensor (+ optional normalize)
# ---------------------------
@triton.jit
def _resize_crop_norm_kernel(
    src_ptr, dst_ptr,                          # uint8  float32
    B: tl.int32, H: tl.int32, W: tl.int32,     # input image size (same within group)
    O: tl.int32,                                # target output size (square)
    scale: tl.float32, offy: tl.float32, offx: tl.float32,
    sb_in: tl.int32, sc_in: tl.int32, sh_in: tl.int32, sw_in: tl.int32,
    sb_out: tl.int32, sc_out: tl.int32, sh_out: tl.int32, sw_out: tl.int32,
    m0, m1, m2, s0, s1, s2,                    # per-channel norm (use mean=0,std=1 to disable)
    inv255: tl.float32,
    BLOCK: tl.constexpr
):
    n_elems = B * 3 * O * O
    offs    = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask    = offs < n_elems

    # —— (b, c, y, x) from linear index
    b  = offs // sb_out
    r1 = offs - b * sb_out
    c  = r1   // sc_out
    r2 = r1   - c * sc_out
    y  = r2   // sh_out
    x  = r2   - y * sh_out

    # —— Map output (y,x) to input with "resize-then-center-crop" inverse mapping
    fy = (offy + y.to(tl.float32)) / scale
    fx = (offx + x.to(tl.float32)) / scale

    y0 = tl.floor(fy)
    x0 = tl.floor(fx)
    y1 = y0 + 1
    x1 = x0 + 1

    # —— Clamp
    y0i = tl.where(y0 < 0, 0, tl.where(y0 > H - 1, H - 1, y0)).to(tl.int32)
    x0i = tl.where(x0 < 0, 0, tl.where(x0 > W - 1, W - 1, x0)).to(tl.int32)
    y1i = tl.where(y1 < 0, 0, tl.where(y1 > H - 1, H - 1, y1)).to(tl.int32)
    x1i = tl.where(x1 < 0, 0, tl.where(x1 > W - 1, W - 1, x1)).to(tl.int32)

    bi = b.to(tl.int32)
    ci = c.to(tl.int32)

    # —— Load 4 neighbors for bilinear
    off00 = bi * sb_in + ci * sc_in + y0i * sh_in + x0i * sw_in
    off01 = bi * sb_in + ci * sc_in + y0i * sh_in + x1i * sw_in
    off10 = bi * sb_in + ci * sc_in + y1i * sh_in + x0i * sw_in
    off11 = bi * sb_in + ci * sc_in + y1i * sh_in + x1i * sw_in

    v00 = tl.load(src_ptr + off00, mask=mask).to(tl.float32)
    v01 = tl.load(src_ptr + off01, mask=mask).to(tl.float32)
    v10 = tl.load(src_ptr + off10, mask=mask).to(tl.float32)
    v11 = tl.load(src_ptr + off11, mask=mask).to(tl.float32)

    wy  = fy - y0
    wx  = fx - x0
    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    # —— Linear blend → [0,1]
    val = (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11) * inv255

    # —— Normalize per channel: (x - mean) / std
    mean  = tl.where(ci == 0, m0, tl.where(ci == 1, m1, m2))
    inv_s = tl.where(ci == 0, s0, tl.where(ci == 1, s1, s2))
    out   = (val - mean) * inv_s

    tl.store(dst_ptr + offs, out, mask=mask)


def _run_triton_group(
    x_u8: torch.Tensor,
    img_size: int,
    do_normalize: bool,
    device: torch.device
) :
    if x_u8.dtype != torch.uint8 or not x_u8.is_cuda or x_u8.ndim != 4 or x_u8.shape[1] != 3:
        raise ValueError("Expect CUDA uint8 tensor of shape [B,3,H,W]")

    B, _, H, W = x_u8.shape
    short_side  = min(H, W)
    scale       = float(img_size) / short_side
    new_H       = int(round(H * scale))
    new_W       = int(round(W * scale))
    off_y       = max(0.0, (new_H - img_size) * 0.5)
    off_x       = max(0.0, (new_W - img_size) * 0.5)

    y = torch.empty((B, 3, img_size, img_size), device=device, dtype=torch.float32)

    # Strides
    sb_in, sc_in, sh_in, sw_in = x_u8.stride()
    sb_out, sc_out, sh_out, sw_out = y.stride()

    BLOCK = 1024
    grid  = (triton.cdiv(y.numel(), BLOCK),)

    if do_normalize:
        m = _MEAN;  s = _INV_STD
    else:
        # mean=0, inv_std=1 
        m = (0.0, 0.0, 0.0);  s = (1.0, 1.0, 1.0)

    _resize_crop_norm_kernel[grid](
        x_u8, y,
        B, H, W, img_size,
        scale, off_y, off_x,
        sb_in, sc_in, sh_in, sw_in,
        sb_out, sc_out, sh_out, sw_out,
        *m, *s,
        1.0 / 255.0,
        BLOCK=BLOCK,
    )
    return y


@triton.jit
def _normalize_kernel(src_ptr, dst_ptr,
                      N: tl.int32,  # total elements = B*C*H*W
                      sb: tl.int32, sc: tl.int32, sh: tl.int32, sw: tl.int32,
                      m0, m1, m2, s0, s1, s2,
                      is_u8: tl.int32,
                      BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # unravel to (b,c,y,x) strides
    b = offs // sb
    r1 = offs - b * sb
    c = r1 // sc
    r2 = r1 - c * sc
    y = r2 // sh
    x = r2 - y * sh

    base = b * sb + c * sc + y * sh + x * sw
    val  = tl.load(src_ptr + base, mask=mask).to(tl.float32)
    if is_u8 == 1:
        val = val * (1.0 / 255.0)

    mean  = tl.where(c == 0, m0, tl.where(c == 1, m1, m2))
    inv_s = tl.where(c == 0, s0, tl.where(c == 1, s1, s2))
    out   = (val - mean) * inv_s
    tl.store(dst_ptr + base, out, mask=mask)


def _normalize_gpu(x: torch.Tensor):
    """
    x: [C,H,W] or [B,C,H,W], dtype uint8/float*, on CUDA
    return: float32, same shape, normalized by ImageNet stats
    """
    if not x.is_cuda:
        raise ValueError("normalize expects CUDA tensor")
    if x.ndim == 3:
        x = x.unsqueeze(0)  # [1,C,H,W]
        squeeze = True
    elif x.ndim == 4:
        squeeze = False
    else:
        raise ValueError("normalize expects [C,H,W] or [B,C,H,W]")

    if x.shape[1] != 3:
        raise ValueError("normalize expects 3 channels")

    out = x.clone()
    N = out.numel()
    sb, sc, sh, sw = out.stride()
    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)

    is_u8 = 1 if out.dtype == torch.uint8 else 0
    if out.dtype != torch.float32:
        out = out.contiguous()
    _normalize_kernel[grid](
        out, out,
        N, sb, sc, sh, sw,
        *_MEAN, *_INV_STD,
        is_u8,
        BLOCK=BLOCK,
    )
    out = out.to(torch.float32)
    return out.squeeze(0) if squeeze else out


class VQGANTransformFuse:
    def __init__(self, img_size: int = 256, device: torch.device = None, do_normalize: bool = True):
        self.img_size = img_size
        self.device = device if device is not None else torch.device("cuda")
        self.do_normalize = bool(do_normalize)

    def __call__(self, x: Union[Image.Image, torch.Tensor, List[Image.Image]]):
        device = self.device

        # Case A
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], Image.Image):
            entries: List[Tuple[Tuple[int,int], int, torch.Tensor]] = []
            for i, im in enumerate(x):
                t = TF.pil_to_tensor(im)  # [C,H,W] uint8
                H, W = t.shape[1], t.shape[2]
                entries.append(((H, W), i, t))

            buckets = {}
            for (H, W), idx, t in entries:
                buckets.setdefault((H, W), []).append((idx, t))

            out_list = [None] * len(x)
            for (H, W), items in buckets.items():
                items.sort(key=lambda it: it[0])
                idxs, tensors = zip(*items)
                batch_u8 = torch.stack(list(tensors), dim=0).to(device, non_blocking=True)  # [B,3,H,W] u8
                batch_y  = _run_triton_group(batch_u8, self.img_size, self.do_normalize, device)
                for k, idx in enumerate(idxs):
                    out_list[idx] = batch_y[k]
            return torch.stack(out_list, dim=0)

        # Case B
        if isinstance(x, Image.Image):
            u8 = TF.pil_to_tensor(x).unsqueeze(0).to(device)  # [1,3,H,W] u8
            y  = _run_triton_group(u8, self.img_size, self.do_normalize, device)
            return y[0]

        # Case C
        if isinstance(x, torch.Tensor):
            if x.ndim == 3:
                x = x.unsqueeze(0)
                squeeze = True
            elif x.ndim == 4:
                squeeze = False
            else:
                raise ValueError("Tensor input must be [C,H,W] or [B,C,H,W]")

            if x.dtype not in (torch.uint8,):
                x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
            x = x.to(device, non_blocking=True)
            y = _run_triton_group(x, self.img_size, self.do_normalize, device)
            return y[0] if squeeze else y

        raise TypeError("Unsupported input type for VQGANTransformFuse")




# --------- default instances (behave like torchvision transforms) ----------
vqgan_transform_fuse = VQGANTransformFuse(img_size=256, do_normalize=True)

