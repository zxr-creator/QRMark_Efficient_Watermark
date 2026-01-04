# QRMark: Efficient Watermark Detection
## Model supported:
| Model | Repo |
|-------|-----------|
| Stable Signature | [Github](https://github.com/facebookresearch/stable_signature) |
| AquaLoRA | [GitHub](https://github.com/Georgefwt/AquaLoRA) |

## Abstract:

Efficient and reliable detection of generated images is critical for the responsible deployment of generative models. Existing approaches primarily focus on improving detection accuracy and robustness under various image transformations and adversarial manipulations, yet they largely overlook the efficiency challenges of watermark detection across large-scale image collections.
To address this gap, we propose QRMark, an efficient and adaptive end-to-end method for detecting embedded image watermarks. The core idea of QRMark is to combine QR Code–inspired error correction with tailored tiling techniques to improve detection efficiency while preserving accuracy and robustness. At the algorithmic level, QRMark employs a Reed–Solomon error correction mechanism to mitigate the accuracy degradation introduced by tiling. At the system level, QRMark implements a resource-aware multi-channel horizontal fusion policy that adaptively assigns more streams to GPU-intensive stages of the detection pipeline. It further employs a tile-based workload interleaving strategy to overlap data-loading overhead with computation and schedules kernels across stages to maximize efficiency. End-to-end evaluations show that QRMark achieves an average 2.43X inference speedup over the sequential baseline.

## 🚀Getting Started:


### Environment Setup:
### For Stable Signature:
``` bash 
git clone https://github.com/zxr-creator/QRMark_Efficient_Watermark.git
cd QRMark_Efficient_Watermark/Stable_Signature
conda create signature python==3.12
pip install -r requirements.txt
conda activate signature
```

Check [Stable Signature](https://github.com/facebookresearch/stable_signature) for further information on set up.

### Using docker for Stable Signature(Optional):

```bash
docker build -f signature.Dockerfile --no-cache --tag signature:1.0.0 .

docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size=16g --cap-add=SYS_ADMIN --security-opt seccomp=unconfined -v $(pwd):/mnt -w /mnt --network host signature:1.0.0
```
### For Aqualora:
```bash
cd QRMark_Efficient_Watermark/AquaLoRA
conda create aqualora python==3.12
pip install -r requirements.txt
conda activate aqualora
```

### Downloaded Original Models

| Model | Download Link |
|-------|---------------|
| Stable Signature | [model](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b.pth) |
| Aqualora | [model](huggingface.co/georgefen/AquaLoRA-Models/tree/main/rob_finetuned) |

### Pretraining 

We following a similar approach with Stable Signature, Please check Stable_Signature/hidden/README.md

### Fine-tuning & Generation

bash Stable_Signature/scripts/generation.sh

### Detection

bash /Stable_Signature/scripts/generation.sh

## BibTeX
If you find [QRMark](https://arxiv.org/abs/2509.02447) useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{zhong2025efficient,
  title={An Efficient and Adaptive Watermark Detection System with Tile-based Error Correction},
  author={Zhong, Xinrui and Feng, Xinze and Zuo, Jingwei and Ye, Fanjiang and Mu, Yi and Guo, Junfeng and Huang, Heng and Lee, Myungjin and Wang, Yuke},
  journal={arXiv preprint arXiv:2509.02447},
  year={2025},
  eprint={2509.02447},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.02447}
}
```