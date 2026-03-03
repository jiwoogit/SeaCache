# SeaCache for FLUX
This is the official implementation of SeaCache for FLUX.

## Installation

This script relies on PyTorch, Diffusers, and common Hugging Face tooling.

```bash
pip install --upgrade \
    diffusers[torch] \
    transformers \
    protobuf \
    tokenizers \
    sentencepiece \
    accelerate \
    safetensors
```

## Usage

The main code is `seacache_generate.py`.
```bash
python seacache_generate.py
```

### Custom Prompt and Ourput Directory

```bash
python seacache_generate.py \
  --prompt "a high-resolution photo of a panda drinking coffee in a cozy cafe" \
  --output_dir ./outputs \
  --seacache_thresh 0.3
```

Arguments:
- `--prompt`: text prompt for FLUX.1-dev  
- `--output_dir`: directory to store generated images  
- `--seacache_thresh`: threshold to adjust cache ratio  

---

## Inference Results
Latency Performance may vary depending on your GPU. In our setup, we used an NVIDIA RTX PRO 6000 Blackwell.

*a high-resolution photo of a panda drinking coffee in a cozy cafe*
![FLUX coffee example](../assets/flux_coffee.png)

|       FLUX.1 [dev]       |   SeaCache (δ = 0.3)   |   SeaCache (δ = 0.6)   |
|:------------------------:|:----------------------:|:----------------------:|
|          18.0 s          |         9.4 s          |         6.4 s          |
