# SeaCache for HunyuanVideo
This is the official implementation of SeaCache for HunyuanVideo.

## Installation
1. Clone the original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) repository and install all required dependencies (Python, CUDA, checkpoints, etc.) as described there.
2. Place the provided SeaCache script (`seacache_sample_video.py`) into the HunyuanVideo repository root.

## Usage
The main code is `seacache_sample_video.py`. Make sure to run the script inside the HunyuanVideo directory.

### Text-to-Video Inference

```bash
cd HunyuanVideo

python3 seacache_generate.py \
    --video-size 720 1280 \
    --video-length 33 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./seacache_results \
    --rel_l1_thresh 0.20
```

Arguments:
- `--video-size`: output resolution `(height width)`
- `--video-length`: number of frames to generate
- `--infer-steps`: number of sampling steps
- `--prompt`: text prompt for video generation
- `--save-path`: directory to store generated videos
- `--rel_l1_thresh`: threshold to adjust cache ratio

> **Note:** You can sweep over `rel_l1_thresh` to find the balanced quality-latency trade-offs.

---
## Inference Results
Latency Performance may vary depending on your GPU. In our setup, we used an single NVIDIA RTX PRO 6000 Blackwell.

> **Note:** `rel_l1_thresh = δ`

- T2V model (720p):

*A cat walks on the grass, realistic style.*
<video src="https://github.com/user-attachments/assets/bf0adf4d-4ace-4684-acc8-0bc4e82910ba" controls muted playsinline width="832"></video>

| Original | SeaCache (δ = 0.2) | SeaCache (δ = 0.4) |
|:---------------:|:-----------------------------------:|:-----------------------------------:|
|   5 min 32 sec   |           2 min 30 sec           |           1 min 40 sec           |
