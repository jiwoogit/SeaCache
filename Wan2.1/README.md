# SeaCache for Wan2.1
This is the official implementation of SeaCache for Wan2.1.

## Installation
1. clone the original [Wan2.1](https://github.com/Wan-Video/Wan2.1) repository and install all required dependencies (Python, CUDA, checkpoints, etc.) as describe there.
2. place the provided SeaCache script (`seacache_generate.py`) into the Wan2.1 repository root.

## Usage
The main code is `seacache_generate.py`. Make sure to run the script inside the Wan2.1 directory.

### Text-to-Video (1.3B model)

    python seacache_generate.py \
        --task t2v-1.3B \
        --size 832*480 \
        --ckpt_dir ./Wan2.1-T2V-1.3B \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
        --base_seed 42 \
        --offload_model True \
        --sample_steps 50 \
        --seacache_thresh 0.20

Arguments:

- `--task`: model task to run (e.g., `t2v-1.3B`)  
- `--size` : output resolution (w*h) (`480p: 832*480, 720p: 1280*720`)
- `--ckpt_dir` : path to the checkpoint directory  
- `--prompt` : text prompt for video generation  
- `--base_seed` : base random seed  
- `--sample_steps` : number of sampling steps for the solver  
- `--seacache_thresh` : threshold to adjust cache ratio

> **Note:** You can sweep over `seacache_thresh` to find the balanced quality-latency trade-offs.

---
## Inference Results
Latency Performance may vary depending on your GPU. In our setup, we used an single NVIDIA RTX PRO 6000 Blackwell.

> **Note:** `seacache_thresh = δ`


- T2V 1.3B model (480p):

*Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.*
<video src="../assets/wan2.1_1.3b_t2v.mp4" controls muted playsinline width="832"></video>

| Original | SeaCache (δ = 0.2) | SeaCache (δ = 0.35) |
|:---------------:|:-----------------------------------:|:-----------------------------------:|
|     2 min 56 sec     |               1min 24 sec s               |               56 sec               |

- T2V 14B model (480p):

*Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.*
<video src="../assets/wan2.1_14b_t2v_480p.mp4" controls muted playsinline width="832"></video>

| Original | SeaCache (δ = 0.2) | SeaCache (δ = 0.4) |
|:---------------:|:-----------------------------------:|:-----------------------------------:|
|   15 min 11 sec   |            7 min 31 sec            |            4 min 53 sec            |

- T2V 14B model (720p):

*Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.*
<video src="../assets/wan2.1_14b_t2v_720p.mp4" controls muted playsinline width="832"></video>

| Original | SeaCache (δ = 0.2) | SeaCache (δ = 0.4) |
|:---------------:|:-----------------------------------:|:-----------------------------------:|
|   51 min 13 sec   |           25 min 32 sec           |           16 min 30 sec           |

- I2V 14B model (720p):

*a bald eagle flying over a tree filled forest*
| Input Image | Output Video |
|:--:|:--:|
| <img src="../assets/a%20bald%20eagle%20flying%20over%20a%20tree%20filled%20forest.jpg" height="256" /> | <video src="../assets/wan2.1_14b_i2v_720p_1.mp4" controls muted playsinline height="256"></video> |

*a basket of french fries in a fryer*
| Input Image | Output Video |
|:--:|:--:|
| <img src="../assets/a%20basket%20of%20french%20fries%20in%20a%20fryer.jpg" height="256" /> | <video src="../assets/wan2.1_14b_i2v_720p_2.mp4" controls muted playsinline height="256"></video> |

| Original | SeaCache (δ = 0.2) | SeaCache (δ = 0.4) |
|:---------------:|:-----------------------------------:|:-----------------------------------:|
|   52 min 5 sec   |           25 min 3 sec           |           16 min 50 sec           |
