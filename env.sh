export EDGE_AI_HOME=/home/ssd/projects/edge-ai

export UV_CACHE_DIR="$EDGE_AI_HOME/cache/uv"
export HF_HOME="$EDGE_AI_HOME/cache/huggingface"
export TORCH_HOME="$EDGE_AI_HOME/cache/torch"

# 避免 ~/.local/lib/python3.8/site-packages 污染 uv 环境
export PYTHONNOUSERSITE=1

# JetPack 5.1.2 CUDA
export CUDA_HOME=/usr/local/cuda-11.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
