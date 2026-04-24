apt update
apt install wget -y
apt install git -y
apt install tmux -y
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt install python3.12 -y && apt install python3.12-venv python3.12-dev -y
python3.12 -m venv myenv

source myenv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install transformers datasets trl accelerate peft huggingface_hub sentencepiece
pip install adam-atan2-pytorch einops wandb tqdm pydantic omegaconf hydra-core trl bitsandbytes flash-linear-attention
pip install --no-deps unsloth unsloth-zoo
pip install arc-agi
echo "done!"