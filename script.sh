pkg update
pkg install build-essential binutils
pkg install rust
pkg install python
pkg install python-pip
pkg install wget git

wget https://huggingface.co/AndriLawrence/gpt2-chatkobi-ai/resolve/main/1SdjAt39Mjfi2mklO.bin
wget https://huggingface.co/AndriLawrence/gpt2-chatkobi-ai/resolve/main/1SdjAt39Mjfi2mklO.meta

git clone --recurse-submodules https://github.com/rustformers/llm.git
cd llm
cargo build --release
cd

git clone https://github.com/andri-jpg/termux-fa
cd termux-fa
wget https://huggingface.co/AndriLawrence/gpt2-chatkobi-ai/resolve/main/1SdjAt39Mjfi2mklO.bin
wget https://huggingface.co/AndriLawrence/gpt2-chatkobi-ai/resolve/main/1SdjAt39Mjfi2mklO.meta
pip install -r requirements.txt
python app.py