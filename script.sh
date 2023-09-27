pkg update
pkg install build-essential binutils
pkg install rust
pkg install python
pkg install python-pip
pkg install wget git

git clone --recurse-submodules https://github.com/rustformers/llm.git
cd llm
cargo build --release
cd


wget https://huggingface.co/AndriLawrence/gpt2-chatkobi-ai/resolve/main/1SdjAt39Mjfi2mklO.bin
wget https://huggingface.co/AndriLawrence/gpt2-chatkobi-ai/resolve/main/1SdjAt39Mjfi2mklO.meta
pip install -r requirements.txt
