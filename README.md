# Setup

## Create virtual environment

### [Option 1] virtualenv

```bash
python3.11 -m venv venv
source ./venv/bin/activate
pip install wheel
pip install --upgrade pip
```

### [Option 2] conda

```bash
conda create -n natten-fwd python=3.11
conda activate natten-fwd
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Install NATTEN from source

You may need to install NATTEN from source. Here's how I did it:

```bash
git clone https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
pip install cmake==3.20.3
CUDACXX=/usr/local/cuda/bin/nvcc make CUDA_ARCH="8.9" WORKERS=2
```