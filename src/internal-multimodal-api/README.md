# Installation Guide


## Python & Environment

```
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```
sudo apt update
sudo apt install -y ffmpeg build-essential
```

```
uv venv .venv
uv sync
```

<!-- ```
source .venv/bin/activate
uv pip install 'torch==2.4.0+cu121' --index-url https://download.pytorch.org/whl/cu121
deactivate
``` -->

```
sudo apt install libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo apt install -y libcudnn8 libcudnn8-dev
```

## Running

```
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```