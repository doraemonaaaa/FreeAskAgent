cd plugins/sam3
uv pip install --no-cache .
uv pip install "numpy>=2.0,<2.8"
uv pip install pycocotools
uv pip install scikit-learn
export UV_CACHE_DIR=$HOME/.cache/uv
uv pip install --reinstall "setuptools<82"
uv pip install --no-cache 'httpx[socks]'