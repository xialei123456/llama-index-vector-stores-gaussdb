# LlamaIndex Vector-Stores Integration: GaussDB

An integration package connecting GaussDB and LlamaIndex, supports quick connection to GaussDB and integrates LlamaIndex workflows.

## Installation
- Before using the GaussDB vector capability in LlamaIndex, you need to install the pygsvector Python SDK.

- git clone this repo, then install with:

```shell
poetry install
```

- install with pip (already been released):
```shell
pip install llama_index-0.1.0-py3-none-any.whl
```

## Usage

```python
from llama_index.vector_stores.gaussdb import GaussVectorStore
from pygsvector import GsVecClient

client = GsVecClient()

# Initialize GaussVectorStore
gauss = GaussVectorStore(
    client=client,
    dim=1536,
    drop_old=True,
    normalize=True,
)
```