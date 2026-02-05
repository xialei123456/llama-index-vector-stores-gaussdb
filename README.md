# LlamaIndex Vector-Stores Integration: GaussDB

An integration package connecting GaussDB and LlamaIndex, supports quick connection to GaussDB and integrates LlamaIndex workflows.

## Installation
- Before using the GaussDB vector capability in LlamaIndex, you need to install the pygsvector Python SDK. Such as:

```shell
pip install pygsvector-0.1.0-py3-none-any.whl
```

- Then, git clone this repo, install with:

```shell
poetry install
```

- Or, install with pip (already been released):
```shell
pip install llama_index-0.1.0-py3-none-any.whl
```

## Quick Start

### Setup OpenAI

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Prepare documents

```shell
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)
```

### Initialization

```python
from pygsvector import GsVecClient
from llama_index.vector_stores.gaussdb import GaussVectorStore

client = GsVecClient(
    uri="10.25.106.116:6899",
    user="llamaindex_gv",
    password=" ",
    db_name="postgres",
)

vector_store = GaussVectorStore.from_params(
    client=client,
    dim=1024,  # openai embedding dimension
    table_name="llama_vector",
    vidx_config={
        "pq_nseg": 1,
        "pq_nclus": 16,
        "num_parallels": 10,
        "enable_pq": True,
    },
    drop_old=True,
)
```

### Create the index

```python
from llama_index.core import StorageContext, VectorStoreIndex

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
```

### Query the index

```python
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What did the author do?")

print(response.response)
print("来源节点:", [src.node.get_content()[:100] for src in response.source_nodes])
```
