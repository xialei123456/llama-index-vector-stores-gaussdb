import math
import logging
import json
from typing import Any, Optional, List, NamedTuple

import numpy as np
from llama_index.core.utils import iter_batch
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreQueryMode,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.core.bridge.pydantic import PrivateAttr

from sqlalchemy import Column, JSON, text, func, TEXT
from sqlalchemy.dialects.postgresql import JSONB

from pygsvector import GsVecClient, FLOATVECTOR, AsyncGsVecClient, IndexType

DEFAULT_GAUSS_BATCH_SIZE = 100
DEFAULT_GAUSS_VECTOR_TABLE_NAME = "llama_vector"
DEFAULT_GAUSS_VECTOR_METRIC_TYPE = "cosine"
DEFAULT_GAUSS_VECTOR_INDEX_TYPE = IndexType.GSDISKANN
DEFAULT_GAUSS_VECTOR_INDEX_PARAM = {
    "pq_nseg": 1,
    "pq_nclus": 16,
    "queue_size": 100,
    "num_parallels": 10,
    "using_clustering_for_parallel": False,
    "lambda_for_balance": 0.00001,
    "enable_pq": True,
    "subgraph_count": 0
}

DEFAULT_GAUSS_PRIMARY_FIELD = "id"
DEFAULT_GAUSS_DOCID_FIELD = "doc_id"
DEFAULT_GAUSS_DOC_FIELD = "text"
DEFAULT_GAUSS_METADATA_FIELD = "metadata"
DEFAULT_GAUSS_VEC_FIELD = "embedding"

DEFAULT_GAUSS_VEC_INDEX_NAME = f"{DEFAULT_GAUSS_VECTOR_TABLE_NAME}_index"
DEFAULT_GAUSS_SPARSE_INDEX_NAME = f"{DEFAULT_GAUSS_VECTOR_TABLE_NAME}_bm25_index"
DEFAULT_GAUSS_SPARSE_INDEX_PARAM = {
    "num_parallels": 16
}


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict
    similarity: float


logger = logging.getLogger(__name__)


def _normalize(vector: List[float]) -> List[float]:
    arr = np.array(vector)
    norm = np.linalg.norm(arr)
    arr = arr / norm
    return arr.tolist()


class GaussVectorStore(BasePydanticVectorStore):
    """
    GaussDB Vector Store.

    Examples:
        `pip install llama-index-vector-stores-gaussdb`

        ```python
        from llama_index.vector_stores.gaussdb import GaussVectorStore

        # Setup ObVecClient
        from pygsvector import GsVecClient

        client = GsVecClient(
            uri="127.0.0.1:8000",
            user="your_username",
            password="your_password",
            db_name="test_db"
        )

        vector_store = GaussVectorStore(
            client=client,
            dim=1024,
            drop_old=True,
            normalize=True,
            table_name="my_vector_table"
        )
        ```

    """
    stores_text: bool = True

    _client: GsVecClient = PrivateAttr()
    _aclient: AsyncGsVecClient = PrivateAttr()
    _dim: int = PrivateAttr()
    _table_name: str = PrivateAttr()
    _vidx_metric_type: str = PrivateAttr()
    _vidx_config: Optional[dict] = PrivateAttr()
    _drop_old: Optional[bool] = PrivateAttr()
    _vidx_type: IndexType = PrivateAttr()
    _primary_field: str = PrivateAttr()
    _doc_id_field: str = PrivateAttr()
    _text_field: str = PrivateAttr()
    _metadata_field: str = PrivateAttr()
    _vector_field: str = PrivateAttr()
    _vidx_name: str = PrivateAttr()
    _partitions: Optional[Any] = PrivateAttr()
    _extra_columns: Optional[List[Column]] = PrivateAttr()
    _normalize: bool = PrivateAttr()
    _use_jsonb: bool = PrivateAttr()
    _enable_sparse: bool = PrivateAttr()
    _sparse_field: str = PrivateAttr()
    _sparse_idx_name: str = PrivateAttr()
    _sparse_idx_config: Optional[dict] = PrivateAttr()
    _is_async_init: bool = PrivateAttr()

    def __init__(
        self,
        dim: int,
        table_name: str = DEFAULT_GAUSS_VECTOR_TABLE_NAME,
        vidx_metric_type: str = DEFAULT_GAUSS_VECTOR_METRIC_TYPE,
        vidx_config: Optional[dict] = None,
        drop_old: bool = False,
        *,
        client: Optional[GsVecClient] = None,
        aclient: Optional[AsyncGsVecClient] = None,
        primary_field: str = DEFAULT_GAUSS_PRIMARY_FIELD,
        doc_id_field: str = DEFAULT_GAUSS_DOCID_FIELD,
        text_field: str = DEFAULT_GAUSS_DOC_FIELD,
        metadata_field: str = DEFAULT_GAUSS_METADATA_FIELD,
        vector_field: str = DEFAULT_GAUSS_VEC_FIELD,
        vidx_name: str = DEFAULT_GAUSS_VEC_INDEX_NAME,
        partitions: Optional[Any] = None,
        extra_columns: Optional[List[Column]] = None,
        normalize: bool = False,
        use_jsonb: bool = False,
        enable_sparse: bool = False,
        sparse_field: str = DEFAULT_GAUSS_DOC_FIELD,
        sparse_idx_name: str = DEFAULT_GAUSS_SPARSE_INDEX_NAME,
        sparse_idx_config: Optional[dict] = None,
    ) -> None:
        """
        Constructor.

        Args:
            client (GsVecClient): GaussDB vector client instance.
            aclient (AsyncGsVecClient): Async GaussDB vector client instance.
            dim (int): Dimension of embedding vector.
            table_name (str): Table name. Defaults to "llama_vector".
            vidx_metric_type (str): Metric method of distance between vectors.
                This parameter takes values in `l2` and `cosine`. Defaults to `cosine`.
            vidx_config (Optional[dict]): Which index params to use.
                Now GaussDB supports gsdiskann only. Refer to `DEFAULT_GAUSS_VECTOR_INDEX_PARAM`
            drop_old (bool): Whether to drop the current table. Defaults to False.
            *,
            primary_field (str): Name of the primary key column. Defaults to "id".
            doc_id_field (str): Name of the doc id column. Defaults to "doc_id".
            text_field (str): Name of the text column. Defaults to "text".
            metadata_field (Optional[str]): Name of the metadata column. Defaults to "metadata".
            vector_field (str): Name of the vector column. Defaults to "embedding".
            vidx_name (str): Name of the vector index. Defaults to "llama_vector_index".
            partitions (Optional[Any]): Partition strategy of table.
            extra_columns (Optional[List[Column]]): Extra sqlalchemy columns to add to the table.
            normalize (bool): normalize vector or not. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.
            enable_sparse (bool): A boolean flag to enable or disable BM25 full-text search. Defaults to False.
            sparse_field (str): Name of BM25 full-text search column. Defaults to "text".
            sparse_idx_name (str): Name of the BM25 index. Defaults to "llama_vector_bm25_index".
            sparse_idx_config (Optional[dict]): The configuration used to build the BM25 index. Defaults to None.

        """
        super().__init__()

        if client is None and aclient is None:
            raise ValueError("Either client or aclient should be provided.")

        self._client = client
        self._aclient = aclient

        self._dim = dim
        self._table_name = table_name
        self._vidx_metric_type = vidx_metric_type
        if self._vidx_metric_type not in {"l2", "cosine"}:
            raise ValueError(f"vidx_metric_type should be l2 or cosine. ")
        self._vidx_type = DEFAULT_GAUSS_VECTOR_INDEX_TYPE
        self._vidx_config = vidx_config or DEFAULT_GAUSS_VECTOR_INDEX_PARAM
        self._drop_old = drop_old

        self._primary_field = primary_field
        self._doc_id_field = doc_id_field
        self._text_field = text_field
        self._metadata_field = metadata_field
        self._vector_field = vector_field
        self._vidx_name = vidx_name
        self._partitions = partitions
        self._extra_columns = extra_columns
        self._normalize = normalize
        self._use_jsonb = use_jsonb

        self._enable_sparse = enable_sparse
        self._sparse_field = sparse_field
        self._sparse_idx_name = sparse_idx_name
        self._sparse_idx_config = sparse_idx_config or DEFAULT_GAUSS_SPARSE_INDEX_PARAM

        self._is_async_init = False

        if self._client is not None:
            self._create_table_with_index()

    def _create_table_with_index(self):
        """Create table and create index."""
        if self._drop_old:
            self._client.drop_table_if_exist(table_name=self._table_name)

        if self._client.check_table_exists(self._table_name):
            logger.info(f"table {self._table_name} already exists, skip")
            return

        metadata_dtype = JSONB if self._use_jsonb else JSON

        cols = [
            Column(self._primary_field, TEXT, primary_key=True, autoincrement=False),
            Column(self._doc_id_field, TEXT),
            Column(self._text_field, TEXT),
            Column(self._metadata_field, metadata_dtype),
            Column(self._vector_field, FLOATVECTOR(self._dim)),
        ]
        if self._extra_columns:
            cols.extend(self._extra_columns)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self._vector_field,
            index_type=self._vidx_type,
            index_name=self._vidx_name,
            metric_type=self._vidx_metric_type,
            params=self._vidx_config
        )

        if self._enable_sparse:
            index_params.add_index(
                field_name=self._sparse_field,
                index_type=IndexType.BM25,
                index_name=self._sparse_idx_name,
                num_parallels=self._sparse_idx_config.get("num_parallels", 16)
            )

        self._client.create_table_with_index_params(
            table_name=self._table_name,
            columns=cols,
            indexes=None,
            index_params=index_params,
            partitions=self._partitions,
        )

    async def _acreate_table_with_index(self):
        """Create table and create index."""
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        if self._drop_old:
            await self._aclient.drop_table_if_exist(table_name=self._table_name)

        if await self._aclient.check_table_exists(self._table_name):
            logger.info(f"table {self._table_name} already exists, skip")
            return

        metadata_dtype = JSONB if self._use_jsonb else JSON

        cols = [
            Column(self._primary_field, TEXT, primary_key=True, autoincrement=False),
            Column(self._doc_id_field, TEXT),
            Column(self._text_field, TEXT),
            Column(self._metadata_field, metadata_dtype),
            Column(self._vector_field, FLOATVECTOR(self._dim)),
        ]
        if self._extra_columns:
            cols.extend(self._extra_columns)

        index_params = self._aclient.prepare_index_params()
        index_params.add_index(
            field_name=self._vector_field,
            index_type=self._vidx_type,
            index_name=self._vidx_name,
            metric_type=self._vidx_metric_type,
            params=self._vidx_config
        )

        if self._enable_sparse:
            index_params.add_index(
                field_name=self._sparse_field,
                index_type=IndexType.BM25,
                index_name=self._sparse_idx_name,
                num_parallels=self._sparse_idx_config.get("num_parallels", 16)
            )

        await self._aclient.create_table_with_index_params(
            table_name=self._table_name,
            columns=cols,
            indexes=None,
            index_params=index_params,
            partitions=self._partitions,
        )

    async def _async_init(self):
        if not self._is_async_init:
            await self._acreate_table_with_index()
            self._is_async_init = True

    @staticmethod
    def _to_gauss_operator(operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return "="
        elif operator == FilterOperator.NE:
            return "!="
        elif operator == FilterOperator.GT:
            return ">"
        elif operator == FilterOperator.LT:
            return "<"
        elif operator == FilterOperator.GTE:
            return ">="
        elif operator == FilterOperator.LTE:
            return "<="
        elif operator == FilterOperator.IN:
            return "IN"
        elif operator == FilterOperator.NIN:
            return "NOT IN"
        elif operator == FilterOperator.ANY:
            return "?|"
        elif operator == FilterOperator.ALL:
            return "?&"
        elif operator == FilterOperator.TEXT_MATCH:
            return "LIKE"
        elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
            return "ILIKE"
        elif operator == FilterOperator.CONTAINS:
            return "@>"
        elif operator == FilterOperator.IS_EMPTY:
            return "IS NULL"
        else:
            msg = f"Operator {operator} is not supported in GaussVectorStore"
            logger.error(msg)
            raise ValueError(msg)

    def _to_gauss_filter(self, metadata_filters: Optional[MetadataFilters] = None) -> str:
        filters = []
        for filter in metadata_filters.filters:
            if isinstance(filter, MetadataFilters):
                filters.append(f"({self._to_gauss_filter(filter)})")
                continue
            elif filter.operator in [
                FilterOperator.EQ,
                FilterOperator.NE,
                FilterOperator.GT,
                FilterOperator.LT,
                FilterOperator.GTE,
                FilterOperator.LTE
            ]:
                # Check if value is a number. If so, cast the metadata value to a float
                # This is necessary because the metadata is stored as a string
                try:
                    filters.append(
                        f"({self._metadata_field}->>'{filter.key}')::float "
                        f"{self._to_gauss_operator(filter.operator)} {float(filter.value)}"
                    )
                except ValueError:
                    filters.append(
                        f"{self._metadata_field}->>'{filter.key}' "
                        f"{self._to_gauss_operator(filter.operator)} '{filter.value}'"
                    )
            elif filter.operator in [FilterOperator.IN, FilterOperator.NIN]:
                filter_value = ",".join(f"'{v}'" for v in filter.value)
                filters.append(
                    f"{self._metadata_field}->>'{filter.key}' "
                    f"{self._to_gauss_operator(filter.operator)} ({filter_value})"
                )
            elif filter.operator in [FilterOperator.ANY, FilterOperator.ALL]:
                filter_value = ",".join(f"'{v}'" for v in filter.value)
                filters.append(
                    f"{self._metadata_field}::jsonb->'{filter.key}' "
                    f"{self._to_gauss_operator(filter.operator)} ARRAY[{filter_value}]"
                )
            elif filter.operator in [FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE]:
                filters.append(
                    f"{self._metadata_field}->>'{filter.key}' "
                    f"{self._to_gauss_operator(filter.operator)} '%{filter.value}%'"
                )
            elif filter.operator == FilterOperator.CONTAINS:
                filters.append(
                    f"{self._metadata_field}::jsonb->'{filter.key}' "
                    f"{self._to_gauss_operator(filter.operator)} '[\"{filter.value}\"]'"
                )
            elif filter.operator == FilterOperator.IS_EMPTY:
                filters.append(
                    f"({self._metadata_field}->>'{filter.key}') "
                    f"{self._to_gauss_operator(filter.operator)}"
                )
            else:
                msg = f"Operator {filter.operator} is not supported by GaussDB"
                logger.error(msg)
                raise ValueError(msg)
        return f" {metadata_filters.condition.value} ".join(filters)

    def _parse_metric_type_str_to_dist_func(self) -> Any:
        if self._vidx_metric_type == "l2":
            return func.l2_distance
        elif self._vidx_metric_type == "cosine":
            return func.cosine_distance
        else:
            msg = f"Invalid vector index metric type: {self._vidx_metric_type}"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _cosine_similarity(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return 1.0 - distance

    @staticmethod
    def _euclidean_similarity(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return 1.0 - distance / math.sqrt(2)

    def _parse_distance_to_similarities(self, distance: float) -> float:
        """Convert distance into relevance score"""
        if self._vidx_metric_type == "cosine":
            return self._cosine_similarity(distance)
        elif self._vidx_metric_type == "l2":
            return self._euclidean_similarity(distance)
        else:
            raise ValueError(f"Metric Type {self._vidx_metric_type} is not supported")

    @classmethod
    def from_params(
        cls,
        dim: int,
        table_name: str = DEFAULT_GAUSS_VECTOR_TABLE_NAME,
        vidx_metric_type: str = DEFAULT_GAUSS_VECTOR_METRIC_TYPE,
        vidx_config: Optional[dict] = None,
        drop_old: bool = False,
        *,
        client: Optional[GsVecClient] = None,
        aclient: Optional[AsyncGsVecClient] = None,
        primary_field: str = DEFAULT_GAUSS_PRIMARY_FIELD,
        doc_id_field: str = DEFAULT_GAUSS_DOCID_FIELD,
        text_field: str = DEFAULT_GAUSS_DOC_FIELD,
        metadata_field: str = DEFAULT_GAUSS_METADATA_FIELD,
        vector_field: str = DEFAULT_GAUSS_VEC_FIELD,
        vidx_name: str = DEFAULT_GAUSS_VEC_INDEX_NAME,
        partitions: Optional[Any] = None,
        extra_columns: Optional[List[Column]] = None,
        normalize: bool = False,
        use_jsonb: bool = False,
        enable_sparse: bool = False,
        sparse_field: str = DEFAULT_GAUSS_DOC_FIELD,
        sparse_idx_name: str = DEFAULT_GAUSS_SPARSE_INDEX_NAME,
        sparse_idx_config: Optional[dict] = None,
    ) -> "GaussVectorStore":
        """Construct from params.

        Args:
            client (GsVecClient): GaussDB vector client instance.
            aclient (AsyncGsVecClient): Async GaussDB vector client instance.
            dim (int): Dimension of embedding vector.
            table_name (str): Table name. Defaults to "llama_vector".
            vidx_metric_type (str): Metric method of distance between vectors.
                This parameter takes values in `l2` and `cosine`. Defaults to `cosine`.
            vidx_config (Optional[dict]): Which index params to use.
                Now GaussDB supports gsdiskann only. Refer to `DEFAULT_GAUSS_VECTOR_INDEX_PARAM`.
            drop_old (bool): Whether to drop the current table. Defaults to False.
            *,
            primary_field (str): Name of the primary key column. Defaults to "id".
            doc_id_field (str): Name of the doc id column. Defaults to "doc_id".
            text_field (str): Name of the text column. Defaults to "text".
            metadata_field (Optional[str]): Name of the metadata column. Defaults to "metadata".
            vector_field (str): Name of the vector column. Defaults to "embedding".
            vidx_name (str): Name of the vector index. Defaults to "llama_vector_index".
            partitions (Optional[Any]): Partition strategy of table.
            extra_columns (Optional[List[Column]]): Extra sqlalchemy columns to add to the table.
            normalize (bool): normalize vector or not. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.
            enable_sparse (bool): A boolean flag to enable or disable BM25 full-text search. Defaults to False.
            sparse_field (str): Name of BM25 full-text search column. Defaults to "text".
            sparse_idx_name (str): Name of the BM25 index. Defaults to "llama_vector_bm25_index".
            sparse_idx_config (Optional[dict]): The configuration used to build the BM25 index. Defaults to None.

        Returns:
            GaussVectorStore: Instance of GaussVectorStore constructed from params.

        """
        return cls(
            client=client,
            dim=dim,
            table_name=table_name,
            vidx_metric_type=vidx_metric_type,
            vidx_config=vidx_config,
            drop_old=drop_old,
            aclient=aclient,
            primary_field=primary_field,
            doc_id_field=doc_id_field,
            text_field=text_field,
            metadata_field=metadata_field,
            vector_field=vector_field,
            vidx_name=vidx_name,
            partitions=partitions,
            extra_columns=extra_columns,
            normalize=normalize,
            use_jsonb=use_jsonb,
            enable_sparse=enable_sparse,
            sparse_field=sparse_field,
            sparse_idx_name=sparse_idx_name,
            sparse_idx_config=sparse_idx_config
        )

    @classmethod
    def class_name(cls) -> str:
        return "GaussVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    @property
    def aclient(self) -> Any:
        """Get async client."""
        return self._aclient

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None
    ) -> List[BaseNode]:
        """
        Get nodes from vector store.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete.
                Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters.
                Defaults to None.

        Returns:
            List[BaseNode]: List of text nodes.

        """
        if self._client is None:
            raise ValueError("Client should be initialized.")

        if filters is not None:
            filter = self._to_gauss_filter(filters)
        else:
            filter = None

        res = self._client.get(
            table_name=self._table_name,
            ids=node_ids,
            where_clause=[text(filter)] if filter is not None else None,
            output_column_names=[self._text_field, self._metadata_field],
        )

        return [
            metadata_dict_to_node(
                metadata=(json.loads(r[1]) if not isinstance(r[1], dict) else r[1]),
                text=r[0],
            )
            for r in res.fetchall()
        ]

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None
    ) -> List[BaseNode]:
        """Asynchronously get nodes from vector store."""
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        await self._async_init()

        if filters is not None:
            filter = self._to_gauss_filter(filters)
        else:
            filter = None

        res = await self._aclient.get(
            table_name=self._table_name,
            ids=node_ids,
            where_clause=[text(filter)] if filter is not None else None,
            output_column_names=[self._text_field, self._metadata_field],
        )

        return [
            metadata_dict_to_node(
                metadata=(json.loads(r[1]) if not isinstance(r[1], dict) else r[1]),
                text=r[0],
            )
            for r in res.fetchall()
        ]

    def add(
        self,
        nodes: List[BaseNode],
        batch_size: Optional[int] = None,
        extras: Optional[List[dict]] = None
    ) -> List[str]:
        """
        Add nodes with embedding to vector store.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings to insert.
            batch_size (Optional[int]): Insert nodes in batch.
            extras (Optional[List[dict]]): If `extra_columns` is set when initializing GaussVectorStore,
                you can add nodes with extra infos.

        Returns:
            List[str]: List of ids inserted.

        """
        if self._client is None:
            raise ValueError("Client should be initialized.")

        batch_size = batch_size or DEFAULT_GAUSS_BATCH_SIZE

        extra_data = extras or [{} for _ in nodes]
        if len(nodes) != len(extra_data):
            error_msg = "length of extras should be the same as nodes."
            logger.error(error_msg)
            raise ValueError(error_msg)

        data = [
            {
                self._primary_field: node.id_,
                self._doc_id_field: node.ref_doc_id or None,
                self._text_field: node.get_content(metadata_mode=MetadataMode.NONE),
                self._metadata_field: node_to_metadata_dict(node, remove_text=True),
                self._vector_field: (
                    node.get_embedding()
                    if not self._normalize
                    else _normalize(node.get_embedding())
                ),
                **extra,
            }
            for node, extra in zip(nodes, extra_data)
        ]

        for data_batch in iter_batch(data, batch_size):
            self._client.insert(table_name=self._table_name, data=data_batch)

        return [node.id_ for node in nodes]

    async def async_add(
        self,
        nodes: List[BaseNode],
        batch_size: Optional[int] = None,
        extras: Optional[List[dict]] = None
    ) -> List[str]:
        """ Asynchronously add nodes with embedding to vector store."""
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        await self._async_init()

        batch_size = batch_size or DEFAULT_GAUSS_BATCH_SIZE

        extra_data = extras or [{} for _ in nodes]
        if len(nodes) != len(extra_data):
            error_msg = "length of extras should be the same as nodes."
            logger.error(error_msg)
            raise ValueError(error_msg)

        data = [
            {
                self._primary_field: node.id_,
                self._doc_id_field: node.ref_doc_id or None,
                self._text_field: node.get_content(metadata_mode=MetadataMode.NONE),
                self._metadata_field: node_to_metadata_dict(node, remove_text=True),
                self._vector_field: (
                    node.get_embedding()
                    if not self._normalize
                    else _normalize(node.get_embedding())
                ),
                **extra,
            }
            for node, extra in zip(nodes, extra_data)
        ]

        for data_batch in iter_batch(data, batch_size):
            await self._aclient.insert(table_name=self._table_name, data=data_batch)

        return [node.id_ for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if self._client is None:
            raise ValueError("Client should be initialized.")

        self._client.delete(
            table_name=self._table_name,
            where_clause=[text(f"{self._doc_id_field} = '{ref_doc_id}'")]
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronously delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        await self._async_init()

        await self._aclient.delete(
            table_name=self._table_name,
            where_clause=[text(f"{self._doc_id_field} = '{ref_doc_id}'")]
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any
    ) -> None:
        """
        Deletes nodes from vector store.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete.
                Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters.
                Defaults to None.

        """
        if self._client is None:
            raise ValueError("Client should be initialized.")

        if filters is not None:
            filter = self._to_gauss_filter(filters)
        else:
            filter = None

        self._client.delete(
            table_name=self._table_name,
            ids=node_ids,
            where_clause=[text(filter)] if filter is not None else None
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any
    ) -> None:
        """
        Asynchronously deletes nodes from vector store.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete.
                Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters.
                Defaults to None.

        """
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        await self._async_init()

        if filters is not None:
            filter = self._to_gauss_filter(filters)
        else:
            filter = None

        await self._aclient.delete(
            table_name=self._table_name,
            ids=node_ids,
            where_clause=[text(filter)] if filter is not None else None
        )

    def clear(self) -> None:
        """Clear all nodes from configured vector store."""
        if self._client is None:
            raise ValueError("Client should be initialized.")

        self._client.perform_raw_text_sql(f"TRUNCATE TABLE {self._table_name}")

    async def aclear(self) -> None:
        """Asynchronously clear all nodes from configured vector store."""
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        await self._async_init()

        await self._aclient.perform_raw_text_sql(f"TRUNCATE TABLE {self._table_name}")

    @staticmethod
    def _dedup_results(results: List[DBEmbeddingRow]) -> List[DBEmbeddingRow]:
        """Deduplicate and combine dense and sparse results"""
        seen_ids = set()
        deduped_results = []
        for result in results:
            if result.node_id not in seen_ids:
                deduped_results.append(result)
                seen_ids.add(result.node_id)
        return deduped_results

    @staticmethod
    def _db_rows_to_query_result(rows: List[DBEmbeddingRow]) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for db_embedding_row in rows:
            node = metadata_dict_to_node(
                metadata=db_embedding_row.metadata,
                text=db_embedding_row.text,
            )
            nodes.append(node)
            similarities.append(db_embedding_row.similarity)
            ids.append(db_embedding_row.node_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _default_search(self, query: VectorStoreQuery, **kwargs: Any) -> List[DBEmbeddingRow]:
        """Perform default search: dense embedding search."""
        if query.filters is not None:
            filter = self._to_gauss_filter(query.filters)
        else:
            filter = None

        res = self._client.ann_search(
            table_name=self._table_name,
            vec_data=(
                query.query_embedding
                if not self._normalize
                else _normalize(query.query_embedding)
            ),
            vec_column_name=self._vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            topk=query.similarity_top_k,
            output_column_names=[
                self._primary_field,
                self._text_field,
                self._metadata_field
            ],
            where_clause=[text(filter)] if filter is not None else None,
        )

        return [
            DBEmbeddingRow(
                node_id=r[0],
                text=r[1],
                metadata=(json.loads(r[2]) if not isinstance(r[2], dict) else r[2]),
                similarity=self._parse_distance_to_similarities(r[3])
            )
            for r in res.fetchall()
        ]

    async def _adefault_search(self, query: VectorStoreQuery, **kwargs: Any) -> List[DBEmbeddingRow]:
        """Asynchronously perform default search: dense embedding search."""
        if query.filters is not None:
            filter = self._to_gauss_filter(query.filters)
        else:
            filter = None

        res = await self._aclient.ann_search(
            table_name=self._table_name,
            vec_data=(
                query.query_embedding
                if not self._normalize
                else _normalize(query.query_embedding)
            ),
            vec_column_name=self._vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            topk=query.similarity_top_k,
            output_column_names=[
                self._primary_field,
                self._text_field,
                self._metadata_field
            ],
            where_clause=[text(filter)] if filter is not None else None,
        )

        return [
            DBEmbeddingRow(
                node_id=r[0],
                text=r[1],
                metadata=(json.loads(r[2]) if not isinstance(r[2], dict) else r[2]),
                similarity=self._parse_distance_to_similarities(r[3])
            )
            for r in res.fetchall()
        ]

    def _sparse_search(self, query: VectorStoreQuery, **kwargs: Any) -> List[DBEmbeddingRow]:
        """Perform BM25 full-text search."""
        if query.filters is not None:
            filter = self._to_gauss_filter(query.filters)
        else:
            filter = None

        res = self._client.bm25_search(
            table_name=self._table_name,
            search_text=query.query_str,
            column_name=self._text_field,
            with_score=True,
            topk=query.sparse_top_k,
            output_column_names=[
                self._primary_field,
                self._text_field,
                self._metadata_field
            ],
            where_clause=[text(filter)] if filter is not None else None,
        )

        return [
            DBEmbeddingRow(
                node_id=r[0],
                text=r[1],
                metadata=(json.loads(r[2]) if not isinstance(r[2], dict) else r[2]),
                similarity=r[3]
            )
            for r in res.fetchall()
        ]

    async def _asparse_search(self, query: VectorStoreQuery, **kwargs: Any) -> List[DBEmbeddingRow]:
        """Asynchronously perform BM25 full-text search."""
        if query.filters is not None:
            filter = self._to_gauss_filter(query.filters)
        else:
            filter = None

        res = await self._aclient.bm25_search(
            table_name=self._table_name,
            search_text=query.query_str,
            column_name=self._text_field,
            with_score=True,
            topk=query.sparse_top_k,
            output_column_names=[
                self._primary_field,
                self._text_field,
                self._metadata_field
            ],
            where_clause=[text(filter)] if filter is not None else None,
        )

        return [
            DBEmbeddingRow(
                node_id=r[0],
                text=r[1],
                metadata=(json.loads(r[2]) if not isinstance(r[2], dict) else r[2]),
                similarity=r[3]
            )
            for r in res.fetchall()
        ]

    def _hybrid_search(self, query: VectorStoreQuery, **kwargs: Any) -> List[DBEmbeddingRow]:
        """Perform hybrid search."""
        if query.alpha is not None:
            logger.warning("GaussDB hybrid search does not support alpha parameter.")

        dense_result = self._default_search(query, **kwargs)
        sparse_result = self._sparse_search(query, **kwargs)
        results = dense_result + sparse_result
        return self._dedup_results(results)

    async def _ahybrid_search(self, query: VectorStoreQuery, **kwargs: Any) -> List[DBEmbeddingRow]:
        """Asynchronously perform hybrid search."""
        if query.alpha is not None:
            logger.warning("GaussDB hybrid search does not support alpha parameter.")

        dense_result = await self._adefault_search(query, **kwargs)
        sparse_result = await self._asparse_search(query, **kwargs)
        results = dense_result + sparse_result
        return self._dedup_results(results)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        if self._client is None:
            raise ValueError("Client should be initialized.")

        if query.mode == VectorStoreQueryMode.DEFAULT:
            results = self._default_search(query, **kwargs)
        elif query.mode in [
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH
        ]:
            if self._enable_sparse is False:
                raise ValueError(f"The query mode {query.mode} requires sparse, but enable_sparse is False.")
            results = self._sparse_search(query, **kwargs)
        elif query.mode == VectorStoreQueryMode.HYBRID:
            if self._enable_sparse is False:
                raise ValueError(f"The query mode {query.mode} requires sparse, but enable_sparse is False.")
            results = self._hybrid_search(query, **kwargs)
        else:
            raise ValueError(f"GaussDB does not support {query.mode} yet.")

        return self._db_rows_to_query_result(results)

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Asynchronously query vector store."""
        if self._aclient is None:
            raise ValueError("Async client should be initialized.")

        await self._async_init()

        if query.mode == VectorStoreQueryMode.DEFAULT:
            results = await self._adefault_search(query, **kwargs)
        elif query.mode in [
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH
        ]:
            if self._enable_sparse is False:
                raise ValueError(f"The query mode {query.mode} requires sparse, but enable_sparse is False.")
            results = await self._asparse_search(query, **kwargs)
        elif query.mode == VectorStoreQueryMode.HYBRID:
            if self._enable_sparse is False:
                raise ValueError(f"The query mode {query.mode} requires sparse, but enable_sparse is False.")
            results = await self._ahybrid_search(query, **kwargs)
        else:
            raise ValueError(f"GaussDB does not support {query.mode} yet.")

        return self._db_rows_to_query_result(results)
