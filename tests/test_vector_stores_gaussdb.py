import logging
from typing import List, Dict

import pytest
from sqlalchemy import Column, TEXT

from llama_index.vector_stores.gaussdb import GaussVectorStore

try:
    from pygsvector import GsVecClient

    CONN_ARGS: Dict[str, str] = {
        "uri": "10.25.106.116:6899",
        "user": "llamaindex_gv",
        "password": "Gauss_234",
        "db_name": "postgres",
    }

    # test client
    client = GsVecClient(**CONN_ARGS)

    gaussdb_available = True
except Exception as e:
    gaussdb_available = False

from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery, VectorStoreQueryMode, FilterOperator,
)

ADA_TOKEN_COUNT = 1024

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    # Pad or trim the list to make it of length ADA_TOKEN_COUNT
    return ascii_values[:ADA_TOKEN_COUNT] + [0.0] * (
        ADA_TOKEN_COUNT - len(ascii_values)
    )


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    """Return a list of TextNodes with embeddings."""
    return [
        TextNode(
            text="foo",
            id_="ffddfb6b-2cad-48ec-917e-6a7dfaba3c9e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "theme": "FOO",
                "location": 1,
                "tags": ["report", "revenue", "profit"],
            },
            embedding=text_to_embedding("foo"),
        ),
        TextNode(
            text="bar",
            id_="6f74bc29-8d84-4c3c-b458-e6a082cf1938",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "theme": "BAR",
                "location": 2,
                "tags": ["report", "revenue"],
            },
            embedding=text_to_embedding("bar"),
        ),
        TextNode(
            text="baz",
            id_="e99b4f5f-5bc5-4cff-8c4e-ee1d66479a11",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "theme": "BAZ",
                "location": 3,
                "tags": ["recruitment", "plan"],
            },
            embedding=text_to_embedding("baz"),
        ),
    ]


def test_class():
    names_of_base_classes = [b.__name__ for b in GaussVectorStore.__mro__]
    assert GaussVectorStore.__name__ in names_of_base_classes


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_init_client():
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
    )


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_init_client_from_params(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore.from_params(
        client=client,
        dim=1024,
    )


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_init_client_all_params(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore.from_params(
        client=client,
        dim=1024,
        table_name="test_llama_vector",
        vidx_metric_type="cosine",
        vidx_config={
            "pq_nseg": 128,
            "pq_nclus": 16,
            "num_parallels": 50,
            "enable_pq": True,
            "using_clustering_for_parallel": False
        },
        drop_old=True,
        primary_field="test_id",
        doc_id_field="test_doc_id",
        text_field="test_text",
        metadata_field="test_metadata",
        vector_field="test_embedding",
        vidx_name="test_llama_vector_index",
        extra_columns=[
            Column("extra_column1", TEXT),
            Column("extra_column2", TEXT),
        ],
        normalize=True,
        use_jsonb=True,
        enable_sparse=True,
        sparse_field="test_text",
        sparse_idx_name="test_llama_vector_bm25_index",
        sparse_idx_config={"num_parallels": 16},
    )


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_add_node(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
    )

    gaussdb.add(node_embeddings)


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_search_with_cosine_distance(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
    )

    gaussdb.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=1)

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_search_with_l2_distance(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
        vidx_metric_type="l2",
    )

    gaussdb.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=1)

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 1.0
    )
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_delete_doc(node_embeddings: List[TextNode]):
    client = GsVecClient(
        **CONN_ARGS,
    )

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
        vidx_metric_type="l2",
    )

    gaussdb.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=3)

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 3
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id

    gaussdb.delete(ref_doc_id="test-1")

    q = VectorStoreQuery(query_embedding=text_to_embedding("baz"), similarity_top_k=3)

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 1.0
    )
    assert result.ids is not None and result.ids[0] == node_embeddings[2].node_id


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_delete_nodes_and_get_nodes(node_embeddings: List[TextNode]):
    client = GsVecClient(
        **CONN_ARGS,
    )

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
    )

    gaussdb.add(node_embeddings)

    result = gaussdb.get_nodes()
    assert len(result) == 3

    result = gaussdb.get_nodes(
        node_ids=[
            node_embeddings[1].id_,
            node_embeddings[2].id_,
        ],
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">"),
            ]
        ),
    )
    assert len(result) == 1

    gaussdb.delete_nodes(
        node_ids=[
            node_embeddings[1].id_,
            node_embeddings[2].id_,
        ],
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">"),
            ]
        ),
    )
    result = gaussdb.get_nodes(
        node_ids=[
            node_embeddings[0].id_,
            node_embeddings[1].id_,
            node_embeddings[2].id_,
        ],
    )
    assert len(result) == 2
    assert (
        result[0].id_ == node_embeddings[0].id_
        and result[1].id_ == node_embeddings[1].id_
    )


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_clear(node_embeddings: List[TextNode]):
    client = GsVecClient(
        **CONN_ARGS,
    )

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
    )

    gaussdb.add(node_embeddings)
    gaussdb.clear()


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_search_with_filter(node_embeddings: List[TextNode]):
    client = GsVecClient(
        **CONN_ARGS,
    )

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
    )

    gaussdb.add(node_embeddings)

    # >=
    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">="),
            ]
        ),
    )

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 2
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[1].text
        and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )

    if hasattr(FilterOperator, "ANY"):
        # in
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="theme", value=["FOO", "BAR"], operator="in"),
                ]
            ),
        )

        result = gaussdb.query(q)
        assert result.nodes is not None and len(result.nodes) == 2
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
            and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[1].text
        )

        # any
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="theme", value=["FOO", "BAR"], operator="any"),
                ]
            ),
        )

        result = gaussdb.query(q)
        assert result.nodes is not None and len(result.nodes) == 2
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
            and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[1].text
        )

        # text_match
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="theme", value="BA", operator="text_match"),
                ]
            ),
        )

        result = gaussdb.query(q)
        assert result.nodes is not None and len(result.nodes) == 2

        # contains
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="tags", value=["report"], operator="contains"),
                ]
            ),
        )

        result = gaussdb.query(q)
        assert result.nodes is not None and len(result.nodes) == 2
        assert (
            result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
            == node_embeddings[0].text
        )

        # is_empty
        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="theme", value=None, operator="is_empty"),
                ]
            ),
        )

        result = gaussdb.query(q)
        assert result.nodes is not None and len(result.nodes) == 0

        q = VectorStoreQuery(
            query_embedding=text_to_embedding("foo"),
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="labels", value=None, operator="is_empty"),
                ]
            ),
        )

        result = gaussdb.query(q)
        assert result.nodes is not None and len(result.nodes) == 3

    # and
    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">="),
                MetadataFilter(key="theme", value="BAZ", operator="=="),
            ],
            condition="and",
        ),
    )

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )

    # or
    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">="),
                MetadataFilter(key="theme", value="FOO", operator="=="),
            ],
            condition="or",
        ),
    )

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 3
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
        and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[1].text
        and result.nodes[2].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_sparse_query(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
        enable_sparse=True,
    )

    gaussdb.add(node_embeddings)

    q = VectorStoreQuery(
        sparse_top_k=3,
        query_str="foo",
        mode=VectorStoreQueryMode.TEXT_SEARCH,
    )

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert result.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == node_embeddings[0].text
    assert result.similarities is not None and result.similarities[0] == pytest.approx(1.0, abs=0.05)
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id


@pytest.mark.skipif(not gaussdb_available, reason="gaussdb is not available")
def test_hybrid_query(node_embeddings: List[TextNode]):
    client = GsVecClient(**CONN_ARGS)

    # Initialize GaussVectorStore
    gaussdb = GaussVectorStore(
        client=client,
        dim=1024,
        drop_old=True,
        normalize=True,
        enable_sparse=True,
    )

    gaussdb.add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=1,
        sparse_top_k=1,
        query_str="baz",
        mode=VectorStoreQueryMode.HYBRID,
    )

    result = gaussdb.query(q)
    assert result.nodes is not None and len(result.nodes) == 2
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
        and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 1
        and result.similarities[1] == pytest.approx(1.0, abs=0.05)
    )
    assert (
       result.ids is not None and result.ids[0] == node_embeddings[0].node_id
       and result.ids[1] == node_embeddings[2].node_id
    )
