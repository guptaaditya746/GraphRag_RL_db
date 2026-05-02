from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag.base import QueryParam, QueryResult
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import (
    cypher_query,
    ensure_limit,
    extract_cypher,
    validate_readonly_cypher,
)


@pytest.mark.offline
def test_extract_cypher_strips_fenced_blocks():
    llm_output = """```cypher
MATCH (n)-[r:DIRECTED]-(m)
RETURN n.entity_id AS source, m.entity_id AS target
```"""

    assert (
        extract_cypher(llm_output)
        == "MATCH (n)-[r:DIRECTED]-(m)\nRETURN n.entity_id AS source, m.entity_id AS target"
    )


@pytest.mark.offline
def test_validate_readonly_cypher_accepts_match_query():
    validate_readonly_cypher(
        "MATCH (n)-[r:DIRECTED]-(m) RETURN n.entity_id AS source, m.entity_id AS target LIMIT 5"
    )


@pytest.mark.offline
@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DELETE n RETURN n",
        "CREATE (n) RETURN n",
        "MERGE (n {entity_id: 'A'}) RETURN n",
        "MATCH (n) SET n.flag = true RETURN n",
    ],
)
def test_validate_readonly_cypher_rejects_write_clauses(cypher: str):
    with pytest.raises(ValueError):
        validate_readonly_cypher(cypher)


@pytest.mark.offline
def test_ensure_limit_adds_limit():
    assert ensure_limit("MATCH (n) RETURN n.entity_id AS entity_id", 7).endswith(
        "LIMIT 7"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cypher_query_refuses_storage_without_execute_cypher(monkeypatch):
    monkeypatch.setattr(
        "lightrag.operate.kg_query",
        AsyncMock(
            return_value=QueryResult(
                content="Knowledge Graph Data",
                raw_data={"metadata": {"query_mode": "mix"}},
            )
        ),
    )

    llm_model = AsyncMock(return_value="MATCH (n) RETURN n.entity_id AS entity_id")
    result = await cypher_query(
        query="List entities",
        knowledge_graph_inst=SimpleNamespace(),
        entities_vdb=SimpleNamespace(),
        relationships_vdb=SimpleNamespace(),
        text_chunks_db=SimpleNamespace(),
        query_param=QueryParam(mode="cypher", top_k=5),
        global_config={"llm_model_func": llm_model},
        hashing_kv=None,
        system_prompt=None,
        chunks_vdb=SimpleNamespace(),
    )

    assert result is not None
    assert result.raw_data is not None
    assert result.raw_data["status"] == "failure"
    assert "Neo4j" in result.raw_data["message"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cypher_query_executes_with_mocked_graph(monkeypatch):
    monkeypatch.setattr(
        "lightrag.operate.kg_query",
        AsyncMock(
            return_value=QueryResult(
                content="Knowledge Graph Data",
                raw_data={
                    "metadata": {
                        "query_mode": "mix",
                        "keywords": {
                            "high_level": ["people"],
                            "low_level": ["alice"],
                        },
                    }
                },
            )
        ),
    )

    llm_model = AsyncMock(
        side_effect=[
            "```cypher\nMATCH (n) WHERE toLower(n.entity_id) CONTAINS toLower(\"alice\") RETURN n.entity_id AS entity_id\n```",
            "Found matching rows for Alice.",
        ]
    )
    execute_cypher = AsyncMock(return_value=[{"entity_id": "Alice"}])
    graph = SimpleNamespace(execute_cypher=execute_cypher)

    result = await cypher_query(
        query="Find Alice",
        knowledge_graph_inst=graph,
        entities_vdb=SimpleNamespace(),
        relationships_vdb=SimpleNamespace(),
        text_chunks_db=SimpleNamespace(),
        query_param=QueryParam(mode="cypher", top_k=8),
        global_config={"llm_model_func": llm_model},
        hashing_kv=None,
        system_prompt=None,
        chunks_vdb=SimpleNamespace(),
    )

    assert result is not None
    assert result.content == "Found matching rows for Alice."
    assert result.raw_data is not None
    assert result.raw_data["status"] == "success"
    assert result.raw_data["data"]["results"] == [{"entity_id": "Alice"}]
    assert result.raw_data["data"]["cypher_query"].endswith("LIMIT 8")
    assert result.raw_data["metadata"]["query_mode"] == "cypher"
    assert result.raw_data["metadata"]["result_count"] == 1

    execute_cypher.assert_awaited_once()
    assert execute_cypher.await_args.args[0].endswith("LIMIT 8")
    assert execute_cypher.await_args.kwargs["limit"] == 8


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("NEO4J_URI") or not os.getenv("NEO4J_PASSWORD"),
    reason="Neo4j integration environment is not configured",
)
async def test_neo4j_execute_cypher_integration():
    from lightrag.kg.neo4j_impl import Neo4JStorage

    async def _dummy_embedding(*args, **kwargs):
        return []

    initialize_share_data()
    storage = Neo4JStorage(
        namespace="test_cypher_query_mode",
        workspace="test_cypher_query_mode",
        global_config={"working_dir": "/tmp/test_cypher_query_mode"},
        embedding_func=_dummy_embedding,
    )

    try:
        await storage.initialize()
        rows = await storage.execute_cypher(
            "MATCH (n) RETURN count(n) AS total_nodes",
            limit=10,
        )
    finally:
        await storage.finalize()

    assert len(rows) == 1
    assert "total_nodes" in rows[0]
    assert isinstance(rows[0]["total_nodes"], int)
