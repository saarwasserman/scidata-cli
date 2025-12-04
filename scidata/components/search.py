import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List

import numpy as np
from opensearchpy import OpenSearch, AsyncOpenSearch, AWSV4SignerAuth
from opensearchpy.helpers import async_bulk
from openai import OpenAI, AsyncOpenAI

from scidata.config import settings
from scidata import logger

@dataclass
class BaseDocument:
    id: str
    embedding: List[float] | None

    def __post_init__(self):
        if self.id is None:
            raise ValueError("must provide unique <id> field")

class HybridSearchApp:

    OPENSEARCH_INDEX_BODY = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text"
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 100,
                            "m": 16
                        }
                    }
                }
            }
        }
    }

    def __init__(self, index: str, field: str, model: str):
        """Initialize hybrid search app

        Args:
            index (str): db index name for a specific app
            field (str): the field name that will hold the actual text content to work on
            model (str): the model name to use for embeddings
        """
        # user must use context manager so that clients are properly opened/closed
        self.db_index = index
        self.field = field
        self.model = model

        self.opensearch_client = None
        self.openai_client = None

    async def __aenter__(self):
        
        self.opensearch_client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=(settings.opensearch_username, settings.opensearch_password),
            use_ssl=True,
            verify_certs=False
        )

        # create index if not exists
        try:
            index_exists = await self.opensearch_client.indices.exists(index=self.db_index)
            if not index_exists:
                logger.info("Creating index", extra={"index": self.db_index})
                await self.create_db_index()
        except Exception as e:
            logger.warning("Could not check if index exists, will attempt to create", extra={"error": str(e)})
            try:
                await self.create_db_index()
            except Exception as create_error:
                logger.debug("Index creation failed or already exists", extra={"error": str(create_error)})

        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key,
                                         organization=settings.openai_organization_id,
                                         project=settings.openai_project_id)
        
        logger.debug("Initialized OpenSearch and OpenAI clients")

        return self

    async def __aexit__(self, exc_type, exc_value, tb):
        if self.opensearch_client:
            await self.opensearch_client.close()
        if self.openai_client:
            await self.openai_client.close()

    async def create_db_index(self):
        """Create OpenSearch index with hybrid search mappings"""

        try:
            createRes = await self.opensearch_client.indices.create(
                index=self.db_index, body=HybridSearchApp.OPENSEARCH_INDEX_BODY)
            getRes = await self.opensearch_client.indices.get(self.db_index)
            logger.debug("Index created successfully", extra={"index": self.db_index, "status": "created"})
        except Exception as e:
            logger.error("Failed to create index", extra={"index": self.db_index, "error": str(e)})

    ### TODO: Chunking Methods ###
    def chunks(self):
        pass

    ### TODO: Embeddings Methods - Batch <-> Pipeline, Async/Sync <-> InPlace ###
        
    # async def index_batch(self, objs: List[BaseDocument]):
    #     documents = []
    #     for obj in objs:
    #         documents.append({
    #             "_op_type": "index",
    #             "_index": self.db_index,
    #             "_id": obj.id,
    #             "_source": asdict(obj)
    #         })

    #     if documents:
    #         try:
    #             async for ok, action in async_bulk(self.opensearch_client, documents):
    #                 if not ok:
    #                     logger.error("Bulk action failed", extra={"action": action})
    #             logger.info("indexed batch of documents", extra={"count": len(documents)})
    #         except Exception as e:
    #             logger.error("Bulk indexing failed", exc_info=True, extra={"count": len(documents)})
    #             raise


    async def index(self, obj: BaseDocument):
        # update embedding
        try:
            embedding = await self.create_embedding(getattr(obj, self.field))
            obj.embedding = embedding
            data=asdict(obj)

            await self.opensearch_client.index(
                index=self.db_index, id=obj.id, body=data)
            logger.debug("indexed document", extra={"doc_id": obj.id})
        except Exception as e:
            logger.error("failed to index document", exc_info=True, extra={"doc_id": obj.id})
            raise

    def normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to 0-1 range using min-max normalization"""
        arr = np.array(embedding, dtype=np.float32)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Avoid division by zero
        if max_val == min_val:
            return np.full_like(arr, 0.5, dtype=np.float32).tolist()
        
        normalized = (arr - min_val) / (max_val - min_val)
        return normalized.tolist()

    async def create_embedding(self, content: str, normalize: bool = True) -> List[float]:
            """ retrieves an embedding for the movie description
            Args:
                description: content to embed

            Returns:
                embedding of the description
            """

            logger.debug("creating embedding", extra={"content": content})

            embedding = None
            try:
                response = await self.openai_client.embeddings.create(
                    input=content,
                    model=self.model
                )
            except Exception as e:
                logger.error("Error creating embedding", extra={"error": str(e), "content": content})
                raise e

            embedding = response.data[0].embedding
            if normalize:
                embedding = self.normalize(embedding)

            return embedding
    
    def create_embedding_batch(self, contents: List[str], normalize: bool = True):
        pass

    async def search_by_vector(self, query: str, amount: int, filter: dict | None = None):
        """ get similar movies by description"""
        try:
            embedding = await self.create_embedding(query)
            body = {
                "size": amount,
                "_source": {
                    "excludes": ["embedding"],
                },
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": amount
                        }
                    }
                }
            }

            logger.debug("searched by vector", extra={"query": query, "amount": amount})
            results = await self.opensearch_client.search(index=self.db_index, body=body)
            return results
        except Exception as e:
            logger.error("vector search failed", exc_info=True, extra={"query": query, "amount": amount})
            raise
    
    async def search_by_keywords(self, query: str, amount: int, filter: dict | None = None):
        """ get similar contents by keywords"""
        try:
            body = {
                "size": amount,
                "_source": {
                    "excludes": ["embedding"],
                },
                "query": {
                    "match": {
                        self.field: query
                    }
                }
            }

            logger.debug("searched by keywords", extra={"query": query, "amount": amount})
            results = await self.opensearch_client.search(index=self.db_index, body=body)
            return results
        except Exception as e:
            logger.error("keyword search failed", exc_info=True, extra={"query": query, "amount": amount})
            raise
    
    async def search_by_hybrid(self, query: str, amount: int, filter: dict | None = None, vector_alpha: float = 0.5):
        """ get similar contents by keywords and vector similarity"""

        keywords_alpha = 1.0 - vector_alpha    

        vector_results = await self.search_by_vector(query, amount, filter)
        vector_scores = {hit["_id"]: self.normalize(hit["_score"]) for hit in vector_results["hits"]["hits"]}

        keywords_results = await self.search_by_keywords(query, amount, filter)
        keywords_scores = {hit["_id"]: self.normalize(hit["_score"]) for hit in keywords_results["hits"]["hits"]}
        
        print("vector scores:", vector_scores)  # --- IGNORE ---
        print("keywords scores:", keywords_scores)  # --- IGNORE ---

        all_ids = set(vector_scores.keys()) | set(keywords_scores.keys())
        combined_scores = {}

        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            k_score = keywords_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = vector_alpha * v_score + (1 - vector_alpha) * k_score

        logger.debug("combined scores calculated", extra={"combined_scores": combined_scores})


    def event_batch_embeddings_completed(self, batch_id: str):
        pass

    
    async def delete(self, doc_id: str):
        """ Delete document from index based on it's doc id """
        try:
            await self.opensearch_client.delete(index=self.db_index, id=doc_id)
            logger.info("deleted document", extra={"doc_id": doc_id})
        except Exception as e:
            logger.error("failed to delete document", exc_info=True, extra={"doc_id": doc_id})
