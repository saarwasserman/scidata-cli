import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List

import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.helpers import bulk
from openai import OpenAI

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

    def __enter__(self):
        
        self.opensearch_client = OpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=(settings.opensearch_username, settings.opensearch_password),
            use_ssl=True,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
        )

        # create index if not exists
        if not self.opensearch_client.indices.exists(index=self.db_index):
            logger.info("Creating index", extra={"index": self.db_index})
            self.create_db_index()

        self.openai_client = OpenAI(api_key=settings.openai_api_key,
                                    organization=settings.openai_organization_id,
                                    project=settings.openai_project_id)
        

        logger.debug("Initialized OpenSearch and OpenAI clients")

        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.opensearch_client.close()
        self.openai_client.close()

    def create_db_index(self):
        """Create OpenSearch index with hybrid search mappings"""

        with OpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=(settings.opensearch_username, settings.opensearch_password),
            use_ssl=True,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
        ) as opensearch_client:
            try:
                createRes = opensearch_client.indices.create(
                    index=self.db_index, body=HybridSearchApp.OPENSEARCH_INDEX_BODY)
                getRes = opensearch_client.indices.get(self.db_index)
                logger.debug("Index created successfully", extra={"index": self.db_index, "status": "created"})
            except Exception as e:
                logger.error("Failed to create index", extra={"index": self.db_index, "error": str(e)})

    ### TODO: Chunking Methods ###
    def chunks(self):
        pass

    ### TODO: Embeddings Methods - Batch <-> Pipeline, Async/Sync <-> InPlace ###
        
    def index_batch(self, objs: List[BaseDocument]):
        documents = []
        for obj in objs:
            documents.append({
                "_op_type": "index",
                "_index": self.db_index,
                "_id": obj.id,
                "_source": asdict(obj)
            })

        if documents:
            bulk(self.opensearch_client, documents)

        logger.info("indexed batch of documents", extra={"count": len(documents)})


    def index(self, obj: BaseDocument):
        # update embedding
        try:
            embedding = self.create_embedding(getattr(obj, self.field))
            obj.embedding = embedding
            data=asdict(obj)

            self.opensearch_client.index(
                index=self.db_index, id=obj.id, body=data)
            logger.debug("indexed document", extra={"doc_id": obj.id})
        except Exception as e:
            logger.error("failed to index document", exc_info=True, extra={"doc_id": obj.id})
            raise

    def normalize(self, embedding: List[float]) -> List[float]:
        nrr = np.array(embedding, dtype=np.float32)  # Ensure proper type
        norm = np.linalg.norm(nrr)
        if norm == 0:
            return embedding

        return (nrr / norm).tolist()

    def create_embedding(self, content: str, normalize: bool = True) -> List[float]:
            """ retrieves an embedding for the movie description
            Args:
                description: content to embed

            Returns:
                embedding of the description
            """

            logger.debug("creating embedding", extra={"content": content})

            embedding = None
            try:
                response = self.openai_client.embeddings.create(
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

    def search_by_vector(self, query: str, amount: int, filter: dict | None = None):
        """ get similar movies by description"""
        try:
            embedding = self.create_embedding(query)
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
            results = self.opensearch_client.search(index=self.db_index, body=body)
            return results
        except Exception as e:
            logger.error("vector search failed", exc_info=True, extra={"query": query, "amount": amount})
            raise
    
    def search_by_keywords(self, query: str, amount: int, filter: dict | None = None):
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
            results = self.opensearch_client.search(index=self.db_index, body=body)
            return results
        except Exception as e:
            logger.error("keyword search failed", exc_info=True, extra={"query": query, "amount": amount})
            raise
    
    def search_by_hybrid(self, query: str, amount: int, filter: dict | None = None, vector_alpha: float = 0.5):
        """ get similar contents by keywords and vector similarity"""

        keywords_alpha = 1.0 - vector_alpha    

        vector_results = self.search_by_vector(query, amount, filter)
        vector_scores = {hit["_id"]: self.normalize(hit["_score"]) for hit in vector_results["hits"]["hits"]}

        keywords_results = self.search_by_keywords(query, amount, filter)
        keywords_scores = {hit["_id"]: self.normalize(hit["_score"]) for hit in keywords_results["hits"]["hits"]}
        
        all_ids = set(vector_scores.keys()) | set(keywords_scores.keys())
        combined_scores = {}

        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            k_score = keywords_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = vector_alpha * v_score + (1 - vector_alpha) * k_score

        logger.debug("combined scores calculated", extra={"combined_scores": combined_scores})


    def event_batch_embeddings_completed(self, batch_id: str):
        pass

    
    def delete(self, doc_id: str):
        """ Delete document from index based on it's doc id """
        try:
            self.opensearch_client.delete(index=self.db_index, id=doc_id)
            logger.info("deleted document", extra={"doc_id": doc_id})
        except Exception as e:
            logger.error("failed to delete document", exc_info=True, extra={"doc_id": doc_id})
