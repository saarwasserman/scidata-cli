""" Movies Hybrid Search App CLI """

from dataclasses import dataclass
from typing import List

import click
from scidata.components.search import HybridSearchApp, BaseDocument


@dataclass
class Movie(BaseDocument):
    title: str
    description: str
    genres: List[str]


# TODO: Implement Chunking Method
class MoviesSearchApp(HybridSearchApp):

    OPENSEARCH_INDEX_NAME = "movies-index"
    EMBEDDINGS_MODEL = "text-embedding-3-small"
    CONTENT_FIELD = "description"

    # TODO: check if fields of init are necessary. demand them in base class
    def __init__(self):
        super().__init__(index=MoviesSearchApp.OPENSEARCH_INDEX_NAME,
                         field=MoviesSearchApp.CONTENT_FIELD,
                         model=MoviesSearchApp.EMBEDDINGS_MODEL)

    def chunks(self):
        pass

    def index_documents(self, data_filepath: str, embeddings_filepath: str):
        documents = []
        for movie in movies.values():
            documents.append({
                "_op_type": "index",
                "_index": self.index,
                # TODO: take the dataclass and convert to dict with asdict.  only add the embedding later
                "_id": movie["Rank"],
                "_source": {
                    "title": movie["Title"],
                    "genres": movie["Genre"],
                    "description": movie["Description"],
                    "embedding": movie["Description_embed"]
                }
            })

        if documents:
            bulk(self.opensearch_client, documents)

    def create_batch_file(self, csv_filepath):
        """creates a .jsonl batch file of actions to upload to openai platform
        for embeddings retrieval

        Args:
            csv_file (str): csv to parse desired embedding for
        """
        requests = []
        with open(csv_file, "r") as movies_file:
            reader = csv.DictReader(movies_file, delimiter=",")
            for i, row in enumerate(reader):
                requests.append({
                    "custom_id": row["Rank"],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.model,
                        "input": row["Description"],
                    }
                })

        jsonl_out_filepath = f"{Path(csv_filepath).stem}.json"
        with open(jsonl_out_filepath, "w") as batch_file:
            batch_file.write('\n'.join(map(json.dumps, requests)))


    def populate(self, documents_filepath, embeddings_filepath):
        """
        Stores the movies data from the csv file into opensearch index
        
        :param self: Description
        :param documents_filepath: Description
        :param embeddings_filepath: Description
        """

        movies = {}
        with open(documents_filepath, "r") as movies_file:
            reader = csv.DictReader(movies_file, delimiter=",")
            for row in reader:
                movies[row["Rank"]] = row

            with open(embeddings_filepath, "r") as embed_file:
                for row in embed_file:
                    result = json.loads(row)
                    movies[result["custom_id"]
                           ]["Description_embed"] = result["response"]["body"]["data"][0]["embedding"]

        # create documents for bulk send to opensearch
        documents = []
        for movie in movies.values():
            documents.append({
                "_op_type": "index",
                "_index": self.db_index,
                # TODO: take the dataclass and convert to dict with asdict.  only add the embedding later
                "_id": movie["Rank"],
                "_source": {
                    "title": movie["Title"],
                    "genres": movie["Genre"],
                    "description": movie["Description"],
                    "embedding": movie["Description_embed"]
                }
            })

        if documents:
            bulk(self.opensearch_client, documents)

    def add(self, movie_data):
        """ Add movie to DB """

        embedding = self.create_embedding(movie_data["Description"])

        movie = Movie(
            id=movie_data["Rank"],
            title=movie_data["Title"],
            description=movie_data["Description"],
            genres=movie_data["Genre"].split(","),
            embedding=embedding
        )

        self.index(movie)
