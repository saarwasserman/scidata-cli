# Apps

## Movies

Find similar movies to your description

### Tools

1. OpenAI Embeddings - create embedding for movie's description
2. Opensearch - store movies documents including embedding for knn search

### Dataset

See datasets/movies.csv

### General Flow

once created the movies index in open search and mappeed the descripteion embed screen as vector field
you will be able to use knn vector search

1. Get embeddings based on movie description (can use batch api of openai)
2. Parse dataset's csv for relevant fields and create documents that include the "description_embed" field for embedding
3. Index the documents (can use bulk operation)
4. For Search: input a description you like, create embedding for it and search based on it, related movies will be retrieved
5. you can add more movies to opensearch db


# Usage

uv run python -m apps.console movies add --movie_id=2 --description="a drama about a family in the wild west" --title="living in the wild" --genre=western --genre=drama