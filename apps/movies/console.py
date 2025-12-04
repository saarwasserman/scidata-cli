import click
import asyncio
from typing import List
from apps.movies.search import Movie, MoviesSearchApp

### CLI ###

@click.group(name="movies", invoke_without_command=False)
@click.pass_context
def cli_movies_search(ctx):
    """movies embeddings app"""
    # Store the app in context for later use
    ctx.ensure_object(dict)
    ctx.meta['app'] = None


@cli_movies_search.command(name="init")
@click.option("--documents_filepath", help="jsonl file to upload", required=True)
@click.pass_context
def init(ctx, documents_filepath: str):
    """ init the opensearch index and documents (include embeddings) """
    asyncio.run(_init_async(documents_filepath))


async def _init_async(documents_filepath: str):
    async with MoviesSearchApp() as app:
        await app.create_db_index()
        # await app.populate(documents_filepath)


@cli_movies_search.command(name="add")
@click.option("--movie_id", type=int, help="description of a movie", required=True)
@click.option("--title", type=str, help="description of a movie", required=True)
@click.option("--description", type=str, help="description of a movie", required=True)
@click.option("--genre", type=str, multiple=True, help="description of a movie", required=True)
@click.pass_context
def add_movie(ctx, movie_id: int, title: str, description: str, genre: List[str]):
    """Add movie to DB"""
    asyncio.run(_add_movie_async(movie_id, title, description, genre))


async def _add_movie_async(movie_id: int, title: str, description: str, genre: List[str]):
    async with MoviesSearchApp() as app:
        await app.index(Movie(
            id=str(movie_id),
            title=title,
            description=description,
            genres=list(genre),
            embedding=None
        ))


@cli_movies_search.command(name="search-vector")
@click.option("--description", help="description of a movie", required=True)
@click.option("--amount", help="max number of movies you would like to see", default=3, required=True)
@click.pass_context
def search_by_vector(ctx, description: str, amount: int):
    """ get similar movies by description"""
    asyncio.run(_search_by_vector_async(description, amount))


async def _search_by_vector_async(description: str, amount: int):
    async with MoviesSearchApp() as app:
        results = await app.search_by_vector(description, amount)
        print(results["hits"]["hits"])


@cli_movies_search.command(name="search-keywords")
@click.option("--query", help="keyword search query", required=True)
@click.option("--amount", help="max number of movies you would like to see", default=3, required=True)
@click.pass_context
def search_by_keywords(ctx, query: str, amount: int):
    """ get similar movies by keywords"""
    asyncio.run(_search_by_keywords_async(query, amount))


async def _search_by_keywords_async(query: str, amount: int):
    async with MoviesSearchApp() as app:
        results = await app.search_by_keywords(query, amount)
        print(results["hits"]["hits"])


@cli_movies_search.command(name="search-hybrid")
@click.option("--query", help="keyword search query", required=True)
@click.option("--amount", help="max number of movies you would like to see", default=3, required=True)
@click.pass_context
def search_by_hybrid(ctx, query: str, amount: int):
    """retrives results using hybrid search (keywords + vector similarity)"""
    asyncio.run(_search_by_hybrid_async(query, amount))


async def _search_by_hybrid_async(query: str, amount: int):
    async with MoviesSearchApp() as app:
        await app.search_by_hybrid(query, amount)


@cli_movies_search.command(name="delete")
@click.option("--movie_id", help="ID of the movie to delete", required=True)
@click.pass_context
def delete_movie(ctx, movie_id: str):
    """Remove a movie from the database"""
    asyncio.run(_delete_movie_async(movie_id))


async def _delete_movie_async(movie_id: str):
    async with MoviesSearchApp() as app:
        print("---------")
        print(app)
        await app.delete(movie_id)


def main():
    cli_movies_search()

if __name__ == "__main__":
    main()
