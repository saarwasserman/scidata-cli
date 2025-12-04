import click
from typing import List
from apps.movies.search import Movie, MoviesSearchApp

### CLI ###

@click.group(name="movies", invoke_without_command=False)
@click.pass_context
def cli_movies_search(ctx):
    """movies embeddings app"""
    app = MoviesSearchApp().__enter__()
    ctx.ensure_object(dict)
    ctx.obj = app
    
    def cleanup(exception=None):
        app.__exit__(None, None, None)
    
    ctx.call_on_close(cleanup)


@cli_movies_search.command(name="init")
@click.option("--documents_filepath", help="jsonl file to upload", required=True)
@click.pass_obj
def init(app: MoviesSearchApp, documents_filepath: str):
    """ init the opensearch index and documents (include embeddings) """
    app.create_index()
    app.populate(documents_filepath)


@cli_movies_search.command(name="add")
@click.option("--movie_id", type=int, help="description of a movie", required=True)
@click.option("--title", type=str, help="description of a movie", required=True)
@click.option("--description", type=str, help="description of a movie", required=True)
@click.option("--genre", type=str, multiple=True, help="description of a movie", required=True)
@click.pass_obj
def add_movie(app: MoviesSearchApp, movie_id: int, title: str, description: str, genre: List[str]):
    """Add movie to DB"""

    app.index(Movie(
        id=str(movie_id),
        title=title,
        description=description,
        # the genre param is set to 'multiple' so for clearance it is in singular form
        genres=genre,
        embedding=None
    ))


@cli_movies_search.command(name="search-vector")
@click.option("--description", help="description of a movie", required=True)
@click.option("--amount", help="max number of movies you would like to see", default=3, required=True)
@click.pass_obj
def search_by_vector(app: MoviesSearchApp, description: str, amount: int):
    """ get similar movies by description"""
    results = app.search_by_vector(description, amount)


@cli_movies_search.command(name="search-keywords")
@click.option("--query", help="keyword search query", required=True)
@click.option("--amount", help="max number of movies you would like to see", default=3, required=True)
@click.pass_obj
def search_by_keywords(app: MoviesSearchApp, query: str, amount: int):
    """ get similar movies by keywords"""
    body = {
        "size": amount,
        "_source": {
            "excludes": ["embedding"],
        },
        "query": {
            "match": {
                "description": query
            }
        }
    }
    results = app.opensearch_client.search(index=app.db_index, body=body)


@cli_movies_search.command(name="search-hybrid")
@click.option("--query", help="keyword search query", required=True)
@click.option("--amount", help="max number of movies you would like to see", default=3, required=True)
@click.pass_obj
def search_by_hybrid(app: MoviesSearchApp, query: str, amount: int):
    """retrives results using hybrid search (keywords + vector similarity)"""

    results = app.search_by_hybrid(query, amount)

@cli_movies_search.command(name="delete")
@click.option("--movie_id", help="ID of the movie to delete", required=True)
@click.pass_obj
def delete_movie(app: MoviesSearchApp, movie_id: str):
    """Remove a movie from the database"""
    app.delete(movie_id)


def main():
    cli_movies_search()

if __name__ == "__main__":
    main()
