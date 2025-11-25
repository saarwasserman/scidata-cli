""" OpenAI embeddings utilities """

from typing import List

from openai import OpenAI
import click

from scidata.config import settings


@click.group(name="embeddings")
def cli_openai_embeddings():
    """embeddings utils"""
    pass  # pylint: disable=unnecessary-pass


def create_embedding(input_text: str, model: str) -> List[float]:
    """Retrieves

    Args:
        input_text (str): text to embed
        model (str): openai model for embeddings

    Returns:
        List[float]: a vector of floats (embeddding) 
    """
    openai_client = OpenAI(api_key=settings.openai_api_key,
                       organization=settings.openai_organization_id,
                       project=settings.openai_project_id)

    # create embedding
    response = openai_client.embeddings.create(
        input=input_text,
        model=model,
    )

    openai_client.close()  
    return response.data[0].embedding


@cli_openai_embeddings.command(name="create")
@click.option("--text", help="input text", required=True)
@click.option("--model", type=click.Choice(["text-embedding-3-small"]), help="model", required=True)
def cli_create_embedding(text: str, model: str):
    """create embedding"""
    create_embedding(text, model)
