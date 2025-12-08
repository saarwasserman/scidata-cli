"""entrypoint for all cli commands
"""

import click

#from nblade import settings  # pylint: disable=unused-import
from apps.movies.console import cli_movies_search


# pylint: disable=missing-function-docstring
@click.group(name="apps")
def cli():
    pass

def main():
    cli.add_command(cli_movies_search)
    cli()

if __name__ == "__main__":
    main()