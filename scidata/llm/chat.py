""" OpenAI chat utilities """


from openai import OpenAI
import click

from scidata.config import settings


@click.group(name="chat")
def cli_openai_chat():
    """chat utils"""
    pass  # pylint: disable=unnecessary-pass


def send_user_message(message):  
    openai_client = OpenAI(api_key=settings.openai_api_key,
                       organization=settings.openai_organization_id,
                       project=settings.openai_project_id)

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {   
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )

    print({ "answer": completion.choices[0].message.content})
    openai_client.close()  


@cli_openai_chat.command(name="send")
@click.option("--message", help="user message", required=True)
def cli_send_user_message(message):
    """send user message"""
    send_user_message(message)
