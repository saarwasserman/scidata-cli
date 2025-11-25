""" OpenAI batch and files utilities """

import os
import csv
import json
from typing import Literal

import click
from openai import OpenAI


@click.group(name="batch")
def cli_openai_batch():
    """batch utils"""
    pass  # pylint: disable=unnecessary-pass



openai_client = OpenAI()


@cli_openai_batch.command(name="upload")
@click.option("--filepath", help="jsonl file to upload", required=True)
def upload_file_for_batch(filepath: str):
    batch_file = None
    with open(filepath, "rb") as input_file:
        batch_file = openai_client.files.create(
            file=input_file,
            purpose="batch"
        )
    
    print(batch_file)


@cli_openai_batch.command(name="create")
@click.option("--file_id", help="jsonl id file in openai storage", required=True)
@click.option("--endpoint", help="endpoint for batch", required=True)
def create_batch(file_id: str, endpoint: Literal["chat/completions", "embeddings"]):
    batch = openai_client.batches.create(
        input_file_id=file_id,
        endpoint=f"/v1/{endpoint}",
        completion_window="24h"
    )
    
    print(batch)


@cli_openai_batch.command(name="show_jobs")
def show_batch_jobs_ids():
    pass
    

@cli_openai_batch.command(name="status")
@click.option("--batch_id", help="show status of the batch", required=True)
def check_batch_status(batch_id):
    batch_job = openai_client.batches.retrieve(batch_id)
    print(batch_job)


@cli_openai_batch.command(name="output")
@click.option("--batch_id", help="download output file for batch_id", required=True)
def check_batch_status(batch_id):
    batch_job = openai_client.batches.retrieve(batch_id)
    if batch_job.status == "completed":
        result = openai_client.files.content(batch_job.output_file_id).content
        filename = batch_job.output_file_id + ".jsonl"

        with open(filename, 'wb') as file:
            file.write(result)
            print(f"wrote results to file: {filename}")    
        
    else:
        print(f"batch status: {batch_job.status}")
