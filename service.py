import os

import torch

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

# download_model(os.environ["model_name"], dir='/cache/')

torch.multiprocessing.set_start_method('spawn', force=True)

"""
Preparation

loading some data, models into this python script...
"""
dataset_path = os.environ["dataset_path"]
reader_path = os.environ["reader_path"]

df = pdf_converter(directory_path=dataset_path)
df = df[~df['paragraphs'].isnull()]
df = filter_paragraphs(df)

"""
The function that will be called when the processor processes.
"""
def f(intent=None, userinput=None, **other_arguments_from_context):
    cdqa_pipeline = QAPipeline(reader=reader_path)
    cdqa_pipeline.fit_retriever(df=df)

    if not intent:
        raise ValueError("A key named 'intent' must be set in the contexts map!")
    question = userinput if userinput else intent
    prediction = cdqa_pipeline.predict(query=question)

    """ Return values get collecteĸd in a dict, that will be 
        written into the context after this processing step"""

    del cdqa_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "question": question,
        "answer": prediction[0],
        "document_title": prediction[1],
        "paragraph": prediction[2]
    }
