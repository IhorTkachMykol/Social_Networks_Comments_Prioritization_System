# Import libraries
import re
import pandas as pd
# import logging
from haystack.nodes import TransformersQueryClassifier
from transformers import pipeline

# logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)

pipe = pipeline("fill-mask", model="ukr-models/xlm-roberta-base-uk")


keyword_classifier = TransformersQueryClassifier(model_name_or_path="ukr-models/xlm-roberta-base-uk")


def detect_keywords(query):
    '''This function will classificate keywords (output_2) and statements or
    question (output_1) comments

    Args::
        query (str): user's comment.
    Returns:
        0 or 1: 0 if input query is statement or question, 1 if it is a keyword
    '''
    query = re.sub(r'[!?]',"", query)
    result = keyword_classifier.run(query=query)
    result = 0 if result[1]== "output_1" else 1
    return result

words = ["хто тобі це сказв", "привіт", "ти віриш в нло", "хотів би звернути увагу, що це не правда", "клас", "топчик",
         "скільки це коштує", "найліпше відео в ютуб", "це фейк", "іди в сраку", "лох",]

for word in words:
    print(word, detect_keywords(word))