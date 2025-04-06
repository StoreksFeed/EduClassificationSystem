import sys

import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import pymorphy3

def get_model(model_name):
    """
    Get a model from the global scope or download it if it doesn't exist.

    Args:
        model_name (str): Name of the model to be loaded.

    Returns:
        list: List of the tokenizer and model objects.
    """

    if model_name not in globals():
        globals()[model_name] = [
            AutoTokenizer.from_pretrained(model_name),
            AutoModel.from_pretrained(model_name)
        ]

    return globals()[model_name]


def get_morph():
    """
    Get morphological tools from the global scope or download it if it doesn't exist.

    Args:
        model_name (str): Name of the model to be loaded.

    Returns:
        list: List of the tokenizer and model objects.
    """

    if 'morph' not in globals():
        nltk.download('stopwords')
        globals()['morph'] = [
            set(stopwords.words('russian')),
            pymorphy3.MorphAnalyzer()
        ]

    return globals()['morph']

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python download.py <options>')
    elif sys.argv[1] == 'get_model':
        get_model(sys.argv[2])
    elif sys.argv[1] == 'get_morph':
        get_morph()
    else:
        print('Not a valid option')
