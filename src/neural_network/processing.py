import re
from typing import List, Dict
import spacy
import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from pandas import DataFrame
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def downsample_dataset(dataframe: DataFrame, desired_count: int) -> DataFrame:
    category_counts = dataframe['label_high'].value_counts()
    balanced_df = pd.DataFrame(columns=dataframe.columns)
    for category in dataframe['label_high'].unique():
        category_data = dataframe[dataframe['label_high'] == category]

        if category_counts[category] > desired_count:
            sampled_data = category_data.sample(desired_count,
                                                random_state=42)
        else:
            sampled_data = category_data
        balanced_df = pd.concat([balanced_df, sampled_data], ignore_index=True)
    return balanced_df


def filter_dataset(dataframe: DataFrame) -> DataFrame:
    category_filter = dataframe[(dataframe['label_high'] == "tu interpolska") |
                                (dataframe['label_high'] == "odpowiedzi niestandardowe") |
                                (dataframe['label_high'] == "prawo podatkowe") |
                                (dataframe['label_high'] == "prawo konstytucyjne") |
                                (dataframe['label_high'] == "prawo miädzynarodowe")].index
    dataframe.drop(category_filter, inplace=True)
    return dataframe


def encode_labels(categories: List[str], labels: Dict):
    return to_categorical([labels[category] for category in categories], num_classes=len(labels))


def split_too_long_sentences(sentences: List[str], categories: List[str], threshold: int):
    new_sentences = []
    new_categories = []
    for i in range(len(sentences)):
        if len(sentences[i].split()) > threshold:
            words = sentences[i].split()
            half = len(words) // 2
            first_half = ' '.join(words[:half])
            second_half = ' '.join(words[half:])
            new_sentences.append(first_half)
            new_categories.append(categories[i])
            new_sentences.append(second_half)
            new_categories.append(categories[i])
        else:
            new_sentences.append(sentences[i])
            new_categories.append(categories[i])
    return new_sentences, new_categories


def remove_connecting_words(text, connecting_words: List[str]):
    text = text.split()
    text = [word for word in text if word not in connecting_words]
    text = ' '.join(text)
    return text


def replace_numbers_with_word(sentence) -> str:
    pattern = r'\d+'
    replaced_sentence = re.sub(pattern, 'liczba', sentence)
    return replaced_sentence


def tokenize_sentences(sentences: List[str]):  # TODO czy tutaj można dospermić? chyba tak
    nlp = spacy.load('pl_core_news_lg')
    tokenized_data = [[token.lemma_ for token in nlp(sentence)] for sentence in sentences]
    vocabulary = {word: idx for idx, word in enumerate(set(word for sentence in tokenized_data for word in sentence))}
    indexed_data = [[vocabulary[word] for word in sentence] for sentence in tokenized_data]
    padded_sequences = pad_sequences(indexed_data)
    padded_sequences = np.array(padded_sequences)

    print("padded_sequences", padded_sequences)
    print("word index", len(vocabulary))  # ile jest słów w słowniku

    return padded_sequences, vocabulary
