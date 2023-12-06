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
    # dataframe.drop(['categories', 'text_full', 'lp'], axis=1, inplace=True)
    return dataframe


def tokenize_sentences(sentences: List[str], tokenizer: Tokenizer = None):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences)
    padded_sequences = np.array(padded_sequences)
    return padded_sequences


def encode_labels(categories: List[str], labels: Dict):
    return to_categorical([labels[category] for category in categories], num_classes=len(labels))


def cut_too_long_sentences(sentences: List[str], main_categories: List[str], sub_categories: List[str], threshold: int):
    while True:  # TODO pętla nieskończona - czacha
        max_sen_word_count = 0
        for sentence in sentences:
            words = len(sentence.split())
            if words > max_sen_word_count:
                max_sen_word_count = words

        if max_sen_word_count <= threshold:
            break

        new_sentences = []
        new_main_categories = []
        new_sub_categories = []
        for i in range(len(sentences)):
            words_count = len(sentences[i].split())
            if words_count > threshold:
                midpoint = len(sentences[i]) // 2  # TODO ucinanie w środku słów - giga gówno
                first_half = sentences[i][:midpoint]
                second_half = sentences[i][midpoint:]

                new_sentences.append(first_half)
                new_main_categories.append(main_categories[i])
                new_sub_categories.append(sub_categories[i])

                new_sentences.append(second_half)
                new_main_categories.append(main_categories[i])
                new_sub_categories.append(sub_categories[i])

            else:
                new_sentences.append(sentences[i])
                new_main_categories.append(main_categories[i])
                new_sub_categories.append(sub_categories[i])
        sentences = new_sentences
    return new_sentences, new_main_categories, new_sub_categories


def remove_connecting_words(text, connecting_words: List[str]):
    text = text.split()
    text = [word for word in text if word not in connecting_words]
    text = ' '.join(text)
    return text


def replace_numbers_with_word(sentence) -> str:
    pattern = r'\d+'
    replaced_sentence = re.sub(pattern, 'number', sentence)
    return replaced_sentence


def replace_polish_letters(input: str) -> str:
    polish_to_latin = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', 'Ą': 'A',
        'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ó': 'O',
        'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    }
    for polish, latin in polish_to_latin.items():
        input = input.replace(polish, latin)
    return input


def tokenize_sentences(sentences: List[str]):  # TODO czy tutaj można dospermić? chyba tak
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(sentences)
    # sequences = tokenizer.texts_to_sequences(sentences)
    # padded_sequences = pad_sequences(sequences)
    # padded_sequences = np.array(padded_sequences)
    #
    # print("padded_sequences", padded_sequences)
    # print("word index", len(tokenizer.word_index))  # ile jest słów w słowniku
    #
    # return padded_sequences, tokenizer.word_index

    nlp = spacy.load('pl_core_news_md')
    tokenized_data = [[token.lemma_ for token in nlp(sentence)] for sentence in sentences]
    vocabulary = {word: idx for idx, word in enumerate(set(word for sentence in tokenized_data for word in sentence))}
    indexed_data = [[vocabulary[word] for word in sentence] for sentence in tokenized_data]
    padded_sequences = pad_sequences(indexed_data)
    padded_sequences = np.array(padded_sequences)

    print("padded_sequences", padded_sequences)
    print("word index", len(vocabulary))  # ile jest słów w słowniku

    return padded_sequences, vocabulary
