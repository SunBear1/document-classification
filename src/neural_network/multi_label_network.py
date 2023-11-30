from typing import Tuple, List, Any, Dict
import requests
import wandb
import re
import statistics
import pprint
import numpy as np
import pandas as pd
from keras.src.layers import GlobalMaxPool1D
from pandas import DataFrame
from tensorflow.keras.models import Model
from keras.src.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import AdamW, Adam
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GRU, Dense, LSTM, Dropout, \
    Activation, SpatialDropout1D
from wandb.keras import WandbCallback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

from src.neural_network.processing import downsample_dataset, filter_dataset, cut_too_long_sentences, \
    replace_polish_letters, replace_numbers_with_word, remove_connecting_words, tokenize_sentences, encode_labels

DATASET_FILE_URL = "https://raw.githubusercontent.com/SunBear1/document-classification/master/data/complete_dataset.csv"
CONNECTING_WORDS_FILE_URL = "https://raw.githubusercontent.com/SunBear1/document-classification/master/data/connecting_words.lst"
DATASET_FILE_PATH = "dataset_complete.csv"
CONNECTING_WORDS_FILE_PATH = "connecting_words.lst"

MAX_SENTENCE_LENGTH = 0


def prepare_data(hyper_parameters: Dict, connecting_words: List[str]) -> DataFrame:
    global MAX_SENTENCE_LENGTH
    df = pd.read_csv(DATASET_FILE_PATH, sep=",")
    df = df.dropna()
    tmp_labels_mid = df["label_mid"].tolist()
    tmp_labels_high = df["label_high"].tolist()
    tmp_text = df["text"].tolist()

    import ast
    sentences = []
    labels = []
    for i in range(len(df)):
        text_list = ast.literal_eval(tmp_text[i])
        for sentence in text_list:
            sentence = sentence.strip().lower()
            punctuation_pattern = re.compile(r'[^\w\s]')
            split_sentence = punctuation_pattern.sub('', sentence)
            if len(split_sentence) > MAX_SENTENCE_LENGTH:
                MAX_SENTENCE_LENGTH = len(split_sentence)
            sentences.append(split_sentence)
            labels.append((tmp_labels_mid[i], tmp_labels_high[i]))

    dataset = pd.DataFrame({"text": sentences, "labels": labels})
    dataset_duplicates = dataset.drop_duplicates()
    return dataset_duplicates


def get_features(text_series):
    """
    transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)


def prediction_to_label(prediction):
    tag_prob = [(labels[i], prob) for i, prob in enumerate(prediction.tolist())]
    return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))


# download_dataset()
with open(CONNECTING_WORDS_FILE_PATH, "r") as f:
    connecting_words = f.read().splitlines()
dataset = prepare_data({}, connecting_words)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(dataset['labels'])
labels = multilabel_binarizer.classes_

maxlen = MAX_SENTENCE_LENGTH
max_words = 9000
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(dataset['text'])


def get_features(text_series):
    """
    transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)


def prediction_to_label(prediction):
    tag_prob = [(labels[i], prob) for i, prob in enumerate(prediction.tolist())]
    return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))


x = get_features(dataset['text'])
y = multilabel_binarizer.transform(dataset['labels'])
print(x.shape)

print(f"List of sub categories: {set(dataset['labels'].tolist())}")
print(f"Number of occurrences per main category")
print(pd.Series(dataset["text"].tolist()).value_counts())

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=9000)

x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.333, random_state=9000)

# main_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset['labels'].tolist()),
#                                           y=dataset['labels'].tolist())
#
# print(main_class_weights)
num_classes = len(np.unique(dataset['labels'].tolist()))
print(num_classes)

model = Sequential()
model.add(Embedding(max_words, 20, input_length=maxlen))
model.add(Dropout(0.1))
model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300,
          batch_size=128)
