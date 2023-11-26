from typing import Tuple, List
import requests
import wandb
import re
import statistics
import random
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping, LearningRateScheduler
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import losses
from tensorflow.keras.optimizers import AdamW, Adam
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GRU, Dense, LSTM, Dropout, \
    Activation, SpatialDropout1D
from wandb.keras import WandbCallback

wandb.login()

# GLOBAL HYPERPARAMETER OPTIONS
# is_down_sampled = [True, False]
# polish_chars_removed = [True, False]
# numbers_replaced_with_single_word = [True, False]
# nr_of_epochs = [40, 50]
# test_val_size = 0.3 To chyba nie podlega parametryzacji, jako że 70/20/10 pozwala nam zachować godność i wiarygodność pomiarów
# val_size = 0.33 tu tak samo
# threshold_of_cutting_sentences = [60, 55, 65]
# learning_rate = [0.001, 0.0005]
# output_dim = [48, 54, 58, 64, 72]
# batch_size = [64, 128, 256]
# random_state = 42
# dense_layer_neurons = [32, 64]
# optimizer = ["Adam", "AdamW"]

for lstm_units in [96]:

    # HYPERPARAMETERS FOR THIS RUN:
    hyper_parameters = {
        "is_down_sampled": False,
        "polish_chars_removed": False,
        "numbers_replaced_with_single_word": False,
        "nr_of_epochs": 35,
        "test_val_size": 0.3,
        "val_size": 0.33,
        "threshold_of_cutting_sentences": 62,
        "learning_rate": 0.001,
        "output_dim": 60,
        "batch_size": 256,
        "random_state": 42,
        # "dense_layer_neurons": 32,
        "lstm_units": lstm_units,
        "optimizer": "AdamW",
    }
    print(hyper_parameters)


    # def create_model(input_dim: str, input_length: str, num_classes: int):
    #     model = Sequential()
    #     model.add(Embedding(input_dim=input_dim, output_dim=hyper_parameters["output_dim"], input_length=input_length))
    #     model.add(LSTM(units=96))
    #     model.add(Dense(num_classes, activation='softmax'))
    #     return model


    def create_model(input_dim: int, input_length: int, num_classes: int):
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=hyper_parameters["output_dim"], input_length=input_length))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=64))
        # model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))
        return model


    def lr_schedule(epoch):
        return hyper_parameters["learning_rate"] * 0.1 ** (epoch // 10)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # def create_model(input_dim: str, input_length: str, num_classes: int):
    #     model = Sequential()
    #     model.add(Embedding(input_dim=input_dim, output_dim=hyper_parameters["output_dim"], input_length=input_length))
    #     # model.add(SpatialDropout1D(0.7))
    #     # model.add(LSTM(, dropout=0.7, recurrent_dropout=0.7))
    #     model.add(LSTM(units=128))
    #     model.add(Dropout(0.2))
    #     model.add(LSTM(units=64))
    #     model.add(Dropout(0.2))
    #     model.add(LSTM(units=32))
    #     # model.add(GlobalMaxPooling1D())
    #     # model.add(Dense(hyper_parameters["dense_layer_neurons"], activation='relu'))
    #     # model.add(Dropout(0.2))
    #     model.add(Dense(num_classes, activation='softmax'))
    #     return model


    # optimzer spec
    if hyper_parameters["optimizer"] == "AdamW":
        optimizer = AdamW(learning_rate=hyper_parameters["learning_rate"])
    if hyper_parameters["optimizer"] == "Adam":
        optimizer = Adam(learning_rate=hyper_parameters["learning_rate"])

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('NO GPU found')
    else:
        print('Found GPU at: {}'.format(device_name))

    dataset_file_url = "https://raw.githubusercontent.com/SunBear1/document-classification/master/data/dataset.csv"

    response = requests.get(dataset_file_url)
    local_file_path = "dataset.csv"

    if response.status_code == 200:
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
        print(f"File '{local_file_path}' has been successfully downloaded.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    connecting_words_file_url = "https://raw.githubusercontent.com/SunBear1/document-classification/master/data/connecting_words.lst"
    local_file_path = "connecting_words.lst"
    response = requests.get(connecting_words_file_url)

    if response.status_code == 200:
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
        print(f"File '{local_file_path}' has been successfully downloaded.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    with open("connecting_words.lst", "r") as f:
        CONNECTING_WORDS = f.read().splitlines()


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


    def cut_too_long_sentences(sentences: List[str], categories: List[str], threshold: int):
        new_sentences = []
        new_categories = []
        for i in range(len(sentences)):
            words_count = len(sentences[i].split())
            if words_count > threshold:
                midpoint = len(sentences[i]) // 2
                first_half = sentences[i][:midpoint]
                second_half = sentences[i][midpoint:]
                new_sentences.append(first_half)
                new_categories.append(categories[i])
                new_sentences.append(second_half)
                new_categories.append(categories[i])
            else:
                new_sentences.append(sentences[i])
                new_categories.append(categories[i])
        return new_sentences, new_categories


    def preprocess_sentences(text: List[str]):
        for i in range(len(text)):
            sentence = text[i]
            sentence = sentence.lower()
            sentence = re.sub(r'[^a-zA-Ząćęłńóśźż\s]', '', sentence)
            if hyper_parameters["polish_chars_removed"]:
                sentence = replace_polish_letters(sentence)
            if hyper_parameters["numbers_replaced_with_single_word"]:
                sentence = replace_numbers_with_word(sentence)
            sentence = ' '.join(sentence.split())
            sentence = remove_connecting_words(sentence)
            text[i] = sentence
        return text


    def remove_connecting_words(text):
        text = text.split()
        text = [word for word in text if word not in CONNECTING_WORDS]
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


    def get_values_from_dataset(path: str) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(path, sep=";")
        df = df.dropna()
        df = filter_dataset(df)
        if hyper_parameters["is_down_sampled"]:
            df = downsample_dataset(df, 225)
        balanced_df = df
        x = balanced_df["text_full"].tolist()
        y = balanced_df["label_high"].tolist()
        x = preprocess_sentences(text=x)
        x, y = cut_too_long_sentences(sentences=x, categories=y,
                                      threshold=hyper_parameters["threshold_of_cutting_sentences"])
        temp = list(zip(x, y))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        x, y = list(res1), list(res2)
        return x, y


    processed_sentences, categories = get_values_from_dataset(path="dataset.csv")
    print("---------------------Welcome---------------------")
    print(f"List of categories: {list(set(categories))}")
    print(f"Number of occurrences per category")
    print(pd.Series(categories).value_counts())

    sentence_lengths = []
    max_sen_word_count = 0
    sum = 0

    for sentence in processed_sentences:
        words = len(sentence.split())
        sum += words
        sentence_lengths.append(words)
        if words > max_sen_word_count:
            max_sen_word_count = words

    print("Max words in a sentence", max_sen_word_count)
    print("Avarge words for sentence", statistics.mean(sentence_lengths))
    sentence_lengths = sorted(sentence_lengths)
    print("Median of words for sentence", statistics.median(sentence_lengths))

    print(f"Sample of Y data: {categories[0]} and X data: {processed_sentences[0]}")

    print("-------------X-SET-------------")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_sentences)
    sequences = tokenizer.texts_to_sequences(processed_sentences)
    padded_sequences = pad_sequences(sequences)
    print("padded_sequences", padded_sequences)
    print("word index", len(tokenizer.word_index))
    padded_sequences = np.array(padded_sequences)

    print("-------------Y-SET-------------")
    label_to_index = {category: idx for idx, category in enumerate(set(categories))}
    print("label_to_index", label_to_index)
    categories_encoded = [label_to_index[category] for category in categories]
    print("categories_encoded", categories_encoded)
    labels = np.array(categories_encoded)
    print("number of classes", len(label_to_index))
    print("input dim", len(tokenizer.word_index))
    print("input_length", padded_sequences.shape[1])
    print("---------CLASS-WEIGHTS---------")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    print("Label weights specifics", np.unique(categories), np.unique(labels), class_weights)

    X_train, X_temp, Y_train, Y_temp = train_test_split(padded_sequences, labels,
                                                        test_size=hyper_parameters["test_val_size"],
                                                        random_state=hyper_parameters["random_state"])
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=hyper_parameters["val_size"],
                                                    random_state=hyper_parameters["random_state"])

    unique_elements, counts = np.unique(Y_val, return_counts=True)

    # Display results
    for element, count in zip(unique_elements, counts):
        print(f"{element}: {count} occurrences")

    print("-------------MODEL-SPEC-------------")
    model = create_model(input_dim=len(tokenizer.word_index) + 1, input_length=padded_sequences.shape[1],
                         num_classes=len(label_to_index))

    hyper_parameters["model_architecture"] = model.get_config()

    model.compile(optimizer=optimizer, loss=losses.SparseCategoricalCrossentropy(),
                  metrics=["sparse_categorical_accuracy"])

    model.summary()

    wandb.init(project="IUI&PUG", entity="gourmet", config=hyper_parameters, group="LSTM")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=hyper_parameters["nr_of_epochs"],
              batch_size=hyper_parameters["batch_size"], class_weight=dict(enumerate(class_weights)),
              callbacks=[WandbCallback(), EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0001), lr_scheduler])

    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    wandb.log({'test_accuracy': test_accuracy})
    wandb.finish()
