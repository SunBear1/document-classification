from typing import Tuple, List, Any
import requests
import wandb
import re
import statistics
import random
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping, LearningRateScheduler
from keras.src.layers import GlobalAveragePooling1D
from keras.src.utils import to_categorical
from numpy import ndarray, dtype
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

LABELS = {
    "prawo medyczne": 0,
    "prawo pracy": 1,
    "prawo cywilne": 2,
    "prawo administracyjne": 3,
    "prawo farmaceutyczne": 4,
    "prawo karne": 5,
}


def program(hyper_parameters):
    # HYPERPARAMETERS FOR THIS RUN:
    print(hyper_parameters)

    # def create_model(input_dim: int, input_length: int, num_classes: int):
    #     model = Sequential()
    #     model.add(Embedding(input_dim=input_dim, output_dim=hyper_parameters["output_dim"], input_length=input_length))
    #     model.add(Dropout(0.1))
    #     model.add(LSTM(units=128))
    #     model.add(Dense(num_classes, activation='softmax'))
    #     return model

    def lr_schedule(epoch):
        return hyper_parameters["learning_rate"] * 0.1 ** (epoch // 30)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    def create_model(input_dim: str, input_length: str, num_classes: int):
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=hyper_parameters["output_dim"], input_length=input_length))
        # model.add(Conv1D(128, 5, activation='relu'))
        # tf.keras.layers.Conv1D(128, 5, activation='relu'),
        model.add(GlobalAveragePooling1D())
        # model.add(Dense(24, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        return model

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
            # sentence = ' '.join(sentence.split())
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
        x, y = cut_too_long_sentences(sentences=x, categories=y,
                                      threshold=hyper_parameters["threshold_of_cutting_sentences"]) # XD
        x, y = cut_too_long_sentences(sentences=x, categories=y,
                                      threshold=hyper_parameters["threshold_of_cutting_sentences"])  # XDDDD
        return x, y

    def tokenize_sentences(sentences: List[str], tokenizer: Tokenizer = None):
        sequences = tokenizer.texts_to_sequences(sentences)
        padded_sequences = pad_sequences(sequences)
        # print("padded_sequences", padded_sequences)
        # print("word index", len(tokenizer.word_index))
        padded_sequences = np.array(padded_sequences)
        return padded_sequences

    def encode_labels(categories: List[str]):
        #categories_encoded = [LABELS[category] for category in categories]
        categories_encoded = to_categorical([LABELS[category] for category in categories], num_classes=len(LABELS))
        # print("categories_encoded", categories_encoded)
        labels = np.array(categories_encoded)
        # print("number of classes", len(LABELS))
        # print("input dim", len(tokenizer.word_index))
        # print("input_length", padded_sequences.shape[1])
        return categories_encoded

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

    print("---------CLASS-WEIGHTS---------")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(categories), y=categories)
    print("Label weights specifics", np.unique(categories), np.unique(categories), class_weights)

    print("-------------X-SET-------------")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_sentences)
    sequences = tokenizer.texts_to_sequences(processed_sentences)
    padded_sequences = pad_sequences(sequences)
    print("padded_sequences", padded_sequences)
    print("word index", len(tokenizer.word_index))
    padded_sequences = np.array(padded_sequences)

    X_train_human_readable, X_temp_human_readable = train_test_split(processed_sentences,
                                                                     test_size=hyper_parameters["test_val_size"])
    X_val_human_readable, X_test_human_readable = train_test_split(X_temp_human_readable,
                                                                   test_size=hyper_parameters["val_size"])

    X_train, X_temp, Y_train_raw, Y_temp_raw = train_test_split(padded_sequences, categories,
                                                                test_size=hyper_parameters["test_val_size"])
    X_val, X_test, Y_val_raw, Y_test_raw = train_test_split(X_temp, Y_temp_raw,
                                                            test_size=hyper_parameters["val_size"])

    print("-------------Y-SET-------------")
    Y_train = encode_labels(Y_train_raw)
    Y_val = encode_labels(Y_val_raw)
    Y_test = encode_labels(Y_test_raw)

    print("-------------MODEL-SPEC-------------")
    model = create_model(input_dim=len(tokenizer.word_index) + 1, input_length=X_train.shape[1],
                         num_classes=len(LABELS))

    hyper_parameters["model_architecture"] = model.get_config()

    # model.compile(optimizer=optimizer, loss=losses.SparseCategoricalCrossentropy(),
    #               metrics=["sparse_categorical_accuracy"])

    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
    model.compile(optimizer=optimizer, loss=losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])

    model.summary()
    wandb.init(project="IUI&PUG", entity="gourmet", config=hyper_parameters, group="Basic")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=hyper_parameters["nr_of_epochs"],
              batch_size=hyper_parameters["batch_size"], class_weight=dict(enumerate(class_weights)),
              callbacks=[EarlyStopping(monitor='val_loss', patience=200, min_delta=0.0001), WandbCallback()])

    # AFTER TRAINING
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    wandb.log({'test_accuracy': test_accuracy})
    predictions = model.predict(X_val)
    y_pred = np.argmax(predictions, axis=1)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # cm = confusion_matrix(Y_val, y_pred)
    # plt.figure(figsize=(16, 16))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS.keys(),
    #             yticklabels=LABELS.keys())
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

    # predictions = model.predict(X_val)
    # y_pred = np.argmax(predictions, axis=1)
    # misclassified_idx = np.where(y_pred != Y_val)[0]
    # print("label_to_index", LABELS)
    # key_list = list(LABELS.keys())
    # val_list = list(LABELS.values())

    # for idx in misclassified_idx[:10]:  # Display the first 5 misclassified examples
    #     print(f"True label: {key_list[val_list.index(Y_val[idx])]}, Predicted label: "
    #           f"{key_list[val_list.index(y_pred[idx])]}, Sentence: {X_val_human_readable[idx]}")

    wandb.log({'test_accuracy': test_accuracy})
    wandb.finish()


hyper_params = {
    "is_down_sampled": False,
    "polish_chars_removed": False,
    "numbers_replaced_with_single_word": False,
    "nr_of_epochs": 80,
    "test_val_size": 0.3,
    "val_size": 0.33,
    "threshold_of_cutting_sentences": 25,
    "learning_rate": 0.001,
    "output_dim": 64,
    "batch_size": 128,
    "random_state": 42,
    "dense_layer_neurons": 48,
    # "lstm_units": lstm_units,
    "optimizer": "AdamW",
}
program(hyper_params)

# for down_sampled in [False, True]:
#     for polish_removed in [False, True]:
#         for numbers_as_word in [False, True]:
#             for threshold_for_sentences in [40, 60, 100, 900]:
#                 for output_dim in [32, 64, 128]:
#                     hyper_params = {
#                         "is_down_sampled": down_sampled,
#                         "polish_chars_removed": polish_removed,
#                         "numbers_replaced_with_single_word": numbers_as_word,
#                         "nr_of_epochs": 200,
#                         "test_val_size": 0.3,
#                         "val_size": 0.33,
#                         "threshold_of_cutting_sentences": threshold_for_sentences,
#                         "learning_rate": 0.001,
#                         "output_dim": output_dim,
#                         "batch_size": 128,
#                         "random_state": 42,
#                         "dense_layer_neurons": 64,
#                         # "lstm_units": lstm_units,
#                         "optimizer": "AdamW",
#                     }
#                     program(hyper_params)

# print(hyper_params["is_down_sampled"])
# hyper_params["is_down_sampled"] = True
# print(hyper_params["is_down_sampled"])
