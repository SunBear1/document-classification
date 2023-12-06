from typing import Tuple, List, Any, Dict
import requests
import wandb
import re
import statistics
import pprint
import numpy as np
import pandas as pd
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

from src.neural_network.processing import downsample_dataset, filter_dataset, cut_too_long_sentences, \
    replace_polish_letters, replace_numbers_with_word, remove_connecting_words, tokenize_sentences, encode_labels

# wandb.login()

DATASET_FILE_URL = "https://raw.githubusercontent.com/SunBear1/document-classification/master/data/complete_dataset.csv"
CONNECTING_WORDS_FILE_URL = "https://raw.githubusercontent.com/SunBear1/document-classification/master/data/connecting_words.lst"
DATASET_FILE_PATH = "../../data/complete_dataset.csv"
CONNECTING_WORDS_FILE_PATH = "../../data/connecting_words.lst"
LABELS = {
    "prawo medyczne": 0,
    "prawo pracy": 1,
    "prawo cywilne": 2,
    "prawo administracyjne": 3,
    "prawo farmaceutyczne": 4,
    "prawo karne": 5,
}


def download_dataset():
    response = requests.get(DATASET_FILE_URL)
    if response.status_code == 200:
        with open(DATASET_FILE_PATH, 'wb') as file:
            file.write(response.content)
        print(f"File '{DATASET_FILE_PATH}' has been successfully downloaded.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    response = requests.get(CONNECTING_WORDS_FILE_URL)
    if response.status_code == 200:
        with open(CONNECTING_WORDS_FILE_PATH, 'wb') as file:
            file.write(response.content)
        print(f"File '{CONNECTING_WORDS_FILE_PATH}' has been successfully downloaded.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def prepare_data(hyper_parameters: Dict, connecting_words: List[str]) -> Tuple[Any, Any]:
    df = pd.read_csv(DATASET_FILE_PATH, sep=",")
    df = df.dropna()
    df = filter_dataset(df)
    if hyper_parameters["is_down_sampled"]:
        df = downsample_dataset(df, 225)

    # TODO wywalenie powtórek, dodanie ewentualnie nowej kolumny do labelek, wywalić zdania które są skaftem
    # TODO Zaciąganie kolumny text zamiast text_full i rozdział zdań na mniejsze. Ref multi_label_network.py

    sentences = df["text_full"].tolist()
    categories = df["label_high"].tolist()
    # sentences, main_categories, sub_categories = cut_too_long_sentences(sentences=sentences, # TODO fix this
    #                                                                     main_categories=main_categories,
    #                                                                     sub_categories=sub_categories,
    #                                                                     threshold=hyper_parameters[
    #                                                                         "threshold_of_cutting_sentences"])

    for i in range(len(sentences)):
        sentence = sentences[i]

        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Ząćęłńóśźż\s]', '', sentence)
        if hyper_parameters["polish_chars_removed"]:
            sentence = replace_polish_letters(sentence)
        if hyper_parameters["numbers_replaced_with_single_word"]:
            sentence = replace_numbers_with_word(sentence)
        sentence = ' '.join(sentence.split())  # TODO  questionable
        sentence = remove_connecting_words(sentence, connecting_words)

        sentences[i] = sentence

    sentence_lengths = []
    max_sen_word_count = 0
    _sum = 0
    for sentence in sentences:
        words = len(sentence.split())
        _sum += words
        sentence_lengths.append(words)
        if words > max_sen_word_count:
            max_sen_word_count = words

    print(f"List of categories: {list(set(categories))}")
    print(f"Number of occurrences per main category")
    print(pd.Series(categories).value_counts())
    print("Max words in a sentence", max_sen_word_count)
    print("Average words for sentence", statistics.mean(sentence_lengths))
    print("Median of words for sentence", statistics.median(sorted(sentence_lengths)))
    print("Dataset sample:")
    for _ in range(10):
        i = np.random.randint(0, len(sentences))
        print(f"Main category: {categories[i]},  Sentence: {sentences[i]}")

    return sentences, categories


def solve_the_document_classification_problem(hyper_parameters: Dict, wandb_group: str):
    print("---------------------PREPARING-ENVIRONMENT---------------------")
    pprint.pprint(hyper_parameters)

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('NO GPU found')
    else:
        print('Found GPU at: {}'.format(device_name))

    # download_dataset() # jeśli chcesz żeby ciągnął z neta to odkomentuj
    with open(CONNECTING_WORDS_FILE_PATH, "r") as f:
        connecting_words = f.read().splitlines()

    print("-----------------------WELCOME-----------------------")
    processed_sentences, categories = prepare_data(hyper_parameters, connecting_words)

    x_train_human_readable, x_temp_human_readable = train_test_split(processed_sentences,
                                                                     test_size=hyper_parameters["test_val_size"],
                                                                     shuffle=False)
    x_val_human_readable, x_test_human_readable = train_test_split(x_temp_human_readable,
                                                                   test_size=hyper_parameters["val_size"],
                                                                   shuffle=False)

    print("--------------------CLASS-WEIGHTS--------------------")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(categories),
                                         y=categories)
    print("class_weights", class_weights)

    print("--------------------TOKENIZATION--------------------")
    main_padded_sequences, word_index = tokenize_sentences(sentences=processed_sentences)

    x_train, x_temp, y_train, y_temp = train_test_split(main_padded_sequences,
                                                        categories,
                                                        test_size=hyper_parameters[
                                                            "test_val_size"])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                    test_size=hyper_parameters["val_size"])

    print("-------------Y-SET-------------")
    y_train = encode_labels(y_train, LABELS)
    y_val = encode_labels(y_val, LABELS)
    y_test = encode_labels(y_test, LABELS)

    print("-------------MODEL-SPEC-------------")
    model = create_model(input_dim=len(word_index) + 1, input_length=x_train.shape[1],
                         num_classes=len(LABELS), output_dim=hyper_parameters["output_dim"])  # TODO co robi output dim?

    hyper_parameters["model_architecture"] = model.get_config()

    def lr_schedule(epoch):
        return hyper_parameters["learning_rate"] * 0.1 ** (epoch // hyper_parameters["scheduler_threshold"])

    lr_scheduler = LearningRateScheduler(lr_schedule)

    if hyper_parameters["optimizer"] == "AdamW":
        optimizer = AdamW(learning_rate=hyper_parameters["learning_rate"])
    if hyper_parameters["optimizer"] == "Adam":
        optimizer = Adam(learning_rate=hyper_parameters["learning_rate"])

    model.compile(optimizer=optimizer, loss=hyper_parameters["loss"],
                  metrics=hyper_parameters["metrics"])

    model.summary()
    # wandb.init(project="IUI&PUG", entity="gourmet", config=hyper_parameters, group=wandb_group)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=hyper_parameters["nr_of_epochs"],
              batch_size=hyper_parameters["batch_size"], class_weight=dict(enumerate(class_weights)),
              callbacks=[EarlyStopping(monitor='val_loss', patience=200, min_delta=0.0001
                                       # )
                                       # ,
                                       #       WandbCallback(save_model=False, log_weights=True,
                                       #                     labels=LABELS.keys(), input_type="auto",
                                       #                     output_type="auto"
                                       ), lr_scheduler])
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    # wandb.log({'test_accuracy': test_accuracy})
    # wandb.finish()

    # AFTER TRAINING
    if hyper_parameters["post_training_info"]:
        predictions = model.predict(x_val)
        y_pred = np.argmax(predictions, axis=1)

        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(16, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS.keys(),
                    yticklabels=LABELS.keys())
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        predictions = model.predict(x_val)
        y_pred = np.argmax(predictions, axis=1)
        misclassified_idx = np.where(y_pred != y_val)[0]
        print("label_to_index", LABELS)
        key_list = list(LABELS.keys())
        val_list = list(LABELS.values())

        for idx in misclassified_idx[:10]:
            print(f"True label: {key_list[val_list.index(y_val[idx])]}, Predicted label: "
                  f"{key_list[val_list.index(y_pred[idx])]}, Sentence: {x_val_human_readable[idx]}")


def create_model(input_dim: int, input_length: int, num_classes: int, output_dim: int):
    model = Sequential()  # to jest zawsze
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))  # to jest zawsze
    model.add(GlobalMaxPooling1D())  # tutaj potrzebne jest coś to zmieni wymiar z 3D na 2D
    #model.add(Dropout(0.1))  # element do dospermiania
    model.add(Dense(32, activation="relu"))  # element do dospermiania
    #model.add(Dropout(0.1))  # element do dospermiania
    model.add(Dense(num_classes, activation='softmax'))  # to jest zawsze
    return model


hyper_params = {
    "is_down_sampled": False,
    "polish_chars_removed": False,
    "numbers_replaced_with_single_word": False,
    "nr_of_epochs": 100,
    "test_val_size": 0.3,
    "val_size": 0.33,
    "threshold_of_cutting_sentences": 25,
    "learning_rate": 0.001,
    "output_dim": 64,  # TODO to można dospermić
    "batch_size": 128,
    "model_config": create_model,
    "optimizer": "AdamW",
    "scheduler_threshold": 30,  # jak chcesz go wyłączyć to daj wielką liczbe
    "loss": "categorical_crossentropy",  # TODO sprawdzić inne lossy jak np sparse_categorical_crossentropy
    "metrics": ["accuracy"],  # TODO sprawdzić inne metryki jak np sparse_categorical_accuracy albo categorical_accuracy
    "post_training_info": False,
    "wandb_group": "LSTM"  # pamiętaj o zmianie tego kiedy zmieniasz model z dense na lstm
}

solve_the_document_classification_problem(hyper_parameters=hyper_params, wandb_group="LSTM")
