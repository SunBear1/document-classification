from typing import Tuple, List

import re
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

with open("data/connecting_words.lst", "r") as f:
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


def get_values_from_dataset(path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, sep=";")
    df = df.dropna()
    df = filter_dataset(df)
    balanced_df = downsample_dataset(df, 225)
    x = balanced_df["text_full"].tolist()
    y = balanced_df["label_high"].tolist()
    return x, y


def preprocess_sentence(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Ząćęłńóśźż\s]', '', text)
    text = ' '.join(text.split())
    text = remove_connecting_words(text)
    return text


def remove_connecting_words(text):
    text = text.split()
    text = [word for word in text if word not in CONNECTING_WORDS]
    text = ' '.join(text)
    return text


if __name__ == "__main__":
    sentences, categories = get_values_from_dataset(path="data/dataset.csv")
    print("---------------------Welcome---------------------")
    print(f"List of categories: {list(set(categories))}")
    print(f"Number of occurrences per category")
    print(pd.Series(categories).value_counts())
    processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_sentences, categories, test_size=0.10, random_state=125
    )

    tfidf_vectorizer = TfidfVectorizer()
    x_train_featured = tfidf_vectorizer.fit_transform(X_train)
    x_test_featured = tfidf_vectorizer.transform(X_test)

    model = MultinomialNB()

    model.fit(x_train_featured, y_train)

    y_pred = model.predict(x_test_featured)
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print("---------------------Results---------------------")
    print(f"Sample text:  {X_test[3]}")
    predicted = model.predict(x_test_featured[3])
    print("Actual Value:", y_test[3])
    print("Predicted Value:", predicted[0])

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("-------------------------------------------------")
