from typing import Tuple, List

import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


def get_values_from_dataset(path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, sep=";")
    df = df.dropna()
    x = df["text_full"].tolist()
    y = df["label_high"].tolist()
    return x, y


def preprocess_sentence(text):
    # Convert to lowercase
    text = text.lower()
    text = re.sub(r'[^a-zA-Ząćęłńóśźż\s]', '', text)
    text = ' '.join(text.split())
    return text


if __name__ == "__main__":
    sentences, categories = get_values_from_dataset(path="DATASET.csv")
    processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_sentences, categories, test_size=0.33, random_state=125
    )

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    model = MultinomialNB()

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print(X_test[3])
    predicted = model.predict(X_test_tfidf[3])
    print("Actual Value:", y_test[3])
    print("Predicted Value:", predicted[0])

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
