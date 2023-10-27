from typing import Tuple, List

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def get_values_from_dataset(path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, sep=";")
    df = df.dropna()
    x = df["text_full"].tolist()
    y = df["label_high"].tolist()
    return x, y

if __name__ == "__main__":

    sentences, categories = get_values_from_dataset(path="DATASET.csv")

    sentences_split = []
    for sentence in sentences:
        sentences_split.append(sentence.split(" "))

    X_train, X_test, y_train, y_test = train_test_split(
        sentences_split, categories, test_size=0.33, random_state=125
    )

    # Build a Gaussian Classifier
    model = GaussianNB()

    # Model training
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
