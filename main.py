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

CONNECTING_WORDS = ['czy', 'w', 'a', 'o', 'po', 'ich', 'jest', 'dla', 'i', '', 'z', 'na', 'do', 'że', 'od', 'by', 'je',
                    'się', 'żeby', 'co', 'które', 'te', 'lub', 'niż', 'przez', 'gdy', 'bo', 'jak', 'być', 'bez', 'albo',
                    'tak', 'ten', 'tylko', 'więc', 'już', 'która', 'mi', 'nad', 'ale', 'poza', 'raz', 'razem', 'to', 'wiec',
                    'właśnie', 'wszystko', 'każdy', 'kiedy', 'lecz', 'mają', 'może', 'na', 'nawet', 'nim', 'nów', 'od', 'około',
                    'prawie', 'przecież', 'są', 'tego', 'to', 'trzeba', 'tu', 'w', 'we', 'z', 'za']


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
    text = remove_connecting_words(text)
    return text


def remove_connecting_words(text):
    text = text.split()
    text = [word for word in text if word not in CONNECTING_WORDS]
    text = ' '.join(text)
    return text


if __name__ == "__main__":
    sentences, categories = get_values_from_dataset(path="dataset.csv")
    processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_sentences, categories, test_size=0.15, random_state=125
    )

    tfidf_vectorizer = TfidfVectorizer()
    x_train_featured = tfidf_vectorizer.fit_transform(X_train)
    x_test_featured = tfidf_vectorizer.transform(X_test)

    model = MultinomialNB()

    model.fit(x_train_featured, y_train)

    y_pred = model.predict(x_test_featured)
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print(X_test[3])
    predicted = model.predict(x_test_featured[3])
    print("Actual Value:", y_test[3])
    print("Predicted Value:", predicted[0])

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
