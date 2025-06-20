import re
from pathlib import Path
from itertools import chain

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer


random_seed = 26
n_features = 3000
stop_words = stopwords.words('english')
stemmer = PorterStemmer()


def create_hansard_df():
    """
    Read hansard40000 data and subsets and renames dataframe

    Returns:
        df : subset df with renamed column values
    """
    filepath = Path.cwd() / "p2-texts" / "hansard40000.csv"
    df = pd.read_csv(filepath)

    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

    party_counts = df['party'].value_counts()
    top_4_parties = party_counts[party_counts.index != 'Speaker'].nlargest(4).index
    df = df[df['party'].isin(top_4_parties)]

    df = df[df['speech_class'].str.lower() == 'speech']
    df = df[df['speech'].str.len() >= 1000]

    return df

def vectorize_and_split_dataset(vectorizer, df):
    """
    vectorizes speech text and splits the dataset into train and test portions.

    args:
        vectorizer: TfidfVectorizer instance
        df: hansard dataframe
    
    returns: List containing train-test split of inputs
    """
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']

    # split dataset into train(80%) and test(20%)
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_seed
    )

def report_evaluation_metrics(y_test, predictions):
    """
    prints out evaluation metrics for a given

    args:
        y_test:
        predictions:
    returns: None
    """
    print("F1 Score:", f1_score(y_test, predictions, average="macro"))
    print("Classification Report:")
    print(
        classification_report(y_test, predictions, zero_division=0)
    )  # Ignore division by zero warnings

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Fits the model, evaluates it against test data and prints the report

    Args:
        model, X_train, X_test, y_train, y_test

    Returns: None
    """
    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)
    report_evaluation_metrics(y_test, model_predictions)

def super_tokenizer(text):
    """
    custom text tokenizer function

    Args:
        text: str
    Returns
        combined list of unigrams, bigrams and trigrams.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower()) # Ignore specials characters like punctuations and make text case insensitive
    tokens = word_tokenize(text)

    unigrams = [stemmer.stem(word) for word in tokens if (len(word) > 2 and word not in stop_words)] # Ignore 2 character words and stop words
    bigrams = ['_'.join(gram) for gram in ngrams(unigrams, 2)]
    trigrams = ['_'.join(gram) for gram in ngrams(unigrams, 3)]

    result = list(chain(unigrams, bigrams, trigrams))
    return result


if __name__ == "__main__":
    # (a) Read hansard40000 data and subsets and renames dataframe
    df = create_hansard_df()
    print(f"df.shape:  {df.shape} \n")


    # (b) Vectorize the speeches with TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=n_features, stop_words="english"
    )
    X_train, X_test, y_train, y_test = vectorize_and_split_dataset(vectorizer, df)


    # (c) Train RandomForest and SVM with linear kernel classifiers
    print("Fit Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=300, random_state=random_seed)
    train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)

    # SVM Classifier (Linear Kernel)
    print("\nFit SVM (Linear Kernel)...")
    svm = SVC(kernel="linear", random_state=random_seed)
    train_and_evaluate_model(svm, X_train, X_test, y_train, y_test)


    # (d) Classifiers using Tfidfvectorizer with unigrams
    ngram_vectorizer = TfidfVectorizer(
        max_features=n_features, stop_words="english", ngram_range=(1, 3)
    )
    X_train, X_test, y_train, y_test = vectorize_and_split_dataset(ngram_vectorizer, df)
    print("\nFit Random Forest Classifier that uses ngram vectorizer...")
    rf = RandomForestClassifier(n_estimators=300, random_state=random_seed)
    train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)

    # SVM Classifier (Linear Kernel)
    print("\nFit SVM (Linear Kernel) that uses ngram vectorizer...")
    svm = SVC(kernel="linear", random_state=random_seed)
    train_and_evaluate_model(svm, X_train, X_test, y_train, y_test)


    # # (e) Classifiers using Tfidfvectorizer with custom tokenizer
    custom_vectorizer = TfidfVectorizer(
        tokenizer=super_tokenizer,
        max_features=n_features,
        sublinear_tf=True,
        token_pattern=None,  # disable default regex warning
    )
    X_train, X_test, y_train, y_test = vectorize_and_split_dataset(custom_vectorizer, df)
    print("\nFit Random Forest Classifier with super tokenizer...")
    rf = RandomForestClassifier(n_estimators=300, random_state=random_seed)
    train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)

    # SVM Classifier (Linear Kernel) - Best performing classifier
    print("\nFit SVM (Linear Kernel) with super tokenizer...")
    svm = SVC(kernel="linear", random_state=random_seed)
    train_and_evaluate_model(svm, X_train, X_test, y_train, y_test)
