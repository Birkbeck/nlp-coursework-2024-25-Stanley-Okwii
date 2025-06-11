from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score


random_seed = 26
n_features = 3000


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
    """
    print("F1 Score:", f1_score(y_test, predictions, average="macro"))
    print("Classification Report:")
    print(
        classification_report(y_test, predictions, zero_division=0)
    )  # Ignore division by zero warnings

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)
    report_evaluation_metrics(y_test, model_predictions)


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
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=300, random_state=random_seed)
    train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)

    # SVM Classifier (Linear Kernel)
    print("\nTraining SVM (Linear Kernel)...")
    svm = SVC(kernel="linear", random_state=random_seed)
    train_and_evaluate_model(svm, X_train, X_test, y_train, y_test)
