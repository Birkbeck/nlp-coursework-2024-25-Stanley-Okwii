from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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

if __name__ == "__main__":
    # (a) Read hansard40000 data and subsets and renames dataframe
    df = create_hansard_df()
    print(f"df.shape:  {df.shape} \n")


    # (b) Vectorise the speeches with TfidfVectorizer
    random_seed = 26
    n_features = 3000
    vectorizer = TfidfVectorizer(
        max_features=n_features, stop_words="english"
    )
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']

    # split dataset into train(80%) and test(20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_seed
    )
    print(f"Train dataset shape: {X_train.shape} \n")
    print(f"Test dataset shape: {X_test.shape} \n")  

