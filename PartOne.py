#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import os
from pathlib import Path

import nltk
import spacy
import pandas as pd

# Download the spaCy English pipeline if not already present
english_pipeline = "en_core_web_sm"
spacy_info = spacy.info()

if(english_pipeline not in spacy_info.get('pipelines', {})):
    print("Downloading en_core_web_sm pipeline")
    spacy.cli.download(english_pipeline)

nlp = spacy.load(english_pipeline)
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    novels = []
    # Find all .txt files in the novels directory
    text_files = Path.glob(path, '*.txt')
    for f in text_files:
        with open(f, 'r') as file:
            filename = file.name.split("novels/")[1]
            base_name = filename.replace('.txt', '')
            parts = base_name.split('-')
            title = parts[0].replace('_', ' ')
            author = parts[1]
            year = parts[2]
            content = file.read()

            novel_info = {
                "text": content,
                "title": title,
                "author": author,
                "year": year,
            }
            novels.append(novel_info)
    df = pd.DataFrame(novels)
    df.sort_values(by='year').reset_index(drop=True, inplace=True)
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    os.makedirs(store_path, exist_ok=True) # Does not error if directory exists
    pickle = Path(store_path / out_name)
    if not pickle.exists():
        try:
            df['parsed_docs'] = df['text'].apply(lambda x: nlp(x).to_bytes())
        except Exception as exc:
            raise exc
        df.to_pickle(store_path / out_name)
    # else:
    #     df = pd.read_pickle(store_path / out_name)
    #     df['parsed_docs'] = df['parsed_docs'].apply(lambda x: spacy.tokens.Doc(x.vocab).from_bytes(x))
    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    # print(df.head())
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

