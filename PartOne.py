import math
from pathlib import Path
from collections import Counter

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
    sentence_count = len(nltk.sent_tokenize(text))
    words = nltk.word_tokenize(text)
    word_count = len(words)
    syllable_count = sum(count_syl(word, d) for word in words if word.isalpha())

    if not sentence_count and not word_count:
        return 0.0
    return (0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59)


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    count = 0
    if word in d:
        count = len([pr for pr in d[word][0] if pr[-1].isdigit()]) # count vowels in the word's first pronunciation
    else:
        vowels = set("aeiouy") # y is sometimes considered a vowel especially in syllable estimation (https://www.grammarly.com/blog/grammar/vowels/)
        count = sum(curr in vowels and prev not in vowels for prev, curr in zip(' ' + word, word))
    return count


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
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
    """
    Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting DataFrame to a pickle file
    """
    # Create the pickles folder if it doesn't exist
    Path(store_path).mkdir(parents=True, exist_ok=True)
    pickle = Path(store_path / out_name)
    if not pickle.exists():
        df['parsed'] = df['text'].apply(lambda x: nlp(x))
        print("Pickle-ing dataframe...\n")
        pd.to_pickle(df, store_path / out_name)
        return df
    else:
        print("Using pickled dataframe...\n")
        _df = pd.read_pickle(store_path / out_name)
        return _df


def nltk_ttr(text):
    """
    Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.
    """
    type_token_ratio = 0.0
    tokens = nltk.word_tokenize(text)
    word_tokens = [token.lower() for token in tokens if not token.isalpha()] # filter out punctuations and ignore case
    num_tokens = len(word_tokens)
    num_unique_tokens = len(set(word_tokens))

    if num_tokens:
        type_token_ratio = num_unique_tokens/num_tokens
    return type_token_ratio


def get_ttrs(df):
    """
    helper function to add ttr to a dataframe
    """
    results = {}
    for _, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """
    helper function to add fk scores to a dataframe
    """
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for _, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def calculate_pmi_score(word, target_word, unigram_freq, bigram_freq):
    """
    Calculates pmi scores for each word and target_word pair using unigram and bigram frequencies.

    Reference: https://stackoverflow.com/questions/22118350/python-sentiment-analysis-using-pointwise-mutual-information
    """
    try:
        word_prob = unigram_freq[word] / float(sum(unigram_freq.values()))
        target_word_prob = unigram_freq[target_word] / float(sum(unigram_freq.values()))
        bigram_prob = bigram_freq[word] / float(sum(bigram_freq.values()))
        return round(math.log(bigram_prob / float(word_prob*target_word_prob)), 3)
    except Exception as exc:
        return 0.0


def subjects_by_verb_pmi(doc, target_verb):
    """
    Returns a list of the ten most common syntactic subjects target_verb in the text,
    ordered by their Pointwise Mutual Information (PMI) scores.
    """
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    unique_subjects = set()
    subjects = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB" and token.lemma_ == target_verb:
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ in ("NOUN", "PROPN", "PRON"):
                        unique_subjects.add(child.text.lower())
                        subjects.append(child.text.lower())
                        break # one subject per sentence

    unigram_freq = Counter(tokens)
    bigram_freq = Counter(unique_subjects)

    pmi_scores = [{subject: calculate_pmi_score(subject, target_verb, unigram_freq, bigram_freq)} for subject in subjects]
    top_10 = sorted(pmi_scores, key=lambda x: next(iter(x.values())), reverse=True)[:10]
    
    return top_10


def subjects_by_verb_count(doc, verb):
    """
    Extracts the most common subjects of a given verb in a parsed document.
    Returns a list.
    """
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            # Find subject dependencies (nsubj and nsubjpass)
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(child.text.lower())
    return [subject for (subject,_) in Counter(subjects).most_common(10)]


def adjective_counts(df):
    """
    Extracts the 10 most common adjectives in each parsed document.
    Returns a dictionary: {title: [(adj, freq), ...]}.
    """
    results = {}
    for _, row in df.iterrows():
        doc = row["parsed"]
        adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
        top_adjectives = Counter(adjectives).most_common(10)
        results[row["title"]] = top_adjectives
    return results


def common_syntactic_objects(doc):
    """
    Find the ten most common syntactic objects overall in the text
    """
    dependencies = Counter([token.dep_ for token in doc])
    top_dependencies = [dep.lower() for (dep,_) in dependencies.most_common(10)]
    return top_dependencies


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path)
    print(df.head(), "\n\n")
    nltk.download("cmudict")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    df = parse(df)
    print(df.head(), "\n\n")
    print(get_ttrs(df), "\n\n")
    print(get_fks(df), "\n\n")

    print(adjective_counts(df))

    # The title of each novel and a list of the ten most common syntactic objects overall in the text.
    for _, row in df.iterrows():
        print(row["title"], "\n")
        print(common_syntactic_objects(row["parsed"]))
        print("\n")

    # # The title of each novel and a list of the ten most common syntactic subjects of the verb ‘to hear’ (in any tense) in the text, ordered by their frequency.
    for _, row in df.iterrows():
        print(row["title"], "\n")
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")
    
    #  the ten most common syntactic subjects of the verb ‘to hear’ (in any tense) in the text, ordered by their Pointwise Mutual Information.
    for _, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
