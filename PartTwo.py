from pathlib import Path
import pandas as pd


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
    # a) Read hansard40000 data and subsets and renames dataframe
    df = create_hansard_df()
    print('df.shape \n', df.shape)
