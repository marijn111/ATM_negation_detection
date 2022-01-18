import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer



def tokenize(df: pd.Dataframe, column:str) -> pd.DataFrame:
    df['tokenized'] = df.apply(lambda row: word_tokenize(row[column]), axis=1)
    return df


def lemmatize(df: pd.Dataframe, column:str) -> pd.DataFrame:
    df['lemmatized'] = df.apply(lambda row: lemmatize(row[column]), axis=1)
    return df