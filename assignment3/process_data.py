import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#TODO: MAKE sentence

def make_sentence(df: pd.Dataframe) -> pd.DataFrame:
    sentence_numbers =
    return


def lowercase_column(df: pd.Dataframe, column:str) -> pd.DataFrame:
    df[column] = df[column].astype(str)
    df['lowered_text']= df[column].str.lower()
    return df


def remove_stopwords(df: pd.Dataframe, column:str) -> pd.DataFrame:
    stop = stopwords.words('english')
    newStopWords = []
    stop = stop.extend(newStopWords)
    df[column] = df[column].astype(str)
    df['cleaned_text'] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df


def tokenize_sentences(df: pd.Dataframe, column:str) -> pd.DataFrame:
    df[column] = df[column].astype(str)
    df['sentence'] = df.apply(lambda row: sent_tokenize(row[column]), axis=1)
    return df


def tokenize(df: pd.Dataframe, column:str) -> pd.DataFrame:
    df[column] = df[column].astype(str)
    df['tokenized'] = df.apply(lambda row: word_tokenize(row[column]), axis=1)
    return df


def lemmatizer(df: pd.Dataframe, column:str) -> pd.DataFrame:
    df['lemmatized'] = df.apply(lambda row: WordNetLemmatizer(row[column]), axis=1)
    return df


def stem_sentences(df: pd.Dataframe, column:str) -> pd.DataFrame:
    stemmer = SnowballStemmer("english")
    df['stemmed'] = df[column].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
    return df