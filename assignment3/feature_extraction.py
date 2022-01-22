import pandas as pd
from nltk import pos_tag, pos_tag_sents

#TODO: MAKE sentence - pos tagger - dependancy tree - N-grams - affix detection - Count of negation cue sub-tokens - Chunking -

negation_suffix = ['less']
negation_prefix = ['de', 'dis', 'il', 'im' ,'in', 'ir', 'mis', 'non', 'un', 'anti']


def chunking():

    return


def pos_tagger_sentences(df: pd.DataFrame, column:str) -> pd.DataFrame:
    texts = df[column].tolist()
    tagged_texts = pos_tag_sents(texts)
    df['POS'] = tagged_texts
    return df


def pos_tagger(df: pd.DataFrame, column:str) -> pd.DataFrame:
    texts = df[column].tolist()
    tagged_texts = pos_tag(texts)
    df['POS'] = tagged_texts
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    return df