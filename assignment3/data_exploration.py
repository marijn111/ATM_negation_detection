import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from loguru import logger

#TODO: Aantal zinnen, aantal token, aantal negations, (word cloud)


def count_nans(df: pd.Dataframe, column:str) -> pd.DataFrame:

    return


def count_negations(df: pd.Dataframe) -> pd.DataFrame:
    negations = len(df[df['value']== 'B-NEG'])
    return negations


def count_tokens(df: pd.Dataframe) -> pd.DataFrame:
    tokens = len(df['token'])
    return tokens


def count_sentences(df: pd.Dataframe) -> pd.DataFrame:
    sentences = len(df['sentence_number'].unique())
    return sentences


def word_cloud(df: pd.Dataframe, column:str) -> pd.DataFrame:
    wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                          collocations=False, stopwords=STOPWORDS).generate(' '.join(df[column]))
    plot_cloud(wordcloud)


def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off");


def data_exploration(df: pd.Dataframe) -> pd.DataFrame:
    logger.info('Explore data')
    logger.info("amount of negation in dataset =" + count_negations(df))
    logger.info("amount of sentences in dataset =" + count_sentences(df))
    logger.info("amount of tokens in dataset =" + count_tokens(df))
    word_cloud(df, 'token')

    return df