import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from loguru import logger

#TODO: Aantal zinnen, aantal token, aantal negations, (word cloud)


def count_nans(df: pd.DataFrame, column:str) -> pd.DataFrame:

    return


def count_negations(df: pd.DataFrame) -> pd.DataFrame:
    negations = len(df[df['value']== 'B-NEG'])
    return negations


def count_tokens(df: pd.DataFrame) -> pd.DataFrame:
    tokens = len(df['token'])
    return tokens


def count_sentences(df: pd.DataFrame) -> pd.DataFrame:
    sentences = len(df['sentence_number'].unique())
    return sentences


def word_cloud(df: pd.DataFrame, column:str) -> pd.DataFrame:
    wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                          collocations=False, stopwords=STOPWORDS).generate(' '.join(df[column]))
    plot_cloud(wordcloud)


def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off");


def data_exploration(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Explore data')
    logger.info("amount of negation in dataset = " + str(count_negations(df)))
    logger.info("amount of sentences in dataset = " + str(count_sentences(df)))
    logger.info("amount of tokens in dataset = " + str(count_tokens(df)))
    word_cloud(df, 'token')

    return df