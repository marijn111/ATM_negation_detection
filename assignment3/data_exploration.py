import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

#TODO: Aantal zinnen, aantal token, aantal negations, (word cloud)


def count_nans(df: pd.Dataframe, column:str) -> pd.DataFrame:

    return


def word_cloud(df: pd.Dataframe, column:str) -> pd.DataFrame:
    wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                          collocations=False, stopwords=STOPWORDS).generate(' '.join(df[column]))
    plot_cloud(wordcloud)


def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off");