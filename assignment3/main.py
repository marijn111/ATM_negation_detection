import os
from loguru import logger


from assignment3.process_data import *
from assignment3.data_exploration import data_exploration
from assignment3.load_data import preprocess_data
import pandas as pd

input_path = "./dataset/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"
output_path = "./dataset/processed_corpus.csv"


def load_file(filename: str, column_names: list) -> pd.DataFrame:
    """
    Reads file and name the columns. Return Dataframe
    """
    df = pd.read_csv(filename, sep='\t')
    df.columns = column_names
    return df


if __name__ == '__main__':
    """
    Main function with continues user question
    """
    #Read, explore and process data
    logger.info('Read training data')
    df = load_file('dataset/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', ['document', 'sentence_number', 'location', 'token', 'value'])
    data_exploration(df)
    preprocess_data(input_path, output_path)
    #Read training data
    training_data = load_file('dataset/processed_corpus.csv', ['document', 'sentence ID', 'token ID', 'token', 'cue', 'POS', 'LEMMA', 'TAG', 'DEP', 'STOP', 'NER', 'AFFIX', 'CONTR', 'EXPR'])

