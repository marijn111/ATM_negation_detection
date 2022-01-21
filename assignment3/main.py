import os
from loguru import logger


from assignment3.process_data import *
from assignment3.data_exploration import data_exploration
import pandas as pd

def file_names()-> list:
    path = ''
    file_list = []
    for root, dirs, files in os.walk("."):
        for filename in files:
            file_list.append(filename)
    return file_list


def load_files(file_list: list) -> pd.DataFrame:
    path = ''
    for file in file_list:
        logger.info(f"Reading {file}")
    return df


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
    logger.info('Read training data')
    df = load_file('dataset/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', ['document', 'sentence_number', 'location', 'token', 'value'])
    data_exploration(df)
    preprocess_data(df)

