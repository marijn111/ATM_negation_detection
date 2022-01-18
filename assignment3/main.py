from loguru import logger
import pandas as pd

from assignment3.process_data import *


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
