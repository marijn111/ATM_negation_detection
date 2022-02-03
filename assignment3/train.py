import pickle

from joblib import dump, load
import math

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.pyplot as plt

import pandas as pd


class TrainModel:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = None
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_train_all = []
        self.y_train_all = []
        self.y_pred = []
        self.dataset_split = 70
        self.save_directory = 'models'
        self.df_error = pd.DataFrame()

    def load_processed_corpus(self, path):
        df = pd.read_csv(path, sep='\t',
                         names=['document', 'sentence ID', 'token ID', 'token', 'cue', 'POS', 'LEMMA', 'TAG', 'DEP', 'STOP',
                                'NER', 'AFFIX', 'CONTR', 'EXPR'], header=None)

        self.df = df

    def model_init(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            # all_possible_transitions=True
        )

        self.model = crf

    def word2features(self, df_token):

        features = {
            'token': df_token['token'].values[0],
            'pos': df_token['POS'].values[0],
            'lemma': df_token['LEMMA'].values[0],
            'tag': df_token['TAG'].values[0],
            'dep': df_token['DEP'].values[0],
            'isstop': df_token['STOP'].values[0],
            'ner': df_token['NER'].values[0],
            'isaffix': df_token['AFFIX'].values[0],
            'iscontr': df_token['CONTR'].values[0],
            'isexpr': df_token['EXPR'].values[0]

        }

        return features

    def sent2features(self, sent):
        return [self.word2features(sent.loc[sent['token ID'] == token_id, :]) for token_id in sent['token ID'].unique()]

    def sent2labels(self, sent):
        return [sent.at[i, 'cue'] for i in sent.index]

    def sent2tokens(self, sent):
        return [token for token, postag, label in sent]

    def get_train_test_data(self):
        documents = self.df['document'].unique()

        for document in documents:
            print(f'[INFO] Currently processing document: {document}')
            doc_df = self.df.loc[self.df['document'] == document, :]
            sentences = doc_df['sentence ID'].unique()

            for sentence_id in sentences:
                sentence_df = doc_df.loc[doc_df['sentence ID'] == sentence_id, :]
                self.X_train_all.append(self.sent2features(sentence_df))
                self.y_train_all.append(self.sent2labels(sentence_df))

    def split_train_test_data(self):
        dataset_length = len(self.X_train_all)
        split = math.floor(dataset_length * (self.dataset_split/100))

        self.X_train = self.X_train_all[:split]
        self.X_test = self.X_train_all[split:]
        self.y_train = self.y_train_all[:split]
        self.y_test = self.y_train_all[split:]

        self.store_data('x_test.pkl', self.X_test)
        self.store_data('y_test.pkl', self.y_test)

    def fit_model(self):
        print('[INFO] Fitting the model...')
        self.model.fit(self.X_train, self.y_train)

    def save_model(self):
        print('[INFO] Saving the model...')
        dump(self.model, f'./{self.save_directory}/crf.joblib')

    def load_model(self):
        self.model = load(f'./{self.save_directory}/crf.joblib')

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        self.store_data('y_pred.pkl', self.y_pred)

    def evaluation(self):
        print('[INFO] Running the evaluation...')
        labels = list(self.model.classes_)
        labels.remove('O')

        print('[INFO] flat f1_score...')
        print(metrics.flat_f1_score(self.y_test, self.y_pred,
                              average='weighted', labels=labels))

        print('\n')
        print('[INFO] per-class results...')
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(
            self.y_test, self.y_pred, labels=sorted_labels
        ))

    def store_data(self, filename, data):
        with open(f'./{self.save_directory}/{filename}', 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        with open(f'./{self.save_directory}/{filename}', 'rb') as f:
            data = pickle.load(f)
        return data

    def get_data(self):
        predictions = [item for sublist in self.y_pred for item in sublist]
        len_train = len(self.df) - len(predictions)
        df = self.df[len_train:]
        df['predictions'] = predictions
        prediction_list = []
        for index, row in df.iterrows():
            if row['cue'] == row['predictions']:
                prediction_list.append('True')
            else:
                prediction_list.append('False')
        df['correct_pred'] = prediction_list
        self.df_error = df
        df.to_csv('df_predictions.csv')

    def confusion_matrix(self):
        contingency_matrix = pd.crosstab(self.df_error['predictions'], self.df_error['cue'])
        plt.clf()
        ax = sn.heatmap(contingency_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix of CRF model\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=100)
        plt.show()


def main(input_path):
    model_class = TrainModel()
    model_class.load_processed_corpus(input_path)
    model_class.model_init()
    model_class.get_train_test_data()
    model_class.split_train_test_data()
    model_class.fit_model()
    model_class.save_model()
    #model_class.load_model()
    model_class.predict()
    model_class.evaluation()
    model_class.get_data()
    model_class.confusion_matrix()




if __name__ == '__main__':
    #processed_corpus_path = "./dataset/processed_corpus_bio.csv"
    processed_corpus_path = "./dataset/processed_corpus.csv"
    main(processed_corpus_path)
