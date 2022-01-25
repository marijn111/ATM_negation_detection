from joblib import dump, load

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter

import pandas as pd


class TrainModel:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = None
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

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
            'token': df_token['token'],
            'pos': df_token['POS'],
            'lemma': df_token['LEMMA'],
            'tag': df_token['TAG'],
            'dep': df_token['DEP'],
            'isstop': df_token['STOP'],
            'ner': df_token['NER'],
            'isaffix': df_token['AFFIX'],
            'iscontr': df_token['CONTR'],
            'isexpr': df_token['EXPR']

        }

        return features

    def sent2features(self, sent):
        return [self.word2features(sent.iloc[i, :]) for i in sent.index]

    def sent2labels(self, sent):
        return [sent.iloc[i, sent.columns.get_loc('cue')] for i in sent.index]

    # def sent2tokens(self, sent):
    #     return [token for token, postag, label in sent]

    def get_train_test_data(self):
        documents = self.df['document'].unique()

        for document in documents:
            doc_df = self.df.loc[self.df['document'] == document, :]
            sentences = doc_df['sentence ID'].unique()

            for sentence_id in sentences:
                sentence_df = doc_df.loc[doc_df['sentence ID'] == sentence_id, :]
                self.X_train.append(self.sent2features(sentence_df))
                self.y_train.append(self.sent2labels(sentence_df))

        # X_test = [sent2features(s) for s in test_sents]
        # y_test = [sent2labels(s) for s in test_sents]

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)

    def save_model(self):
        dump(self.model, './models/crf.joblib')

    def load_model(self):
        self.model = load('./models/crf.joblib')

    def evaluation(self):
        labels = list(self.model.classes_)
        labels.remove('O')

        y_pred = self.model.predict(self.X_test)
        print('flat f1_score')
        print(metrics.flat_f1_score(self.y_test, y_pred,
                              average='weighted', labels=labels))

        print('per-class results')
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(
            self.y_test, y_pred, labels=sorted_labels, digits=3
        ))


def main(input_path):
    model_class = TrainModel()
    model_class.load_processed_corpus(input_path)
    return True


if __name__ == '__main__':
    processed_corpus_path = "./dataset/processed_corpus.csv"
    main(processed_corpus_path)
