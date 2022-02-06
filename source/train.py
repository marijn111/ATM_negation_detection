import pickle

import scipy.stats
from joblib import dump, load
import math

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.metrics import make_scorer

import pandas as pd

plt.style.use('ggplot')


class TrainModel:
    """
    This is the main CRF model, and all helper functions included. Each function has its own abstract usecase so that
    it remains clear what happens where in the code.
    """
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
        self.save_directory = 'test_set_cardboard'

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

        self.store_data('x_train.pkl', self.X_train)
        self.store_data('y_train.pkl', self.y_train)

    def get_test_data(self):
        self.X_test = self.X_train_all
        self.y_test = self.y_train_all

        self.store_data('x_test_cardboard.pkl', self.X_test)
        self.store_data('y_test_cardboard.pkl', self.y_test)

    def fit_model(self):
        print('[INFO] Fitting the model...')
        self.model.fit(self.X_train, self.y_train)

    def save_model(self):
        print('[INFO] Saving the model...')
        dump(self.model, f'./models/crf.joblib')

    def load_model(self):
        self.model = load(f'./models/crf.joblib')

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        self.store_data('y_pred_circle.pkl', self.y_pred)

    def evaluation(self):
        print('[INFO] Running the evaluation...')
        labels = list(self.model.classes_)
        labels.remove('O')

        print('[INFO] flat f1_score...')
        print(metrics.flat_f1_score(self.y_test, self.y_pred,
                              average='weighted', labels=labels))

        print('\n')
        print('[INFO] per-class results...')
        self.sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(
            self.y_test, self.y_pred, labels=self.sorted_labels
        ))

    def store_data(self, filename, data):
        with open(f'./{self.save_directory}/{filename}', 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        with open(f'./{self.save_directory}/{filename}', 'rb') as f:
            data = pickle.load(f)
        return data

    def hyperopt(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        labels = list(self.model.classes_)
        labels.remove('O')

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=labels)

        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=1,
                                n_iter=50,
                                scoring=f1_scorer)

        rs.fit(self.X_train, self.y_train)
        print(rs.best_params_)

        self.rs = rs

    def check_parameter_space(self):
        print('best params:', self.rs.best_params_)
        print('best CV score:', self.rs.best_score_)
        print('model size: {:0.2f}M'.format(self.rs.best_estimator_.size_ / 1000000))

        _x = [s['c1'] for s in self.rs.cv_results_['params']]
        _y = [s['c2'] for s in self.rs.cv_results_['params']]
        _c = [s for s in self.rs.cv_results_['mean_test_score']]

        fig = plt.figure()
        fig.set_size_inches(12, 12)
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('C1')
        ax.set_ylabel('C2')
        ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
            min(_c), max(_c)
        ))

        ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])

        fig.savefig('./plots/search_space.png')

        print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    def check_best_estimator(self):
        labels = list(self.model.classes_)
        labels.remove('O')
        self.sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        self.best_estimator = self.rs.best_estimator_
        y_pred = self.best_estimator.predict(self.X_test)
        print(metrics.flat_classification_report(
            self.y_test, y_pred, labels=self.sorted_labels, digits=3
        ))

    def print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def check_classifier_learning(self):
        print("Top likely transitions:")
        self.print_transitions(Counter(self.best_estimator.transition_features_).most_common(20))

        print("\nTop unlikely transitions:")
        self.print_transitions(Counter(self.best_estimator.transition_features_).most_common()[-20:])

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def check_state_features(self):
        print("Top positive:")
        self.print_state_features(Counter(self.best_estimator.state_features_).most_common(30))

        print("\nTop negative:")
        self.print_state_features(Counter(self.best_estimator.state_features_).most_common()[-30:])


def main(input_path):
    model_class = TrainModel()
    model_class.load_processed_corpus(input_path)
    model_class.model_init()
    model_class.get_train_test_data()
    model_class.get_test_data()
    # model_class.fit_model()
    # model_class.save_model()
    # model_class.predict()
    # model_class.evaluation()

    # model_class.load_model()


if __name__ == '__main__':
    processed_corpus_path = "./dataset/processed_corpus_test_set_cardboard.csv"
    main(processed_corpus_path)
